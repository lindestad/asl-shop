"""Causal streaming student for the app's live path, distilled from an offline OpenFS teacher.

Accuracy-versus-latency study on one architecture family: the runtime's own causal TCN
(model.neural.FingerspellingNet, 91-column asl-shop-xy-wrist-v2 contract) with a dilation stack
that fixes the causal receptive field, so receptive field, parameter count and per-arrival CPU
cost can be traded against CER on the FSboard test split.  `--receptive-frames` takes one value or
several; each value is trained in turn, kept under `--max-parameters`, and recorded with its own
checkpoint, algorithmic latency (frames, ms at 10-15 Hz capture, wall-clock history on the app's
nominal 30 Hz model grid, buffered frames the app must hold) and metrics.

License separation is explicit, because the teacher side is research-only ("Commercial use of the
source code, model checkpoints, or derivative works is strictly prohibited" —
data/external/openfs/source/LICENSE, data/openfs-checkpoints/*):
- the student starts from random weights, trains only on our own packed FSboard timeline, and never
  loads an upstream tensor; the teacher is used only offline, as a source of per-frame labels and
  per-frame posteriors.  Nothing under model/ imports the teacher, and this script never imports
  scripts/finetune_openfs_transfer.py;
- every teacher artifact is recorded by path plus sha256 (plus the refined-label file's sha256) in
  the run report and copied into the student checkpoint payload, with research_only=true whenever
  teacher-derived signal reached the weights, so any reviewer can trace a checkpoint back to the
  research weights that taught it — or confirm that none did;
- `--teacher` accepts the released checkpoint directory or a fine-tuned teacher checkpoint; it is
  never written into a payload carrying the app contract.

Index mapping lives in exactly one place: the teacher consumes compressed detection order (one token
per signing-hand detection, arange positional index), the student consumes gap-preserving packed
rows.  teacher_tokens() returns the tokens *and* the packed row each token came from, and
student_rows() is its inverse; align_rows() re-applies the packer's temporal augmentation to those
rows.  Nothing else converts between the two orders, and padded batch slots never enter a mapping.

Losses: CTC on the packed transcripts; an auxiliary per-frame cross-entropy against teacher-derived
frame labels (data/openfs-refined/<split>.npz from scripts/refine_openfs_frame_labels.py; a clip it
does not cover falls back to the pack's own per-frame targets, then to <split>_handshape.npz when
that pack carries frame indices, and the per-clip counts land in the report); and a KL term against
teacher posteriors restricted to frames where the teacher holds a letter (--distillation-blank also
distills blank/nothing frames).  --no-distill drops the KL term, --no-frame-labels drops the
per-frame loss entirely, so the two ablations isolate soft labels and teacher frame labels.

Real runs (VPS V100-SXM3; per-row accuracy and latency in docs/streaming-latency-2026-10-19.md; 61
frames reproduces the shipped model's parameter count exactly, so the distill/no-distill pair is a
fixed-latency comparison).  The pairing that produced usable supervision is the released checkpoint
with --teacher-input-representation compressed, which needs --openfs-source and --trusted-teacher
and does exercise the compressed<->gaps index mapping.  Our own gaps-timeline recogniser
(data/models/openfs-recognizer-gaps/recognizer.pt, validation CER 7.08%) consumes the same 91-channel
gap-preserving timeline as the student, so its posteriors would land on packed rows directly, but a
probe through this harness found them unusable: argmax over 200 training clips decoded to 95.6% CER,
almost all blank frames, against 15.0% for the released compressed teacher on the same clips.  That
recogniser's wiring (--teacher-input-representation gaps --teacher-channels 91 --teacher-positions
timeline) is unit-tested; its input preprocessing is not, so check it before distilling from it.
Same --seed and --limit-clips make the two 61-frame arms see identical clips:

  .venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --teacher data/openfs-checkpoints/recognizer_fsboard --trusted-teacher --teacher-input-representation compressed --frame-classifier data/openfs-checkpoints/frame_classifier_fsboard --openfs-source data/external/openfs/source --teacher-cache data/models/streaming-student-rf61-distill/teacher-posteriors.npz --receptive-frames 61 --limit-clips 20000 --epochs 30 --output data/models/streaming-student-rf61-distill/student.pt --report data/streaming-student-rf61-distill.report.json
  .venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --no-distill --receptive-frames 61 --limit-clips 20000 --epochs 30 --output data/models/streaming-student-rf61-nodelearn/student.pt --report data/streaming-student-rf61-nodelearn.report.json
  .venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --teacher data/openfs-checkpoints/recognizer_fsboard --trusted-teacher --teacher-input-representation compressed --frame-classifier data/openfs-checkpoints/frame_classifier_fsboard --openfs-source data/external/openfs/source --teacher-cache data/models/streaming-student-rf61-distill/teacher-posteriors.npz --receptive-frames 200 600 --limit-clips 20000 --epochs 30 --output data/models/streaming-student-sweep/student.pt --report data/streaming-student-sweep.report.json

Evaluate an existing student (same metric path, no training) and merge rows into one table:

  .venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --evaluate-checkpoint data/models/neural-2026-10-12/sequence.pt --report data/streaming-student-baseline.report.json
  .venv/bin/python scripts/train_streaming_student.py --summarize baseline:data/streaming-student-baseline.report.json:data/streaming-latency-baseline.benchmark.json --summarize rf61-distill:data/streaming-student-rf61-distill.report.json:data/streaming-latency-rf61-distill.benchmark.json --report data/streaming-latency.report.json

CPU latency itself is never measured here: scripts/benchmark_neural.py --sequence <student.pt>
times the real streaming path and its json is what --summarize reads.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

# Set before torch builds its caching allocator; same reason as scripts/train_openfs.py.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.neural import CONTRACT,HANDSHAPES,VOCAB,FingerspellingNet,decode_ctc,load_checkpoint
from model.offline_recognizer import PAD_SEG
from model.recognition_metrics import ErrorTally
from scripts.prepare_openfs_labels_v3 import CONTRACT_V3,UPSTREAM_CHARS,refined_lookup,upstream_target
from scripts.train_openfs import read_data
from scripts.train_openfs_recognizer import normalized_box,timeline_indices

LICENSE=('OpenFS Non-Commercial Research License: the released recogniser, the released frame '
         'classifier and any label or posterior derived from them are research-only and never enter '
         'app code; a student trained with that signal is a research-only derivative work')
DEFAULT_DILATIONS=(1,2,4,8)
CAPTURE_HZ=(10,15)
GRID_MS=1000/30  # the packed contract's nominal rate, which model.timeline resamples capture onto
UPSTREAM_BLANK=0                       # released frame classifier class 0 is the blank phi token
UNMAPPED=-100                          # what the packer emits for a class with no student letter
TEACHER_CLASSES=len(UPSTREAM_CHARS)+1  # frame classifier emits 60 letters + the blank token


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- receptive field, one definition

def dilation_stack(receptive_frames,kernel=5):
    """Dilations whose causal `kernel`-tap stack reaches at least receptive_frames.  A causal conv
    with dilation d and kernel k extends the field by (k-1)*d, so the stack's field is exactly
    1 + sum((k-1)*d): double the dilation while it still fits, then one fill dilation closes the
    remainder, which lands on the smallest reachable field at or above the request."""
    stack=[]
    dilation=1
    while 1+(kernel-1)*(sum(stack)+dilation)<=receptive_frames:
        stack.append(dilation);dilation*=2
    remaining=receptive_frames-1-(kernel-1)*sum(stack)
    if remaining>0:
        stack.append(-(-remaining//(kernel-1)))
    return tuple(stack)


def receptive_field(dilations,kernel=5):
    return 1+sum((kernel-1)*int(dilation) for dilation in dilations)


def model_receptive_field(network):
    """Field measured off the built layers (kernel size and dilation read from every conv), never
    from the requested value."""
    field=1
    for block in network.blocks:
        kernel,=block.conv.kernel_size
        dilation,=block.conv.dilation
        field+=(kernel-1)*dilation
    return field


def app_latency(dilations):
    """What the app would pay for a causal stack: the model timeline is the pack's nominal 30 Hz
    grid (model/timeline.py resamples 10-15 Hz capture onto it), so `field` frames of history sit
    `field * 33.3 ms` behind a decision, while the per-arrival budget is 67-100 ms."""
    field=receptive_field(dilations)
    return dict(receptive_field_frames=field,
                algorithmic_latency_frames=field-1,
                algorithmic_latency_ms_at_10hz=round((field-1)*1000/10,1),
                algorithmic_latency_ms_at_15hz=round((field-1)*1000/15,1),
                history_ms_on_model_grid=round((field-1)*GRID_MS,1),
                buffered_model_frames_required=field,
                note=('the model is causal, so it adds no look-ahead delay; `algorithmic latency` '
                      'is the history a decision at frame t needs, and the app must buffer that '
                      'many of its 30 Hz model frames (segment cap is 900)'))


# ------------------------------------------------- teacher index space <-> student index space (one)

def detected_rows(values,base):
    """Packed gap-preserving rows where this hand slot has a detection; every other row keeps a
    zero presence flag, which is what makes a frame gap visible to the student."""
    return np.flatnonzero(values[:,base+44]>0)


def timeline_segments(values):
    """Hand identity per packed row the way scripts/train_openfs_recognizer.py packs it: 0 right,
    1 left, 2 when that frame carries no detection at all."""
    right,left=values[:,44]>0,values[:,89]>0
    segments=np.full(len(values),2,dtype=np.int64)
    segments[right]=0
    segments[left]=1
    segments[right&left]=0 if right.sum()>=left.sum() else 1
    return segments


def teacher_tokens(values,representation,hand=None,positions='arange',channels=42):
    """One clip of packed rows -> (poses, segments, positions, rows), where `rows` is the packed
    gap-preserving student row of every teacher token.  That fourth value is the only bridge
    between the teacher's compressed detection order and the student's timeline.
    compressed: upstream and frame-label staging parity — the signing hand's detections in
    increasing frame order, bbox-normalised +-0.5 landmarks, segment 0, arange index, so a 30-frame
    detection gap disappears from the index.
    gaps: either every packed row (91 channels, hand identity as segment) or that hand's
    detections (42 channels), positioned by the accumulated column-90 deltas so gaps survive.
    positions: 'arange' (compressed detection order), 'timeline' (raw accumulated frame index,
    what scripts/train_openfs_recognizer.py trained the gaps recogniser on) or 'rebased' (the same
    index shifted so the clip starts at 0, what scripts/finetune_openfs_transfer.py used)."""
    right,left=detected_rows(values,0),detected_rows(values,45)
    if hand is None: hand=0 if len(right)>=len(left) else 1
    base,track=((0,right) if hand==0 else (45,left))
    if representation=='gaps':
        segments=timeline_segments(values)
        if channels==91:
            rows=np.arange(len(values),dtype=np.int64)
            poses=np.ascontiguousarray(values[:,:91],dtype=np.float32)
        else:
            rows=track
            poses=normalized_box(values[track][:,base:base+42].reshape(-1,21,2)).reshape(-1,42)
            segments=segments[track]
        index=timeline_indices(values)[rows]
        if positions=='rebased' and len(index): index=index-index[0]
        if positions=='arange': index=np.arange(len(rows),dtype=np.float64)
        return poses,segments,index.astype(np.float64),rows
    rows=track
    poses=normalized_box(values[rows][:,base:base+42].reshape(-1,21,2)).reshape(-1,42)
    segments=np.zeros(len(rows),dtype=np.int64)
    if positions=='arange': index=np.arange(len(rows),dtype=np.float64)
    else:
        index=timeline_indices(values)[rows]
        if positions=='rebased' and len(index): index=index-index[0]
    return poses,segments,index.astype(np.float64),rows


def student_rows(rows,length):
    """Inverse of teacher_tokens: packed student row -> teacher token index, -1 for a row no
    teacher token came from (undetected frames under compressed)."""
    result=np.full(int(length),-1,dtype=np.int64)
    result[np.asarray(rows,dtype=np.int64)]=np.arange(len(rows),dtype=np.int64)
    return result


def padded_teacher(sets,channels,pad_segment=0):
    """Batch teacher tokens with the upstream sentinels (-2 pose, 1024 position).  Each clip's
    student rows stay in their own list, never inside a padded array, so a pad slot has no
    student-row meaning at all.  `pad_segment` matters: the released model masks pads by the -2
    pose sentinel, but our own Recognizer masks only segments==PAD_SEG, so a recognizer-kind
    teacher must pad with 32 or garbage pose tokens would sit unmasked inside the encoder's
    self-attention and contaminate the real tokens' memory."""
    longest=max((len(pose) for pose,_,_,_ in sets),default=0)
    poses=np.full((len(sets),longest,channels),-2,dtype=np.float32)
    segments=np.full((len(sets),longest),pad_segment,dtype=np.int64)
    positions=np.full((len(sets),longest),1024,dtype=np.float64)
    rows=[]
    for position,(pose,segment,index,student) in enumerate(sets):
        if pose.shape[1]!=channels:
            raise ValueError(f'Teacher tokens carry {pose.shape[1]} channels, expected {channels}.')
        poses[position,:len(pose)]=pose
        segments[position,:len(pose)]=segment
        positions[position,:len(pose)]=index
        rows.append(student)
    return poses,segments,positions,rows


def align_rows(rows,probabilities,sampled,length):
    """Move per-token teacher rows onto an augmented (temporally resampled) timeline: every row
    joins the nearest sampled frame, and rows that land on one frame average their posteriors."""
    rows=np.asarray(rows,dtype=np.int64)
    if not len(rows):
        return rows,probabilities
    sampled=np.asarray(sampled,dtype=np.int64)
    if len(sampled)==length and np.array_equal(sampled,np.arange(length)):
        return rows,probabilities
    right=np.clip(np.searchsorted(sampled,rows,side='left'),0,len(sampled)-1)
    left=np.clip(right-1,0,len(sampled)-1)
    nearer=np.where(np.abs(sampled[left]-rows)<=np.abs(sampled[right]-rows),left,right)
    target=nearer.astype(np.int64)
    width=probabilities.shape[1]
    dense=np.zeros((length,width),dtype=np.float32)
    counts=np.zeros(length,dtype=np.float32)
    np.add.at(dense,target,probabilities.astype(np.float32))
    np.add.at(counts,target,1.)
    keep=counts>0
    dense[keep]/=counts[keep,None]
    return np.flatnonzero(keep),dense[keep]


# ---------------------------------------------------------------- teacher weights, offline only

def class_matrix():
    """Released frame-classifier class ids (0..60, the head has len(UPSTREAM_CHARS)+1 classes) ->
    this contract's CTC token space.  Class 0 (blank phi) is our blank token 0, letters map through
    the packer's own table, and the extra class 60 the released head can emit keeps a zero row, so
    its mass is dropped (and the posterior is renormalised without it)."""
    table=np.zeros((TEACHER_CLASSES,len(VOCAB)),dtype=np.float32)
    for upstream_id in range(len(UPSTREAM_CHARS)):
        target=int(upstream_target(upstream_id))
        if target!=UNMAPPED:
            table[upstream_id,target]=1.
    return table


def resolve_teacher_positions(kind,recorded,requested):
    """Which positional index the teacher was trained on, so --teacher-positions auto cannot leak
    'auto' into teacher_tokens (an unknown value silently selects the timeline branch there).
    upstream: arange detection order.  Our gaps recogniser (scripts/train_openfs_recognizer.py):
    raw accumulated frame deltas, not rebased.  Fine-tuned gaps42/gaps91 adapters
    (scripts/finetune_openfs_transfer.py): the same deltas rebased per hand.  compressed/control42:
    arange."""
    if requested!='auto':
        return requested
    if kind=='upstream':
        return 'arange'
    if recorded=='gaps':
        return 'timeline'
    if recorded in ('gaps42','gaps91'):
        return 'rebased'
    return 'arange'


def load_teacher(args):
    """Released recogniser or a fine-tuned teacher, plus the released per-frame classifier head.
    Research-only: reachable only from this script, and never from anything under model/."""
    teacher=args.teacher
    provenance=dict(kind=None,representation=args.teacher_input_representation,
                    positions=args.teacher_positions,channels=None,paths={})
    source=None
    classifier_module=None
    import importlib
    import yaml
    if args.openfs_source:
        source=Path(args.openfs_source).resolve()
        if not (source/'lib/models/recognizer.py').is_file():
            raise ValueError('OpenFS source checkout is missing lib/models/recognizer.py.')
        # the released frame-classifier constructor is needed even when the recogniser side is one
        # of our own checkpoints, so it is imported here, not inside the released-checkpoint branch
        sys.path.insert(0,str(source))
        classifier_module=importlib.import_module('lib.models.frame_classifier')
        if Path(classifier_module.__file__).resolve()!=(source/'lib/models/frame_classifier.py').resolve():
            raise ValueError(f'Another OpenFS module is already loaded: {classifier_module.__file__}')
    if teacher.is_dir():
        if not source or not args.trusted_teacher:
            raise ValueError('The released teacher needs --openfs-source and --trusted-teacher; it is '
                             'research-only and must never reach app code.')
        recognizer=importlib.import_module('lib.models.recognizer')
        if Path(recognizer.__file__).resolve()!=(source/'lib/models/recognizer.py').resolve():
            raise ValueError(f'Another OpenFS module is already loaded: {recognizer.__file__}')
        config=yaml.safe_load((teacher/'.hydra/config.yaml').read_text(encoding='utf-8'))
        if config['decode']['chars']!=UPSTREAM_CHARS:
            raise ValueError('Teacher checkpoint vocabulary does not match the packed label vocabulary.')
        kwargs={key:value for key,value in config['recognizer'].items()
                if not key.startswith('_') and key!='ckpt_path'}
        network=recognizer.EncoderDecoderTransformer(char_size=TEACHER_CLASSES,**kwargs)
        weights=teacher/'best_model.pth'
        network.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        payload=None
        kind='upstream'
        recorded=None
        channels=42
    else:
        payload=torch.load(teacher,map_location='cpu',weights_only=True)
        if payload.get('task')=='sequence':
            raise ValueError(f'{teacher} is an app model, not a teacher; only the app\'s own '
                             'task="sequence" checkpoints carry the runtime contract '
                             '(see scripts/finetune_openfs_transfer.py).')
        from model.offline_recognizer import Recognizer
        architecture=payload['architecture']
        network=Recognizer(input_dim=architecture['input_dim'],d_model=architecture['d_model'],
                           heads=architecture['heads'],layers=architecture['layers'],
                           feedforward=architecture['feedforward'],
                           dropout=architecture.get('dropout',.1))
        network.load_state_dict(payload['state_dict'],strict=True)
        weights=teacher
        kind='recognizer'
        channels=int(architecture['input_dim'])
        recorded=payload.get('representation') or payload.get('adapter')
        if recorded:
            inferred='compressed' if recorded in ('compressed','control42') else 'gaps'
            if inferred!=args.teacher_input_representation:
                print(json.dumps(dict(warning='Teacher checkpoint was trained with the '
                                              f'{recorded} representation, but '
                                              f'--teacher-input-representation={args.teacher_input_representation}.')),
                          flush=True)
            provenance['trained_representation']=recorded
    provenance['positions']=resolve_teacher_positions(kind,recorded,args.teacher_positions)
    if channels==42 and args.teacher_input_representation=='gaps' and args.teacher_channels=='auto':
        channels=42
    if args.teacher_channels!='auto': channels=args.teacher_channels
    if channels==91 and args.teacher_input_representation!='gaps':
        raise ValueError('A 91-channel teacher only makes sense with --teacher-input-representation gaps.')
    if channels==91 and kind=='upstream':
        raise ValueError('The released checkpoint consumes 42 channels; a 91-channel teacher has to '
                         'be one of our own or a fine-tuned checkpoint.')
    classifier_directory=Path(args.frame_classifier)
    classifier_weights=classifier_directory/'best_model.pth'
    if not classifier_weights.is_file():
        raise ValueError(f'Released frame classifier checkpoint is missing: {classifier_weights}')
    if classifier_module is None:
        raise ValueError('The released per-frame classifier needs --openfs-source, because its '
                         'constructor comes from the pinned upstream checkout.')
    classifier_config=yaml.safe_load((classifier_directory/'.hydra/config.yaml').read_text(encoding='utf-8'))
    classifier_kwargs={key:value for key,value in classifier_config['frame_classifier'].items()
                       if not key.startswith('_') and key!='ckpt_path'}
    classifier=classifier_module.Classifier(char_size=TEACHER_CLASSES,**classifier_kwargs)
    state=torch.load(classifier_weights,map_location='cpu',weights_only=True)
    classifier.load_state_dict(state,strict=True)
    width=classifier_kwargs['d_model']
    expected=['linear.0.weight','linear.0.bias','linear.1.weight','linear.1.bias',
              'linear.4.weight','linear.4.bias']
    if list(state)!=expected or [list(state[key].shape) for key in expected]!=[[width,width],[width],
                                                                              [width],[width],
                                                                              [TEACHER_CLASSES,width],
                                                                              [TEACHER_CLASSES]]:
        raise ValueError('Frame classifier checkpoint does not match the released Table S11 architecture.')
    network.eval().to(args.device)
    classifier.eval().to(args.device)
    provenance.update(kind=kind,channels=channels,parameters=sum(q.numel() for q in network.parameters()),
                      classifier_parameters=sum(q.numel() for q in classifier.parameters()),
                      classifier_classes=TEACHER_CLASSES,
                      paths={'teacher':str(Path(teacher).resolve()),'teacher_sha256':sha256(weights),
                             'frame_classifier':str(classifier_weights.resolve()),
                             'frame_classifier_sha256':sha256(classifier_weights),
                             'upstream_source':str(source) if source else None})
    return dict(network=network,classifier=classifier,kind=kind,channels=channels,payload=payload,
                representation=args.teacher_input_representation,
                positions=provenance['positions'],provenance=provenance,
                pad_segment=0 if kind=='upstream' else PAD_SEG)


def teacher_memory(teacher,poses,segments,positions):
    """Encoder pass for one padded batch; both teacher classes take (poses, segments, positions)
    but disagree on pose rank and position dtype, which is the only place that knows it.  The
    network already lives on --device, so the numpy batch has to follow it there."""
    device=next(teacher['network'].parameters()).device
    if teacher['kind']=='upstream':
        values=torch.as_tensor(poses,device=device)
        values=values.reshape(len(poses),values.shape[1],values.shape[2]//2,2)
        return teacher['network'](values,torch.as_tensor(segments,device=device),
                                  torch.as_tensor(positions,dtype=torch.long,device=device)
                                  )['encoder_memory']
    return teacher['network'].encode(torch.as_tensor(poses,device=device),
                                     torch.as_tensor(segments,device=device),
                                     torch.as_tensor(positions,dtype=torch.float32,device=device))[0]


def token_groups(lengths,batch_size,token_budget,attention_budget):
    """Length-grouped batches capped on tokens and on batch*length^2 attention work."""
    groups=[];group=[]
    for index in np.argsort(np.asarray(lengths),kind='stable'):
        span=int(lengths[int(index)])
        oversized=group and (len(group)+1>batch_size or (len(group)+1)*span>token_budget
                             or (len(group)+1)*span*span>attention_budget)
        if oversized:
            groups.append(group);group=[]
        group.append(int(index))
    if group:
        groups.append(group)
    return groups


@torch.inference_mode()
def teacher_posteriors(teacher,data,order,hand_of=None,*,batch_size=32,token_budget=8000,
                       attention_budget=4000000):
    """One offline encoder+classifier pass over the clips the student will train on; the test split
    is never passed here.  Returns {clip index: (student rows, posteriors already mapped into the
    student's token space, rows aligned to the same student rows as `labels`)} plus bookkeeping."""
    matrix=class_matrix()
    clips=data['clips']
    hand_of=hand_of or {}
    out={}
    started=time.perf_counter()
    tokens=0
    dropped=0.
    blank=0
    letters=0
    skipped=Counter()
    groups=token_groups([max(1,int(clips[index]['end'])-int(clips[index]['start'])) for index in order],
                        batch_size,token_budget,attention_budget)
    for group in groups:
        sets=[]
        for index in group:
            row=clips[index]
            values=np.asarray(data['x'][row['start']:row['end']])
            poses,segments,positions,rows=teacher_tokens(values,teacher['representation'],
                                                         hand=hand_of.get(row['name']),
                                                         positions=teacher['positions'],
                                                         channels=teacher['channels'])
            if not len(rows):
                skipped['clip_without_detections']+=1
                continue
            sets.append((index,poses,segments,positions,rows))
        if not sets:
            continue
        poses,segments,positions,rows=padded_teacher([set_[1:] for set_ in sets],teacher['channels'],
                                                     teacher.get('pad_segment',0))
        memory=teacher_memory(teacher,poses,segments,positions)
        probabilities=teacher['classifier'](memory).float().softmax(-1).cpu().numpy()
        for position,(index,_,_,_,student) in enumerate(sets):
            span=len(student)
            posterior=probabilities[position,:span] @ matrix
            dropped+=float(probabilities[position,:span,len(UPSTREAM_CHARS):].mean())
            total=posterior.sum(-1)
            usable=total>0
            posterior[usable]/=total[usable,None]
            out[index]=(student.astype(np.int32),posterior.astype(np.float16))
            tokens+=span
            hard=posterior.argmax(-1)
            blank+=int((hard==0).sum())
            letters+=int((hard>0).sum())
    seconds=time.perf_counter()-started
    return out,dict(clips=len(out),clips_requested=len(order),teacher_tokens=tokens,
                    teacher_seconds=round(seconds,2),tokens_per_second=round(tokens/max(seconds,1e-9),1),
                    mean_dropped_class_mass=round(dropped/max(tokens,1),6),
                    posterior_blank_frames=blank,posterior_letter_frames=letters,skipped=dict(skipped))


# ---------------------------------------------------------------- per-frame teacher labels

def frame_labels(data,refined,handshape=None):
    """Dense VOCAB-space per-frame labels over the pack's gap-preserving timeline, joined by clip
    name and original frame index.  A clip the refined file covers wins; otherwise the pack's own
    per-frame targets, then <split>_handshape.npz when that pack records frame indices; the count
    per route is reported, because a fallback is a weaker label source, not a free win."""
    labels=np.full(len(data['aux']),-100,dtype=np.int16)
    origin=Counter()
    handshape_token=[0 if label=='nothing' else VOCAB.index(label) for label in HANDSHAPES]
    frames_of={}
    if handshape is not None:
        for clip_id,frame,shape in zip(handshape['clips'],handshape['frames'],handshape['labels']):
            frames_of.setdefault(int(clip_id),[]).append((int(frame),int(shape)))
    for index,row in enumerate(data['clips']):
        start,end=int(row['start']),int(row['end'])
        entry=(refined or {}).get(row['name'])
        if entry is not None and len(entry[0]):
            index_values,upstream_ids,_hand=entry
            if int(index_values.min())<0 or int(index_values.max())>=end-start:
                raise ValueError(f"Refined frame index leaves clip {row['name']}.")
            labels[start:end]=0  # the pack's own convention: a covered clip labels bare frames blank
            labels[start+np.asarray(index_values,dtype=np.int64)]=upstream_target(
                np.asarray(upstream_ids,dtype=np.int64)).astype(np.int16)
            origin['refined']+=1
            continue
        packed=data['aux'][start:end]
        if bool((packed!=-100).any()):
            labels[start:end]=packed
            origin['pack_targets']+=1
            continue
        rows=frames_of.get(index)
        if rows:
            frames_here=np.array([frame for frame,_ in rows],dtype=np.int64)
            if len(frames_here) and (frames_here.min()<0 or frames_here.max()>=end-start):
                raise ValueError(f"Handshape frame index leaves clip {row['name']}.")
            tokens=np.array([int(handshape_token[shape]) for _,shape in rows],dtype=np.int16)
            keep=tokens>=0
            labels[start+frames_here[keep]]=tokens[keep]
            origin['handshape_npz']+=1
            continue
        origin['none']+=1
    covered=labels>=0
    return labels,dict(frame_label_clips=dict(origin),
                       uses_teacher_labels=origin['refined']>0 or origin['handshape_npz']>0,
                       frame_label_rows=int(covered.sum()),
                       frame_label_blank_rows=int((labels==0).sum()),
                       frame_label_letter_rows=int((labels>0).sum()),
                       frame_label_unlabelled_rows=int((labels==-100).sum()),
                       frame_label_note=('refined npz labels are model-derived (released frame '
                                         'classifier), never human held-letter annotation; a refined '
                                         'clip labels frames outside its detections blank, matching '
                                         'the v2 pack convention'))


# ---------------------------------------------------------------- student batches, aligned

def batches(data,batch_size,labels,posteriors,*,augment=False,shuffle=False,order=None,
            distillation=True,seed=None):
    """Packed clips with every supervision channel resampled by the same index map, so teacher rows,
    teacher posteriors and frame labels stay glued to the frames they describe."""
    clips=data['clips']
    indices=np.arange(len(clips)) if order is None else np.asarray(order,dtype=np.int64)
    if shuffle:
        rng=np.random.default_rng(seed)
        indices=rng.permutation(indices)
    chunks=[sorted(indices[i:i+batch_size*50],key=lambda j:clips[j]['end']-clips[j]['start'])
            for i in range(0,len(indices),batch_size*50)]
    groups=[window[i:i+batch_size] for window in chunks for i in range(0,len(window),batch_size)]
    if shuffle:
        random.Random(seed).shuffle(groups)
    for selected in groups:
        frames,target_rows,counts,frame_label_rows,posterior_rows,posterior_values=[],[],[],[],[],[]
        width=0
        for index in selected:
            row=clips[index]
            start,end=int(row['start']),int(row['end'])
            values=np.array(data['x'][start:end],copy=True)
            frame_label=np.array(labels[start:end],dtype=np.int64)
            text=row['text']
            sampled=None
            if augment:
                required=len(text)+sum(a==b for a,b in zip(text,text[1:]))
                length=max(required,round(len(values)*random.uniform(.85,1.15)))
                sampled=np.linspace(0,len(values)-1,length).round().astype(int)
                values,frame_label=values[sampled].copy(),frame_label[sampled]
                for base in (0,45):
                    present=values[:,base+44:base+45]
                    values[:,base:base+42]+=np.random.normal(0,.01,(length,42)).astype(np.float32)*present
                    values[:,base+42:base+44]+=np.random.normal(0,.15,(1,2)).astype(np.float32)*present
                values[:,-1]*=(end-start)/length
            frames.append(torch.from_numpy(values))
            frame_label_rows.append(torch.from_numpy(frame_label))
            target_rows.append(torch.tensor([VOCAB.index(character) for character in text],dtype=torch.int32))
            counts.append(len(text))
            entry=posteriors.get(index) if distillation else None
            if entry is None:
                posterior_rows.append(None)
                posterior_values.append(None)
            else:
                rows,probabilities=entry
                rows=np.asarray(rows,dtype=np.int64)
                probabilities=probabilities.astype(np.float32)
                if sampled is not None:
                    rows,probabilities=align_rows(rows,probabilities,sampled,len(values))
                posterior_rows.append(rows)
                posterior_values.append(torch.from_numpy(probabilities))
                width=max(width,int(posterior_values[-1].shape[1]))
        lengths=torch.tensor([len(value) for value in frames],dtype=torch.int32)
        yield (torch.nn.utils.rnn.pad_sequence(frames,batch_first=True),
               torch.cat(target_rows),lengths,torch.tensor(counts,dtype=torch.int32),
               torch.nn.utils.rnn.pad_sequence(frame_label_rows,batch_first=True,padding_value=-100),
               posterior_rows,posterior_values,selected)


# ---------------------------------------------------------------- losses

def soft_kl(logits,targets,mask,temperature):
    """KL(teacher || student) at masked frames only, with the teacher softened by `temperature`;
    masked-out (blank/nothing and padded) frames contribute nothing but keep the graph alive."""
    probabilities=targets[mask]
    if probabilities.shape[0]==0:
        return logits.sum()*0
    log_probabilities=F.log_softmax(logits[mask].float()/temperature,dim=-1)
    return temperature*temperature*F.kl_div(log_probabilities,probabilities,reduction='batchmean')


def posterior_batch(posterior_values,posterior_rows,lengths,device,distill_blank):
    """Dense (B,T,len(VOCAB)) teacher distribution plus the frames the KL term may use: pad slots
    and, by default, frames where the teacher holds no letter are masked out."""
    batch=int(lengths.shape[0])
    length=int(lengths.max())
    values=torch.zeros((batch,length,len(VOCAB)),dtype=torch.float32)
    for row,(entries,rows) in enumerate(zip(posterior_values,posterior_rows)):
        if entries is None:
            continue
        values[row,np.asarray(rows,dtype=np.int64)]=entries
    used=values.sum(-1)>0
    if not distill_blank:
        used&=values.argmax(-1)>0
    return values.to(device),used.to(device)


# ---------------------------------------------------------------- measurement and checkpoints

@torch.inference_mode()
def evaluate(model,data,args,split,limit=None,seed=42):
    """Greedy CTC over the packed timeline, scored by model.recognition_metrics.ErrorTally: the same
    path scripts/eval_recognition.py uses for the shipped head, so rows compare with its baseline.
    fp32 and no augmentation on purpose, because that is what the CPU runtime does."""
    model.eval()
    clips=data['clips']
    order=np.arange(len(clips))
    if limit:
        order=np.random.default_rng(seed).choice(len(clips),min(int(limit),len(clips)),replace=False)
    order=sorted((int(index) for index in order),key=lambda index:clips[index]['end']-clips[index]['start'])
    tally=ErrorTally()
    examples=[]
    started=time.perf_counter()
    for start in range(0,len(order),args.head_batch_size):
        group=order[start:start+args.head_batch_size]
        lengths=[int(clips[index]['end']-clips[index]['start']) for index in group]
        values=torch.nn.utils.rnn.pad_sequence([torch.from_numpy(
            np.array(data['x'][clips[index]['start']:clips[index]['end']])) for index in group],batch_first=True)
        predicted=model(values.to(args.device)).argmax(-1).cpu().numpy()
        for row,index in enumerate(group):
            truth=clips[index]['text']
            guess=decode_ctc(predicted[row,:lengths[row]])
            tally.add(truth,guess)
            if len(examples)<20:
                examples.append(dict(name=clips[index]['name'],truth=truth,prediction=guess))
    summary=tally.summary()
    summary.update(split=split,seconds=round(time.perf_counter()-started,2),clips_evaluated=summary['clips'],
                   decoder='greedy CTC on the gap-preserving timeline; no dictionary or language correction',
                   examples=examples)
    return summary


def checkpoint(model,args,path,metrics,epoch,extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(contract=CONTRACT,task='sequence',labels=VOCAB,hidden=args.hidden,
                 dilations=list(model.dilations),
                 receptive_field_frames=model_receptive_field(model),
                 state_dict={key:value.detach().cpu() for key,value in model.state_dict().items()},
                 metrics=metrics,epoch=epoch,seed=args.seed,dataset=str(args.data.resolve()),
                 experimental=True,**extra)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.pt',delete=False) as file:
        temporary=Path(file.name)
    try:
        torch.save(payload,temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path,document):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(document,indent=2),encoding='utf-8')
    temporary.replace(path)


def student_run(args,data,validation,test_data,frame_label,label_info,posteriors,teacher_info,order):
    """Train one model per requested receptive field, sequentially, so at most one training process
    exists at a time and the offline teacher pass is paid once for the whole sweep."""
    blocks=[]
    for requested in args.receptive_frames:
        dilations=dilation_stack(requested)
        model=FingerspellingNet(args.hidden,dilations).to(args.device)
        field=model_receptive_field(model)
        parameters=sum(q.numel() for q in model.parameters())
        if parameters>args.max_parameters:
            raise ValueError(f'rf{requested} has {parameters} parameters, above --max-parameters '
                             f'{args.max_parameters}; lower --hidden or shorten the stack.')
        path=args.output.with_name(f'{args.output.stem}-rf{field}{args.output.suffix}')
        print(json.dumps(dict(config=f'rf{field}',requested_receptive_frames=requested,dilations=dilations,
                              parameters=parameters,
                              **app_latency(dilations),
                              training_clips=len(order),torch=torch.__version__,
                              gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                              amp=args.amp)),flush=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.learning_rate*.05)
        scaler=torch.amp.GradScaler('cuda',enabled=args.amp=='fp16')
        ctc=torch.nn.CTCLoss(blank=0,zero_infinity=False)
        best=float('inf')
        best_metrics=None
        best_epoch=None
        stale=0
        history=[]
        run_started=time.perf_counter()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,args.epochs+1):
            model.train()
            started=time.perf_counter()
            loss_sum=ctc_sum=aux_sum=kl_sum=steps=0
            for frames,targets,lengths,counts,frame_label_batch,posterior_rows,posterior_values,selected in batches(
                    data,args.batch_size,frame_label,posteriors,augment=not args.no_augment,shuffle=True,
                    order=order,distillation=teacher_info is not None,seed=args.seed+epoch):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=args.amp_dtype or torch.float16,enabled=args.amp!='off'):
                    logits=model(frames.to(args.device))
                logits=logits.float()
                loss=ctc(logits.transpose(0,1).contiguous().log_softmax(-1),
                         targets.to(args.device,dtype=torch.int64),lengths.long(),counts.long())
                ctc_value=float(loss.detach())
                if args.auxiliary_frame_weight and bool((frame_label_batch!=-100).any()):
                    auxiliary=F.cross_entropy(logits.reshape(-1,len(VOCAB)),
                                             frame_label_batch.to(args.device).reshape(-1),ignore_index=-100)
                    loss=loss+args.auxiliary_frame_weight*auxiliary
                    aux_sum+=float(auxiliary.detach())
                kl_value=0.
                if teacher_info is not None and args.distillation_weight:
                    probabilities,used=posterior_batch(posterior_values,posterior_rows,lengths,args.device,
                                                       args.distillation_blank)
                    if bool(used.any()):
                        kl=soft_kl(logits,probabilities,used,args.distillation_temperature)
                        loss=loss+args.distillation_weight*kl
                        kl_value=float(kl.detach())
                if not torch.isfinite(loss):
                    detail=dict(config=f'rf{field}',epoch=epoch,step=steps,loss=float(loss.detach()),
                                ctc=ctc_value,kl=kl_value,
                                logits_finite=bool(torch.isfinite(logits).all()),
                                maximum_coordinate=float(frames.abs().max()),
                                clips=[data['clips'][index]['name'] for index in selected])
                    raise ValueError('Non-finite loss: '+json.dumps(detail))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer)
                scaler.update()
                loss_sum+=float(loss.detach())
                ctc_sum+=ctc_value
                kl_sum+=kl_value
                steps+=1
                if steps%200==0:
                    print(json.dumps(dict(config=f'rf{field}',epoch=epoch,steps=steps,
                                          loss=round(loss_sum/steps,5),ctc=round(ctc_sum/steps,5),
                                          frame_ce=round(aux_sum/steps,5),kl=round(kl_sum/steps,5),
                                          training_seconds=round(time.perf_counter()-started,1))),
                          flush=True)
            metrics=evaluate(model,validation,args,'validation',limit=args.validation_limit)
            score=metrics['cer']
            row=dict(config=f'rf{field}',epoch=epoch,loss=loss_sum/steps,ctc=ctc_sum/steps,
                     frame_ce=aux_sum/steps,kl=kl_sum/steps,
                     training_seconds=round(time.perf_counter()-started-metrics['seconds'],2),
                     validation_score=score,validation_letter=metrics['letter_accuracy'],
                     validation_exact=metrics['exact_sequence_accuracy'],
                     elapsed_seconds=round(time.perf_counter()-run_started,2))
            history.append(row)
            print(json.dumps(row))
            if score<best:
                best=score
                best_epoch=epoch
                stale=0
                best_metrics=metrics
                research=dict(research_only=teacher_info is not None or label_info.get('uses_teacher_labels',False),
                              license=LICENSE,
                              trained_from='random initialisation on our own packed FSboard timeline',
                              distilled_from=teacher_info,frame_labels=label_info)
                checkpoint(model,args,path,metrics,epoch,research)
            else:
                stale+=1
            scheduler.step()
            if args.patience and stale>=args.patience:
                print('Validation early stopping',flush=True)
                break
        peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
        gate=args.test_if_validation_cer_below
        if gate is not None and (best_metrics is None or best_metrics['cer'] is None
                                 or not best_metrics['cer'] < gate):
            test=dict(skipped=True,split='test',
                      reason='best validation CER did not beat --test-if-validation-cer-below',
                      threshold=gate,best_validation_cer=best_metrics['cer'] if best_metrics else None,
                      clips=0,clips_evaluated=0,characters=0,errors=0,cer=None,
                      letter_accuracy=None,exact_sequence_accuracy=None)
        else:
            model=load_checkpoint(path,'sequence').to(args.device)
            test=evaluate(model,test_data,args,'test',limit=args.test_limit)
        trained_seconds=round(sum(row['training_seconds'] for row in history),1)
        blocks.append(dict(config=f'rf{field}',requested_receptive_frames=requested,dilations=list(dilations),
                           hidden=args.hidden,parameters=parameters,
                           **app_latency(dilations),checkpoint=str(path.resolve()),
                           best_validation_cer=best,best_validation_epoch=best_epoch,
                           training=dict(epochs=len(history),
                                         seconds_per_epoch=round(trained_seconds/max(len(history),1),2),
                                         training_seconds=trained_seconds,
                                         peak_vram_mib=round(peak/2**20),train_clips=len(order),
                                         batch_size=args.batch_size,augment=not args.no_augment,amp=args.amp,
                                         seed=args.seed,patience=args.patience,
                                         distillation=teacher_info is not None,
                                         distillation_weight=args.distillation_weight if teacher_info else 0.,
                                         distillation_temperature=args.distillation_temperature,
                                         distillation_blank_frames=args.distillation_blank,
                                         auxiliary_frame_weight=args.auxiliary_frame_weight),
                           validation=without_examples(best_metrics or {}),history=history,
                           test=without_examples(test)))
        write(args.report,dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),script=__file__,
                               contract=CONTRACT,dataset=args.dataset_summary,frame_labels=label_info,
                               teacher=teacher_info,configs=blocks,torch=torch.__version__,
                               gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                               license=LICENSE))
    return blocks


def without_examples(block):
    return {key:value for key,value in block.items() if key!='examples'}


# ---------------------------------------------------------------- reporting modes

def benchmark_block(args,checkpoint):
    """Wall-clock arrival latency through the real streaming runtime, measured by the existing
    scripts/benchmark_neural.py harness: it replays public packed test landmarks frame by frame
    through model.live_neural.NeuralRuntime with this checkpoint swapped in as the sequence model.
    Nothing here times anything itself; the harness json is the record."""
    harness=ROOT/'scripts'/'benchmark_neural.py'
    with tempfile.TemporaryDirectory() as directory:
        json_path=Path(directory)/'benchmark.json'
        command=[sys.executable,str(harness),'--packed',str(args.data),'--sequence',str(checkpoint),
                 '--handshape',str(args.handshape),
                 '--threads',*[str(value) for value in args.benchmark_threads],
                 '--frames',*[str(value) for value in args.benchmark_frames],'--report',str(json_path)]
        if args.packed_generation:
            command+=['--generation',args.packed_generation]
        subprocess.run(command,check=True,capture_output=True,text=True)
        measurements=json.loads(json_path.read_text(encoding='utf-8'))['measurements']
    rows=[{key:measurement[key] for key in
           ('threads','segment_frames','p50_ms','p95_ms','tail100_p50_ms','tail100_p95_ms')}
          for measurement in measurements]
    fastest={threads:min((row['p50_ms'] for row in rows if row['threads']==threads),default=None)
             for threads in args.benchmark_threads}
    return dict(source='scripts/benchmark_neural.py',scope='CPU arrival latency through the live '
            'streaming runtime; excludes camera/detector/IPC',harness_command=command,
            fastest_p50_ms_by_threads=fastest,measurements=rows)


def evaluate_only(args):
    """Score an existing checkpoint on the test split with the same metric path, then stop: the
    released baseline (data/sequence.pt) is a row in this study's tables, not a training run."""
    model=load_checkpoint(args.evaluate_checkpoint,'sequence').to(args.device)  # evaluate() moves
    # the clips to --device, so the checkpoint has to live there too or the first matmul throws
    test_data=read_data(args.data,'test','sequence')
    summary=evaluate(model,test_data,args,'test',limit=args.test_limit)
    document=dict(mode='evaluate',checkpoint=str(args.evaluate_checkpoint.resolve()),
                  contract=CONTRACT,hidden=model.blocks[0].conv.in_channels,
                  dilations=list(model.dilations),
                  parameters=sum(q.numel() for q in model.parameters()),
                  **app_latency(model.dilations),test=without_examples(summary))
    if args.benchmark:
        document['cpu_latency']=benchmark_block(args,args.evaluate_checkpoint)
    write(args.report,document)
    print(json.dumps({key:document[key] for key in
                      ('receptive_field_frames','parameters','dilations')}),flush=True)
    print(json.dumps({key:summary[key] for key in
                      ('cer','exact_sequence_accuracy','errors','characters','clips')}),flush=True)


def summarize(args):
    """Merge run reports (plus their benchmark jsons) into one accuracy-versus-latency table.
    `NAME:report.json[:benchmark.json]`; the first row is meant to be the released baseline, so
    every later row reads as a CER delta at a stated latency, never as a bare number."""
    rows=[]
    for spec in args.summarize:
        parts=spec.split(':')
        if len(parts)<2:
            raise ValueError('--summarize takes NAME:report.json[:benchmark.json]')
        name,report=parts[0],Path(parts[1])
        document=json.loads(report.read_text(encoding='utf-8'))
        benchmark=None
        if len(parts)>2:
            benchmark=json.loads(Path(parts[2]).read_text(encoding='utf-8'))
            if 'measurements' not in benchmark:
                raise ValueError(f'{parts[2]} is not a benchmark_neural.py report')
        configs=document.get('configs') or [document]
        for block in configs:
            test=block.get('test') or {}
            field=block.get('receptive_field_frames')
            if field is None:  # a baseline report predates this script; read the stack off the checkpoint
                checkpoint=block.get('checkpoint') or args.evaluate_checkpoint
                if not checkpoint:
                    raise ValueError(f'{name} has no receptive_field_frames and no checkpoint to read '
                                     'the dilation stack from')
                model=load_checkpoint(Path(checkpoint),'sequence')
                field=model_receptive_field(model)
                block=dict(block,dilations=list(model.dilations),**app_latency(model.dilations))
            row=dict(config=name if len(configs)==1 else f"{name}:{block.get('config','?')}",
                     checkpoint=block.get('checkpoint'),parameters=block.get('parameters'),
                     receptive_field_frames=field,dilations=block.get('dilations'),
                     buffered_model_frames_required=block.get('buffered_model_frames_required'),
                     algorithmic_latency_ms_at_10hz=block.get('algorithmic_latency_ms_at_10hz'),
                     algorithmic_latency_ms_at_15hz=block.get('algorithmic_latency_ms_at_15hz'),
                     history_ms_on_model_grid=block.get('history_ms_on_model_grid'),
                     cer=test.get('cer'),letter_accuracy=test.get('letter_accuracy'),
                     exact_sequence_accuracy=test.get('exact_sequence_accuracy'),
                     deletions=test.get('deletions'),substitutions=test.get('substitutions'),
                     insertions=test.get('insertions'),
                     deletion_rate=test.get('deletion_rate'),
                     substitution_rate=test.get('substitution_rate'),
                     insertion_rate=test.get('insertion_rate'),
                     characters=test.get('characters'),clips=test.get('clips'),
                     validation_cer=(block.get('best_validation_cer') or
                                     (block.get('validation') or {}).get('cer')),
                     seconds_per_epoch=(block.get('training') or {}).get('seconds_per_epoch'),
                     peak_vram_mib=(block.get('training') or {}).get('peak_vram_mib'),
                     distillation=(block.get('training') or {}).get('distillation'),
                     decoder=test.get('decoder'))
            if benchmark is not None:
                row['arrival_p50_ms_best']=min((measurement['p50_ms'] for measurement
                                                in benchmark['measurements']),default=None)
                row['arrival_p95_ms_best']=min((measurement['p95_ms'] for measurement
                                                in benchmark['measurements']),default=None)
                row['arrival_latency_source']=parts[2]
            rows.append(row)
    write(args.report,dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                           scope='FSboard packed test split, greedy CTC, no dictionary or language '
                                 'correction; every row on the identical metric and decode path',
                           latency_note=('algorithmic latency is the causal history a decision needs '
                                         'at capture rate; arrival columns are measured CPU wall-clock '
                                         'per arrival through the live runtime; the app runs the model '
                                         'on a 30 Hz grid, so history_ms_on_model_grid is what a '
                                         'decision actually lags the live text by'),
                           rows=rows))
    print(json.dumps(['config','receptive_field_frames','parameters','cer','exact_sequence_accuracy',
                      'algorithmic_latency_ms_at_15hz','arrival_p50_ms_best']),flush=True)
    for row in rows:
        print(json.dumps({key:row.get(key) for key in
                          ('config','receptive_field_frames','parameters','cer',
                           'exact_sequence_accuracy','algorithmic_latency_ms_at_15hz',
                           'arrival_p50_ms_best')}),flush=True)


def clip_order(data,limit,seed):
    """Deterministic clip subset for capped runs, so the distill/no-distill pair sees identical
    data and the report states exactly which clips were used."""
    if not limit or limit>=len(data['clips']):
        return list(range(len(data['clips'])))
    return sorted(int(index) for index in np.random.default_rng(seed).choice(len(data['clips']),
                                                                             int(limit),replace=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',type=Path,default=ROOT/'data/openfs-packed')
    parser.add_argument('--packed-generation',
                        help='forwarded to scripts/benchmark_neural.py --generation')
    parser.add_argument('--refined-labels',type=Path,default=ROOT/'data/openfs-refined',
                        help='directory of <split>.npz from scripts/refine_openfs_frame_labels.py')
    parser.add_argument('--teacher',type=Path,
                        help='released OpenFS checkpoint directory or a fine-tuned teacher .pt; '
                             'research-only, never reachable from model/')
    parser.add_argument('--teacher-cache',type=Path,
                        help='cache the offline teacher posterior pass to this .npz; reused when it '
                             'describes exactly this dataset, teacher set of weights and clip subset')
    parser.add_argument('--frame-classifier',type=Path,
                        default=ROOT/'data/openfs-checkpoints/frame_classifier_fsboard')
    parser.add_argument('--openfs-source',default=os.environ.get('ASL_OPENFS_SOURCE'),
                        help='pinned upstream checkout (provides the released constructors)')
    parser.add_argument('--trusted-teacher',action='store_true',
                        help='required to load the research-licensed released checkpoint here')
    parser.add_argument('--teacher-input-representation',choices=('compressed','gaps'),
                        default='compressed')
    parser.add_argument('--teacher-positions',choices=('auto','arange','rebased','timeline'),
                        default='arange',help='which positional index the teacher consumes; '
                        'auto reads it off the checkpoint provenance')
    parser.add_argument('--teacher-channels',choices=('auto','42','91'),default='auto')
    parser.add_argument('--receptive-frames',type=int,nargs='+',default=[61],
                        help='causal receptive field(s) in model-grid frames to train; several '
                             'values train each in turn and share one teacher pass')
    parser.add_argument('--hidden',type=int,default=192)
    parser.add_argument('--max-parameters',type=int,default=3_000_000)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--lr','--learning-rate',dest='learning_rate',type=float,default=8e-4)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--head-batch-size',type=int,default=64,help='clips per evaluation batch')
    parser.add_argument('--limit-clips',type=int)
    parser.add_argument('--validation-limit',type=int)
    parser.add_argument('--test-limit',type=int)
    parser.add_argument('--test-if-validation-cer-below',type=float,
                        help='evaluate the test split only if best validation CER, as a fraction, '
                             'is strictly below this value; e.g. --test-if-validation-cer-below 0.1407')
    parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--seed',type=int,default=20261019)
    parser.add_argument('--auxiliary-frame-weight',type=float,default=1.0,
                        help='weight on per-frame CE; 0 drops the teacher frame-label term')
    parser.add_argument('--no-frame-labels',action='store_true',
                        help='ablation: drop the per-frame loss entirely')
    parser.add_argument('--distillation-weight',type=float,default=0.5,help='weight on teacher KL')
    parser.add_argument('--distillation-temperature',type=float,default=2.0)
    parser.add_argument('--distillation-blank',action='store_true',
                        help='also distill frames where the teacher holds no letter')
    parser.add_argument('--no-distill',action='store_true',
                        help='ablation: no teacher pass and no KL term')
    parser.add_argument('--no-augment',action='store_true')
    parser.add_argument('--amp',choices=('auto','bf16','fp16','off'),default='auto')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'data/models/streaming-student/student.pt')
    parser.add_argument('--report',type=Path,default=ROOT/'data/streaming-student.report.json')
    parser.add_argument('--evaluate-checkpoint',type=Path,
                        help='score an existing checkpoint on the test split and exit')
    parser.add_argument('--summarize',action='append',
                        help='NAME:run-report.json[:benchmark.json]; repeatable; merges rows into '
                             '--report as one accuracy-versus-latency table')
    parser.add_argument('--benchmark',action='store_true',
                        help='also run scripts/benchmark_neural.py on each trained checkpoint')
    parser.add_argument('--handshape',type=Path,default=ROOT/'data/handshape.pt',
                        help='runtime handshape checkpoint forwarded to scripts/benchmark_neural.py')
    parser.add_argument('--benchmark-threads',type=int,nargs='+',default=[4])
    parser.add_argument('--benchmark-frames',type=int,nargs='+',default=[300,900])
    args=parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('--device cuda requested but no CUDA device is visible.')
    if args.device=='cpu':
        torch.set_num_threads(min(8,os.cpu_count() or 8))  # a local CPU pass must not starve the box
    args.amp_dtype=None
    if args.amp=='auto':
        args.amp='off' if args.device=='cpu' else ('bf16' if torch.cuda.get_device_capability()[0]>=8
                                                   else 'fp16')
    if args.amp=='bf16': args.amp_dtype=torch.bfloat16
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    if args.summarize:
        summarize(args)
        return
    if args.evaluate_checkpoint:
        evaluate_only(args)
        return

    data=read_data(args.data,'train','sequence')
    validation=read_data(args.data,'validation','sequence')
    test_data=read_data(args.data,'test','sequence')
    order=clip_order(data,args.limit_clips,args.seed)
    refined_path=args.refined_labels/'train.npz'
    refined=refined_lookup(refined_path) if refined_path.is_file() else {}
    handshape=None
    handshape_path=args.data/'train_handshape.npz'
    if handshape_path.is_file():
        with np.load(handshape_path,allow_pickle=False) as source:
            if {'clips','frames','labels'}<=set(source.files):
                handshape={key:source[key] for key in ('clips','frames','labels')}
    if args.no_frame_labels:
        frame_label=np.full(len(data['aux']),-100,dtype=np.int16)
        label_info=dict(frame_label_clips={'disabled':len(order)},uses_teacher_labels=False,
                        frame_label_rows=0,frame_label_note='--no-frame-labels ablation')
    else:
        frame_label,label_info=frame_labels(data,refined,handshape)
    label_info['refined_labels']=str(refined_path.resolve()) if refined else None
    if refined: label_info['refined_labels_sha256']=sha256(refined_path)
    label_info['clip_order']=dict(clips=len(order),seed=args.seed,limit=args.limit_clips,
                                  first=data['clips'][order[0]]['name'],
                                  last=data['clips'][order[-1]]['name'])

    teacher_info=None; posteriors={}; posterior_info={}
    if not args.no_distill and args.distillation_weight:
        if args.teacher is None:
            raise ValueError('Distillation needs --teacher (or --no-distill / '
                             '--distillation-weight 0).')
        teacher=load_teacher(args)
        posteriors,posterior_info=teacher_posteriors_cached(args,teacher,data,order)
        teacher_info=teacher['provenance']
        del teacher
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    elif args.no_distill:
        posterior_info=dict(ablation='--no-distill: no teacher pass, no KL term')

    args.dataset_summary=dict(train_rows=len(data['x']),train_clips=len(data['clips']),
                              validation_clips=len(validation['clips']),
                              test_clips=len(test_data['clips']),training_clips=len(order),
                              teacher_posteriors=posterior_info)
    blocks=student_run(args,data,validation,test_data,frame_label,label_info,posteriors,
                       teacher_info,order)
    if args.benchmark:
        for block in blocks:
            block['cpu_latency']=benchmark_block(args,Path(block['checkpoint']))
            write(args.report,dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),script=__file__,
                                   contract=CONTRACT,dataset=args.dataset_summary,
                                   frame_labels=label_info,teacher=teacher_info,configs=blocks,
                                   torch=torch.__version__,
                                   gpu=torch.cuda.get_device_name(0)
                                       if args.device.startswith('cuda') else None,license=LICENSE))


def teacher_posteriors_cached(args,teacher,data,order):
    """The offline teacher pass is the expensive half of every run, so it is cached and its key
    (dataset directory, teacher hashes, representation, clip subset) is recorded either way: a hit
    has to describe exactly this set of clips and exactly these teacher weights."""
    key=dict(dataset=str(args.data.resolve()),teacher_sha256=teacher['provenance']['paths'],
             representation=teacher['representation'],positions=teacher['positions'],
             channels=teacher['channels'],clips=sorted(int(index) for index in order))
    digest=hashlib.sha256(json.dumps(key,sort_keys=True,default=str).encode()).hexdigest()
    if args.teacher_cache and args.teacher_cache.is_file():
        with np.load(args.teacher_cache,allow_pickle=True) as source:
            if str(source['key'].item())==digest:
                clip_names=[str(value) for value in source['clip_names']]
                posteriors={}
                for name,(rows,posterior) in zip(clip_names,source['records']):
                    index=next(i for i,clip in enumerate(data['clips']) if clip['name']==name)
                    posteriors[index]=(np.asarray(rows,dtype=np.int32),
                                       np.asarray(posterior,dtype=np.float16))
                stats=json.loads(source['stats'].item());stats.update(
                    cache='hit',cache_path=str(args.teacher_cache.resolve()))
                return posteriors,stats
            print(json.dumps(dict(warning='teacher cache key does not match this run; recomputing')),
                  flush=True)
    posteriors,stats=teacher_posteriors(teacher,data,order)
    if args.teacher_cache:
        args.teacher_cache.parent.mkdir(parents=True,exist_ok=True)
        # names and records must be written from ONE iteration order: a cache hit zips them, and
        # sorted(keys) vs dict insertion order mispairs clips, gluing one clip's posteriors onto
        # another clip's rows (which is what crashed rf200 with an out-of-range row index)
        ordered=sorted(posteriors)
        records=np.empty((len(ordered),2),dtype=object)  # per-cell assignment: np.array() would
        for position,index in enumerate(ordered):        # broadcast rows into the posterior shape
            rows,posterior=posteriors[index]
            records[position,0]=np.asarray(rows,dtype=np.int32)
            records[position,1]=np.asarray(posterior,dtype=np.float16)
        temporary=args.teacher_cache.with_name(f'{args.teacher_cache.name}.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream,key=digest,
                     clip_names=[data['clips'][index]['name'] for index in ordered],
                     records=records if posteriors else np.zeros(0,dtype=object),
                     stats=json.dumps(stats))
        temporary.replace(args.teacher_cache)
    stats.update(cache='miss',
                 cache_path=str(args.teacher_cache.resolve()) if args.teacher_cache else None)
    return posteriors,stats


if __name__=='__main__':
    main()

