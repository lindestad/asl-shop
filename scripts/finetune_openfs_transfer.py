"""Transfer the released OpenFS recogniser onto our gap-preserving packed timeline.

Research-only inputs: the released checkpoint and the pinned checkout it needs carry the OpenFS
Non-Commercial Research License ("Commercial use of the source code, model checkpoints, or
derivative works is strictly prohibited", data/external/openfs/source/LICENSE).  This script
imports upstream code the way scripts/evaluate_openfs_reference.py does (--source plus
--trusted-source); nothing here may be imported by model/, electron/, frontend/ or the packaged
backend.  The fine-tuned weights it writes are research-only derivative works: they deliberately
have no contract/task keys, so model.neural.load_checkpoint refuses them.

One set of released weights, three input adapters:

  control42  upstream parity: right- and left-hand detection tracks concatenated, 42 landmark
             coordinates bbox-midpoint / max-abs normalised to +-0.5, positional index = arange
             over detection order (frame gaps discarded).  This is the fidelity control: any gap
             versus the published FSboard test row (CER 6.29% / exact top-1 59.32%, see
             PUBLISHED_FSBOARD) is our own pack plumbing, which is the number that matters most.
             It is also checked per clip against the truth/prediction pairs the published harness
             recorded (--reference-report), which is a stricter version of the same question.
  gaps42     identical tokens and identical normalisation; only the positional index changes, to
             the column-90 frame deltas accumulated along the track, so a 30-frame detection gap
             stays 30 frames.  The positional encoding is index-driven with no learned table, so
             every released tensor loads unchanged and the delta is purely timeline semantics.
  gaps91     every packed timeline row as a token (no-detection frames included), all 91 channels,
             hand identity as the segment id (2 = no hand in that frame).  The pose embedding's
             first layer goes from Linear(42, 256) to Linear(91, 256): that layer's weight and bias
             are re-initialised as a unit and every other released tensor loads unchanged.

Streaming-chunk evaluation, because the app never feeds the recogniser a whole session: fixed
windows of --chunk-frames with --chunk-overlap overlap, one autoregressive decode per window, and
a deterministic merge that gives each character to the window whose centre is nearest to it
(merge_windows).  Normalisation and positional indices are recomputed per window, exactly like a
streaming client would.  Both modes report CER, letter accuracy, exact top-1 and the
deletion/substitution/insertion breakdown plus a per-window-count breakdown, so window-boundary
damage is visible; clips that need a single window must decode identically in both modes, which is
a free check on the chunker itself.

Fine-tuning stays small and never touches the test split: released weights, lr ~1e-5, a few
epochs, length-grouped batches capped by a shared frame budget (attention memory grows as
batch*heads*len^2 over the longest clip), auto amp (bf16 on Ampere and newer, fp16 plus a
GradScaler below it), and an overwrite guard on the output path.  Evaluation always runs fp32 so
the control is compared with the published row on the same arithmetic.

Real runs (VPS V100-SXM3, torch 2.9.1, peak measured per run; measured numbers live in
data/transfer-teacher.report.json):

  .venv/bin/python scripts/finetune_openfs_transfer.py --source data/external/openfs/source --trusted-source --adapter control42 gaps42 gaps91 --epochs 4 --limit-clips 30000 --batch-size 32 --validation-limit 3000 --eval-limit 3000 --report data/transfer-teacher.report.json

Smoke (no fine-tune, one adapter, 60 test clips, both modes):

  .venv/bin/python scripts/finetune_openfs_transfer.py --source data/external/openfs/source --trusted-source --adapter control42 --epochs 0 --eval-limit 60 --report data/transfer-teacher-smoke.report.json
"""
import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

# Before torch initialises its caching allocator; see scripts/train_openfs.py.
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.neural import CONTRACT
from model.offline_recognizer import PAD_FRAME,PAD_POSE,PAD_SEG,attention_entropy_loss,soft_monotonicity_loss
from model.recognition_metrics import ErrorTally
from scripts.eval_recognition import block_of,deltas
from scripts.train_openfs_recognizer import (augment_pose,augment_timeline,normalized_box,read_data,
                                             timeline_indices)

LICENSE=('OpenFS Non-Commercial Research License: the source code and released model checkpoints may '
         'be used, reproduced, modified and distributed solely for non-commercial research and '
         'education, and commercial use of the source code, model checkpoints, or derivative works '
         'is strictly prohibited.')
PUBLISHED_FSBOARD=dict(available=False,cer=.06286748,letter_accuracy=.93713252,
                       exact_sequence_accuracy=.59324335,clips=12521,characters=221100,errors=13900,exact=7428,
                       source='official OpenFS FSboard pretrained recognizer on the FSboard test split, '
                               'greedy autoregressive, no language model (data/openfs-reference.report.json)')
FRAMES_PER_SECOND=30  # nominal fps of the packed timeline; column 90 holds the frame delta in seconds
ADAPTERS=('control42','gaps42','gaps91')
DESCRIPTION={'control42':'upstream parity: 42-dim bbox-normalised tracks, arange detection indices',
             'gaps42':'same tokens and normalisation, positional index accumulated from real frame gaps',
             'gaps91':'whole packed timeline as tokens, 91 channels, released pose embedding re-initialised'}


def arange_positions(count):
    """Upstream index: one step per detection, so a dropped detection shifts nothing."""
    return np.arange(count,dtype=np.float64)


def gap_positions(values,track=None):
    """Timeline position in frames, accumulating the column-90 deltas: a 30-frame detection gap
    stays 30 frames.  With a boolean track the index is rebased onto that hand's first detection,
    matching the released per-hand index that starts at 0."""
    positions=timeline_indices(values)
    if track is not None: positions=positions[track]
    if not len(positions): return positions
    return positions-positions[0]


def window_tokens(values,adapter,rng=None,unmirror=False):
    """One window (or whole clip) of packed rows -> (poses, segments, positions); None when no hand
    is detected at all, because then a 42-dim adapter has nothing to encode.  control42 augments
    per detection track like the released pipeline; gaps42 augments the whole timeline rows so the
    delta column it accumulates stays consistent, and gaps91 uses our trainer's timeline recipe.
    unmirror flips the left hand's x back: our packed contract mirrors it so both hands face the
    same way, and the released model never saw that flip."""
    if unmirror:
        values=np.array(values,copy=True)
        values[:,45:87:2]*=-1
    if adapter=='gaps91':
        if rng is not None: values=augment_timeline(np.array(values,copy=True),rng)
        right,left=values[:,44]>0,values[:,89]>0
        segments=np.full(len(values),2,dtype=np.int64)
        segments[right]=0
        segments[left]=1
        segments[right&left]=0 if right.sum()>=left.sum() else 1
        return np.ascontiguousarray(values[:,:91],dtype=np.float32),segments,gap_positions(values)
    if adapter=='gaps42' and rng is not None: values=augment_timeline(np.array(values,copy=True),rng)
    poses,segments,positions=[],[],[]
    for base,segment in ((0,0),(45,1)):
        present=values[:,base+44]>0
        if not present.any(): continue
        pose=normalized_box(values[present][:,base:base+42].reshape(-1,21,2))
        if adapter=='control42' and rng is not None: pose=augment_pose(pose,rng)
        poses.append(pose.reshape(-1,42))
        segments.append(np.full(len(pose),segment,dtype=np.int64))
        positions.append(gap_positions(values,present) if adapter=='gaps42' else arange_positions(len(pose)))
    if not poses: return None
    return np.concatenate(poses),np.concatenate(segments),np.concatenate(positions)


def clip_tokens(data,index,adapter,augment=False,unmirror=False):
    """Tokens for one packed clip; the transcript reverses with the timeline so the target stays
    aligned (released do_reverse parity, p<0.3 when augmenting)."""
    row=data['clips'][index]
    values=np.array(data['x'][row['start']:row['end']])
    text=row['text']
    rng=clip_tokens.rng if augment else None
    if rng is not None and rng.random()<.3: values,text=values[::-1],text[::-1]
    tokens=window_tokens(values,adapter,rng,unmirror)
    if tokens is None: return None
    return (*tokens,text)
clip_tokens.rng=np.random.default_rng()  # main pins per epoch; the augment helpers need a Generator


def encode_text(text,chars):
    """Released preprocess_word parity: lowercase, one id per character (0..len(chars)-1), then
    BOS=len(chars) in front and EOS=0 at the end.  Characters outside the released alphabet are
    dropped, and alphabet_report counts them so the drop can never be silent."""
    ids=[len(chars)]
    for character in str(text).lower():
        index=chars.find(character)
        if index>=0: ids.append(index)
    return ids+[0]


def decode_text(tokens,chars):
    """Released token handling, mirroring the published harness in
    scripts/evaluate_openfs_reference.py: 0 ends the transcript, len(chars)+1 pads it, and an id at
    or above len(chars) is the start token leaking out, recorded as a replacement character (the
    published run reported zero of those, so CER stays comparable)."""
    result=[];invalid=0
    for token in tokens[1:]:
        token=int(token)
        if token in (0,len(chars)+1): break
        if token>=len(chars):
            result.append('\ufffd');invalid+=1;continue
        result.append(chars[token])
    return ''.join(result),invalid


def reached_limit(tokens,chars):
    """True when a decode ran out of steps before emitting the finish token."""
    return not any(int(token) in (0,len(chars)+1) for token in tokens[1:])


def alphabet_report(clips,chars):
    text=''.join(row['text'] for row in clips).upper()
    outside=sorted({character for character in text if chars.find(character.lower())<0})
    return dict(characters=len(text),outside_alphabet=sum(text.count(character) for character in outside),
                outside_distinct=outside,clips=len(clips),
                longest_transcript=max((len(row['text']) for row in clips),default=0))


def clip_windows(length,size,overlap):
    """Fixed streaming windows over a clip timeline: stride = size*(1-overlap), so consecutive
    windows share exactly size*overlap frames (a short final window shares less) and every frame is
    covered at least once.  Frames inside an overlap are decoded twice on purpose; merge_windows
    decides which copy survives.  A trailing window that the previous one already covered is
    dropped, so the last window is the short one rather than a redundant repeat."""
    if size<1: raise ValueError('Window size must be positive.')
    if not 0<=overlap<1: raise ValueError('Overlap must be in [0, 1).')
    stride=max(1,int(round(size*(1-overlap))))
    result=[];start=0
    while start<length:
        end=min(length,start+size)
        result.append((start,end))
        if end>=length: break
        start+=stride
    return result


def merge_windows(windows):
    """Deterministic overlap merge: each character is placed at its linear position inside its
    window's timeline span, and only the window whose centre is nearest to that position keeps it
    (an exact tie goes to the earlier window).  For uniform windows that cuts every shared region
    at its midpoint, so nothing is concatenated twice.  Windows are sorted by start and only the
    ones that decoded something compete, so the result depends on the geometry alone and a silent
    window cannot censor its neighbour's characters."""
    speaking=[(start,end,text) for start,end,text in sorted(windows) if text and end>start]
    centres=[(start+end)/2 for start,end,_ in speaking]
    result=[]
    for position,(start,end,text) in enumerate(speaking):
        span=end-start
        for order,character in enumerate(text):
            point=start+(order+.5)*span/len(text)
            owner=min(range(len(speaking)),key=lambda other:(abs(point-centres[other]),other))
            if owner==position: result.append(character)
    return ''.join(result)


def load_released(network,checkpoint,adapter):
    """Released weights into the adapter's network, reporting exactly what loaded.

    control42 and gaps42 must take every tensor, because the network is identical and only index
    semantics change.  gaps91 widens pose_embedding.0 from 42 to 91 inputs: a layer whose weight
    cannot fit is replaced as a unit (weight and bias), rather than keeping a bias trained against
    a different weight."""
    released=torch.load(checkpoint,map_location='cpu',weights_only=True)
    own=network.state_dict()
    dropped=[name for name in released if name not in own]
    if dropped:
        raise ValueError('Released tensors the adapter network has no place for: '
                         +json.dumps(dict(count=len(dropped),first=sorted(dropped)[:6],adapter=adapter)))
    mismatch=[name for name,tensor in released.items() if name in own and own[name].shape!=tensor.shape]
    replaced=set(mismatch)
    for name in mismatch:
        parent=name.rsplit('.',1)[0]
        replaced.update(key for key in own if key.rsplit('.',1)[0]==parent)
    kept={name:tensor for name,tensor in released.items() if name in own and name not in replaced}
    result=network.load_state_dict(kept,strict=False)
    if result.unexpected_keys or set(result.missing_keys)!=replaced:
        raise ValueError('Released checkpoint does not line up with the adapter: '
                         +json.dumps(dict(unexpected=sorted(result.unexpected_keys),
                                          missing=sorted(result.missing_keys),replaced=sorted(replaced),
                                          shape_mismatch=mismatch,adapter=adapter)))
    weights=dict(adapter=adapter,released_tensors=len(released),model_tensors=len(own),
                 loaded_tensors=len(kept),loaded_parameters=sum(tensor.numel() for tensor in kept.values()),
                 reinitialised=sorted(replaced),
                 reinitialised_parameters=sum(own[name].numel() for name in replaced),
                 shape_mismatch=mismatch,loaded_all=not replaced)
    if not weights['loaded_all'] and adapter!='gaps91':
        raise ValueError(f'{adapter} must load every released tensor, but these were re-initialised: '
                         +json.dumps(weights['reinitialised']))
    return weights


def groups_of(items,batch_size,frame_budget):
    """Length-sorted index groups: attention memory grows as batch*heads*len^2, so a group is
    capped by total padded frames as well as by clip count."""
    order=sorted(range(len(items)),key=lambda index:len(items[index][0]))
    result=[];current=[];longest=0
    for index in order:
        length=len(items[index][0])
        if current and ((len(current)+1)*max(longest,length)>frame_budget or len(current)>=batch_size):
            result.append(current);current=[];longest=0
        current.append(index);longest=max(longest,length)
    if current: result.append(current)
    return result


def collate(selected,items,device):
    """Pad one group with the released sentinels (-2 poses, 32 segments, 1024 positions) and hand
    the upstream forward its (B, L, channels, 1) pose tensor."""
    longest=max(len(items[index][0]) for index in selected)
    channels=items[0][0].shape[1]
    poses=np.full((len(selected),longest,channels),PAD_POSE,dtype=np.float32)
    segments=np.full((len(selected),longest),PAD_SEG,dtype=np.int64)
    positions=np.full((len(selected),longest),PAD_FRAME,dtype=np.float64)
    for row,index in enumerate(selected):
        pose,segment,position=items[index][:3]
        poses[row,:len(pose)]=pose
        segments[row,:len(segment)]=segment
        positions[row,:len(position)]=position
    return (torch.from_numpy(poses).unsqueeze(-1).to(device),torch.from_numpy(segments).to(device),
            torch.from_numpy(positions).to(device))


@torch.inference_mode()
def decode(network,chars,items,device,*,max_decode,batch_size,frame_budget):
    """Greedy autoregressive decode with the released generate, fp32 so the control's arithmetic
    matches the published harness."""
    texts=[None]*len(items);truncated=invalid=0
    for selected in groups_of(items,batch_size,frame_budget):
        poses,segments,positions=collate(selected,items,device)
        generated=network.generate(poses,segments,positions,max_len=max_decode,bos_token_id=len(chars),
                                   eos_token_id=0,pad_token_id=len(chars)+1)['output_ids']
        for row,index in enumerate(selected):
            tokens=generated[row].tolist()
            truncated+=reached_limit(tokens,chars)
            texts[index],invalid_count=decode_text(tokens,chars)
            invalid+=invalid_count
    return texts,dict(decode_limit_hits=int(truncated),invalid_special_tokens=int(invalid))


def evaluate(network,chars,data,indexes,adapter,device,mode,options):
    """Decode the same clips once, whole or through streaming windows, and tally the metric set."""
    args=SimpleNamespace(**vars(options))
    args.batch_size=options.eval_batch_size or options.batch_size
    started=time.perf_counter()
    windows_of={}
    if mode=='full-clip':
        items=[];owner=[];empty=0
        for index in indexes:
            tokens=clip_tokens(data,index,adapter,unmirror=options.unmirror_left)
            if tokens is None: empty+=1; continue
            items.append(tokens[:3]);owner.append(index);windows_of[index]=1
        texts,notes=decode(network,chars,items,device,max_decode=options.max_decode,
                           batch_size=args.batch_size,frame_budget=options.frame_budget)
        decoded={index:text for index,text in zip(owner,texts)}
    else:
        items=[];owner=[];empty=0
        for index in indexes:
            row=data['clips'][index]
            spans=clip_windows(row['end']-row['start'],options.chunk_frames,options.chunk_overlap)
            windows_of[index]=len(spans)
            for start,end in spans:
                values=np.array(data['x'][row['start']+start:row['start']+end])
                tokens=window_tokens(values,adapter,unmirror=options.unmirror_left)
                if tokens is None: empty+=1; continue
                items.append(tokens[:3]);owner.append((index,start,end))
        texts,notes=decode(network,chars,items,device,max_decode=options.chunk_max_decode,
                           batch_size=args.batch_size,frame_budget=options.frame_budget)
        rows=defaultdict(list)
        for (index,start,end),text in zip(owner,texts): rows[index].append((start,end,text))
        decoded={index:merge_windows(rows[index]) for index in rows}
    tally=ErrorTally();by_count=defaultdict(ErrorTally)
    for index,text in decoded.items():
        expected=data['clips'][index]['text'].strip().upper()
        tally.add(expected,text.strip().upper())
        by_count[windows_of[index]].add(expected,text.strip().upper())
    summary=block_of(tally,requested=len(indexes),decoded=len(decoded),empty_windows=empty,**notes,
                     mode=mode,windows=sum(windows_of.values()),
                     max_decode=options.max_decode if mode=='full-clip' else options.chunk_max_decode,
                     chunk_frames=options.chunk_frames if mode=='chunked' else None,
                     chunk_overlap=options.chunk_overlap if mode=='chunked' else None,
                     decoder='released greedy autoregressive; no dictionary or language correction',
                     seconds=round(time.perf_counter()-started,1),
                     by_window_count={str(key):value.summary() for key,value in sorted(by_count.items())})
    return summary,decoded


def training_batches(data,chars,adapter,batch_size,frame_budget,augment=False,shuffle=False,limit=None,
                     unmirror=False,rng=None):
    """Length-grouped training batches (groups shuffled, lengths sorted inside a shuffle window)
    with targets in the released token layout: BOS..EOS padded with -100.  rng drives the clip
    sample, the batch order and the augmentation, so a caller that pins it gets the same training
    data for every adapter and every process: without that, a run in another process would fine-tune
    on a different random sample and the adapters would not be paired."""
    rng=rng or np.random.default_rng()  # the augment helpers want a Generator, not the numpy module
    indices=np.arange(len(data['clips']))
    if shuffle: rng.shuffle(indices)
    if limit: indices=indices[:limit]
    windows=[sorted(indices[i:i+batch_size*50],key=lambda j:data['clips'][j]['end']-data['clips'][j]['start'])
             for i in range(0,len(indices),batch_size*50)]
    batches=[]
    for window in windows:
        current=[];longest=0
        for index in window:
            # timeline frames bound the token count for every adapter, so the budget caps padding cost
            length=data['clips'][index]['end']-data['clips'][index]['start']
            if current and ((len(current)+1)*max(longest,length)>frame_budget or len(current)>=batch_size):
                batches.append(current);current=[];longest=0
            current.append(index);longest=max(longest,length)
        if current: batches.append(current)
    if shuffle: rng.shuffle(batches)
    for selected in batches:
        items=[];targets=[];kept=[]
        for index in selected:
            tokens=clip_tokens(data,index,adapter,augment,unmirror)
            if tokens is None: continue
            items.append(tokens[:3])
            targets.append(np.array(encode_text(tokens[3],chars),dtype=np.int64))
            kept.append(index)
        if not items: continue
        padded=np.full((len(items),max(len(target) for target in targets)),-100,dtype=np.int64)
        for row,target in enumerate(targets): padded[row,:len(target)]=target
        yield collate(list(range(len(items))),items,'cpu')+(torch.from_numpy(padded),kept)


def fine_tune_step(network,poses,segments,positions,targets,epoch,options,char_size,pad_character):
    """The paper's three losses in the released layout: logits at step t predict target t+1."""
    from torch.nn import functional as F
    inputs=torch.where(targets[:,:-1]==-100,pad_character,targets[:,:-1])
    labels=targets[:,1:]
    with torch.autocast('cuda',dtype=options.amp_dtype or torch.float16,enabled=options.amp!='off'):
        output=network(poses,segments,positions,inputs)
        logits=output['logits'].float()
    ce=F.cross_entropy(logits.reshape(-1,char_size),labels.reshape(-1))
    loss=options.ce_weight*ce;entropy=monotonicity=0.
    if epoch>options.warm_epoch:
        cross_attns=torch.stack(output['cross_attns'],dim=1)
        if options.entropy_weight:
            entropy=attention_entropy_loss(cross_attns,segments,targets)
            loss=loss+options.entropy_weight*entropy
        if options.monotonicity_weight:
            monotonicity=soft_monotonicity_loss(cross_attns,segments,targets)
            loss=loss+options.monotonicity_weight*monotonicity
    return loss,ce,entropy,monotonicity


def save(network,options,adapter,chars,architecture,weights,metrics,history,best):
    """Research-only checkpoint: no contract/task keys, so no app loader will accept it."""
    options.output.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(task='openfs-transfer',research_only=True,license=LICENSE,adapter=adapter,
                 source_checkpoint=str(options.checkpoint.resolve()),alphabet=chars,
                 architecture=dict(input_dim=91 if adapter=='gaps91' else 42,d_model=architecture['d_model'],
                                   heads=architecture['nhead'],layers=architecture['num_layers'],
                                   feedforward=architecture['dim_feedforward'],
                                   activation=architecture['activation']),
                 weights=weights,state_dict={key:value.detach().cpu() for key,value in network.state_dict().items()},
                 metrics=metrics,best_validation_cer=best,history=history,seed=options.seed,
                 dataset=str(options.data.resolve()),experimental=True,
                 note='research-only derivative of the released OpenFS checkpoint; not an app model')
    with tempfile.NamedTemporaryFile(dir=options.output.parent,suffix='.pt',delete=False) as file:
        temporary=Path(file.name)
    try:
        torch.save(payload,temporary)
        temporary.replace(options.output)
    finally:
        temporary.unlink(missing_ok=True)
    options.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


def run_fine_tuning(network,chars,train,validation,adapter,options,architecture,weights):
    """Released weights, small lr, the paper's losses, validation-only selection; the test split is
    never read here."""
    optimizer=torch.optim.Adam(network.parameters(),lr=options.learning_rate)
    scaler=torch.amp.GradScaler('cuda',enabled=options.amp=='fp16')
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,options.lr_step,gamma=options.lr_gamma)
    char_size,pad_character=len(chars)+1,len(chars)+1  # 61 logits, id 61 pads the character embedding
    history=[];run_started=time.perf_counter();step=0;best=float('inf')
    for epoch in range(1,options.epochs+1):
        network.train();started=time.perf_counter()
        loss_sum=ce_sum=ent_sum=mono_sum=steps=clips=0
        # pinned per epoch: the same clip sample and the same augment decisions for every adapter,
        # so control42 and the gap adapters see identical training data in identical order
        epoch_rng=np.random.default_rng(options.seed+epoch)
        clip_tokens.rng=epoch_rng
        for poses,segments,positions,targets,kept in training_batches(train,chars,adapter,options.batch_size,
                                                                      options.train_frame_budget,
                                                                      augment=not options.no_augment,shuffle=True,
                                                                      limit=options.limit_clips,
                                                                      unmirror=options.unmirror_left,
                                                                      rng=epoch_rng):
            poses,segments=poses.to(options.device),segments.to(options.device)
            positions,targets=positions.to(options.device),targets.to(options.device)
            optimizer.zero_grad(set_to_none=True)
            loss,ce,entropy,monotonicity=fine_tune_step(network,poses,segments,positions,targets,epoch,
                                                        options,char_size,pad_character)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss: '+json.dumps(dict(epoch=epoch,step=step,adapter=adapter,
                        loss=float(loss.detach()),clips=[train['clips'][index]['name'] for index in kept[:4]],
                        frames=int(segments.shape[1]),targets=int(targets.shape[1]),amp=options.amp)))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum+=loss.item();ce_sum+=ce.item()
            ent_sum+=entropy.detach().item() if torch.is_tensor(entropy) else 0.
            mono_sum+=monotonicity.detach().item() if torch.is_tensor(monotonicity) else 0.
            steps+=1;step+=1;clips+=len(kept)
            if steps%50==0:
                print(json.dumps(dict(adapter=adapter,epoch=epoch,steps=steps,loss=round(loss_sum/steps,5),
                                      seconds=round(time.perf_counter()-started,1))),flush=True)
        network.eval()
        metrics,_=evaluate(network,chars,validation,list(range(min(options.validation_limit,
                                                                len(validation['clips'])))),adapter,
                           options.device,'full-clip',options)
        row=dict(adapter=adapter,epoch=epoch,batches=steps,clips=clips,loss=loss_sum/max(steps,1),
                 ce=ce_sum/max(steps,1),attention_entropy=ent_sum/max(steps,1),
                 monotonicity=mono_sum/max(steps,1),validation_cer=metrics['cer'],
                 validation_letter=metrics['letter_accuracy'],
                 validation_exact=metrics['exact_sequence_accuracy'],
                 validation_clips=metrics['clips'],validation_seconds=metrics['seconds'],
                 training_seconds=round(time.perf_counter()-started-metrics['seconds'],1),
                 elapsed_seconds=round(time.perf_counter()-run_started,1))
        history.append(row);print(json.dumps(row),flush=True)
        if row['validation_cer']<best:
            best=row['validation_cer']
            save(network,options,adapter,chars,architecture,weights,metrics,history,best)
        scheduler.step()
    return history,best


def load_reference(options):
    """Instantiate the released class exactly as scripts/evaluate_openfs_reference.py does."""
    source=options.source.resolve()
    if not (source/'lib/models/recognizer.py').is_file():
        raise ValueError('OpenFS source checkout is missing lib/models/recognizer.py.')
    sys.path.insert(0,str(source))
    module=importlib.import_module('lib.models.recognizer')
    if Path(module.__file__).resolve()!=(source/'lib/models/recognizer.py').resolve():
        raise ValueError('Another OpenFS/lib module is already loaded; restart with the intended checkout.')
    config=yaml.safe_load((options.checkpoint.parent/'.hydra/config.yaml').read_text(encoding='utf-8'))
    chars=config['decode']['chars']
    if not isinstance(chars,str) or not chars or len(set(chars))!=len(chars):
        raise ValueError('Released checkpoint alphabet is invalid.')
    architecture={key:value for key,value in config['recognizer'].items()
                  if not key.startswith('_') and key!='ckpt_path'}
    try:
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):  # a vendored checkout without git metadata is still usable
        commit='unknown'
    return module.EncoderDecoderTransformer,chars,architecture,commit


def published_row(options):
    if options.reference_report and options.reference_report.is_file():
        return json.loads(options.reference_report.read_text(encoding='utf-8'))
    return dict(PUBLISHED_FSBOARD)


def reference_matches(options,evaluation,selection,decoded):
    """Compare the control's full-clip predictions with the per-clip predictions the published
    harness recorded for the same clips (same released weights, so a mismatch is our plumbing)."""
    wanted={entry['name']:(entry['truth'].upper(),entry['prediction'].upper())
            for entry in options.published.get('examples',[])}
    position={evaluation['clips'][index]['name']:index for index in selection}
    rows=[]
    for name,(truth,expected) in wanted.items():
        index=position.get(name)
        if index is None or index not in decoded: continue
        rows.append(dict(name=name,truth=truth,reference=expected,ours=decoded[index].upper(),
                         identical=decoded[index].upper()==expected))
    return dict(clips=len(rows),identical=sum(row['identical'] for row in rows),
                mismatches=[{key:row[key] for key in ('name','truth','reference','ours')}
                            for row in rows if not row['identical']])


def main():
    p=argparse.ArgumentParser(description='Fine-tune the released OpenFS recogniser onto the packed FSboard timeline.')
    p.add_argument('--data',type=Path,default=ROOT/'data/openfs-packed')
    p.add_argument('--source',type=Path,help='pinned OpenFS checkout (research-licensed)')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'data/openfs-checkpoints/recognizer_fsboard/best_model.pth')
    p.add_argument('--trusted-source',action='store_true',
                   help='explicitly acknowledge loading the research-licensed upstream source and checkpoint')
    p.add_argument('--adapter',nargs='+',choices=ADAPTERS,default=['control42'])
    p.add_argument('--output',type=Path,help='single fine-tune target; defaults to --run-prefix-<adapter>/recognizer.pt')
    p.add_argument('--run-prefix',default='data/models/openfs-transfer')
    p.add_argument('--epochs',type=int,default=4,help='0 evaluates the released weights only')
    p.add_argument('--limit-clips',type=int)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--lr-step',type=int,default=3)
    p.add_argument('--lr-gamma',type=float,default=.3)
    p.add_argument('--ce-weight',type=float,default=1.)
    p.add_argument('--entropy-weight',type=float,default=.8)
    p.add_argument('--monotonicity-weight',type=float,default=1.)
    p.add_argument('--warm-epoch',type=int,default=0,help='the two auxiliary losses start after this epoch')
    p.add_argument('--no-augment',action='store_true')
    p.add_argument('--fp32',action='store_true',help='disable autocast (same as --amp off)')
    p.add_argument('--amp',choices=['auto','bf16','fp16'],default='auto',
                   help='auto: bf16 on Ampere and newer, fp16 with a GradScaler below it (Volta has no bf16 '
                        'tensor cores and flash attention needs sm_80)')
    p.add_argument('--frame-budget',type=int,default=2400,help='max padded frames in one decode group')
    p.add_argument('--train-frame-budget',type=int,default=6000,help='max padded frames in one training batch')
    p.add_argument('--max-decode',type=int,default=128,help='full-clip AR steps (the published harness used 128)')
    p.add_argument('--chunk-frames',type=int,default=300,help='streaming window length in timeline frames')
    p.add_argument('--chunk-overlap',type=float,default=.5)
    p.add_argument('--chunk-max-decode',type=int,default=48,help='AR steps per streaming window')
    p.add_argument('--eval-split',default='test')
    p.add_argument('--eval-limit',type=int,help='clips to evaluate; default is the whole split')
    p.add_argument('--eval-reference',action='store_true',
                   help='evaluate only the clips the reference report recorded predictions for, which turns '
                        'the control adapter into a per-clip plumbing check against the published run')
    p.add_argument('--unmirror-left',action='store_true',
                   help='flip the left hand x back: our packed contract mirrors it so both hands face the same '
                        'way, and the released model never saw that flip')
    p.add_argument('--validation-limit',type=int,default=3000)
    p.add_argument('--eval-batch-size',type=int,help='defaults to --batch-size')
    p.add_argument('--eval-modes',nargs='+',choices=['full-clip','chunked'],default=['full-clip','chunked'])
    p.add_argument('--reference-report',type=Path,default=ROOT/'data/openfs-reference.report.json',
                   help='published run to compare the control against: headline numbers plus per-clip examples')
    p.add_argument('--report',type=Path,default=ROOT/'data/transfer-teacher.report.json')
    p.add_argument('--merge-report',action='store_true',help='fold these results into an existing report file')
    p.add_argument('--merge-reports',type=Path,nargs='*',default=[],
                   help='report files from other runs of this script to fold into the written report; '
                        'nothing is recomputed, the adapter blocks are copied and the deltas rebuilt')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--force',action='store_true',help='replace an existing --output; throwaway runs only')
    args=p.parse_args()
    if not args.trusted_source:
        p.error('Loading the research-licensed upstream source and checkpoint needs --trusted-source.')
    if not args.source: p.error('--source is required (pinned OpenFS checkout).')
    if len(args.adapter)>1 and args.output: p.error('--output only works with a single --adapter.')
    args.outputs={adapter:args.output or Path(f'{args.run_prefix}-{adapter}/recognizer.pt') for adapter in args.adapter}
    if args.epochs:
        for adapter,output in args.outputs.items():
            if output.exists() and not args.force:
                raise ValueError(f'{output} already exists; use a fresh run name (never overwrite checkpoints).')
    cuda=args.device.startswith('cuda')
    amp='off' if args.fp32 or not cuda else args.amp
    if amp=='auto': amp='bf16' if cuda and torch.cuda.get_device_capability(args.device)[0]>=8 else 'fp16'
    args.amp=amp; args.amp_dtype={'bf16':torch.bfloat16,'fp16':torch.float16}.get(amp)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    clip_tokens.rng=np.random.default_rng(args.seed)  # run_fine_tuning re-pins this per epoch
    torch.set_num_threads(4)
    started=time.perf_counter()
    EncoderDecoder,chars,architecture,commit=load_reference(args)
    args.alphabet=chars
    args.published=published_row(args)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    schema=json.loads((args.data/'dataset.json').read_text(encoding='utf-8'))
    if schema['contract']!=CONTRACT: raise ValueError('Packed dataset contract mismatch; prepare it again.')
    splits=[set(part['signers']) for part in schema['partitions']]
    if any(a&b for i,a in enumerate(splits) for b in splits[i+1:]): raise ValueError('Signer leakage in dataset metadata.')
    evaluation=read_data(args.data,args.eval_split)
    if args.eval_reference:
        wanted=[entry['name'] for entry in args.published.get('examples',[])]
        position={row['name']:index for index,row in enumerate(evaluation['clips'])}
        selection=[position[name] for name in wanted if name in position]
        if not selection: raise ValueError('The reference report lists no clips from this split.')
        selection_note=f'the {len(selection)} clips the reference report recorded predictions for'
    elif args.eval_limit:
        selection=np.random.default_rng(args.seed).choice(len(evaluation['clips']),
                                                          min(args.eval_limit,len(evaluation['clips'])),
                                                          replace=False).tolist()
        selection_note=f'seed{args.seed} random subset'
    else:
        selection=list(range(len(evaluation['clips'])))
        selection_note='all clips'
    train=read_data(args.data,'train') if args.epochs else None
    validation=read_data(args.data,'validation') if args.epochs else None
    report=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),torch=torch.__version__,numpy=np.__version__,
                device=args.device,gpu=torch.cuda.get_device_name(0) if cuda else None,amp=args.amp,
                seed=args.seed,commands=[' '.join(sys.argv)],
                license=dict(terms=LICENSE,packaged=False,checkpoint=str(args.checkpoint.resolve()),
                             source=str(args.source.resolve()),source_commit=commit,
                             derivatives='fine-tuned weights under data/models/openfs-transfer-* are research-only '
                                         'derivative works; they carry no contract/task keys, so the app backend '
                                         'cannot load them'),
                released=dict(checkpoint=str(args.checkpoint.resolve()),architecture=architecture,
                              alphabet=chars,alphabet_size=len(chars),
                              tensors=len(state),parameters=sum(tensor.numel() for tensor in state.values())),
                published={key:args.published[key] for key in
                           ('cer','letter_accuracy','exact_sequence_accuracy','clips','characters','source')
                           if key in args.published},
                pack=dict(data=str(args.data.resolve()),contract=schema['contract'],
                          partitions={part['split']:dict(clips=part['clips'],frames=part['frames'])
                                      for part in schema['partitions']}),
                evaluation=dict(split=args.eval_split,clips=len(selection),characters=sum(
                                len(evaluation['clips'][index]['text']) for index in selection),
                                selection=selection_note,
                                modes=args.eval_modes,chunk_frames=args.chunk_frames,
                                chunk_overlap=args.chunk_overlap,max_decode=args.max_decode,
                                chunk_max_decode=args.chunk_max_decode,frame_budget=args.frame_budget,
                                eval_batch_size=args.eval_batch_size or args.batch_size,
                                unmirror_left=args.unmirror_left,
                                alphabet=alphabet_report([evaluation['clips'][index] for index in selection],chars)),
                fine_tune=dict(epochs=args.epochs,limit_clips=args.limit_clips,batch_size=args.batch_size,
                               learning_rate=args.learning_rate,lr_step=args.lr_step,lr_gamma=args.lr_gamma,
                               warm_epoch=args.warm_epoch,ce_weight=args.ce_weight,
                               entropy_weight=args.entropy_weight,monotonicity_weight=args.monotonicity_weight,
                               augment=not args.no_augment,train_frame_budget=args.train_frame_budget,
                               validation_limit=args.validation_limit,
                               test_split_used='never for training or model selection'),
                adapters={},comparison={})
    if args.merge_report and args.report.is_file():
        previous=json.loads(args.report.read_text(encoding='utf-8'))
        report={**previous,**report,
                'adapters':{**previous.get('adapters',{}),**report['adapters']},
                'commands':previous.get('commands',[])+report['commands'],
                'merged_runs':previous.get('merged_runs',[])+[report['generated']]}
    peak_overall=0
    for adapter in args.adapter:
        adapter_started=time.perf_counter()
        args.output=args.outputs[adapter]
        network=EncoderDecoder(char_size=len(chars)+1,
                               **{**architecture,'input_dim':91 if adapter=='gaps91' else 42})
        weights=load_released(network,args.checkpoint,adapter)
        network.to(args.device).eval()
        if cuda: torch.cuda.reset_peak_memory_stats()
        print(json.dumps(dict(adapter=adapter,input_dim=network.input_dim,weights=weights)),flush=True)
        block=dict(description=DESCRIPTION[adapter],input_dim=network.input_dim,weights=weights,
                   released={},fine_tuned={})
        per_mode={}
        for mode in args.eval_modes:
            summary,decoded=evaluate(network,chars,evaluation,selection,adapter,args.device,mode,args)
            block['released'][mode]=summary;per_mode[mode]=decoded
            emit(adapter,'released',mode,summary)
        if 'full-clip' in per_mode and 'chunked' in per_mode:
            shared=[index for index in per_mode['chunked'] if index in per_mode['full-clip']
                    if len(clip_windows(evaluation['clips'][index]['end']-evaluation['clips'][index]['start'],
                                        args.chunk_frames,args.chunk_overlap))==1]
            block['released']['single_window_agreement']=dict(
                note='clips that need only one streaming window: their chunked text should equal the '
                     'full-clip text, which is the plumbing check rather than an accuracy claim',
                clips=len(shared),identical=sum(per_mode['chunked'][index]==per_mode['full-clip'][index]
                                                for index in shared))
        if adapter=='control42':
            matches=reference_matches(args,evaluation,selection,per_mode.get('full-clip',{}))
            if matches['clips']: block['released']['reference_examples']=matches
        if args.epochs:
            history,best=run_fine_tuning(network,chars,train,validation,adapter,args,architecture,weights)
            block['fine_tune']=dict(history=history,best_validation_cer=best,checkpoint=str(args.output.resolve()))
            payload=torch.load(args.output,map_location='cpu',weights_only=True)
            network.load_state_dict(payload['state_dict'],strict=True)
            network.eval()
            for mode in args.eval_modes:
                summary,decoded=evaluate(network,chars,evaluation,selection,adapter,args.device,mode,args)
                block['fine_tuned'][mode]=summary
                emit(adapter,'fine_tuned',mode,summary)
        if cuda:
            peak=torch.cuda.max_memory_allocated();peak_overall=max(peak_overall,peak)
            block['peak_vram_mib']=round(peak/2**20)
        block['seconds']=round(time.perf_counter()-adapter_started,1)
        del network
        if cuda: torch.cuda.empty_cache()
        report['adapters'][adapter]=block
    compare(report,args.published)
    report['peak_vram_mib']=round(peak_overall/2**20)
    report['seconds']=round(time.perf_counter()-started,1)
    if args.merge_reports:
        for path in args.merge_reports:
            merge_into(report,json.loads(path.read_text(encoding='utf-8')),str(path))
        compare(report,args.published)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(args.report),seconds=report['seconds'],peak_vram_mib=report['peak_vram_mib'],
                          adapters=sorted(report['adapters']),
                          control_vs_published=report['comparison'].get('control_vs_published'))),flush=True)


def compare(report,published):
    """Paired deltas: the same clips, the same weights, one thing changed at a time.  Run again
    after merging so cross-adapter deltas appear even when the adapters ran in separate processes."""
    comparison=report.setdefault('comparison',{})
    for key in [key for key in comparison if key.endswith(('_minus_control42_released','_minus_control42_fine_tuned'))
                or key.startswith('chunked_cost_') or key=='chunked_cost']:
        del comparison[key]
    for stage in ('released','fine_tuned'):
        control=report['adapters'].get('control42',{}).get(stage,{}).get('full-clip')
        for adapter in report['adapters']:
            block=report['adapters'][adapter].get(stage,{})
            if not block.get('full-clip'): continue
            if adapter!='control42' and control:
                comparison[f'{adapter}_minus_control42_{stage}']=deltas(control,block['full-clip'])
            if block.get('chunked'):
                comparison.setdefault(f'chunked_cost_{stage}',{})[adapter]=deltas(block['full-clip'],
                                                                                 block['chunked'])
    control=report['adapters'].get('control42',{}).get('released',{}).get('full-clip')
    if control and 'cer' in published and 'exact_sequence_accuracy' in published:
        comparison['control_vs_published']=dict(
            cer_points=round(100*(published['cer']-control['cer']),3),
            exact_sequence_points=round(100*(control['exact_sequence_accuracy']-
                                             published['exact_sequence_accuracy']),3),
            published=dict(cer=published['cer'],exact_sequence_accuracy=published['exact_sequence_accuracy'],
                           clips=published.get('clips'),characters=published.get('characters')),
            ours=dict(cer=round(control['cer'],5),letter_accuracy=round(control['letter_accuracy'],5),
                      exact_sequence_accuracy=round(control['exact_sequence_accuracy'],5),
                      clips=control['clips'],characters=control['characters']),
            note='both rows use the released weights on the FSboard test split, so the gap is our '
                 'pack plumbing plus decode-time sampling of the same clips')


def merge_into(report,other,label):
    """Fold another run of this script into the report we are about to write, so one run per
    adapter still produces one file.  Each run's own settings are kept under merged_from."""
    for adapter,block in other.get('adapters',{}).items():
        report['adapters'].setdefault(adapter,block)
    merged=report.setdefault('merged_from',[])
    if not any(row['report']==label for row in merged):
        merged.append(dict(report=label,adapters=sorted(other.get('adapters',{})),
                           evaluation=other.get('evaluation',{}),peak_vram_mib=other.get('peak_vram_mib'),
                           seconds=other.get('seconds')))
    report['commands']=report.get('commands',[])+[command for command in other.get('commands',[])
                                                  if command not in report.get('commands',[])]


def emit(adapter,stage,mode,summary):
    print(json.dumps(dict(adapter=adapter,stage=stage,mode=mode,
                          **{key:summary[key] for key in ('cer','letter_accuracy','exact_sequence_accuracy',
                                                          'deletions','substitutions','insertions','clips','seconds')})),
          flush=True)


if __name__=='__main__':
    main()
