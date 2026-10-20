"""Offline (non-causal) OpenFS-recipe trainer over the packed FSboard contract.

Reproduction of the published recogniser (arXiv 2602.22949: 93.7 letter accuracy / 59.4 exact
top-1 on the FSboard test split) inside this repo's own harness: transformer encoder +
autoregressive cross-attention decoder with dual-level positional encodings and the three
paper losses (character cross-entropy, signing-hand attention entropy, soft monotonicity)
written from the published formulas.  CTC is not used.  `--input-representation` isolates
what our contract changes relative to the paper:

  compressed  upstream reproduction: per-hand detection tracks concatenated in detection
              order, arange positional indices (frame gaps discarded), bbox-midpoint /
              max-abs +-0.5 normalisation, 42 channels.
  gaps        the app's real domain: the whole packed timeline (including no-detection
              frames) as tokens, positional indices accumulated from the column-90 frame
              deltas (irregular timing preserved), all 91 wrist-relative channels.
  wrist2d     same timeline and indices as gaps, but only the 42 wrist-relative landmark
              coordinates of the clip's dominant hand — ablation for whether the other
              49 packed channels matter.

Full runs (schedule serially, one per 5080; the test split is never touched here):

  .venv/bin/python scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation compressed --output data/models/openfs-recognizer-compressed/recognizer.pt
  .venv/bin/python scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation gaps        --output data/models/openfs-recognizer-gaps/recognizer.pt
  .venv/bin/python scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation wrist2d     --output data/models/openfs-recognizer-wrist2d/recognizer.pt

Measured on this machine (torch 2.14.1, single RTX 5080):
- Training with inline augmentation at batch 16, measured on single 3,000-clip epochs: compressed 6.7 s,
  gaps 6.6 s, wrist2d 5.3 s — 450-560 clips/s; single-process augmentation is the limiter (the paper's
  pipeline reaches 482-648 clips/s with 16 dataloader workers).  One epoch over 126,131 clips is
  ~4.5-5 min of training, so budget ~11-13 h for the paper's 100 epochs and ~2.5-3 h for a 20-epoch
  probe (--lr-step 16 keeps the decay in the last fifth).  --no-augment is faster but the SOTA recipe
  relies on the augmentation.
- Validation pass over all 12,015 clips (greedy AR decode, batch 32): ~75 s, so budget ~1.5 min per epoch.
- Peak torch.cuda.max_memory_allocated: batch 8 smokes peak at 554 MiB, batch 16 at 1.0-1.2 GB; add
  ~0.4 GB for the CUDA context.  Attention memory grows as batch*heads*len^2 over the longest clip, so
  batch 32 needs roughly 2.5-3 GB and batch 64 roughly 4-5 GB: use those only on a largely idle card
  (batch 32 OOM'd here while another process held ~13 GB of the 16 GB).

Smoke run (one variant, ~1 min, well under 1.2 GB; prints the untrained baseline first and
peak torch.cuda.max_memory_allocated at the end; --warm-epoch 0 makes the two auxiliary
losses fire immediately, which the paper's real schedule only does from epoch 6):

  .venv/bin/python scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation gaps --output data/models/openfs-recognizer-gaps/smoke.pt --epochs 1 --limit-clips 400 --validation-limit 100 --batch-size 8 --evaluate-initial --warm-epoch 0

Score an existing checkpoint on any split without training (the report goes to
<output stem>.eval-report.json, the checkpoint is never rewritten):

  .venv/bin/python scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation compressed --output data/models/openfs-recognizer-compressed/test-eval.pt --resume data/models/openfs-recognizer-compressed/recognizer.pt --evaluate-only --evaluate-splits test
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# Set before torch initializes its caching allocator, which otherwise reserves
# about twice what the sequence model keeps live (measured on the causal TCN:
# 1230 MiB reserved, 560 MiB live).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.neural import CONTRACT,VOCAB
from model.offline_recognizer import (EOS,BOS,PAD_CHAR,PAD_FRAME,PAD_POSE,PAD_SEG,Recognizer,
                                      attention_entropy_loss,decode_text,encode_text,
                                      soft_monotonicity_loss)
from scripts.train_neural import edit_distance


def read_data(directory,split):
    return dict(x=np.load(directory/f'{split}_frames.npy',mmap_mode='r'),
                clips=json.loads((directory/f'{split}_clips.json').read_text(encoding='utf-8')))


def resample(values,length):
    """Linear temporal resampling of a (T, features) or (T, joints, 2) array (upstream parity)."""
    positions=np.linspace(0,len(values)-1,length)
    low=np.floor(positions).astype(np.int64)
    high=np.minimum(low+1,len(values)-1)
    weight=(positions-low).reshape((-1,)+(1,)*(values.ndim-1))
    return (values[low]*(1-weight)+values[high]*weight).astype(np.float32)


def affine_matrix(rng):
    """2D counterpart of upstream's 3D affine (scale 0.8-1.2, shear +-0.2, rotation +-30 deg)."""
    angle=math.radians(rng.uniform(-30,30))
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    shear=np.array([[1,rng.uniform(-.2,.2)],[0,1]])
    scale=np.diag(rng.uniform(.8,1.2,2).astype(np.float32))
    return rotation@shear@scale,rng.uniform(-.1,.1,2).astype(np.float32)


def augment_pose(pose,rng):
    """Upstream augmentation parity on a (T,21,2) track: resample 0.5-1.5 at p<0.8, affine at
    p<0.75, temporal mask p<0.5 (20-40% of frames), spatial mask p<0.5 (5-10% of joints)."""
    if rng.random()<.8:
        pose=resample(pose,max(1,int(len(pose)*rng.uniform(.5,1.5))))
    if rng.random()<.75:
        matrix,shift=affine_matrix(rng)
        pose=pose@matrix+shift
    if rng.random()<.5:
        size=int(rng.uniform(.2,.4)*len(pose))
        if size>=len(pose):
            pose[:]=0
        else:
            start=int(rng.integers(0,len(pose)-size))
            pose[start:start+size]=0
    if rng.random()<.5:
        size=int(rng.uniform(.05,.1)*21)
        if size:
            pose[:,rng.choice(21,size,replace=False)]=0
    return pose


def augment_timeline(values,rng):
    """The same augmentation applied to whole packed (T,91) rows for the timeline variants;
    the two hand slots share one affine so the merged tokens stay self-consistent."""
    if rng.random()<.8:
        length=max(1,int(len(values)*rng.uniform(.5,1.5)))
        scale=len(values)/length
        values=np.concatenate((resample(values[:,:90],length),(resample(values[:,-1:],length)*scale)),axis=1)
        values[:,44]*=(values[:,44]>.5)
        values[:,89]*=(values[:,89]>.5)
    if rng.random()<.75:
        matrix,shift=affine_matrix(rng)
        for base in (0,45):
            pose=values[:,base:base+42].reshape(-1,21,2)
            pose=pose@matrix+shift
            values[:,base:base+42]=pose.reshape(-1,42)
            for trajectory in (42,87):
                values[:,trajectory:trajectory+2]=values[:,trajectory:trajectory+2]@matrix+shift
    if rng.random()<.5:
        size=int(rng.uniform(.2,.4)*len(values))
        if size>=len(values):
            values[:,:90]=0
        else:
            start=int(rng.integers(0,len(values)-size))
            values[start:start+size,:90]=0
    if rng.random()<.5:
        size=int(rng.uniform(.05,.1)*21)
        if size:
            for joint in rng.choice(21,size,replace=False):
                for base in (0,45):
                    values[:,base+2*joint:base+2*joint+2]=0
    return np.clip(values,-32,32)


def timeline_indices(values):
    """Position along the original clip timeline: accumulating the column-90 frame deltas keeps
    irregular detection spacing (1/30 s gaps in this packed contract) instead of compressing it."""
    return np.cumsum(values[:,-1],dtype=np.float64)*30


def normalized_box(pose):
    """Upstream normalisation: per frame translate the landmark bounding-box midpoint to the
    origin and divide by the maximum absolute coordinate, scaled to +-0.5."""
    pose=pose.copy()
    pose-=0.5*(pose.min(axis=1,keepdims=True)+pose.max(axis=1,keepdims=True))
    return pose/(np.abs(pose).max(axis=1,keepdims=True)+1e-6)*.5


def compressed_tokens(values,text,rng=None):
    """Upstream representation: right- and left-hand detections concatenated as separate
    tracks, arange indices per track (inter-detection gaps discarded), 42 channels."""
    if rng is not None and rng.random()<.3:
        values,text=values[::-1],text[::-1]
    poses,segments,positions=[],[],[]
    for base,segment in ((0,0),(45,1)):
        present=values[:,base+44]>0
        if not present.any():
            continue
        pose=normalized_box(values[present][:,base:base+42].reshape(-1,21,2))
        if rng is not None:
            pose=augment_pose(pose,rng)
        poses.append(pose.reshape(-1,42))
        segments.append(np.full(len(pose),segment,dtype=np.int64))
        positions.append(np.arange(len(pose),dtype=np.float64))
    if not poses:
        poses.append(np.zeros((1,42),dtype=np.float32))
        segments.append(np.array([2]))
        positions.append(np.zeros(1))
    return np.concatenate(poses),np.concatenate(segments),np.concatenate(positions),text


def timeline_tokens(values,text,representation,rng=None):
    """Our packed timeline as tokens: every frame (including no-detection ones) stays, the
    position index accumulates the real frame gaps, and hand identity becomes the segment id."""
    if rng is not None and rng.random()<.3:
        values,text=values[::-1],text[::-1]
    if rng is not None:
        values=augment_timeline(values.copy(),rng)
    right,left=values[:,44]>0,values[:,89]>0
    dominant=0 if right.sum()>=left.sum() else 1
    segments=np.full(len(values),2,dtype=np.int64)
    segments[right]=0
    segments[left]=1
    segments[right&left]=dominant
    if representation=='wrist2d':
        pose=values[:,0:42] if dominant==0 else values[:,45:87]
    else:
        pose=values[:,:91]
    return pose.astype(np.float32),segments,timeline_indices(values),text


def clip_tokens(data,index,representation,augment=False):
    row=data['clips'][index]
    values=np.array(data['x'][row['start']:row['end']],copy=augment)
    rng=clip_tokens.rng if augment else None
    if representation=='compressed':
        return compressed_tokens(values,row['text'],rng)
    return timeline_tokens(values,row['text'],representation,rng)
clip_tokens.rng=np.random


def batches(data,batch_size,representation,augment=False,shuffle=False,limit=None):
    indices=np.arange(len(data['clips']))
    if shuffle: np.random.shuffle(indices)
    if limit: indices=indices[:limit]
    # Shuffle windows, then sort within each window to reduce padded GPU work.
    chunks=[sorted(indices[i:i+batch_size*50],key=lambda j:data['clips'][j]['end']-data['clips'][j]['start'])
            for i in range(0,len(indices),batch_size*50)]
    groups=[window[i:i+batch_size] for window in chunks for i in range(0,len(window),batch_size)]
    if shuffle: random.shuffle(groups)
    for selected in groups:
        poses,segments,positions,targets=[],[],[],[]
        longest=1
        for index in selected:
            pose,segment,position,text=clip_tokens(data,index,representation,augment)
            poses.append(pose);segments.append(segment);positions.append(position)
            targets.append(np.array(encode_text(text),dtype=np.int64))
            longest=max(longest,len(pose),len(targets[-1]))
        width=poses[0].shape[1]
        padded=np.full((len(selected),longest,width),PAD_POSE,dtype=np.float32)
        padded_segments=np.full((len(selected),longest),PAD_SEG,dtype=np.int64)
        padded_positions=np.full((len(selected),longest),PAD_FRAME,dtype=np.float64)
        padded_targets=np.full((len(selected),max(len(t) for t in targets)),-100,dtype=np.int64)
        for i,(pose,segment,position,target) in enumerate(zip(poses,segments,positions,targets)):
            padded[i,:len(pose)]=pose
            padded_segments[i,:len(segment)]=segment
            padded_positions[i,:len(position)]=position
            padded_targets[i,:len(target)]=target
        yield (torch.from_numpy(padded),torch.from_numpy(padded_segments),
               torch.from_numpy(padded_positions),torch.from_numpy(padded_targets),selected)


def summarize(stats):
    return dict(cer=stats['errors']/stats['characters'],
                letter_accuracy=1-stats['errors']/stats['characters'],
                exact_sequence_accuracy=stats['exact']/stats['clips'],**stats)


@torch.inference_mode()
def evaluate(model,data,args,split):
    model.eval()
    started=time.perf_counter()
    stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    examples=[]
    for batch in batches(data,getattr(args,'validation_batch_size',None) or args.batch_size,args.input_representation,
                         limit=args.validation_limit):
        poses,segments,positions,_,selected=batch
        decoded=model.generate(poses.to(args.device),segments.to(args.device),
                               positions.to(args.device),max_chars=args.max_decode).cpu()
        for i,index in enumerate(selected):
            row=data['clips'][index]
            actual=decode_text(decoded[i])
            error=edit_distance(row['text'],actual)
            for key in ('overall',row['signer']):
                stats[key]['errors']+=error;stats[key]['characters']+=len(row['text'])
                stats[key]['exact']+=actual==row['text'];stats[key]['clips']+=1
            if len(examples)<30:
                examples.append(dict(name=row['name'],truth=row['text'],prediction=actual))
    metrics=summarize(stats.pop('overall'))
    metrics.update(split=split,per_signer={k:summarize(v) for k,v in stats.items()},examples=examples,
                   decoder='greedy autoregressive; no dictionary or language correction',
                   seconds=round(time.perf_counter()-started,3))
    return metrics


class Ema:
    """Exponential moving average of the trainable weights: the shadow copy is what gets
    evaluated and checkpointed while --ema-decay is set.  The decay ramps up from 0.1 so the
    shadow does not lag badly over the first hundred steps; live weights are swapped out
    around each validation pass and restored before the next epoch."""

    def __init__(self,model,decay):
        self.decay,self.step_index,self.live=decay,0,{}
        self.shadow={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self,model):
        self.step_index+=1
        factor=min(self.decay,(1.+self.step_index)/(10.+self.step_index))
        for name,p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(factor).add_(p.detach(),alpha=1.-factor)

    @torch.no_grad()
    def swap_in(self,model):
        self.live={name:p.detach().clone().cpu() for name,p in model.named_parameters()}
        for name,p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self,model):
        for name,p in model.named_parameters():
            if name in self.live:
                p.copy_(self.live[name].to(p.device))


def checkpoint(model,args,metrics,epoch):
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(contract=CONTRACT,task='recognizer',labels=VOCAB,
                 representation=args.input_representation,max_decode=args.max_decode,
                 architecture=dict(input_dim=args.input_dim,d_model=args.d_model,heads=args.heads,
                                   layers=args.layers,feedforward=args.feedforward,dropout=args.dropout),
                 state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 metrics=metrics,epoch=epoch,seed=args.seed,dataset=str(args.data.resolve()),
                 experimental=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent,suffix='.pt',delete=False) as file:
        temporary=Path(file.name)
    try:
        torch.save(payload,temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description='Train the offline OpenFS-recipe recogniser on the packed FSboard contract.')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--input-representation',dest='input_representation',
                   choices=['compressed','gaps','wrist2d'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--d-model',dest='d_model',type=int,default=256)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--feedforward',type=int,default=2048)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--learning-rate',type=float,default=.0001)
    p.add_argument('--lr-step',type=int,default=80)
    p.add_argument('--lr-gamma',type=float,default=.1)
    p.add_argument('--lr-schedule',choices=['step','cosine'],default='step')
    p.add_argument('--warmup-epochs',type=int,default=0)
    p.add_argument('--ema-decay',dest='ema_decay',type=float,
                   help='keep an exponential moving average of the weights (e.g. 0.9995 at ~1.5k '
                        'steps/epoch); the shadow copy is what gets validated and checkpointed')
    p.add_argument('--ce-weight',type=float,default=1.)
    p.add_argument('--entropy-weight',type=float,default=.8)
    p.add_argument('--monotonicity-weight',type=float,default=1.)
    p.add_argument('--warm-epoch',type=int,default=5,help='auxiliary losses start after this epoch (paper: 5)')
    p.add_argument('--max-decode',type=int,default=48)
    p.add_argument('--patience',type=int,default=0,help='0 disables early stopping (paper schedule)')
    p.add_argument('--no-augment',action='store_true')
    p.add_argument('--fp32',action='store_true',help='disable autocast (same as --amp off)')
    p.add_argument('--amp',choices=['auto','bf16','fp16'],default='auto',help="auto: bf16 on Ampere and newer, fp16 "
                   "with a GradScaler below it (Volta has no bf16 tensor cores, and flash attention needs sm_80)")
    p.add_argument('--validation-batch-size',type=int,help='defaults to --batch-size; lower it if validation decode '
                   'on the longest clips does not fit next to other jobs')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit-clips',type=int)
    p.add_argument('--train-slice',type=int,help='use the first N clips of the train split; for fixed standard slices')
    p.add_argument('--validation-limit',type=int)
    p.add_argument('--evaluate-initial',action='store_true',help='evaluate the untrained model first')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-only',dest='evaluate_only',action='store_true',
                   help='train nothing: load --resume and score --evaluate-splits. Writes <output stem>.eval-report.json '
                        'and never writes a checkpoint, so point --output at a fresh stem (e.g. .../test-eval.pt).')
    p.add_argument('--evaluate-splits',dest='evaluate_splits',nargs='+',default=['validation'],
                   choices=['train','validation','test'])
    p.add_argument('--force',action='store_true',help='replace an existing --output; for throwaway smoke runs only')
    args=p.parse_args()
    if args.evaluate_only and not args.resume:
        raise ValueError('--evaluate-only needs --resume <checkpoint> to score.')
    if args.output.exists() and not (args.force or args.resume):
        raise ValueError(f'{args.output} already exists; use a fresh run name (never overwrite checkpoints).')
    args.input_dim=91 if args.input_representation=='gaps' else 42
    args.validation_batch_size=args.validation_batch_size or args.batch_size
    cuda=args.device.startswith('cuda')
    amp='off' if args.fp32 or not cuda else args.amp
    if amp=='auto': amp='bf16' if cuda and torch.cuda.get_device_capability(args.device)[0]>=8 else 'fp16'
    args.amp=amp; args.amp_dtype={'bf16':torch.bfloat16,'fp16':torch.float16}.get(amp)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    clip_tokens.rng=np.random.default_rng(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    schema=json.loads((args.data/'dataset.json').read_text())
    if schema['contract']!=CONTRACT or schema['labels']!=VOCAB:
        raise ValueError('Packed dataset vocabulary/contract mismatch; prepare it again.')
    sets=[set(part['signers']) for part in schema['partitions']]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Signer leakage in dataset metadata.')
    train,validation=read_data(args.data,'train'),read_data(args.data,'validation')
    if args.train_slice and args.limit_clips:
        raise ValueError('--train-slice and --limit-clips cannot be combined; use one clip-capping mode.')
    if args.train_slice and args.train_slice < len(train['clips']):
        train=dict(train,clips=train['clips'][:args.train_slice])
    model=Recognizer(input_dim=args.input_dim,d_model=args.d_model,heads=args.heads,layers=args.layers,
                     feedforward=args.feedforward,dropout=args.dropout).to(args.device)
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu',weights_only=True)
        if payload.get('representation')!=args.input_representation:
            raise ValueError('Checkpoint input representation differs from the requested one.')
        model.load_state_dict(payload['state_dict'],strict=True)
    parameters=sum(q.numel() for q in model.parameters())
    print(json.dumps(dict(task='recognizer',representation=args.input_representation,parameters=parameters,
                          torch=torch.__version__,device=args.device,amp=args.amp,ema_decay=args.ema_decay,
                          gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                          training=len(train['clips']),validation=len(validation['clips']))),flush=True)
    if args.evaluate_only:
        evaluations=[evaluate(model,read_data(args.data,name),args,name) for name in args.evaluate_splits]
        for metrics in evaluations:
            print(json.dumps({key:metrics[key] for key in ('split','cer','letter_accuracy','exact_sequence_accuracy',
                                                            'clips','characters','errors','exact','seconds')}),flush=True)
        peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
        report=dict(task='recognizer',mode='evaluate-only',representation=args.input_representation,parameters=parameters,
                    checkpoint=str(args.resume.resolve()),evaluations=evaluations,dataset=schema,
                    torch=torch.__version__,peak_vram_mib=round(peak/2**20),
                    gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
        report_path=args.output.with_suffix('.eval-report.json')
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(report=str(report_path),peak_vram_mib=report['peak_vram_mib'])),flush=True)
        return
    baseline=None
    if args.evaluate_initial:
        baseline=evaluate(model,validation,args,'validation-untrained')
        print(json.dumps({k:baseline[k] for k in ('split','cer','letter_accuracy','exact_sequence_accuracy','seconds')}),flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp=='fp16')
    if args.lr_schedule=='cosine':
        warmup=max(args.warmup_epochs,0)
        def factor(step_index,epochs=args.epochs,warmup=warmup):
            epoch=step_index+1
            if warmup and epoch<=warmup:
                return epoch/warmup
            progress=min((epoch-warmup)/max(epochs-warmup,1),1.)
            return .5*(1.+math.cos(math.pi*progress))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    else:
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.lr_step,gamma=args.lr_gamma)
    ema=Ema(model,args.ema_decay) if args.ema_decay else None
    best=float('inf');stale=0;history=[];run_started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();started=time.perf_counter()
        loss_sum=ce_sum=ent_sum=mono_sum=steps=0
        for batch in batches(train,args.batch_size,args.input_representation,
                             augment=not args.no_augment,shuffle=True,limit=args.limit_clips):
            poses,segments,positions,targets,selected=batch
            poses,segments=poses.to(args.device),segments.to(args.device)
            positions,targets=positions.to(args.device),targets.to(args.device)
            inputs=torch.where(targets[:,:-1]==-100,PAD_CHAR,targets[:,:-1])
            labels=targets[:,1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=args.amp_dtype or torch.float16,enabled=args.amp!='off'):
                logits,cross_attns=model(poses,segments,positions,inputs)
            ce=torch.nn.functional.cross_entropy(logits.float().reshape(-1,len(VOCAB)),labels.reshape(-1))
            loss=args.ce_weight*ce;ent=mono=0.
            if epoch>args.warm_epoch:
                if args.entropy_weight:
                    ent=attention_entropy_loss(cross_attns,segments,targets)
                    loss=loss+args.entropy_weight*ent
                if args.monotonicity_weight:
                    mono=soft_monotonicity_loss(cross_attns,segments,targets)
                    loss=loss+args.monotonicity_weight*mono
            if not torch.isfinite(loss):
                detail=dict(epoch=epoch,step=steps,loss=float(loss.detach()),representation=args.input_representation,
                            logits_finite=bool(torch.isfinite(logits).all()),maximum_coordinate=float(poses.abs().max()),
                            lengths=segments.shape[1],target_lengths=(targets!=-100).sum(1).tolist(),
                            clips=[train['clips'][i]['name'] for i in selected])
                raise ValueError('Non-finite loss: '+json.dumps(detail))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer)
            scaler.update()
            if ema: ema.update(model)
            loss_sum+=loss.item();ce_sum+=ce.item()
            ent_sum+=ent.detach().item() if torch.is_tensor(ent) else 0.
            mono_sum+=mono.detach().item() if torch.is_tensor(mono) else 0.
            steps+=1
            if steps%20==0:
                print(json.dumps(dict(epoch=epoch,steps=steps,loss=round(loss_sum/steps,5),
                                      training_seconds=round(time.perf_counter()-started,1))),flush=True)
        if ema: ema.swap_in(model)
        metrics=evaluate(model,validation,args,'validation')
        row=dict(epoch=epoch,loss=loss_sum/steps,ce=ce_sum/steps,attention_entropy=ent_sum/steps,
                 monotonicity=mono_sum/steps,training_seconds=round(time.perf_counter()-started-metrics['seconds'],2),
                 validation_score=metrics['cer'],validation_letter=metrics['letter_accuracy'],
                 validation_exact=metrics['exact_sequence_accuracy'],
                 elapsed_seconds=round(time.perf_counter()-run_started,2))
        history.append(row);print(json.dumps(row),flush=True)
        if metrics['cer']<best:
            best=metrics['cer'];stale=0;checkpoint(model,args,metrics,epoch)
        else: stale+=1
        if ema: ema.restore(model)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        scheduler.step()
        if args.patience and stale>=args.patience:
            print('Validation early stopping',flush=True);break
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report=dict(task='recognizer',representation=args.input_representation,parameters=parameters,amp=args.amp,
                ema_decay=args.ema_decay,
                checkpoint=str(args.output.resolve()) if best<float('inf') else None,
                best_validation_cer=best,training_seconds=round(time.perf_counter()-run_started,2),
                epochs=len(history),history=history,baseline=baseline,dataset=schema,
                train_clips=len(train['clips']),train_slice=args.train_slice,
                torch=torch.__version__,peak_vram_mib=round(peak/2**20),
                gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
    report_path=args.output.with_suffix('.report.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report_path),peak_vram_mib=report['peak_vram_mib'],best_validation_cer=best,
                          final_validation={k:history[-1][k] for k in ('validation_score','validation_letter','validation_exact')}
                          if history else None)),flush=True)


if __name__=='__main__':
    main()
