"""Beam-search decode of a trained offline OpenFS recogniser: same contract, weights and
Levenshtein harness as evaluate() in train_openfs_recognizer, only the AR inference changes
(batched beam search, sum of log-probabilities optionally divided by generated-length**alpha
via --length-penalty, 0 = plain sum, no dictionary or language correction).  Hypotheses retire
on EOS and duplicate prefixes keep their best score.  With --tta-rot-degrees, each checkpoint
additionally decodes the pose timeline rotated by the listed angles (train-time affine branch,
deterministic) and all model x rotation frame log-posteriors are fused geometrically.

  python scripts/beam_eval_openfs_recognizer.py --checkpoint <recognizer.pt> --data data/openfs-packed \
      --splits validation --beam-size 8 --output <stem>   # writes <stem>.beam-eval-report.json
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.neural import CONTRACT
from model.offline_recognizer import BOS,EOS,PAD_CHAR,PAD_SEG,Recognizer,decode_text
from scripts.train_neural import edit_distance
from scripts.train_openfs_recognizer import batches,read_data,summarize


def rotate_pose(pose,segments,degrees):
    """Train-time affine branch (scripts.train_openfs_recognizer.augment_timeline) evaluated
    deterministically at inference: rotate pose and trajectory columns about the origin by
    `degrees`, leaving presence flags, frame deltas and PAD_POSE rows untouched."""
    angle=math.radians(degrees)
    matrix=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]],
                        dtype=pose.dtype,device=pose.device)
    rotated=pose.clone()
    for base in (0,45):
        hand=rotated[:,:,base:base+42].reshape(rotated.shape[0],rotated.shape[1],21,2)@matrix
        rotated[:,:,base:base+42]=hand.reshape(rotated.shape[0],rotated.shape[1],42)
    rotated[:,:,42:44]=rotated[:,:,42:44]@matrix
    rotated[:,:,87:89]=rotated[:,:,87:89]@matrix
    mask=(segments!=PAD_SEG).unsqueeze(-1)
    return torch.where(mask,rotated,pose)


@torch.inference_mode()
def beam_decode(models,poses,segments,positions,beam_size,max_chars,length_penalty=0.0,weights=None,fusion='logprob',rotations=(0.0,)):
    n=len(poses)
    pose_variants=[rotate_pose(poses,segments,degrees) for degrees in rotations]
    encoded=[[(memory.repeat_interleave(beam_size,dim=0),padding.repeat_interleave(beam_size,dim=0))
              for memory,padding in (model.encode(variant,segments,positions) for variant in pose_variants)]
             for model in models]
    device=encoded[0][0][0].device
    samples=torch.arange(n,device=device).repeat_interleave(beam_size)
    beam_rows=torch.arange(beam_size,device=device)
    prefixes=torch.full((n*beam_size,1),BOS,dtype=torch.long,device=device)
    scores=torch.zeros(n,beam_size,device=device)
    active=torch.ones(n,beam_size,dtype=torch.bool,device=device)
    finished_scores=torch.full((n,),-float('inf'),device=device)
    completed_prefixes=torch.full((n,max_chars+1),PAD_CHAR,dtype=torch.long,device=device)
    completed_lengths=torch.zeros(n,dtype=torch.long,device=device)
    finished_norm=torch.full((n,),-float('inf'),device=device) if length_penalty else None
    for _ in range(max_chars):
        length=prefixes.shape[1]
        logp=None
        for weight,(model,encodings) in zip(weights,zip(models,encoded)):
            for step_memory,step_padding in encodings:
                logits,_=model.decode(step_memory,step_padding,prefixes)
                model_logp=F.log_softmax(logits[:,-1].float(),dim=-1)
                if fusion=='prob':
                    model_term=weight*model_logp.exp()
                    logp=model_term if logp is None else logp+model_term
                else:
                    logp=weight*model_logp if logp is None else logp+weight*model_logp
        logp=(logp/(sum(weights)*len(rotations))).view(n,beam_size,-1)
        if fusion=='prob':
            logp=logp.clamp_min(1e-12).log()
        candidate=(scores.unsqueeze(-1)+logp.masked_fill(~active.unsqueeze(-1),-float('inf'))).flatten(1)
        top,indices=torch.topk(candidate,beam_size,dim=-1)
        vocab=candidate.shape[-1]//beam_size
        source=indices//vocab
        token=(indices%vocab).reshape(-1)
        source_flat=source.reshape(-1)
        src_rows=torch.add(samples*beam_size,source_flat)
        score_flat=top.reshape(-1)
        valid=active.gather(1,source) & torch.isfinite(score_flat).reshape(n,beam_size)
        valid_flat=valid.reshape(-1)
        cand_prefixes=torch.full((n*beam_size,length+1),PAD_CHAR,dtype=torch.long,device=device)
        if valid_flat.any():
            valid_rows=valid_flat.nonzero(as_tuple=True)[0]
            cand_prefixes[valid_rows,:length]=prefixes[src_rows[valid_rows]]
            cand_prefixes[valid_rows,length]=token[valid_rows]
        cand_scores=score_flat.reshape(n,beam_size)
        eos_mask=valid&(token.reshape(n,beam_size)==EOS)
        eos_scores=torch.where(eos_mask,cand_scores,torch.full_like(cand_scores,-float('inf')))
        best_eos_scores,best_eos=eos_scores.max(dim=-1)
        if length_penalty:
            cand_norm=best_eos_scores/float(length)**length_penalty
            improved=cand_norm>finished_norm
        else:
            improved=best_eos_scores>finished_scores
        if improved.any():
            rows=improved.nonzero(as_tuple=True)[0]
            completed_prefixes[rows,:length+1]=cand_prefixes[(rows*beam_size+best_eos[rows])]
            completed_lengths[rows]=length+1
            finished_scores[rows]=best_eos_scores[rows]
            if length_penalty:
                finished_norm[rows]=cand_norm[rows]
        active_mask=valid&(token.reshape(n,beam_size)!=EOS)
        new_prefixes=torch.full((n*beam_size,length+1),PAD_CHAR,dtype=torch.long,device=device)
        new_scores=torch.full((n,beam_size),-float('inf'),device=device)
        new_active=torch.zeros(n,beam_size,dtype=torch.bool,device=device)
        active_flat=active_mask.reshape(-1)
        if active_flat.any():
            active_rows=active_flat.nonzero(as_tuple=True)[0]
            keys=torch.cat((cand_prefixes[active_rows],samples[active_rows].unsqueeze(-1)),dim=1)
            unique_keys,inverse=torch.unique(keys,dim=0,return_inverse=True)
            unique_scores=torch.full((unique_keys.shape[0],),-float('inf'),device=device)
            unique_scores.scatter_reduce_(0,inverse,score_flat[active_flat],reduce='amax',include_self=True)
            unique_samples=unique_keys[:,-1]
            unique_prefixes=unique_keys[:,:-1]
            for sample in unique_samples.unique().tolist():
                selected=unique_samples==sample
                selected_indices=selected.nonzero(as_tuple=True)[0]
                count=min(int(selected_indices.numel()),beam_size)
                selected_scores,selected_local=torch.topk(unique_scores[selected_indices],k=count)
                picked=selected_indices[selected_local]
                row_start=sample*beam_size
                new_prefixes[row_start:row_start+count,:length+1]=unique_prefixes[picked]
                new_scores[sample,:count]=selected_scores
                new_active[sample,:count]=True
        prefixes,scores,active=new_prefixes,new_scores,new_active
        if not bool(active.any()):
            break
    outputs=[]
    prefix_rows=prefixes.tolist()
    for i in range(n):
        completed_length=int(completed_lengths[i])
        if length_penalty:
            best_score=float(finished_norm[i])
            if completed_length>1:
                best_score/=float(completed_length-1)**length_penalty
        else:
            best_score=float(finished_scores[i])
        best=tuple(completed_prefixes[i,1:completed_length].tolist()) if completed_length else None
        active_rows=active[i].tolist()
        score_rows=scores[i].tolist()
        for j,is_active in enumerate(active_rows):
            if not is_active or not math.isfinite(score_rows[j]):
                continue
            prefix=prefix_rows[i*beam_size+j]
            if length_penalty:
                candidate_score=score_rows[j]/float(max(len(prefix)-1,1))**length_penalty
            else:
                candidate_score=score_rows[j]
            if candidate_score>best_score:
                best_score=candidate_score
                best=tuple(prefix[1:])
        outputs.append(best if best is not None else ())
    return outputs


@torch.inference_mode()
def score(models,data,representation,beam_size,max_chars,batch_size,device,split,limit,length_penalty=0.0,weights=None,fusion='logprob',rotations=(0.0,)):
    for model in models:
        model.eval()
    started=time.perf_counter()
    stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    examples=[]
    for batch in batches(data,batch_size,representation,limit=limit):
        poses,segments,positions,_,selected=batch
        decoded=beam_decode(models,poses.to(device),segments.to(device),positions.to(device),beam_size,max_chars,length_penalty,weights,fusion,rotations)
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
                   decoder=f'beam search width {beam_size}; length penalty {length_penalty}; ensemble of {len(models)} checkpoints ({"arithmetic mean of frame probabilities" if fusion=="prob" else "weighted mean of log-posteriors"}, weights {weights}); input rotations {list(rotations)}; no dictionary or language correction',
                   seconds=round(time.perf_counter()-started,3))
    return metrics


def main():
    p=argparse.ArgumentParser(description='Beam-search evaluation of an offline OpenFS recogniser checkpoint.')
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True,help='one or more checkpoints; multiple checkpoints are ensembled by averaging frame log-posteriors')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['validation'],choices=['train','validation','test'])
    p.add_argument('--beam-size',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-chars',type=int,help='defaults to the checkpoint max_decode (48)')
    p.add_argument('--length-penalty',type=float,default=0.0,help='divide log-probability scores by generated-length**alpha when ranking hypotheses of different lengths; 0 reproduces plain sum-of-log-probabilities')
    p.add_argument('--ensemble-weights',type=float,nargs='+',help='per-checkpoint weights for the mean of frame log-posteriors (default: all 1)')
    p.add_argument('--fusion',choices=['logprob','prob'],default='logprob',help='ensemble fusion rule: logprob = weighted mean of frame log-posteriors (geometric); prob = weighted arithmetic mean of frame probabilities, then log')
    p.add_argument('--tta-rot-degrees',type=float,nargs='+',default=[0.0],help='deterministic input-rotation ensemble: decode with the pose timeline rotated by each degree value and fuse all model x rotation frame log-posteriors with the ensemble weights; a single 0.0 reproduces the identity-only behaviour')
    p.add_argument('--output',type=Path,required=True,help='report stem; never loads or writes a checkpoint')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    payloads=[torch.load(path,map_location='cpu',weights_only=True) for path in args.checkpoint]
    for path,payload in zip(args.checkpoint,payloads):
        if payload.get('contract')!=CONTRACT:
            raise ValueError(f'Checkpoint contract mismatch: {path}.')
    representation=payloads[0]['representation']
    if any(payload['representation']!=representation for payload in payloads):
        raise ValueError('Ensembled checkpoints must share the same input representation.')
    architectures=[payload['architecture'] for payload in payloads]
    labels=payloads[0]['labels']
    if any(payload['labels']!=labels for payload in payloads):
        raise ValueError('Ensembled checkpoints must share the same label vocabulary.')
    max_chars=args.max_chars or payloads[0].get('max_decode',48)
    weights=args.ensemble_weights or [1.0]*len(payloads)
    if len(weights)!=len(payloads):
        raise ValueError('--ensemble-weights must give one weight per --checkpoint.')
    models=[]
    for architecture,payload in zip(architectures,payloads):
        model=Recognizer(**architecture).to(args.device)
        model.load_state_dict(payload['state_dict'],strict=True)
        models.append(model)
    parameters=sum(q.numel() for model in models for q in model.parameters())
    print(json.dumps(dict(task='recognizer',mode='beam-eval',representation=representation,
                          parameters=parameters,ensemble_size=len(models),ensemble_weights=weights,beam_size=args.beam_size,length_penalty=args.length_penalty,fusion=args.fusion,
                          tta_rot_degrees=args.tta_rot_degrees,
                          ensemble_architectures=architectures,
                          torch=torch.__version__,device=args.device,
                          gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)),flush=True)
    if args.fusion=='prob' and any(degrees!=0.0 for degrees in args.tta_rot_degrees):
        raise ValueError('--tta-rot-degrees currently supports only --fusion logprob.')
    evaluations=[score(models,read_data(args.data,name),representation,args.beam_size,max_chars,
                       args.batch_size,args.device,name,args.limit,args.length_penalty,weights,args.fusion,
                       tuple(args.tta_rot_degrees)) for name in args.splits]
    for metrics in evaluations:
        print(json.dumps({key:metrics[key] for key in ('split','cer','letter_accuracy','exact_sequence_accuracy',
                                                        'clips','characters','errors','exact','seconds')}),flush=True)
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report=dict(task='recognizer',mode='beam-eval',representation=representation,parameters=parameters,
                ensemble_size=len(models),ensemble_weights=weights,beam_size=args.beam_size,length_penalty=args.length_penalty,fusion=args.fusion,max_chars=max_chars,
                tta_rot_degrees=args.tta_rot_degrees,
                checkpoints=[str(path.resolve()) for path in args.checkpoint],
                evaluations=evaluations,dataset=json.loads((args.data/'dataset.json').read_text()),
                torch=torch.__version__,peak_vram_mib=round(peak/2**20),
                gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
    report_path=args.output.with_suffix('.beam-eval-report.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report_path),peak_vram_mib=report['peak_vram_mib'])),flush=True)


if __name__=='__main__':
    main()
