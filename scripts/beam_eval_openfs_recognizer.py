"""Beam-search decode of a trained offline OpenFS recogniser: same contract, weights and
Levenshtein harness as evaluate() in train_openfs_recognizer, only the AR inference changes
(batched beam search, sum of log-probabilities, no length normalisation, no dictionary or
language correction).  Hypotheses retire on EOS and duplicate prefixes keep their best score.

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
from model.offline_recognizer import BOS,EOS,PAD_CHAR,Recognizer,decode_text
from scripts.train_neural import edit_distance
from scripts.train_openfs_recognizer import batches,read_data,summarize


@torch.inference_mode()
def beam_decode(model,poses,segments,positions,beam_size,max_chars):
    memory,padding=model.encode(poses,segments,positions)
    device=memory.device
    n=len(poses)
    memory=memory.repeat_interleave(beam_size,dim=0)
    padding=padding.repeat_interleave(beam_size,dim=0)
    samples=torch.arange(n,device=device).repeat_interleave(beam_size)
    beam_rows=torch.arange(beam_size,device=device)
    prefixes=torch.full((n*beam_size,1),BOS,dtype=torch.long,device=device)
    scores=torch.zeros(n,beam_size,device=device)
    active=torch.ones(n,beam_size,dtype=torch.bool,device=device)
    finished_scores=torch.full((n,),-float('inf'),device=device)
    completed_prefixes=torch.full((n,max_chars+1),PAD_CHAR,dtype=torch.long,device=device)
    completed_lengths=torch.zeros(n,dtype=torch.long,device=device)
    for _ in range(max_chars):
        length=prefixes.shape[1]
        logits,_=model.decode(memory,padding,prefixes)
        logp=F.log_softmax(logits[:,-1].float(),dim=-1).view(n,beam_size,-1)
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
        improved=best_eos_scores>finished_scores
        if improved.any():
            rows=improved.nonzero(as_tuple=True)[0]
            completed_prefixes[rows,:length+1]=cand_prefixes[(rows*beam_size+best_eos[rows])]
            completed_lengths[rows]=length+1
            finished_scores[rows]=best_eos_scores[rows]
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
        best_score=float(finished_scores[i])
        best=tuple(completed_prefixes[i,1:int(completed_lengths[i])].tolist()) if int(completed_lengths[i]) else None
        active_rows=active[i].tolist()
        score_rows=scores[i].tolist()
        for j,is_active in enumerate(active_rows):
            if not is_active or not math.isfinite(score_rows[j]):
                continue
            if score_rows[j]>best_score:
                best_score=score_rows[j]
                best=tuple(prefix_rows[i*beam_size+j][1:])
        outputs.append(best if best is not None else ())
    return outputs


@torch.inference_mode()
def score(model,data,representation,beam_size,max_chars,batch_size,device,split,limit):
    model.eval()
    started=time.perf_counter()
    stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    examples=[]
    for batch in batches(data,batch_size,representation,limit=limit):
        poses,segments,positions,_,selected=batch
        decoded=beam_decode(model,poses.to(device),segments.to(device),positions.to(device),beam_size,max_chars)
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
                   decoder=f'beam search width {beam_size}; no dictionary or language correction',
                   seconds=round(time.perf_counter()-started,3))
    return metrics


def main():
    p=argparse.ArgumentParser(description='Beam-search evaluation of an offline OpenFS recogniser checkpoint.')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['validation'],choices=['train','validation','test'])
    p.add_argument('--beam-size',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-chars',type=int,help='defaults to the checkpoint max_decode (48)')
    p.add_argument('--output',type=Path,required=True,help='report stem; never loads or writes a checkpoint')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if payload.get('contract')!=CONTRACT:
        raise ValueError('Checkpoint contract mismatch.')
    representation=payload['representation']
    max_chars=args.max_chars or payload.get('max_decode',48)
    model=Recognizer(**payload['architecture']).to(args.device)
    model.load_state_dict(payload['state_dict'],strict=True)
    parameters=sum(q.numel() for q in model.parameters())
    print(json.dumps(dict(task='recognizer',mode='beam-eval',representation=representation,
                          parameters=parameters,beam_size=args.beam_size,
                          torch=torch.__version__,device=args.device,
                          gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)),flush=True)
    evaluations=[score(model,read_data(args.data,name),representation,args.beam_size,max_chars,
                       args.batch_size,args.device,name,args.limit) for name in args.splits]
    for metrics in evaluations:
        print(json.dumps({key:metrics[key] for key in ('split','cer','letter_accuracy','exact_sequence_accuracy',
                                                        'clips','characters','errors','exact','seconds')}),flush=True)
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report=dict(task='recognizer',mode='beam-eval',representation=representation,parameters=parameters,
                beam_size=args.beam_size,max_chars=max_chars,checkpoint=str(args.checkpoint.resolve()),
                evaluations=evaluations,dataset=json.loads((args.data/'dataset.json').read_text()),
                torch=torch.__version__,peak_vram_mib=round(peak/2**20),
                gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
    report_path=args.output.with_suffix('.beam-eval-report.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report_path),peak_vram_mib=report['peak_vram_mib'])),flush=True)


if __name__=='__main__':
    main()
