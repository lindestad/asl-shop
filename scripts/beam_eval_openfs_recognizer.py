"""Beam-search decode of a trained offline OpenFS recogniser: same contract, weights and
Levenshtein harness as evaluate() in train_openfs_recognizer, only the AR inference changes
(batched beam search, sum of log-probabilities optionally divided by generated-length**alpha
via --length-penalty, 0 = plain sum, no dictionary or language correction unless --lm-weight > 0
adds a train-split character n-gram shallow-fusion, named explicitly).  Hypotheses retire
on EOS and duplicate prefixes keep their best score.  With --tta-rot-degrees, each checkpoint
additionally decodes the pose timeline rotated by the listed angles (train-time affine branch,
deterministic) and all model x rotation frame log-posteriors are fused geometrically.

  python scripts/beam_eval_openfs_recognizer.py --checkpoint <recognizer.pt> --data data/openfs-packed \
      --splits validation --beam-size 8 --output <stem>   # writes <stem>.beam-eval-report.json

Mixed-representation ensembles are allowed when every checkpoint carries the gap-preserving
timeline (gaps, wrist2d): each model encodes its own representation of the identical clip
batch (same segments/positions) and all frame log-posteriors fuse as before; `compressed`
checkpoints use a different timeline and stay single-representation.

Every decode additionally collects the beam hypothesis pool (per-sample record-setting
EOS retirements plus surviving active prefixes, deduplicated, best score kept) and reports
two selections in one pass: the historical argmax-score (MAP) output as the primary metric
and a minimum-Bayes-risk (MBR) output that minimises the expected unit-cost Levenshtein
distance under the softmax-weighted pool posterior - the selection criterion matches the
evaluation metric directly.
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
from model.neural import CONTRACT,VOCAB
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
        if base+42>rotated.shape[2]:
            continue
        hand=rotated[:,:,base:base+42].reshape(rotated.shape[0],rotated.shape[1],21,2)@matrix
        rotated[:,:,base:base+42]=hand.reshape(rotated.shape[0],rotated.shape[1],42)
    if rotated.shape[2]>=44:
        rotated[:,:,42:44]=rotated[:,:,42:44]@matrix
    if rotated.shape[2]>=89:
        rotated[:,:,87:89]=rotated[:,:,87:89]@matrix
    mask=(segments!=PAD_SEG).unsqueeze(-1)
    return torch.where(mask,rotated,pose)


def build_ngram_lm(texts, order, device):
    """Witten-Bell interpolated character n-gram log-probability table over the recogniser
    character vocabulary, built from TRAIN-split reference texts only.  Context states are the
    character indices 1..V-1 plus a start boundary V; output classes are the same characters
    plus an end class V (the model's EOS "").  Contexts are coded most-recent-first with the
    boundary padding short strings, matching context_codes() below."""
    V = len(VOCAB)
    K = V + 1
    E = V + 1
    counts = [dict() for _ in range(order + 1)]
    for text in texts:
        sequence = [VOCAB.index(character) for character in text] + [V]
        for level in range(1, order + 1):
            table = counts[level]
            for position, token in enumerate(sequence):
                context = 0
                for offset in range(level - 1):
                    index = position - 1 - offset
                    context = context * K + (sequence[index] if index >= 0 else V)
                bucket = table.get(context)
                if bucket is None:
                    bucket = table[context] = {}
                bucket[token] = bucket.get(token, 0) + 1
    probs = torch.full((E,), 1.0 / E, dtype=torch.float64)
    for level in range(1, order + 1):
        contexts = torch.zeros((K ** (level - 1), E), dtype=torch.float64)
        for context, bucket in counts[level].items():
            for token, count in bucket.items():
                contexts[context, token] = count
        rows = contexts.sum(dim=1)
        unique = (contexts > 0).sum(dim=1)
        gamma = torch.where(rows > 0, rows / (rows + unique), torch.zeros_like(rows))
        smoothed = torch.where(rows.unsqueeze(1) > 0,
                               contexts / rows.clamp_min(1.0).unsqueeze(1),
                               torch.zeros_like(contexts))
        if level == 1:
            backoff = probs.unsqueeze(0)
        else:
            backoff = probs[(torch.arange(K ** (level - 1)) // K)]
        probs = gamma.unsqueeze(1) * smoothed + (1.0 - gamma).unsqueeze(1) * backoff
    return probs.log().reshape(-1).to(device=device, dtype=torch.float32)


def context_codes(prefixes, order, device):
    """Most-recent-first flat context code (base K = len(VOCAB)+1) of the last order-1 tokens of
    each prefix row; BOS/PAD/EOS slots become the start boundary state V."""
    K = len(VOCAB) + 1
    length = prefixes.shape[1]
    codes = torch.zeros(prefixes.shape[0], dtype=torch.long, device=device)
    for offset in range(order - 1):
        column = length - 1 - offset
        if column < 0:
            values = torch.full((prefixes.shape[0],), K - 1, dtype=torch.long, device=device)
        else:
            values = prefixes[:, column]
            values = torch.where((values >= 1) & (values <= K - 2), values, K - 1)
        codes = codes * K + values
    return codes


def _clean(ids):
    """Character-token ids of a hypothesis with BOS/EOS/PAD removed, so completed and
    still-active hypotheses dedupe on the same key."""
    return tuple(index for index in ids if index not in (EOS,PAD_CHAR,BOS))


def mbr_select(pool):
    """Minimum-Bayes-risk selection over the hypothesis pool: dedupe by character ids
    (best score kept), weight each unique hypothesis by softmax(pool scores), and return
    the ids minimising expected unit-cost Levenshtein distance; ties break to higher score."""
    unique={}
    for score_value,ids in pool:
        if ids not in unique or score_value>unique[ids]:
            unique[ids]=score_value
    if not unique:
        return ()
    hypotheses=list(unique)
    scores=[unique[key] for key in hypotheses]
    top=max(scores)
    weights=[math.exp(score-top) for score in scores]
    total=sum(weights)
    weights=[weight/total for weight in weights]
    best=0
    best_cost=None
    for i,candidate in enumerate(hypotheses):
        cost=sum(weight*edit_distance(list(candidate),list(other))
                 for weight,other in zip(weights,hypotheses))
        if best_cost is None or cost<best_cost-1e-9 or (abs(cost-best_cost)<=1e-9 and scores[i]>scores[best]):
            best,best_cost=i,cost
    return hypotheses[best]


@torch.inference_mode()
def beam_decode(models,poses_by_model,segments,positions,beam_size,max_chars,length_penalty=0.0,weights=None,fusion='logprob',rotations=(0.0,),lm=None):
    n=len(poses_by_model[0])
    encoded=[]
    for model,poses in zip(models,poses_by_model):
        pose_variants=[rotate_pose(poses,segments,degrees) for degrees in rotations]
        encoded.append([(memory.repeat_interleave(beam_size,dim=0),padding.repeat_interleave(beam_size,dim=0))
                        for memory,padding in (model.encode(variant,segments,positions) for variant in pose_variants)])
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
    pool=[[] for _ in range(n)]
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
        if lm is not None:
            lm_table,lm_weight,lm_order=lm
            E=len(VOCAB)+1
            codes=context_codes(prefixes,lm_order,device)
            tokens=torch.arange(logp.shape[-1],device=device)
            tokens=torch.where(tokens==0,E-1,tokens)
            lm_row=lm_table[(codes.unsqueeze(1)*E+tokens).reshape(-1)].view(n,beam_size,-1)
            logp=logp+lm_weight*lm_row
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
            for sample in rows.tolist():
                pool[sample].append((float(best_eos_scores[sample]),
                                     _clean(cand_prefixes[sample*beam_size+int(best_eos[sample]),1:length+1].tolist())))
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
    end_lm=None
    if lm is not None:
        lm_table,lm_weight,lm_order=lm
        E=len(VOCAB)+1
        end_lm=(lm_weight*lm_table[context_codes(prefixes,lm_order,device)*E+(E-1)]).view(n,beam_size).tolist()
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
            if end_lm is not None:
                candidate_score+=end_lm[i][j]
            pool[i].append((candidate_score,_clean(prefix[1:])))
            if candidate_score>best_score:
                best_score=candidate_score
                best=tuple(prefix[1:])
        outputs.append(best if best is not None else ())
    return outputs,pool


@torch.inference_mode()
def score(models,data,representations,beam_size,max_chars,batch_size,device,split,limit,length_penalty=0.0,weights=None,fusion='logprob',rotations=(0.0,),lm=None,lm_description=None):
    for model in models:
        model.eval()
    started=time.perf_counter()
    stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    mbr_stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    pool_sizes=[]
    examples=[]
    unique=list(dict.fromkeys(representations))
    iterators=[batches(data,batch_size,rep,limit=limit) for rep in unique]
    for batch_tuple in zip(*iterators):
        base=batch_tuple[0]
        segments,positions,selected=base[1],base[2],base[4]
        for batch in batch_tuple[1:]:
            if batch[4]!=selected or not torch.equal(batch[1],segments) or not torch.equal(batch[2],positions):
                raise ValueError('Mixed-representation batches diverged; gap-preserving timelines are not aligned.')
        poses_by_rep={rep:batch[0].to(device) for rep,batch in zip(unique,batch_tuple)}
        decoded,pools=beam_decode(models,[poses_by_rep[rep] for rep in representations],segments.to(device),positions.to(device),beam_size,max_chars,length_penalty,weights,fusion,rotations,lm)
        for i,index in enumerate(selected):
            row=data['clips'][index]
            actual=decode_text(decoded[i])
            error=edit_distance(row['text'],actual)
            for key in ('overall',row['signer']):
                stats[key]['errors']+=error;stats[key]['characters']+=len(row['text'])
                stats[key]['exact']+=actual==row['text'];stats[key]['clips']+=1
            mbr_actual=decode_text(mbr_select(pools[i]))
            mbr_error=edit_distance(row['text'],mbr_actual)
            for key in ('overall',row['signer']):
                mbr_stats[key]['errors']+=mbr_error;mbr_stats[key]['characters']+=len(row['text'])
                mbr_stats[key]['exact']+=mbr_actual==row['text'];mbr_stats[key]['clips']+=1
            pool_sizes.append(len(set(ids for _,ids in pools[i])))
            if len(examples)<30:
                examples.append(dict(name=row['name'],truth=row['text'],prediction=actual,mbr_prediction=mbr_actual))
    metrics=summarize(stats.pop('overall'))
    mbr_metrics=summarize(mbr_stats.pop('overall'))
    metrics['mbr']=mbr_metrics
    metrics.update(split=split,per_signer={k:summarize(v) for k,v in stats.items()},
                   mbr_per_signer={k:summarize(v) for k,v in mbr_stats.items()},
                   mean_pool_size=round(sum(pool_sizes)/max(len(pool_sizes),1),3),examples=examples,
                   decoder=f'beam search width {beam_size}; length penalty {length_penalty}; ensemble of {len(models)} checkpoints over representations {representations} ({"arithmetic mean of frame probabilities" if fusion=="prob" else "weighted mean of log-posteriors"}, weights {weights}); input rotations {list(rotations)}; {lm_description if lm_description else "no dictionary or language correction"}; selection: argmax-score (MAP) primary and minimum-Bayes-risk (expected unit-cost Levenshtein under the softmax-weighted beam pool posterior) reported alongside',
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
    p.add_argument('--lm-weight',type=float,default=0.0,help='shallow-fusion weight lambda of a Witten-Bell character n-gram language model trained on the TRAIN split reference texts; the LM log-probability is added to the fused model log-posteriors at every beam step (including EOS); 0.0 disables the LM and reproduces the no-language-model behaviour exactly')
    p.add_argument('--lm-order',type=int,default=4,help='character n-gram order for --lm-weight')
    p.add_argument('--output',type=Path,required=True,help='report stem; never loads or writes a checkpoint')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    payloads=[torch.load(path,map_location='cpu',weights_only=True) for path in args.checkpoint]
    for path,payload in zip(args.checkpoint,payloads):
        if payload.get('contract')!=CONTRACT:
            raise ValueError(f'Checkpoint contract mismatch: {path}.')
    representations=[payload['representation'] for payload in payloads]
    for representation in representations:
        if representation not in ('gaps','wrist2d'):
            raise ValueError('Ensembled checkpoints must use gap-preserving representations (gaps, wrist2d).')
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
    lm=None
    lm_description=None
    if args.lm_weight:
        train_texts=[row['text'] for row in read_data(args.data,'train')['clips']]
        lm=(build_ngram_lm(train_texts,args.lm_order,args.device),args.lm_weight,args.lm_order)
        lm_description=(f'character {args.lm_order}-gram Witten-Bell interpolated LM built from the '
                        f'{len(train_texts)} train-split reference texts, shallow-fused with weight {args.lm_weight}')
    print(json.dumps(dict(task='recognizer',mode='beam-eval',representations=representations,
                          parameters=parameters,ensemble_size=len(models),ensemble_weights=weights,beam_size=args.beam_size,length_penalty=args.length_penalty,fusion=args.fusion,
                          tta_rot_degrees=args.tta_rot_degrees,
                          lm_weight=args.lm_weight,lm_order=args.lm_order,
                          ensemble_architectures=architectures,
                          torch=torch.__version__,device=args.device,
                          gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)),flush=True)
    if args.fusion=='prob' and any(degrees!=0.0 for degrees in args.tta_rot_degrees):
        raise ValueError('--tta-rot-degrees currently supports only --fusion logprob.')
    evaluations=[score(models,read_data(args.data,name),representations,args.beam_size,max_chars,
                       args.batch_size,args.device,name,args.limit,args.length_penalty,weights,args.fusion,
                       tuple(args.tta_rot_degrees),lm,lm_description) for name in args.splits]
    for metrics in evaluations:
        print(json.dumps({key:metrics[key] for key in ('split','cer','letter_accuracy','exact_sequence_accuracy',
                                                        'clips','characters','errors','exact','seconds')}),flush=True)
        mbr=dict(metrics['mbr']);mbr['selection']='mbr'
        print(json.dumps(mbr),flush=True)
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report=dict(task='recognizer',mode='beam-eval',representations=representations,parameters=parameters,
                ensemble_size=len(models),ensemble_weights=weights,beam_size=args.beam_size,length_penalty=args.length_penalty,fusion=args.fusion,max_chars=max_chars,
                tta_rot_degrees=args.tta_rot_degrees,
                lm_weight=args.lm_weight,lm_order=args.lm_order,
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
