"""Offline breakdown plus character-LM rescoring for both fingerspelling decoders.

Reports deletion/substitution/insertion counts and target-length accuracy for our causal
CTC head and for the upstream OpenFS reference, then adds a character n-gram language
model fit on training-split transcripts only. Every LM source is filtered against the
validation and test clip names, so held-out text cannot reach the counts, and the LM
weight is picked on the validation split before any test number is produced.
The upstream checkout is imported for the reference baseline only; nothing here is
packaged into the app backend.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

# Same allocator hint the training script uses; keeps the CUDA reservation close to live use.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.char_lm import CharNgramLM, ctc_prefix_beam, transcripts_from_files
from model.neural import VOCAB, decode_ctc, load_checkpoint
from model.recognition_metrics import ErrorTally
from scripts.evaluate_openfs_reference import preprocess

DEFAULT_LM_SOURCES=[ROOT/'data/openfs-packed/train_clips.json',
                    ROOT/'data/external/openfs/metadata/daun_v3-train.json',
                    ROOT/'data/external/openfs/metadata/dmk_v3-train.json']
ALPHABET=''.join(VOCAB[1:])


def clips_of(data,split):
    return json.loads((data/f'{split}_clips.json').read_text(encoding='utf-8'))


def draw(clips,limit,seed):
    indices=np.arange(len(clips))
    if limit: indices=np.random.default_rng(seed).choice(len(clips),min(int(limit),len(clips)),replace=False)
    return [int(index) for index in indices]


def by_clip_length(clips,indices):
    return sorted(indices,key=lambda index:clips[index]['end']-clips[index]['start'])


def target_stats(clips):
    tally=ErrorTally()
    for row in clips: tally.add(row['text'],row['text'])
    return dict(clips=len(clips),characters=sum(len(row['text']) for row in clips),
                target_lengths=tally.summary()['length_clips'])


def fit_language_model(args,held_out_names):
    """Fit on training text only, then report validation perplexity for each candidate order."""
    corpus=transcripts_from_files(args.lm_source,exclude_names=held_out_names)
    dev_texts=[row['text'] for row in clips_of(args.data,'validation')]
    perplexity={}
    best_order=best_model=None
    for order in args.lm_order_sweep:
        candidate=CharNgramLM(order=order,backoff_weight=args.lm_backoff,alphabet=ALPHABET).fit(corpus['texts'])
        perplexity[str(order)]=round(candidate.perplexity(dev_texts),4)
        if best_model is None or perplexity[str(order)]<perplexity[str(best_order)]: best_model,best_order=candidate,order
        else: del candidate
        gc.collect()
    return best_model,dict(type='character n-gram with stupid backoff',order=best_order,
                           order_selection='lowest validation perplexity',order_perplexity=perplexity,
                           backoff_weight=args.lm_backoff,unigram_pseudo=best_model.unigram_pseudo,
                           alphabet=ALPHABET,alphabet_size=len(ALPHABET),
                           lm_train_source=[str(path) for path in args.lm_source],
                           lm_source_records=corpus['sources'],lm_transcripts=len(corpus['texts']),
                           lm_characters=sum(len(text) for text in corpus['texts']),
                           held_out_names_checked=len(held_out_names),
                           held_out_records_excluded=corpus['excluded_records'],
                           leak_guard='every LM source is filtered against the validation and test clip names',
                           dev_split='validation',dev_clips=len(dev_texts),
                           dev_characters=sum(len(text) for text in dev_texts),
                           dev_perplexity=perplexity[str(best_order)])


def language_scores(model):
    """Score hypothesis texts once each: (with end token, without end token)."""
    cache={}
    def score(text):
        value=cache.get(text)
        if value is None:
            value=(model.score(text),model.score(text,include_end=False))
            cache[text]=value
        return value
    return score


def rank_hypotheses(hypotheses,combo):
    """Apply one rescoring rule to (text, acoustic, lm_with_end, lm_without_end) rows."""
    kind=combo[0]
    if kind=='add':
        _,weight,bonus,end=combo
        return max(hypotheses,key=lambda row:row[1]+weight*(row[2] if end else row[3])+bonus*len(row[0]))
    _,band,bonus,end=combo
    top=max(row[1] for row in hypotheses)
    pool=[row for row in hypotheses if row[1]>=top-band]
    return max(pool,key=lambda row:(row[2] if end else row[3])+bonus*len(row[0]))


def rescore_rows(rows,combos):
    """Re-rank stored n-best hypotheses at every setting, sharing pair metrics via cache.

    The additive rule (LM weight plus the classic per-character insertion bonus) can hurt a
    strong autoregressive model because the LM adds a per-character length penalty; the
    tie-band rule only lets the LM choose among hypotheses whose acoustic scores are nearly
    tied. Both families are tuned on the validation split only.
    """
    chosen={combo:set() for combo in combos}
    for expected,hypotheses in rows:
        for combo in combos:
            chosen[combo].add((expected,rank_hypotheses(hypotheses,combo)[0]))
    cache={}
    tallies={}
    for combo,pairs in chosen.items():
        tally=ErrorTally()
        for expected,actual in pairs:
            pair=cache.get((expected,actual))
            if pair is None:
                pair=cache[(expected,actual)]=ErrorTally();pair.add(expected,actual)
            tally.merge(pair)
        tallies[combo]=tally
    return tallies


def selection_combos(weights,ends,bonuses,bands):
    combos=[('add',weight,bonus,end) for weight in weights for bonus in bonuses for end in ends
            if not (weight==0.0 and bonus!=0.0)]
    combos+=[('band',band,bonus,end) for band in bands for bonus in bonuses[:3] for end in ends]
    return combos


def combo_label(combo):
    kind,first,bonus,end=combo
    if kind=='add':
        return f'add weight={first:g} bonus={bonus:g} end={end}'
    return f'band={first:g} bonus={bonus:g} end={end}'


def pick_setting(tallies):
    summaries={key:tally.summary() for key,tally in tallies.items()}
    table={combo_label(key):dict(cer=round(summary['cer'],5),
                                 exact_sequence_accuracy=round(summary['exact_sequence_accuracy'],5),
                                 deletions=summary['deletions'],substitutions=summary['substitutions'],
                                 insertions=summary['insertions'])
           for key,summary in summaries.items()}
    best=min(summaries,key=lambda key:(round(summaries[key]['cer'],6),-round(summaries[key]['exact_sequence_accuracy'],6)))
    return table,best


@torch.inference_mode()
def head_frames(model,directory,split,order,clips,batch_size,device):
    """Yield (clip, per-frame log probabilities); causal padding never enters a slice."""
    frames=np.load(directory/f'{split}_frames.npy',mmap_mode='r')
    for start in range(0,len(order),batch_size):
        group=order[start:start+batch_size]
        lengths=[clips[index]['end']-clips[index]['start'] for index in group]
        values=torch.nn.utils.rnn.pad_sequence([torch.from_numpy(np.array(frames[clips[index]['start']:clips[index]['end']]))
                                               for index in group],batch_first=True).to(device)
        probabilities=model(values).log_softmax(-1).float().cpu().numpy()
        for row,index in enumerate(group):
            yield clips[index],probabilities[row,:lengths[row]]


def head_pass(model,args,split,order,language,*,lm_weight=0.0,fused=False,n_best=1):
    """Greedy CTC plus one prefix beam per clip, keeping n-best rows when asked."""
    clips=clips_of(args.data,split)
    greedy,beam=ErrorTally(),ErrorTally()
    rows=[]
    score=language_scores(language)
    started=time.perf_counter()
    for clip,probabilities in head_frames(model,args.data,split,order,clips,args.head_batch_size,args.device):
        truth=clip['text']
        greedy.add(truth,decode_ctc(probabilities.argmax(-1)))
        hypotheses=ctc_prefix_beam(probabilities,VOCAB,beam_width=args.beam_width,n_best=n_best,top_k=args.top_k,
                                   lm=language if fused else None,lm_weight=lm_weight if fused else 0.0)
        beam.add(truth,hypotheses[0][0])
        if n_best>1: rows.append((truth,[(text,acoustic,*score(text)) for text,acoustic,_ in hypotheses]))
    return dict(greedy=greedy,beam=beam,rows=rows,seconds=round(time.perf_counter()-started,1))


def decode_tokens(tokens,chars):
    """Same token handling as the reference harness: 0 ends the transcript, 61 is padding."""
    result=[]
    for token in tokens[1:]:
        token=int(token)
        if token in (0,len(chars)+1): break
        if token>=len(chars):
            result.append('\ufffd'); continue
        result.append(chars[token])
    return ''.join(result).upper()


@torch.inference_mode()
def decoder_step(network,tokens,memory,source_mask):
    from lib.utils.mask import get_batch_attention_mask
    from lib.utils.pos import sinusoidal_positional_encoding
    length=tokens.shape[1]
    embedding=network.char_embedding(tokens)+sinusoidal_positional_encoding(length,network.d_model,device=tokens.device)
    causal=torch.triu(torch.full((length,length),float('-inf'),device=tokens.device),1)
    output,_=network.decoder(tgt=embedding,memory=memory,tgt_mask=causal,
                             tgt_key_padding_mask=get_batch_attention_mask(tokens,network.char_size),
                             memory_key_padding_mask=source_mask)
    return network.lm_head(output[:,-1,:])


@torch.inference_mode()
def ar_group(network,chars,items,device,*,beam_width,max_tokens,n_best=1):
    """Greedy plus beam search over the upstream AR decoder for one length-sorted group.

    items is a list of (expected text, poses, segments, frame indices). Each entry comes back as
    (expected, greedy text, n-best (text, model log probability) sorted best first).
    """
    import constants
    from lib.utils.mask import get_batch_attention_mask
    bos,eos,pad=len(chars),0,len(chars)+1
    poses=torch.nn.utils.rnn.pad_sequence([item[1] for item in items],batch_first=True,padding_value=-2).to(device)
    segments=torch.nn.utils.rnn.pad_sequence([item[2] for item in items],batch_first=True,padding_value=32).to(device)
    indices=torch.nn.utils.rnn.pad_sequence([item[3] for item in items],batch_first=True,padding_value=1024).to(device)
    memory=network(poses,segments,indices)['encoder_memory']
    source_mask=get_batch_attention_mask(poses,constants.MINUS_TWO_VALUE)

    def search(width):
        rows=len(items)*width
        tokens=[[bos] for _ in range(rows)]
        scores=[0.0 if index%width==0 else float('-inf') for index in range(rows)]
        alive=[True]*rows  # every item owns width slots; only slot 0 starts with a real score
        closed=[[] for _ in items]
        for step in range(max_tokens):
            length=step+1
            prefix=torch.tensor([row+[pad]*(length-len(row)) for row in tokens],dtype=torch.long,device=device)
            logits=decoder_step(network,prefix,memory.repeat_interleave(width,0) if width>1 else memory,
                                source_mask.repeat_interleave(width,0) if width>1 else source_mask)
            probabilities=torch.log_softmax(logits.float(),-1).cpu().numpy()
            for position in range(len(items)):
                base=position*width
                following=[]
                for slot in range(width):
                    if not alive[base+slot] or scores[base+slot]==float('-inf'): continue
                    row=probabilities[base+slot]
                    for token in np.argpartition(-row,width-1)[:width]:
                        token=int(token)
                        state=scores[base+slot]+row[token]
                        if token in (eos,pad): closed[position].append((state,tokens[base+slot]))
                        else: following.append((state,tokens[base+slot]+[token]))
                following.sort(key=lambda entry:-entry[0])
                following=following[:width]
                for slot in range(width):
                    alive[base+slot]=slot<len(following)
                    if alive[base+slot]: scores[base+slot],tokens[base+slot]=following[slot]
                    else: scores[base+slot],tokens[base+slot]=float('-inf'),[bos]
            if not any(alive): break
        for position in range(len(items)):
            for slot in range(width):
                index=position*width+slot
                if alive[index] and scores[index]>float('-inf'): closed[position].append((scores[index],tokens[index]))
        keep=max(width,n_best)
        return [[(decode_tokens(sequence,chars),state) for state,sequence in sorted(entry,key=lambda row:-row[0])[:keep]]
                for entry in closed]

    greedy=search(1)
    beams=greedy if beam_width<2 else search(max(1,beam_width))
    result=[]
    for index in range(len(items)):
        pool=list(beams[index])
        greedy_row=greedy[index][0] if greedy[index] else None
        if greedy_row is not None and greedy_row[0] not in {text for text,_ in pool}:
            pool.append(greedy_row)  # beam search can prune the greedy path; rescore both
        result.append((items[index][0],greedy_row[0] if greedy_row else '',beams[index],pool))
    return result


def ar_pass(network,chars,args,split,order,language,*,n_best):
    """One decoding pass of the reference model over a split, with n-best rows for rescoring."""
    clips=clips_of(args.data,split)
    with np.load(args.ar_npz_dir/f'fsboard_{split}.npz',allow_pickle=True) as data:
        names,right,left=(data[k] for k in ('vidnames','right_hand_poses','left_hand_poses'))
        lookup={str(name):index for index,name in enumerate(names)}
        sizes={name:len(right[index])+len(left[index]) for name,index in lookup.items()}
    order=sorted(order,key=lambda index:sizes[clips[index]['name']])
    greedy,beam=ErrorTally(),ErrorTally()
    rows=[]
    score=language_scores(language)
    started=time.perf_counter()
    for start in range(0,len(order),args.ar_batch_size):
        group=order[start:start+args.ar_batch_size]
        items=[(clips[index]['text'],*preprocess(right[lookup[clips[index]['name']]],left[lookup[clips[index]['name']]]))
               for index in group]
        for truth,guess,candidates,pool in ar_group(network,chars,items,args.device,beam_width=args.ar_beam_width,
                                                    max_tokens=args.ar_max_tokens,n_best=n_best):
            greedy.add(truth,guess)
            beam.add(truth,candidates[0][0] if candidates else '')
            rows.append((truth,[(text,acoustic,*score(text)) for text,acoustic in pool[:n_best+1]]))
    return dict(greedy=greedy,beam=beam,rows=rows,seconds=round(time.perf_counter()-started,1))


def block_of(tally,**extra):
    summary=tally.summary()
    summary.update(extra)
    return summary


def deltas(before,after):
    return dict(cer_points=round(100*(before['cer']-after['cer']),3),
                exact_sequence_points=round(100*(after['exact_sequence_accuracy']-before['exact_sequence_accuracy']),3),
                letter_accuracy_points=round(100*(after['letter_accuracy']-before['letter_accuracy']),3),
                deletion_change=after['deletions']-before['deletions'],
                substitution_change=after['substitutions']-before['substitutions'],
                insertion_change=after['insertions']-before['insertions'],
                deletion_rate_points=round(100*(after['deletion_rate']-before['deletion_rate']),3),
                insertion_rate_points=round(100*(after['insertion_rate']-before['insertion_rate']),3))


def setting_row(summary):
    return dict(cer=round(summary['cer'],5),exact_sequence_accuracy=round(summary['exact_sequence_accuracy'],5),
                deletions=summary['deletions'],substitutions=summary['substitutions'],insertions=summary['insertions'])


def run_head(head,args,language):
    dev_clips=clips_of(args.data,args.select_split)
    dev=by_clip_length(dev_clips,draw(dev_clips,args.select_limit,args.seed))
    selected=head_pass(head,args,args.select_split,dev,language,n_best=args.n_best)
    combos=selection_combos(args.lm_weight_grid,args.lm_end_tokens,args.lm_insert_bonus,args.lm_band_grid)
    table,combo=pick_setting(rescore_rows(selected['rows'],combos))
    weight,bonus,end=combo[1],combo[2],combo[3]
    fusion_draw=by_clip_length(dev_clips,draw(dev_clips,args.fusion_select_limit or args.select_limit,args.seed+1))
    fusion_table={};fusion_best=None
    for candidate in args.lm_weight_grid:
        sweep=head_pass(head,args,args.select_split,fusion_draw,language,lm_weight=candidate,fused=True,n_best=1)
        summary=sweep['beam'].summary()
        fusion_table[f'{candidate:g}']=setting_row(summary)
        if fusion_best is None or (summary['cer'],-summary['exact_sequence_accuracy'])<fusion_best[1]:
            fusion_best=(candidate,(summary['cer'],-summary['exact_sequence_accuracy']))
    fusion_weight=fusion_best[0]
    result=dict(checkpoint=str(args.checkpoint.resolve()),parameters=sum(p.numel() for p in head.parameters()),
                beam_width=args.beam_width,n_best=args.n_best,top_k=args.top_k,
                selection_split=args.select_split,selection_clips=len(dev),selection_beam_seconds=selected['seconds'],
                rescore_rule=combo_label(combo),lm_weight=weight,lm_insert_bonus=bonus,lm_include_end=end,
                lm_band=combo[1] if combo[0]=='band' else None,
                selection_table=table,fusion_weight=fusion_weight,fusion_selection_clips=len(fusion_draw),
                fusion_selection_table=fusion_table)
    for split in args.evaluate_splits:
        clips=clips_of(args.data,split)
        order=by_clip_length(clips,draw(clips,args.limit,args.seed))
        decoded=head_pass(head,args,split,order,language,n_best=args.n_best)
        rescored=rescore_rows(decoded['rows'],[combo])[combo]
        fused=head_pass(head,args,split,order,language,lm_weight=fusion_weight,fused=True,n_best=1)
        tuned=f'{combo_label(combo)}'
        conditions=dict(greedy_ctc=block_of(decoded['greedy'],decoder='greedy CTC'),
                        beam_no_lm=block_of(decoded['beam'],decoder='CTC prefix beam, best label sequence, no language model'),
                        beam_lm_rescored=block_of(rescored,decoder=f'CTC prefix beam n-best rescored, {tuned}'),
                        beam_lm_fused=block_of(fused['beam'],decoder=f'CTC prefix beam with lm shallow fusion at weight {fusion_weight:g}'))
        result[split]=dict(seconds=decoded['seconds'],fused_seconds=fused['seconds'],conditions=conditions,
                           deltas=dict(greedy_ctc_to_beam_no_lm=deltas(conditions['greedy_ctc'],conditions['beam_no_lm']),
                                       beam_no_lm_to_rescored=deltas(conditions['beam_no_lm'],conditions['beam_lm_rescored']),
                                       beam_no_lm_to_fused=deltas(conditions['beam_no_lm'],conditions['beam_lm_fused']),
                                       greedy_ctc_to_fused=deltas(conditions['greedy_ctc'],conditions['beam_lm_fused'])))
        print(json.dumps(dict(split=split,**{key:round(value['cer'],4) for key,value in conditions.items()})),flush=True)
    return result


def load_reference(args):
    import importlib
    import yaml
    source=args.ar_source.resolve()
    if not (source/'lib/models/recognizer.py').is_file():
        raise ValueError('OpenFS source checkout is missing lib/models/recognizer.py.')
    sys.path.insert(0,str(source))
    module=importlib.import_module('lib.models.recognizer')
    config=yaml.safe_load((args.ar_model_dir/'.hydra/config.yaml').read_text(encoding='utf-8'))
    chars=config['decode']['chars']
    kwargs={key:value for key,value in config['recognizer'].items() if not key.startswith('_') and key!='ckpt_path'}
    network=module.EncoderDecoderTransformer(char_size=len(chars)+1,**kwargs).to(args.device).eval()
    network.load_state_dict(torch.load(args.ar_model_dir/'best_model.pth',map_location=args.device,weights_only=True))
    return network,chars


def run_reference(args,language):
    network,chars=load_reference(args)
    dev_clips=clips_of(args.data,args.select_split)
    dev=draw(dev_clips,args.select_limit,args.seed)
    selected=ar_pass(network,chars,args,args.select_split,dev,language,n_best=args.n_best)
    table,combo=pick_setting(rescore_rows(selected['rows'],selection_combos(args.lm_weight_grid,args.lm_end_tokens,
                                                                            args.lm_insert_bonus,args.lm_band_grid)))
    weight,bonus,end=combo[1],combo[2],combo[3]
    result=dict(model='official OpenFS FSboard pretrained recognizer',
                checkpoint=str((args.ar_model_dir/'best_model.pth').resolve()),parameters=sum(p.numel() for p in network.parameters()),
                beam_width=args.ar_beam_width,n_best=args.n_best,max_tokens=args.ar_max_tokens,
                selection_split=args.select_split,selection_clips=len(dev),selection_beam_seconds=selected['seconds'],
                rescore_rule=combo_label(combo),lm_weight=weight,lm_insert_bonus=bonus,lm_include_end=end,
                lm_band=combo[1] if combo[0]=='band' else None,selection_table=table)
    for split in args.evaluate_splits:
        clips=clips_of(args.data,split)
        decoded=ar_pass(network,chars,args,split,draw(clips,args.limit,args.seed),language,n_best=args.n_best)
        rescored=rescore_rows(decoded['rows'],[combo])[combo]
        tuned=combo_label(combo)
        conditions=dict(greedy_ar=block_of(decoded['greedy'],decoder='greedy autoregressive decoding (beam width 1)'),
                        beam_no_lm=block_of(decoded['beam'],decoder='autoregressive beam search, no language model'),
                        beam_lm_rescored=block_of(rescored,decoder=f'autoregressive beam and greedy n-best rescored with {tuned}'))
        result[split]=dict(seconds=decoded['seconds'],conditions=conditions,
                           deltas=dict(greedy_ar_to_beam_no_lm=deltas(conditions['greedy_ar'],conditions['beam_no_lm']),
                                       beam_no_lm_to_rescored=deltas(conditions['beam_no_lm'],conditions['beam_lm_rescored']),
                                       greedy_ar_to_rescored=deltas(conditions['greedy_ar'],conditions['beam_lm_rescored'])))
        print(json.dumps(dict(split=split,**{key:round(value['cer'],4) for key,value in conditions.items()})),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/openfs-packed')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--lm-source',type=Path,action='append')
    p.add_argument('--lm-order-sweep',type=int,nargs='+',default=[3,4,5,6])
    p.add_argument('--lm-backoff',type=float,default=.4)
    p.add_argument('--lm-weight-grid',type=float,nargs='+',default=[0.,.25,.5,.75,1.,1.5,2.,3.,4.])
    p.add_argument('--lm-insert-bonus',type=float,nargs='+',default=[0.,1.,2.,3.,4.5])
    p.add_argument('--lm-band-grid',type=float,nargs='+',default=[0.,1.,2.,3.,5.,8.,12.])
    p.add_argument('--lm-end-tokens',type=json.loads,nargs='+',default=[True,False])
    p.add_argument('--beam-width',type=int,default=12)
    p.add_argument('--n-best',type=int,default=8)
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--head-batch-size',type=int,default=32)
    p.add_argument('--ar-source',type=Path)
    p.add_argument('--ar-model-dir',type=Path)
    p.add_argument('--ar-npz-dir',type=Path,help='defaults to the pinned FSboard raw extraction')
    p.add_argument('--ar-beam-width',type=int,default=8)
    p.add_argument('--ar-max-tokens',type=int,default=64)
    p.add_argument('--ar-batch-size',type=int,default=8)
    p.add_argument('--fusion-select-limit',type=int,default=1000)
    p.add_argument('--trusted-source',action='store_true')
    p.add_argument('--evaluate-splits',nargs='+',default=['test'])
    p.add_argument('--select-split',default='validation')
    p.add_argument('--select-limit',type=int,default=3000)
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--report',type=Path,default=ROOT/'data/lm-rescoring.report.json')
    args=p.parse_args()
    if args.lm_source is None: args.lm_source=DEFAULT_LM_SOURCES
    torch.set_num_threads(4)
    started=time.perf_counter()
    held_out={row['name'] for split in ('validation','test') for row in clips_of(args.data,split)}
    language,lm=fit_language_model(args,held_out)
    print(json.dumps(dict(lm=lm),indent=2),flush=True)
    result=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),torch=torch.__version__,numpy=np.__version__,
                device=args.device,gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                seed=args.seed,lm=lm,
                splits={split:target_stats(clips_of(args.data,split))
                        for split in set(args.evaluate_splits)|{args.select_split}},
                selection=dict(split=args.select_split,limit=args.select_limit,lm_weight_grid=args.lm_weight_grid,
                               insert_bonus_grid=args.lm_insert_bonus,band_grid=args.lm_band_grid,
                               end_token_variants=args.lm_end_tokens,
                               fusion_selection_limit=args.fusion_select_limit,tuned_on_dev_text_only=True))
    if args.checkpoint:
        head=load_checkpoint(args.checkpoint,'sequence').to(args.device).eval()
        result['causal_head']=run_head(head,args,language)
    if args.ar_source:
        if not (args.ar_model_dir and args.trusted_source):
            p.error('--ar-source requires --ar-model-dir and --trusted-source to evaluate the reference model.')
        if args.ar_npz_dir is None:
            args.ar_npz_dir=ROOT/'data/external/openfs/extracted/fingerspelling_fsboard_raw'
        result['openfs_reference']=run_reference(args,language)
    result['seconds']=round(time.perf_counter()-started,1)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(args.report),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    main()
