"""Evaluate a trusted upstream OpenFS checkpoint; never converts it to an app model.

Requires an explicit official source checkout and its research-use checkpoint.
Uses released compressed detection-order positional indices, matching upstream.
Our app model instead retains original frame gaps, so this is a separate baseline.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_neural import edit_distance


def preprocess(right,left):
    poses,segments,indices=[],[],[]
    people=sorted(set(right)|set(left))
    for person_position,person in enumerate(people):
        for side,stream in enumerate((right,left)):
            xy=np.asarray(stream.get(person,[]),dtype=np.float32)
            if not len(xy): continue
            xy=xy[...,:2].copy()
            xy-=(xy.min(axis=1,keepdims=True)+xy.max(axis=1,keepdims=True))*.5
            xy/=np.abs(xy).max(axis=1,keepdims=True)+1e-6
            xy*=.5
            poses.append(torch.from_numpy(xy))
            segments.append(torch.full((len(xy),),2*person_position+side,dtype=torch.int64))
            indices.append(torch.arange(len(xy),dtype=torch.int64))
    if not poses: raise ValueError('No hands in released sequence.')
    return torch.cat(poses),torch.cat(segments),torch.cat(indices)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--model-dir',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-output',type=int,default=128)
    p.add_argument('--trusted-source',action='store_true')
    args=p.parse_args()
    if not args.trusted_source: p.error('Explicitly acknowledge official source/object NPZ loading with --trusted-source.')
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.source.resolve()))
    from lib.models.recognizer import EncoderDecoderTransformer
    config=yaml.safe_load((args.model_dir/'.hydra/config.yaml').read_text())
    chars=config['decode']['chars']
    kwargs={k:v for k,v in config['recognizer'].items() if not k.startswith('_') and k!='ckpt_path'}
    network=EncoderDecoderTransformer(char_size=len(chars)+1,**kwargs).to(args.device).eval()
    network.load_state_dict(torch.load(args.model_dir/'best_model.pth',weights_only=True,map_location=args.device))
    with np.load(args.input,allow_pickle=True) as data:
        names,words,right,left=(data[k] for k in ('vidnames','words','right_hand_poses','left_hand_poses'))
    metadata={}
    for file in args.metadata.glob('*.json'):
        for item in json.loads(file.read_text()): metadata[Path(item['clipFilename']).stem]=item
    selection=np.arange(len(names))
    if args.limit:
        selection=np.random.default_rng(42).choice(selection,min(len(selection),args.limit),replace=False)
    items=[]
    for index in selection:
        poses,segments,indices=preprocess(right[index],left[index])
        items.append(dict(index=int(index),poses=poses,segments=segments,indices=indices))
    items.sort(key=lambda row:len(row['poses']))
    stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
    examples=[];invalid_tokens=0;started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(items),args.batch_size):
            group=items[start:start+args.batch_size]
            poses=torch.nn.utils.rnn.pad_sequence([x['poses'] for x in group],batch_first=True,padding_value=-2).to(args.device)
            segments=torch.nn.utils.rnn.pad_sequence([x['segments'] for x in group],batch_first=True,padding_value=32).to(args.device)
            indices=torch.nn.utils.rnn.pad_sequence([x['indices'] for x in group],batch_first=True,padding_value=1024).to(args.device)
            generated=network.generate(poses,segments,indices,max_len=args.max_output,
                                       bos_token_id=len(chars),eos_token_id=0,pad_token_id=len(chars)+1)['output_ids'].cpu().numpy()
            for row,tokens in zip(group,generated):
                index=row['index'];text=[]
                for token in tokens[1:]:
                    if token in (0,len(chars)+1): break
                    if token>=len(chars):
                        text.append('\ufffd');invalid_tokens+=1
                    else: text.append(chars[int(token)])
                actual=''.join(text).upper();expected=str(words[index]).strip().upper()
                name=str(names[index]);signer=metadata[name]['signerId']
                error=edit_distance(expected,actual)
                for key in ('overall',signer):
                    stats[key]['errors']+=error;stats[key]['characters']+=len(expected)
                    stats[key]['exact']+=expected==actual;stats[key]['clips']+=1
                if len(examples)<30: examples.append(dict(name=name,truth=expected,prediction=actual))
            if (start//args.batch_size+1)%20==0:
                s=stats['overall']
                print(json.dumps(dict(evaluated=s['clips'],cer=s['errors']/s['characters'],seconds=round(time.perf_counter()-started,2))),flush=True)
    def summarize(s): return dict(cer=s['errors']/s['characters'],exact_sequence_accuracy=s['exact']/s['clips'],**s)
    result=summarize(stats.pop('overall'))
    commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    result.update(model='official OpenFS FSboard pretrained recognizer',source_commit=commit,
                  checkpoint=str((args.model_dir/'best_model.pth').resolve()),input=str(args.input.resolve()),
                  decoder='upstream greedy autoregressive; no dictionary/language correction',max_output=args.max_output,
                  device=args.device,seconds=time.perf_counter()-started,per_signer={k:summarize(s) for k,s in stats.items()},
                  examples=examples,invalid_special_tokens=invalid_tokens,
                  sample_selection='all available archive clips' if not args.limit else 'seed42 random subset')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','per_signer')}),flush=True)


if __name__=='__main__': main()
