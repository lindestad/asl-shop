"""GPU training and fixed-split evaluation of the packed FSboard app models."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# Set before torch initializes its caching allocator, which otherwise reserves
# about twice what the sequence model keeps live (1230 MiB reserved, 560 MiB live).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT, HANDSHAPES, VOCAB, HandshapeNet, FingerspellingNet, decode_ctc, load_checkpoint
from scripts.train_neural import edit_distance


def read_data(directory, split, task):
    if task == 'handshape':
        with np.load(directory/f'{split}_handshape.npz') as z:
            return dict(x=z['features'], y=z['labels'], clips=z['clips'])
    return dict(x=np.load(directory/f'{split}_frames.npy',mmap_mode='r'),
                aux=np.load(directory/f'{split}_targets.npy',mmap_mode='r'),
                clips=json.loads((directory/f'{split}_clips.json').read_text(encoding='utf-8')))


def batches(data, batch_size, task, *, shuffle=False, augment=False, limit=None):
    if task == 'handshape':
        indices=np.arange(len(data['y']))
        if shuffle: np.random.shuffle(indices)
        if limit: indices=indices[:limit]
        for start in range(0,len(indices),batch_size):
            selected=indices[start:start+batch_size]
            values=data['x'][selected].copy()
            if augment:
                xy=values.reshape(-1,21,2)
                angles=np.random.normal(0,.07,len(xy))
                c,s=np.cos(angles),np.sin(angles)
                rotation=np.stack((c,-s,s,c),axis=-1).reshape(-1,2,2)
                xy[:]=np.einsum('bji,bik->bjk',xy,rotation)
                values += np.random.normal(0,.01,values.shape).astype(np.float32)
            yield torch.from_numpy(values),torch.from_numpy(data['y'][selected])
        return
    indices=np.arange(len(data['clips']))
    if shuffle: np.random.shuffle(indices)
    if limit: indices=indices[:limit]
    # Shuffle windows, then sort within each window to reduce padded GPU work.
    chunks=[sorted(indices[i:i+batch_size*50],key=lambda j:data['clips'][j]['end']-data['clips'][j]['start'])
            for i in range(0,len(indices),batch_size*50)]
    groups=[window[i:i+batch_size] for window in chunks for i in range(0,len(window),batch_size)]
    if shuffle: random.shuffle(groups)
    for selected in groups:
        frames,aux,targets=[],[],[]
        for index in selected:
            row=data['clips'][index]
            values=np.array(data['x'][row['start']:row['end']],copy=True)
            labels=np.array(data['aux'][row['start']:row['end']],dtype=np.int64)
            text=row['text']
            if augment:
                required=len(text)+sum(a==b for a,b in zip(text,text[1:]))
                length=max(required,round(len(values)*random.uniform(.85,1.15)))
                sampled=np.linspace(0,len(values)-1,length).round().astype(int)
                values,labels=values[sampled].copy(),labels[sampled]
                for base in (0,45):
                    present=values[:,base+44:base+45]
                    values[:,base:base+42]+=np.random.normal(0,.01,(length,42)).astype(np.float32)*present
                    # Translate the camera origin in hand units, consistently over the clip.
                    values[:,base+42:base+44]+=np.random.normal(0,.15,(1,2)).astype(np.float32)*present
                values[:,-1]*=(row['end']-row['start'])/length
            frames.append(torch.from_numpy(values))
            aux.append(torch.from_numpy(labels))
            targets.append(torch.tensor([VOCAB.index(c) for c in text],dtype=torch.int32))
        lengths=torch.tensor([len(v) for v in frames],dtype=torch.int32)
        counts=torch.tensor([len(v) for v in targets],dtype=torch.int32)
        yield (torch.nn.utils.rnn.pad_sequence(frames,batch_first=True),torch.cat(targets),
               lengths,counts,torch.nn.utils.rnn.pad_sequence(aux,batch_first=True,padding_value=-100),selected)


@torch.inference_mode()
def evaluate(model,data,args,split):
    model.eval()
    started=time.perf_counter()
    if args.task=='handshape':
        confusion=np.zeros((len(HANDSHAPES),len(HANDSHAPES)),dtype=np.int64)
        accepted=correct_accepted=total=correct=0
        for values,labels in batches(data,args.batch_size,args.task):
            scores=model(values.to(args.device)).softmax(-1).cpu().numpy()
            expected=labels.numpy(); predicted=scores.argmax(-1)
            np.add.at(confusion,(expected,predicted),1)
            keep=(scores.max(-1)>=.8)&(predicted!=0)
            accepted+=int(keep.sum());correct_accepted+=int(((expected==predicted)&keep).sum())
            total+=len(expected);correct+=int((expected==predicted).sum())
        metrics=dict(accuracy=correct/total,frames=total,threshold=.8,
                     accepted_coverage=accepted/total,accepted_accuracy=correct_accepted/accepted if accepted else None,
                     per_class={label:dict(recall=float(confusion[i,i]/confusion[i].sum()) if confusion[i].sum() else None,
                                           frames=int(confusion[i].sum())) for i,label in enumerate(HANDSHAPES)},
                     label_provenance='agreement with model-derived frame annotations, not human held-letter accuracy',
                     idle_false_positive_rate=None)
    else:
        stats=defaultdict(lambda:dict(errors=0,characters=0,exact=0,clips=0))
        examples=[]
        for batch in batches(data,args.batch_size,args.task):
            values,_,lengths,_,_,selected=batch
            with torch.autocast('cuda',enabled=args.amp and args.device.startswith('cuda')):
                predicted=model(values.to(args.device)).argmax(-1).cpu().numpy()
            for i,index in enumerate(selected):
                row=data['clips'][index]
                actual=decode_ctc(predicted[i,:int(lengths[i])])
                error=edit_distance(row['text'],actual)
                for key in ('overall',row['signer']):
                    stats[key]['errors']+=error;stats[key]['characters']+=len(row['text'])
                    stats[key]['exact']+=actual==row['text'];stats[key]['clips']+=1
                if len(examples)<30:
                    examples.append(dict(name=row['name'],truth=row['text'],prediction=actual))
        def summarize(s):
            return dict(cer=s['errors']/s['characters'],exact_sequence_accuracy=s['exact']/s['clips'],**s)
        metrics=summarize(stats.pop('overall'))
        metrics.update(per_signer={k:summarize(v) for k,v in stats.items()},examples=examples,
                       decoder='greedy CTC; no dictionary or language correction',idle_false_positive_rate=None)
    metrics.update(split=split,seconds=round(time.perf_counter()-started,3))
    return metrics


def checkpoint(model,args,metrics,epoch):
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(contract=CONTRACT,task=args.task,hidden=args.hidden,
                    labels=HANDSHAPES if args.task=='handshape' else VOCAB,
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    metrics=metrics,epoch=epoch,seed=args.seed,dataset=str(args.data.resolve()),
                    experimental=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix='.pt', delete=False) as file:
        temporary=Path(file.name)
    try:
        torch.save(payload,temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--task',choices=['handshape','sequence'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--hidden',type=int,default=192)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--train-limit',type=int)
    p.add_argument('--warmstart-handshape',type=Path)
    p.add_argument('--auxiliary-frame-weight',type=float,default=.2)
    p.add_argument('--evaluate',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    schema=json.loads((args.data/'dataset.json').read_text())
    if schema.get('generation'):
        root=args.data.resolve()
        prepared=(root/schema['generation']).resolve()
        if not prepared.is_relative_to(root):
            raise ValueError('Prepared generation leaves the dataset directory.')
        args.data=prepared
    if schema['contract']!=CONTRACT or schema['labels']!=VOCAB or schema['static_labels']!=HANDSHAPES:
        raise ValueError('Packed dataset preprocessing/vocabulary mismatch; prepare it again.')
    sets=[set(part['signers']) for part in schema['partitions']]
    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Signer leakage in dataset metadata.')
    if args.evaluate:
        model=load_checkpoint(args.evaluate,args.task).to(args.device)
        metrics=evaluate(model,read_data(args.data,'test',args.task),args,'test')
        metrics.update(checkpoint=str(args.evaluate.resolve()),contract=CONTRACT,
                       gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
        print(json.dumps(metrics),flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        return
    train,validation=read_data(args.data,'train',args.task),read_data(args.data,'validation',args.task)
    model=(HandshapeNet if args.task=='handshape' else FingerspellingNet)(args.hidden).to(args.device)
    if args.resume:
        args.hidden=torch.load(args.resume,map_location='cpu',weights_only=True)['hidden']
        model=load_checkpoint(args.resume,args.task).to(args.device)
    elif args.warmstart_handshape and args.task=='sequence':
        handshape=load_checkpoint(args.warmstart_handshape,'handshape')
        encoder=torch.nn.Sequential(*list(handshape.network.children())[:-1])
        model.hand_encoder.load_state_dict(encoder.state_dict())
        with torch.no_grad():
            model.input.weight.zero_();model.input.bias.zero_()
            model.input.weight[:,:args.hidden]=torch.eye(args.hidden,device=args.device)
            model.output.weight.zero_();model.output.bias.fill_(-3.)
            model.output.bias[0]=0
            for i,label in enumerate(HANDSHAPES[1:],1):
                j=VOCAB.index(label)
                model.output.weight[j]=handshape.network[-1].weight[i].to(args.device)
                model.output.bias[j]=handshape.network[-1].bias[i].to(args.device)
    parameters=sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(task=args.task,parameters=parameters,torch=torch.__version__,
                          device=args.device,gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                          training=len(train['y']) if args.task=='handshape' else len(train['clips']),
                          validation=len(validation['y']) if args.task=='handshape' else len(validation['clips']))),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.learning_rate*.05)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and args.device.startswith('cuda'))
    ctc=torch.nn.CTCLoss(blank=0,zero_infinity=False)
    weights=None
    if args.task=='handshape':
        counts=np.bincount(train['y'],minlength=len(HANDSHAPES))
        weights=torch.tensor(np.sqrt(counts.max()/np.maximum(counts,1)),dtype=torch.float32,device=args.device)
        weights[0]=0
    best=float('inf');stale=0;history=[];run_started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();started=time.perf_counter();loss_sum=0;steps=0
        for batch in batches(train,args.batch_size,args.task,shuffle=True,augment=True,limit=args.train_limit):
            values,targets=batch[:2]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=args.amp and args.device.startswith('cuda')):
                logits=model(values.to(args.device))
            if args.task=='handshape':
                loss=F.cross_entropy(logits.float(),targets.to(args.device),weight=weights)
            else:
                lengths,counts,aux,_=batch[2:]
                log_probs=logits.float().log_softmax(-1).transpose(0,1).contiguous()
                # int64 targets select the log-domain native CUDA implementation;
                # cuDNN's probability-domain path can underflow on sharp logits.
                loss=ctc(log_probs,targets.to(args.device,dtype=torch.int64),lengths.long(),counts.long())
                if args.auxiliary_frame_weight and bool((aux!=-100).any()):
                    frame_loss=F.cross_entropy(logits.float().reshape(-1,len(VOCAB)),aux.to(args.device).reshape(-1),ignore_index=-100)
                    loss=loss+args.auxiliary_frame_weight*frame_loss
            if not torch.isfinite(loss):
                detail=dict(step=steps,loss=float(loss.detach()),
                            logits_finite=bool(torch.isfinite(logits).all()),
                            maximum_coordinate=float(values.abs().max()))
                if args.task=='sequence':
                    detail['clips']=[train['clips'][i]['name'] for i in batch[-1]]
                    detail['lengths']=lengths.tolist()
                    detail['target_lengths']=counts.tolist()
                raise ValueError('Non-finite loss: '+json.dumps(detail))
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach());steps+=1
            if steps%200==0:
                print(json.dumps(dict(epoch=epoch,steps=steps,loss=round(loss_sum/steps,5),
                                      training_seconds=round(time.perf_counter()-started,1))),flush=True)
        metrics=evaluate(model,validation,args,'validation')
        score=1-metrics['accuracy'] if args.task=='handshape' else metrics['cer']
        row=dict(epoch=epoch,loss=loss_sum/steps,training_seconds=round(time.perf_counter()-started-metrics['seconds'],2),
                 validation_score=score,validation_exact=metrics.get('exact_sequence_accuracy'),
                 validation_coverage=metrics.get('accepted_coverage'),elapsed_seconds=round(time.perf_counter()-run_started,2))
        history.append(row);print(json.dumps(row),flush=True)
        if score<best:
            best=score;stale=0;checkpoint(model,args,metrics,epoch)
        else: stale+=1
        args.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        scheduler.step()
        if stale>=args.patience:
            print('Validation early stopping',flush=True);break
    model=load_checkpoint(args.output,args.task).to(args.device)
    test=evaluate(model,read_data(args.data,'test',args.task),args,'test')
    report=dict(task=args.task,parameters=parameters,checkpoint=str(args.output.resolve()),
                best_validation_score=best,training_seconds=round(time.perf_counter()-run_started,2),
                epochs=len(history),history=history,test=test,dataset=schema,
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
    report_path=args.report or args.output.with_suffix('.report.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report_path),test={k:v for k,v in test.items() if k not in ('examples','per_signer','per_class')})),flush=True)


if __name__=='__main__':
    main()
