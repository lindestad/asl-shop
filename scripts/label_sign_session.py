"""Make labelling cheap: the current model proposes spans, a human confirms them.

After scripts/collect_signs.py records a session, this replays it through the same streaming path the
app runs and writes annotation.draft.json: idle spans from hand presence, held-letter spans from
stable confident per-frame predictions, J/Z motion spans where the sequence head emitted them under
the app's motion gate, one spelling span per segment that produced text, and activity spans for a
visible hand that never settled on a letter. Travelling time between two claimed spans becomes a
transition span.

Every draft says out loud that it is the model suggesting its own output back: label_provenance is
'model-suggestion', confirmed is false, and each span carries a suggestion_confidence and a note. No
accuracy number may quote a draft; scripts/eval_held_letters.py can score one for plumbing checks and
marks those numbers numbers_from_suggested_labels.

    uv run --extra neural python scripts/label_sign_session.py --session data/sign-sessions/person01-day01
    $EDITOR data/sign-sessions/person01-day01/annotation.draft.json   # correct what is wrong
    uv run --extra neural python scripts/label_sign_session.py --session ... --confirm --by person01

--confirm refuses to promote a draft that does not validate, and writes the confirmed annotation.json
that scripts/eval_held_letters.py scores with human_labels true.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.eval_held_letters import APP_HANDSHAPE_GATE,build_runtime,replay
from scripts.sign_sessions import (ANNOTATION_SCHEMA,MOTION_LETTERS,SignSessionError,read_annotation,
                                   read_session,validate,write_annotation)

DRAFT_NOTICE=('DRAFT: these spans are the current model describing its own output. Nobody has confirmed '
              'them. A person has to correct the letters, split merged spans and delete invented ones '
              'before any accuracy number may quote this file.')
MERGE_GAP_FRAMES=3  # prediction flicker of a few frames should not split one held letter


def _runs(flags,first,last):
    """Half-open runs where flags is True inside [first,last)."""
    runs=[]; start=None
    for index in range(first,last+1):
        on=index<last and bool(flags[index])
        if on and start is None:
            start=index
        if not on and start is not None:
            runs.append((start,index)); start=None
    return runs


def _letter_runs(frames,fps,min_ms,merge_gap=MERGE_GAP_FRAMES):
    """Consecutive-frame runs of one letter, tolerating a few frames of prediction flicker."""
    runs=[]; start=0
    for index in range(1,len(frames)+1):
        if index==len(frames) or frames[index]-frames[index-1]>merge_gap:
            run=frames[start:index]
            if (run[-1]+1-run[0])/fps*1000>=min_ms:
                runs.append((run[0],run[-1]+1,run))
            start=index
    return runs


def _claimed(kind,letter,confidence):
    return (kind,letter,confidence)


def propose_spans(session,data,*,app_gate=APP_HANDSHAPE_GATE,min_held_ms=400,min_motion_ms=250,
                  min_idle_ms=300,min_spelling_chars=2):
    """Frame-level model output turned into a draft of non-overlapping spans.

    Priority per frame: held, then motion, then a decoded spelling run, then travel or leftover
    visible activity, then idle. A draft that overlapped itself would only teach the annotator to
    ignore the validator, so the proposals are a partition of the recording.
    """
    fps=session['measured_fps']; n=len(session['timestamps_ms'])
    claimed=[None]*n
    for segment in data['segments']:
        if not segment['block']:
            continue
        start,end=segment['block']
        letters={}
        for row in data['emissions']:
            if row['path']=='handshape' and not row['retracted'] and row['confidence']>=app_gate \
                    and start<=row['frame']<end:
                letters.setdefault(row['char'],[]).append(row)
        for letter,rows in sorted(letters.items()):
            frames=sorted(row['frame'] for row in rows)
            confidence=round(float(np.mean([row['confidence'] for row in rows])),3)
            for run_start,run_end,_ in _letter_runs(frames,fps,min_held_ms):
                for frame in range(run_start,run_end):
                    if claimed[frame] is None:
                        claimed[frame]=_claimed('held',letter,confidence)
        characters=[row for row in data['emissions'] if row['path']=='sequence' and not row['retracted']
                    and start<=row['frame']<end]
        for letter in sorted(MOTION_LETTERS):
            frames=sorted(row['frame'] for row in characters if row['char']==letter)
            if not frames or (frames[-1]+1-frames[0])/fps*1000<min_motion_ms:
                continue
            for frame in range(frames[0],frames[-1]+1):
                if claimed[frame] is None:
                    claimed[frame]=_claimed('motion',letter,None)
        if len(segment['text'])>=min_spelling_chars and characters:
            first=min(row['frame'] for row in characters); last=max(row['frame'] for row in characters)+1
            if all(claimed[frame] is None for frame in range(first,last)):
                for frame in range(first,last):
                    claimed[frame]=_claimed('spelling',segment['text'],None)
    present=session['present'].any(axis=1)
    spans=[]
    for segment in data['segments']:
        if not segment['block']:
            continue
        start,end=segment['block']
        index=start
        while index<end:
            state=claimed[index] or _claimed('activity',None,None)
            kind,letter,confidence=state
            stop=index
            while stop+1<end and (claimed[stop+1] or _claimed('activity',None,None))==state:
                stop+=1
            travelled=kind=='activity' and index>start and claimed[index-1] is not None \
                      and stop+1<end and claimed[stop+1] is not None
            if travelled:
                kind='transition'
            span=dict(kind=kind,start=index,end=stop+1,note='model suggestion, not confirmed')
            if kind in ('held','motion') and letter is not None:
                span['letter']=letter
            if kind=='spelling' and letter:
                span['text']=letter
                span['note']='model-decoded text for this run; check every character'
            if confidence is not None:
                span['suggestion_confidence']=confidence
            elif kind in ('held','motion','spelling'):
                span['suggestion_confidence']=None
            spans.append(span)
            index=stop+1
    for start,stop in _runs(~present,0,n):
        if (stop-start)/fps*1000>=min_idle_ms:
            spans.append(dict(kind='idle',start=start,end=stop,note='no hand visible in the recording'))
    return sorted(spans,key=lambda span:(span['start'],span['end']))


def draft_for(session,*,spans,models):
    return dict(schema=ANNOTATION_SCHEMA,session_id=session['session_id'],signer_id=session['signer_id'],
                label_provenance='model-suggestion',confirmed=False,notice=DRAFT_NOTICE,
                suggested_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),suggested_by=models,spans=spans)


def show(directory,annotation):
    session=read_session(directory)
    print(f'{directory}: {session["frame_count"]} frames at {session["measured_fps"]} fps, '
          f'{len(annotation.get("spans",[]))} spans, provenance={annotation.get("label_provenance")}, '
          f'confirmed={annotation.get("confirmed") is True}')
    for index,span in enumerate(annotation.get('spans',[])):
        seconds=(span['end']-span['start'])/max(session['measured_fps'],1e-6)
        extra=span.get('letter') or span.get('text') or ''
        print(f'  {index:>3} {span["kind"]:<11} frames {span["start"]:>5}-{span["end"]:<5} {seconds:5.2f}s '
              f'{extra:<12} confidence={span.get("suggestion_confidence")}')
    if annotation.get('notice'):
        print(f'  {annotation["notice"]}')


def _edit(directory):
    editor=os.environ.get('EDITOR')
    if not editor:
        print('  EDITOR is not set; edit annotation.draft.json by hand instead.'); return False
    subprocess.run([editor,str(Path(directory)/'annotation.draft.json')],check=False)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session',dest='sessions',type=Path,action='append',required=True,
                        help='session directory, or a folder of session directories')
    parser.add_argument('--review',action='store_true',help='print the spans instead of proposing them')
    parser.add_argument('--draft',action='store_true',help='with --review, read the draft file')
    parser.add_argument('--confirm',action='store_true',
                        help='promote annotation.draft.json to annotation.json as human-reviewed suggestions')
    parser.add_argument('--by',help='who reviewed the spans; required with --confirm')
    parser.add_argument('--edit',action='store_true',help='open $EDITOR on the draft before confirming')
    parser.add_argument('--app-gate',type=float,default=APP_HANDSHAPE_GATE,
                        help='per-frame letter confidence a held suggestion must reach')
    parser.add_argument('--min-held-ms',type=int,default=400)
    parser.add_argument('--min-motion-ms',type=int,default=250)
    parser.add_argument('--min-idle-ms',type=int,default=300)
    parser.add_argument('--handshape-model',type=Path)
    parser.add_argument('--sequence-model',type=Path)
    parser.add_argument('--skip-sequence',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    arguments=parser.parse_args()
    targets=[]
    for entry in arguments.sessions:
        if (entry/'session.json').is_file():
            targets.append(entry)
        elif entry.is_dir():
            targets.extend(sorted(child for child in entry.iterdir() if (child/'session.json').is_file()))
        else:
            raise SignSessionError(f'no session.json in {entry}')
    if arguments.confirm and not arguments.by:
        parser.error('--confirm needs --by with the name of the person who reviewed the spans')
    runtime=None; logs={}; failed=0
    for directory in targets:
        session=read_session(directory)
        if arguments.review or arguments.confirm:
            if arguments.edit and arguments.confirm:
                _edit(directory)
            annotation=read_annotation(directory,draft=arguments.draft or arguments.confirm)
            if not arguments.confirm:
                show(directory,annotation); continue
            errors,warnings=validate(session,annotation)
            if errors:
                failed+=1
                print(f'REFUSED {directory}: the draft still has contradictions')
                for problem in errors:
                    print(f'  ERROR: {problem}')
                continue
            annotation=dict(annotation)
            annotation.update(label_provenance='human-reviewed-model-suggestions',confirmed=True,
                              labels_by=arguments.by,labelled_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            annotation.pop('notice',None)
            write_annotation(directory,annotation)
            print(f'{directory}: confirmed as human-reviewed-model-suggestions by {arguments.by}')
            for warning in warnings:
                print(f'  note: {warning}')
            continue
        if runtime is None:
            runtime,logs=build_runtime(handshape_model=arguments.handshape_model,
                                       sequence_model=arguments.sequence_model,
                                       skip_sequence=arguments.skip_sequence,threads=arguments.threads)
        data=replay(session,dict(spans=[]),runtime,logs)
        spans=propose_spans(session,data,app_gate=arguments.app_gate,min_held_ms=arguments.min_held_ms,
                            min_motion_ms=arguments.min_motion_ms,min_idle_ms=arguments.min_idle_ms)
        draft=draft_for(session,spans=spans,models=dict(handshape=str(runtime.paths.get('handshape')),
                                                        sequence=str(runtime.paths.get('sequence')),
                                                        app_gate=arguments.app_gate))
        errors,warnings=validate(session,draft)
        write_annotation(directory,draft,draft=True)
        counts={}
        for span in spans:
            counts[span['kind']]=counts.get(span['kind'],0)+1
        print(f'{directory}: proposed {len(spans)} spans {counts} in annotation.draft.json')
        for warning in warnings:
            print(f'  note: {warning}')
        for error in errors:
            print(f'  draft ERROR (fix before confirming): {error}')
        print(f'  next: edit {directory/"annotation.draft.json"}, then --confirm --by <name>')
    if failed:
        print(f'{failed} drafts were not confirmed.')
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
