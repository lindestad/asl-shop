"""Replay recorded app-domain sessions through the real streaming path and score them against spans.

This is the measurement docs/training-results-2026-10-12.md asks for: deliberate held letters, idle
and unrelated hand activity, J/Z motion and natural spelling, replayed from a recording and judged
against the annotation a human confirmed. Replay runs model/live_neural.py on CPU with no camera and
no display, so the gate being scored is the gate the app actually runs. With --via-worker the same
recording is replayed through scripts/recognition_worker.py over the protocol electron/recognition.mjs
uses, which is the desktop-app boundary; it needs a session that kept its video_file.

What the numbers mean:

  held_letters   The app's per-frame letter gate (model/live_neural.py asks for an argmax probability
      of 0.8 or more) is re-applied at each swept threshold. A held span is correct when the modal
      accepted letter equals the annotated letter; a span the app never accepted counts as wrong and
      is also reported separately as coverage. Top-3 asks whether the annotated letter is among the
      three highest mean-probability letters over the frames this threshold accepts, so it moves with
      the threshold the way top-1 does.
  coverage       Fraction of frames where the app emitted something, overall and per annotated span
      kind, so rejection is read per frame type rather than as one number.
  false_accept   Accepted emissions on idle spans and on unrelated-activity spans, per frame and per
      minute. These spans are what calibrate rejection: without them an accuracy number says nothing
      about what the app shows when nobody is signing.
  confusion      Expected class against accepted letter over HANDSHAPES ('nothing' plus the 24 static
      letters). J and Z have no static class, so they can never appear as a column here.
  motion         J and Z spans reported on their own: detection needs the sequence head plus the app's
      local-motion evidence, because today's static head cannot see them at all.
  spelling       CER, exact sequence accuracy and the deletion/substitution/insertion split from
      model/recognition_metrics.py, per annotated run and per whole app segment.

Nothing here is a public benchmark, and it must not be presented as one. If a human never confirmed
the spans, the report says so in its own fields: human_labels is false and
numbers_from_suggested_labels is true, and those numbers only show the current model agreeing with
itself.

    uv run --extra neural python scripts/eval_held_letters.py --session data/sign-sessions/person01-day01
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.recognition_metrics import ErrorTally
from scripts.sign_sessions import (HUMAN_PROVENANCES,SUGGESTED_PROVENANCE,SignSessionError,block_expected,
                                   frame_span_map,hands_at,movement_score,presence_blocks,read_annotation,
                                   read_session,spans_for_block,synthetic_recording,validate)

REPORT_SCHEMA='asl-shop-held-letter-eval/v1'
APP_HANDSHAPE_GATE=.8  # the fixed per-frame letter gate in model/live_neural.py
DEFAULT_THRESHOLDS=(.0,.3,.4,.5,.6,.7,.8,.85,.9,.95,.99)
MOTION_NOTE=('J and Z are movement letters: model/neural.py LETTERS has no class for them, so the static '
             'handshape head can never emit one. Detection here means the sequence head emitted the '
             'letter while the app required local motion evidence in the preceding second.')
SEQUENCE_GATE_NOTE=('model/live_neural.py does not gate sequence characters by model confidence. This '
                    'curve applies a proposed confidence gate to each character the app showed, using the '
                    'strongest per-tick evidence the sequence head gave that frame, so a future gate can '
                    'be calibrated on the same curve. It is not current app behaviour.')
CLAIM_LIMITS=['App-domain calibration on one labelling setup, not a public benchmark and not comparable '
              'to published FSboard numbers, which come from other people\'s clip-level transcripts.',
              'Every frame here comes from a recording made for this harness, so signer, camera, framing '
              'and lighting are the app\'s own conditions; results do not transfer to other setups.',
              'Threshold curves describe the gate as it runs today; picking a threshold from this report '
              'and then quoting the same report overstates what was measured.']


def softmax(table):
    table=np.asarray(table,dtype=np.float64)
    table=table-table.max(axis=-1,keepdims=True)
    weights=np.exp(table)
    return weights/weights.sum(axis=-1,keepdims=True)


class ProbabilityRecorder:
    """Wrap a logit-returning model so replay can see evidence the live gate threw away."""

    def __init__(self,network,*,keep='all'):
        self.network=network; self.keep=keep; self.calls=0; self.probabilities=[]

    def __call__(self,values):
        logits=self.network(values)
        table=logits.detach().cpu().numpy() if hasattr(logits,'detach') else np.asarray(logits)
        if self.keep=='last': self.probabilities=[softmax(table)]
        else: self.probabilities.append(softmax(table))
        self.calls+=1
        return logits


def build_runtime(*,handshape_model=None,sequence_model=None,skip_handshape=False,skip_sequence=False,threads=4):
    """A NeuralRuntime on the app's checkpoints, wrapped to keep frame-level probabilities."""
    import os
    import torch
    from model.live_neural import NeuralRuntime
    if not 1<=threads<=64:
        raise ValueError('thread count must be between 1 and 64')
    torch.set_num_threads(threads)
    if handshape_model: os.environ['ASL_HANDSHAPE_MODEL']=str(handshape_model)
    if sequence_model: os.environ['ASL_SEQUENCE_MODEL']=str(sequence_model)
    runtime=NeuralRuntime(ROOT)
    if skip_handshape: runtime.paths.pop('handshape',None)
    if skip_sequence: runtime.paths.pop('sequence',None)
    runtime.load()
    logs={task:ProbabilityRecorder(runtime.models[task],keep='last' if task=='sequence' else 'all')
          for task in list(runtime.models)}
    for task,wrapper in logs.items():
        runtime.models[task]=wrapper
    return runtime,logs


class WorkerBridge:
    """Speak the app's own newline JSON protocol to scripts/recognition_worker.py."""

    def __init__(self,*,python,script):
        import subprocess
        self.process=subprocess.Popen([str(python),str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,text=True,bufsize=1)

    def analyze(self,*,data_url,session_id,timestamp_ms,finish):
        self.process.stdin.write(json.dumps(dict(id=1,dataUrl=data_url,video=True,sessionId=session_id,
                                                timestampMs=timestamp_ms,finish=finish))+chr(10))
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:
            raise SignSessionError('the recognition worker stopped before answering')
        answer=json.loads(line)
        if 'error' in answer:
            raise SignSessionError(f'recognition worker: {answer["error"]}')
        return answer['result']

    def close(self):
        self.process.kill()


def recorded_frames(session):
    """Re-encode the session's own video so --via-worker feeds the frames the app would have seen."""
    import cv2
    gap=video_gap(session)
    if gap:
        raise SignSessionError(gap)
    capture=cv2.VideoCapture(str(Path(session['directory'])/session['video_file']))
    try:
        while True:
            ok,frame=capture.read()
            if not ok:
                return
            coded=cv2.imencode('.jpg',frame)
            if not coded[0]:
                raise SignSessionError('could not re-encode a recorded frame')
            yield 'data:image/jpeg;base64,'+base64.b64encode(coded[1]).decode()
    finally:
        capture.release()


def video_gap(session):
    """Why --via-worker cannot replay this session, or None when it can.

    The desktop-app boundary replay needs the frames the camera saw, and no recording path in this repo
    keeps those frames yet: scripts/collect_signs.py writes landmark tracks only. Saying so before the
    worker is spawned keeps the failure in the report instead of in a dead subprocess.
    """
    named=session.get('video_file')
    if not named:
        return ('--via-worker replays the recorded video through scripts/recognition_worker.py, and this '
                'session kept no video: session.json has video_file null. Landmark tracks alone cannot '
                'feed that path. Drop --via-worker to replay the landmarks through the same streaming '
                'runtime in-process, or record a session that keeps its source video next to '
                'landmarks.npz and name it in session.json.')
    if not (Path(session['directory'])/named).is_file():
        return (f'--via-worker needs the video this session named, and {named} is not next to '
                f'{session["directory"]}. Drop --via-worker to replay the landmark tracks instead.')
    return None


def segment_id(session,index):
    return f'{session["session_id"]}#seg{index}'[:100]


def _showing(shown,text):
    """Fold one app text update into the characters currently on screen: returns (retracted, added)."""
    current=''.join(row['char'] for row in shown)
    common=0
    while common<len(current) and common<len(text) and current[common]==text[common]:
        common+=1
    for row in shown[common:]:
        row['retracted']=True
    del shown[common:]
    return [dict(frame=None,path='sequence',char=char,confidence=None,retracted=False,segment=None)
            for char in text[common:]]


def replay(session,annotation,runtime,logs,*,limit_frames=None,segment_gap_ms=700,via_worker=False,
           frames=None):
    """Feed one recording through the streaming path; return frame evidence and app emissions.

    Every recorded frame is fed, idle frames included, exactly as the camera feed does. The harness
    passes finish at the last frame of each visible-hand run (the moment a signer would press Finish)
    and then opens a new segment id, because model/live_neural.py stops accumulating once finished.
    Segments are cut from hand presence only, never from the human idle spans, so scoring cannot
    leak the annotation into how the recording is sliced.
    """
    n=len(session['timestamps_ms'])
    if limit_frames: n=min(n,int(limit_frames))
    width,height=int(session['width']),int(session['height'])
    blocks=[block for block in presence_blocks(session,gap_ms=segment_gap_ms) if block[0]<n]
    finishes={block[1]-1 for block in blocks if block[1]-1<n}
    evidence=[]; emissions=[]; segments=[]; shown=[]
    segment=None; index_in_log=0; segment_index=0
    # The handshape recorder keeps every arrival it has seen, so a second session in the same process must
    # index past the first one's frames rather than from the front of the list.
    handshape_offset=len(logs['handshape'].probabilities) if 'handshape' in logs else 0
    for frame in range(n):
        hands=hands_at(session,frame)
        timestamp=float(session['timestamps_ms'][frame])
        finish=frame in finishes
        if segment is None:
            sequence_calls=logs['sequence'].calls if 'sequence' in logs else 0
            segment=dict(index=segment_index,feed_from=frame,feed_to=frame+1,block_from=None,block_to=None,
                         text='',streaming='',method=None,status=None,limit_reached=False,
                         sequence_calls_at_start=sequence_calls)
            segment_index+=1; shown=[]
        segment['feed_to']=frame+1
        if via_worker:
            output=runtime.analyze(data_url=next(frames),session_id=segment_id(session,segment['index']),
                                   timestamp_ms=timestamp,finish=finish)
            seen=output.get('hands') or []
            handshape=output.get('prediction')
        else:
            calls_handshape='handshape' in runtime.models and len(hands)==1
            if calls_handshape: index_in_log+=1
            output=runtime.analyze(hands,width,height,video=True,session_id=segment_id(session,segment['index']),
                                   timestamp_ms=timestamp,finish=finish)
            if calls_handshape and len(logs['handshape'].probabilities)<handshape_offset+index_in_log:
                raise SignSessionError('the live handshape gate no longer calls its model for a single visible '
                                       'hand, so this harness no longer mirrors it: update the harness first')
            seen=hands
            probabilities=None
            if calls_handshape:
                table=np.asarray(logs['handshape'].probabilities[handshape_offset+index_in_log-1])
                probabilities=table.reshape(-1,table.shape[-1])[0]  # the live path scores one hand per arrival
            handshape=None
            if probabilities is not None:
                winner=int(np.argmax(probabilities))
                handshape=dict(letter=None if not winner else _handshape_name(winner),
                               confidence=float(probabilities[winner]),
                               probabilities=[round(float(value),6) for value in probabilities])
        sequence=output['fingerspelling']
        if segment['block_from'] is None and seen:
            start=frame-1
            while start>=segment['feed_from'] and session['present'][start].any():
                start-=1
            segment['block_from']=start+1
        if seen:
            segment['block_to']=frame+1
        segment['streaming']=segment['text']
        segment['text']=sequence.get('text') or ''
        segment['method']=sequence.get('method')
        segment['status']=sequence.get('status')
        segment['limit_reached']=segment['limit_reached'] or sequence.get('status')=='limit'
        row=dict(frame=frame,hands=len(seen),handshape=handshape,sequence_confidence=None,
                 text=segment['text'],segment=segment['index'])
        for pending in _showing(shown,segment['text']):
            pending.update(frame=frame,segment=segment['index'])
            shown.append(pending); emissions.append(pending)
        if handshape and handshape['letter']:
            emissions.append(dict(frame=frame,path='handshape',char=handshape['letter'],
                                  confidence=handshape['confidence'],retracted=False,segment=segment['index']))
        evidence.append(row)
        if finish:
            _collect_sequence_evidence(session,runtime,logs,segment,evidence)
            segments.append(_close_segment(segment)); segment=None
    if segment is not None:
        _collect_sequence_evidence(session,runtime,logs,segment,evidence)
        segments.append(_close_segment(segment))
    for row in emissions:
        if row['path']=='sequence':
            row['confidence']=evidence[row['frame']]['sequence_confidence']
    return dict(evidence=evidence,emissions=emissions,segments=segments,frames=n,
                expected=frame_span_map(annotation,n_frames=n))


def _close_segment(segment):
    """Visible-hand extent of one app segment: where a signer's hand was actually on screen."""
    segment['block']=[segment.pop('block_from'),segment.pop('block_to')] if segment['block_from'] is not None else None
    return segment


def _handshape_name(index):
    from model.neural import HANDSHAPES
    return HANDSHAPES[index]


def _collect_sequence_evidence(session,runtime,logs,segment,evidence):
    """Map the sequence head's per-tick confidence onto the recorded frames of this segment."""
    if not logs.get('sequence') or segment['sequence_calls_at_start']==logs['sequence'].calls:
        return
    ticks=list(getattr(runtime,'model_timestamps',[]))
    if not ticks or not logs['sequence'].probabilities:
        return
    table=np.asarray(logs['sequence'].probabilities[-1])
    table=table[0] if table.ndim==3 else table
    times=session['timestamps_ms']
    for tick in range(min(len(table),len(ticks))):
        position=int(np.searchsorted(times,ticks[tick]))
        candidates=[choice for choice in (position-1,position)
                    if segment['feed_from']<=choice<segment['feed_to'] and 0<=choice<len(times)]
        if not candidates:
            continue
        best=min(candidates,key=lambda choice:abs(times[choice]-ticks[tick]))
        row=evidence[best]
        if row['segment']!=segment['index']:
            continue
        row['sequence_confidence']=max(row['sequence_confidence'] or 0.0,float(table[tick].max()))


def frame_kinds(expected,frames):
    return [(expected[index]['kind'] if index<len(expected) and expected[index] else 'unannotated')
            for index in range(frames)]


def _gate_row(curve,threshold=APP_HANDSHAPE_GATE):
    """The curve row for the gate the app actually runs today, if that threshold was swept."""
    return next((row for row in curve if row['threshold']==threshold),None)


def _rate(part,whole):
    return round(part/whole,6) if whole else None


def _per_minute(count,frames,fps):
    seconds=frames/max(fps,1e-6)
    return round(60*count/max(seconds,1e-6),3)


def held_curve(data,thresholds,*,detail_threshold=APP_HANDSHAPE_GATE):
    """Sweep the per-frame letter gate and report accuracy, coverage and false accepts per setting."""
    evidence=data['evidence']; emissions=data['emissions']; kinds=data['kinds']; fps=data['measured_fps']
    spans=[span for span in data['spans'] if span['kind']=='held']
    frames_by_kind={}
    for kind in kinds:
        frames_by_kind[kind]=frames_by_kind.get(kind,0)+1
    vectors={index:evidence[index]['handshape']['probabilities'] for index in range(len(evidence))
            if evidence[index]['handshape']}
    classes=len(next(iter(vectors.values()))) if vectors else 0
    curve=[]; detail=[]
    for threshold in thresholds:
        accepted=[row for row in emissions if row['path']=='handshape' and not row['retracted']
                  and row['confidence'] is not None and row['confidence']>=threshold]
        by_frame={}
        for row in accepted:
            by_frame.setdefault(row['frame'],[]).append(row)
        span_rows=[]
        for span in spans:
            inside=[row for frame in range(span['start'],span['end']) for row in by_frame.get(frame,())]
            winner=_modal([(row['char'],row['confidence']) for row in inside])
            pooled=np.zeros(classes); weight=0
            for frame in range(span['start'],span['end']):
                values=vectors.get(frame)
                if not values:
                    continue
                vector=np.asarray(values)
                if float(vector[int(np.argmax(vector))])<threshold:
                    continue
                pooled=pooled+vector; weight+=1
            top3=[_handshape_name(index) for index in np.argsort(pooled)[::-1][:3]] if weight else []
            span_rows.append(dict(letter=span['letter'],frames=[span['start'],span['end']],
                                  accepted_emissions=len(inside),
                                  first_accepted_frame=(min(row['frame'] for row in inside) if inside else None),
                                  emitted_letter=winner,correct=winner==span['letter'],
                                  mean_confidence=(round(float(np.mean([row['confidence'] for row in inside])),4)
                                                   if inside else None),
                                  top3=top3,top3_correct=bool(top3) and span['letter'] in top3))
        counted={}
        for row in accepted:
            kind=kinds[row['frame']]
            counted[kind]=counted.get(kind,0)+1
        correct=sum(1 for row in span_rows if row['correct'])
        emitted=sum(1 for row in span_rows if row['accepted_emissions'])
        scored=[row for row in span_rows if row['top3']]
        idle_frames=frames_by_kind.get('idle',0); activity_frames=frames_by_kind.get('activity',0)
        curve.append(dict(threshold=threshold,held_spans=len(span_rows),held_spans_correct=correct,
                          held_spans_with_emission=emitted,held_top1_accuracy=_rate(correct,len(span_rows)),
                          held_top1_accuracy_when_emitted=_rate(correct,emitted),
                          held_top3_spans=len(scored),held_top3_correct=sum(1 for row in scored if row['top3_correct']),
                          held_top3_accuracy=_rate(sum(1 for row in scored if row['top3_correct']),len(scored)),
                          frames=len(evidence),emission_frames=len(by_frame),
                          coverage=_rate(len(by_frame),len(evidence)),
                          emissions_per_minute=_per_minute(len(accepted),len(evidence),fps),
                          emission_frames_by_span_kind=dict(sorted(counted.items())),
                          idle_emissions=counted.get('idle',0),
                          idle_false_accept_rate=_rate(sum(1 for frame,kind in enumerate(kinds)
                                                           if kind=='idle' and frame in by_frame),idle_frames),
                          idle_emissions_per_minute=_per_minute(counted.get('idle',0),idle_frames,fps),
                          activity_emissions=counted.get('activity',0),
                          activity_false_accept_rate=_rate(sum(1 for frame,kind in enumerate(kinds)
                                                               if kind=='activity' and frame in by_frame),
                                                           activity_frames),
                          activity_emissions_per_minute=_per_minute(counted.get('activity',0),activity_frames,fps),
                          transition_emissions=counted.get('transition',0),
                          transition_emissions_per_minute=_per_minute(counted.get('transition',0),
                                                                      frames_by_kind.get('transition',0),fps),
                          ambiguous_hand_frames=sum(1 for row in evidence if row['hands']>1),
                          no_hand_frames=sum(1 for row in evidence if not row['hands'])))
        if threshold==detail_threshold:
            detail=span_rows
    return curve,detail


def _modal(rows):
    """Most frequent accepted letter; ties go to the letter with the higher total confidence."""
    counts={}; weight={}
    for letter,confidence in rows:
        counts[letter]=counts.get(letter,0)+1; weight[letter]=weight.get(letter,0.0)+confidence
    if not counts:
        return None
    return max(counts,key=lambda letter:(counts[letter],round(weight[letter],6),letter))


def sequence_curve(data,thresholds):
    """Sweep a proposed confidence gate over the characters the app put in its text field."""
    kinds=data['kinds']; fps=data['measured_fps']
    characters=[row for row in data['emissions'] if row['path']=='sequence']
    rows=[]
    for threshold in thresholds:
        accepted=[row for row in characters if row['confidence'] is not None and row['confidence']>=threshold]
        entry=dict(threshold=threshold,characters=len(accepted),
                   characters_per_minute=_per_minute(len(accepted),len(data['evidence']),fps),
                   unmeasurable_characters=sum(1 for row in characters if row['confidence'] is None),
                   retracted_characters=sum(1 for row in characters if row['retracted']))
        for kind in ('idle','activity','transition'):
            entry[f'{kind}_characters']=sum(1 for row in accepted if kinds[row['frame']]==kind)
        rows.append(entry)
    return rows


def confusion_matrix(data,threshold=APP_HANDSHAPE_GATE):
    """Expected class against accepted letter over the static handshape vocabulary."""
    from model.neural import HANDSHAPES
    counts={}
    for row in data['emissions']:
        if row['path']!='handshape' or row['retracted'] or row['confidence']<threshold:
            continue
        span=data['expected'][row['frame']]
        kind=span['kind'] if span else 'unannotated'
        if kind=='held':
            expected=span['letter']
        elif kind in ('idle','activity'):
            expected='nothing'
        else:
            continue
        if expected not in HANDSHAPES or row['char'] not in HANDSHAPES:
            continue
        key=(expected,row['char'])
        counts[key]=counts.get(key,0)+1
    cells=[dict(expected=expected,predicted=predicted,count=count) for (expected,predicted),count in
           sorted(counts.items(),key=lambda item:(-item[1],item[0]))]
    return dict(classes=list(HANDSHAPES),
                classes_note=('HANDSHAPES is nothing plus the 24 static letters. J and Z have no static class '
                              'in model/neural.py, so a 26-letter alphabet cannot be covered by this matrix.'),
                threshold=threshold,cells=cells,
                top_confusions=[cell for cell in cells if cell['expected']!=cell['predicted']][:10])


def motion_rows(data):
    """J and Z spans: did the streaming path emit that letter, and what did it show instead?"""
    rows=[]
    for span in data['spans']:
        if span['kind']!='motion':
            continue
        inside=[row for row in data['emissions'] if not row['retracted'] and span['start']<=row['frame']<span['end']]
        segment=next((item for item in data['segments'] if item['block'] and item['block'][0]<=span['start']<item['block'][1]),None)
        text=segment['text'] if segment else ''
        rows.append(dict(letter=span['letter'],frames=[span['start'],span['end']],
                         movement_score=round(movement_score(data['session'],span['start'],span['end']),3),
                         segment_text=text,detected=span['letter'] in text,
                         characters_in_span=sorted({row['char'] for row in inside}),
                         static_letter_emissions=sum(1 for row in inside if row['path']=='handshape'),
                         static_letters_at_app_gate=sum(1 for row in inside if row['path']=='handshape'
                                                        and row['confidence']>=APP_HANDSHAPE_GATE)))
    return rows


def spelling_rows(data):
    """Per annotated spelling run and per whole app segment, with the deletion/substitution/insertion split."""
    runs=[]; pairs=[]
    for span in data['spans']:
        if span['kind']!='spelling':
            continue
        observed=''.join(row['char'] for row in data['emissions'] if row['path']=='sequence'
                         and not row['retracted'] and span['start']<=row['frame']<span['end'])
        pairs.append((span['text'],observed))
        block=ErrorTally(); block.add(span['text'],observed)
        runs.append(dict(text=span['text'],frames=[span['start'],span['end']],observed=observed,**block.summary()))
    segments=[]
    for segment in data['segments']:
        if not segment['block']:
            continue
        start,end=segment['block']
        expected_text,_=block_expected(spans_for_block(dict(spans=data['spans']),start,end))
        if not expected_text:
            continue
        block=ErrorTally(); block.add(expected_text,segment['text'])
        segments.append(dict(frames=[start,end],expected=expected_text,observed=segment['text'],
                             observed_before_finish=segment['streaming'],status=segment['status'],
                             method=segment['method'],app_limit_reached=segment['limit_reached'],
                             **block.summary()))
    return dict(runs=runs,pairs=pairs,segments=segments,
                spelling_spans_outside_replay=sum(1 for span in data['spans'] if span['kind']=='spelling'
                                                  and span['start']>=data['replayed_frames']))


def score_session(session,annotation,data,thresholds):
    """One session's scored report block, with its own label provenance warning."""
    provenance=annotation.get('label_provenance')
    scripted=synthetic_recording(session)
    human=provenance in HUMAN_PROVENANCES and not scripted
    curve,span_rows=held_curve(data,thresholds)
    spelling=spelling_rows(data)
    data['spelling_pairs']=spelling.pop('pairs')
    runs=ErrorTally()
    for expected,observed in data['spelling_pairs']:
        runs.add(expected,observed)
    spelling['runs_summary']=runs.summary()
    segments=ErrorTally()
    for segment in spelling['segments']:
        segments.add(segment['expected'],segment['observed'])
    spelling['segments_summary']=segments.summary()
    motion=motion_rows(data)
    kinds=data['kinds']
    frames_by_kind={}
    for kind in kinds:
        frames_by_kind[kind]=frames_by_kind.get(kind,0)+1
    return dict(session_id=session['session_id'],signer_id=session['signer_id'],
                directory=str(session['directory']),recorded_at=session.get('recorded_at'),
                consent=session.get('consent'),video_file=session.get('video_file'),
                frames=session['frame_count'],replayed_frames=data['replayed_frames'],
                measured_fps=session['measured_fps'],width=session['width'],height=session['height'],
                label_provenance=provenance,confirmed=annotation.get('confirmed') is True,
                synthetic_recording=scripted,
                human_labels=human,numbers_from_suggested_labels=provenance==SUGGESTED_PROVENANCE,
                label_warning=(None if human else
                               ('NOT HUMAN LABELS: this recording is scripted (consent says synthetic), so '
                                'no person produced these frames and no number from it measures accuracy.'
                                if scripted else
                                f'NOT HUMAN LABELS: label_provenance is {provenance!r}. These numbers are not '
                                'a human accuracy measurement.')),
                annotated_frames_by_span_kind=frames_by_kind,
                held_letters=dict(app_gate=APP_HANDSHAPE_GATE,thresholds=list(thresholds),curve=curve,
                                  spans_at_app_gate=span_rows,
                                  confusion_at_app_gate=confusion_matrix(data)),
                motion_letters=dict(note=MOTION_NOTE,spans=motion,
                                    detected=sum(1 for row in motion if row['detected'])),
                natural_spelling=spelling,
                sequence_gate=dict(note=SEQUENCE_GATE_NOTE,thresholds=list(thresholds),
                                   curve=sequence_curve(data,thresholds)),
                segments=[dict(index=segment['index'],feed=[segment['feed_from'],segment['feed_to']],
                               block=list(segment['block']) if segment['block'] else None,text=segment['text'],
                               status=segment['status'],method=segment['method'],
                               app_limit_reached=segment['limit_reached']) for segment in data['segments']],
                app_limit_reached=any(segment['limit_reached'] for segment in data['segments']))


def aggregate(blocks,thresholds,spelling_pairs):
    """Fold every session's counts together, so the headline number cannot be an average of averages."""
    totals={}
    for block in blocks:
        for row in block['held_letters']['curve']:
            key=row['threshold']
            bucket=totals.setdefault(key,{name:0 for name in
                                          ('held_spans','held_spans_correct','held_spans_with_emission','held_top3_spans',
                                           'held_top3_correct','frames','emission_frames','idle_emissions','activity_emissions',
                                           'transition_emissions','ambiguous_hand_frames','no_hand_frames')})
            for name in bucket:
                bucket[name]+=row[name]
    curve=[]
    for threshold in thresholds:
        bucket=totals[threshold]
        curve.append(dict(threshold=threshold,**bucket,
                          held_top1_accuracy=_rate(bucket['held_spans_correct'],bucket['held_spans']),
                          held_top3_accuracy=_rate(bucket['held_top3_correct'],bucket['held_top3_spans']),
                          coverage=_rate(bucket['emission_frames'],bucket['frames'])))
    spelling=ErrorTally()
    for expected,observed in spelling_pairs:
        spelling.add(expected,observed)
    motion=[row for block in blocks for row in block['motion_letters']['spans']]
    confusions={}
    for block in blocks:
        for cell in block['held_letters']['confusion_at_app_gate']['top_confusions']:
            key=(cell['expected'],cell['predicted'])
            confusions[key]=confusions.get(key,0)+cell['count']
    return dict(held_curve=curve,spelling_runs_summary=spelling.summary(),
                motion=dict(spans=len(motion),detected=sum(1 for row in motion if row['detected']),
                            by_letter={letter:dict(spans=sum(1 for row in motion if row['letter']==letter),
                                                   detected=sum(1 for row in motion
                                                                 if row['letter']==letter and row['detected']))
                                       for letter in sorted({row['letter'] for row in motion})},
                            note=MOTION_NOTE),
                top_confusions_app_gate=[dict(expected=expected,predicted=predicted,count=count)
                                         for (expected,predicted),count in
                                         sorted(confusions.items(),key=lambda item:(-item[1],item[0]))][:10])


def label_fields(blocks):
    """The report's own statement about who stood behind the labels."""
    provenances={}
    for block in blocks:
        provenances[block['label_provenance']]=provenances.get(block['label_provenance'],0)+1
    human=bool(blocks) and all(block['human_labels'] for block in blocks)
    suggested=any(block['numbers_from_suggested_labels'] for block in blocks)
    scripted=sum(1 for block in blocks if block.get('synthetic_recording'))
    warning=None
    if suggested:
        warning=('UNCONFIRMED LABELS: at least one scored session carries the current model\'s own suggestions '
                 'as spans. Every number in this report is the model agreeing with itself and must not be '
                 'quoted as accuracy. Confirm the spans with scripts/label_sign_session.py --confirm.')
    elif blocks and not human:
        warning=('NOT HUMAN LABELS: these spans were not confirmed by a person, so this report calibrates the '
                 'harness and the rejection thresholds only, not app accuracy.')
    if scripted:
        warning=(f'{warning} ' if warning else '')+('SCRIPTED RECORDINGS: '
               f'{scripted} scored session(s) carry consent.synthetic, so their frames were scripted rather '
               'than signed; their held-letter numbers only prove the scorer works.')
    return dict(human_labels=human,label_provenances=provenances,synthetic_recordings=scripted,
                numbers_from_suggested_labels=suggested,label_warning=warning)


def collect_sessions(arguments):
    targets=[]
    for entry in arguments.sessions:
        if (entry/'session.json').is_file():
            targets.append(entry)
        elif entry.is_dir():
            targets.extend(sorted(child for child in entry.iterdir() if (child/'session.json').is_file()))
        else:
            raise SignSessionError(f'no session.json in {entry}')
    return targets


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session',dest='sessions',type=Path,action='append',required=True,
                        help='session directory, or a folder of session directories')
    parser.add_argument('--report',type=Path,default=ROOT/'data/held-letter-eval.report.json')
    parser.add_argument('--thresholds',type=float,nargs='+',default=list(DEFAULT_THRESHOLDS),
                        help='emission thresholds to sweep the app letter gate across')
    parser.add_argument('--limit-frames',type=int,help='score only the first N frames of each session')
    parser.add_argument('--segment-gap-ms',type=int,default=700,
                        help='presence gap that opens a new app segment; the app itself resets after 1000 ms')
    parser.add_argument('--handshape-model',type=Path)
    parser.add_argument('--sequence-model',type=Path)
    parser.add_argument('--skip-handshape',action='store_true')
    parser.add_argument('--skip-sequence',action='store_true')
    parser.add_argument('--via-worker',action='store_true',
                        help='replay the session video through scripts/recognition_worker.py over the '
                             'protocol electron/recognition.mjs uses, instead of replaying landmarks in-process')
    parser.add_argument('--python',type=Path,default=Path(sys.executable),help='python for --via-worker')
    parser.add_argument('--require-human',action='store_true',help='refuse sessions nobody confirmed')
    parser.add_argument('--draft',action='store_true',help='score annotation.draft.json (model suggestions)')
    parser.add_argument('--threads',type=int,default=4)
    arguments=parser.parse_args()
    started=time.perf_counter()
    runtime,logs=(None,{}) if arguments.via_worker else build_runtime(
        handshape_model=arguments.handshape_model,sequence_model=arguments.sequence_model,
        skip_handshape=arguments.skip_handshape,skip_sequence=arguments.skip_sequence,threads=arguments.threads)
    bridge=None
    blocks=[]; skipped=[]; pairs=[]
    targets=collect_sessions(arguments)
    try:
        for directory in targets:
            try:
                session=read_session(directory)
                annotation=read_annotation(directory,draft=arguments.draft)
            except (SignSessionError,OSError) as error:
                skipped.append(dict(directory=str(directory),reason=str(error))); continue
            errors,warnings=validate(session,annotation)
            provenance=annotation.get('label_provenance')
            if errors:
                skipped.append(dict(directory=str(directory),reason='annotation is contradictory',problems=errors))
                continue
            if arguments.require_human and (provenance not in HUMAN_PROVENANCES or synthetic_recording(session)):
                skipped.append(dict(directory=str(directory),
                                    reason=(f'label_provenance={provenance!r} is not a human label'
                                            if provenance not in HUMAN_PROVENANCES else
                                            'the recording is scripted (consent.synthetic), so no person signed '
                                            'these frames')))
                continue
            if provenance==SUGGESTED_PROVENANCE:
                print(f'WARNING {directory}: scoring unconfirmed model suggestions; the report marks them',flush=True)
            for warning in warnings:
                print(f'note {directory}: {warning}',flush=True)
            if arguments.via_worker:
                gap=video_gap(session)
                if gap:
                    skipped.append(dict(directory=str(directory),replay='recognition worker protocol',reason=gap))
                    print(f'SKIP {directory}: {gap}',flush=True)
                    continue
            try:
                if arguments.via_worker:
                    bridge=bridge or WorkerBridge(python=arguments.python,script=ROOT/'scripts/recognition_worker.py')
                    data=replay(session,annotation,bridge,logs,limit_frames=arguments.limit_frames,
                                segment_gap_ms=arguments.segment_gap_ms,via_worker=True,
                                frames=recorded_frames(session))
                else:
                    data=replay(session,annotation,runtime,logs,limit_frames=arguments.limit_frames,
                                segment_gap_ms=arguments.segment_gap_ms)
            except (SignSessionError,StopIteration) as error:
                skipped.append(dict(directory=str(directory),reason=f'replay failed: {error}')); continue
            data['session']=session
            data['spans']=annotation['spans']
            data['replayed_frames']=data['frames']
            data['measured_fps']=session['measured_fps']
            data['kinds']=frame_kinds(data['expected'],data['frames'])
            block=score_session(session,annotation,data,arguments.thresholds)
            block['annotation_warnings']=warnings
            blocks.append(block); pairs.extend(data['spelling_pairs'])
            gate=_gate_row(block['held_letters']['curve'])
            summary=dict(session=session['session_id'],frames=data['frames'],
                         held_top1_at_app_gate=(gate or {}).get('held_top1_accuracy'),
                         idle_false_accept_emissions=(gate or {}).get('idle_emissions'),
                         segments=len(data['segments']))
            print(json.dumps(summary),flush=True)
    finally:
        if bridge:
            bridge.close()
    report=dict(schema=REPORT_SCHEMA,generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                app_gate=APP_HANDSHAPE_GATE,thresholds=list(arguments.thresholds),
                replay=('recognition worker protocol over the session recording' if arguments.via_worker
                        else 'landmarks through model/live_neural.py in-process (CPU, no camera, no display)'),
                replay_limit_frames=arguments.limit_frames,segment_gap_ms=arguments.segment_gap_ms,
                models=dict(handshape=str(runtime.paths.get('handshape')) if runtime else None,
                            sequence=str(runtime.paths.get('sequence')) if runtime else None,
                            loaded=sorted(logs) if logs else [],
                            runtime_errors=dict(runtime.errors) if runtime else {},
                            handshape_gate=APP_HANDSHAPE_GATE,
                            note=('the static handshape head is the app per-frame letter emitter; the sequence '
                                  'head is the fingerspelling text emitter')),
                **label_fields(blocks),claim_limits=CLAIM_LIMITS,
                sessions=blocks,aggregate=aggregate(blocks,arguments.thresholds,pairs) if blocks else None,
                skipped_sessions=skipped,
                sessions_requested=len(targets),sessions_scored=len(blocks),
                seconds=round(time.perf_counter()-started,1))
    if not blocks:
        report['label_warning']=('No session was scored; every session was skipped, see skipped_sessions. An '
                                 'empty report is not evidence about the model.')
        report['human_labels']=False
    arguments.report.parent.mkdir(parents=True,exist_ok=True)
    arguments.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(arguments.report),human_labels=report['human_labels'],
                          scored=report['sessions_scored'],skipped=len(report['skipped_sessions']),
                          seconds=report['seconds'])),flush=True)
    return 0 if blocks else 1


if __name__=='__main__':
    sys.exit(main())
