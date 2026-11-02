"""Versioned on-disk format for one app-domain recording session and its timed span annotation.

Layout of one session directory (default root data/sign-sessions/<session-id>):

    session.json     schema, session_id/signer_id, frame rate, frame size, consent, notes
    landmarks.npz    right/left T x 21 x 3 normalized tracks, present T x 2 masks, timestamps_ms
    video_file.mp4   optional recording the landmarks were extracted from (named in session.json)
    annotation.json  timed spans, confirmed only by a human
    annotation.draft.json  model suggestions awaiting review (scripts/label_sign_session.py)

Frames are half-open [start, end) indices into the recorded frame array. Capture timestamps keep
the original clock, so headless replay reproduces the app's 30 Hz resampling exactly. Landmark x/y
are normalized to the recorded frame size (the same convention MediaPipe hands report), z is kept
even though the current 91-dim contract drops it, so extraction can be redone later.

Span kinds: held (one deliberately held static letter), idle (no hand visible), activity (unrelated
hand movement, the rejection negative), motion (J or Z), spelling (a natural fingerspelling run with
the intended text) and transition (travel between letters). A transition may sit inside a spelling
run, otherwise spans must not overlap; held spans exclude J/Z because the static handshape head has
no such class (model/neural.py LETTERS), so those letters are annotated as motion spans instead.

Consent is recorded per session: the FSboard signers consented and were paid, and our own recordings
carry the same kind of record. Provenance separates who labelled the spans: only 'human' and
'human-reviewed-model-suggestions' count as human labels, 'model-suggestion' is an unreviewed draft,
'synthetic-fixture' means no person was recorded at all and 'fsboard-transcript' means the only label
is the published clip transcript, so span timing comes from hand presence rather than from anyone
reading the recording. A session whose consent record says synthetic can never yield a human label,
even after a person confirms its draft spans: confirming what a scripted track does is not a person
signing. This format calibrates the app on its own domain; numbers from it are not a public benchmark.
"""
import json
from pathlib import Path
import time

import numpy as np

SESSION_SCHEMA='asl-shop-sign-session/v1'
ANNOTATION_SCHEMA='asl-shop-sign-annotation/v1'
SPAN_KINDS=('held','idle','activity','motion','spelling','transition')
MOTION_LETTERS=frozenset('JZ')
SPAN_KEYS=frozenset(('kind','start','end','letter','text','suggestion_confidence','note'))
SUGGESTED_PROVENANCE='model-suggestion'
SYNTHETIC_PROVENANCE='synthetic-fixture'
TRANSCRIPT_PROVENANCE='fsboard-transcript'  # the published clip transcript is the only label here
KNOWN_PROVENANCES=frozenset(('human','human-reviewed-model-suggestions',SUGGESTED_PROVENANCE,
                             SYNTHETIC_PROVENANCE,TRANSCRIPT_PROVENANCE))
HUMAN_PROVENANCES=frozenset(('human','human-reviewed-model-suggestions'))
SESSION_FILE='session.json'
LANDMARKS_FILE='landmarks.npz'
ANNOTATION_FILE='annotation.json'
DRAFT_FILE='annotation.draft.json'
MOVEMENT_GATE=.12  # local-motion evidence the live J/Z gate requires in model/live_neural.py


class SignSessionError(ValueError):
    pass


def write_session(directory,*,session_id,signer_id,right,left,present,timestamps_ms,width,height,fps,
                  consent,video_file=None,notes='',recorded_at=None):
    """Write one session bundle; every identity, consent and clock field is required."""
    right=np.asarray(right,dtype=np.float32); left=np.asarray(left,dtype=np.float32)
    present=np.asarray(present); times=np.asarray(timestamps_ms,dtype=np.float64)
    if right.shape!=left.shape or right.ndim!=3 or right.shape[1:]!=(21,3):
        raise SignSessionError('Landmark tracks must be T x 21 x 3 for each hand slot.')
    if present.shape!=(len(right),2) or not np.isin(present,(0,1)).all():
        raise SignSessionError('Presence masks must be T x 2 with 0/1 values.')
    if times.shape!=(len(right),) or not np.isfinite(times).all() or len(times)<2 or np.any(np.diff(times)<=0):
        raise SignSessionError('Capture timestamps must be finite and strictly increasing.')
    if not np.isfinite(right).all() or not np.isfinite(left).all():
        raise SignSessionError('Landmark tracks must be finite.')
    if not session_id or not signer_id or not fps or fps<=0 or width<=0 or height<=0:
        raise SignSessionError('session_id, signer_id, a positive fps and frame size are required.')
    if not isinstance(consent,dict) or consent.get('granted') is not True or not consent.get('note'):
        raise SignSessionError('A session needs a consent record with granted=true and a note.')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/LANDMARKS_FILE,right=right,left=left,
                        present=present.astype(np.uint8),timestamps_ms=times)
    measured=len(times)/max(times[-1]-times[0],1e-6)*1000
    entry=dict(schema=SESSION_SCHEMA,session_id=str(session_id),signer_id=str(signer_id),
               recorded_at=recorded_at or _now(),fps=float(fps),measured_fps=round(float(measured),3),
               width=int(width),height=int(height),landmarks_file=LANDMARKS_FILE,video_file=video_file,
               consent=consent,notes=notes,frame_count=int(len(right)))
    (directory/SESSION_FILE).write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8')
    return entry


def read_session(directory):
    """Load one session bundle and reject structurally broken payloads loudly."""
    directory=Path(directory)
    meta=json.loads((directory/SESSION_FILE).read_text(encoding='utf-8'))
    if meta.get('schema')!=SESSION_SCHEMA:
        raise SignSessionError(f'Unsupported session schema {meta.get("schema")!r}; expected {SESSION_SCHEMA}.')
    with np.load(directory/(meta.get('landmarks_file') or LANDMARKS_FILE)) as data:
        session={'right':data['right'],'left':data['left'],'present':data['present'].astype(bool),
                 'timestamps_ms':data['timestamps_ms']}
    session.update(meta)
    n=len(session['timestamps_ms'])
    problems=[]
    if session['right'].shape!=(n,21,3) or session['left'].shape!=(n,21,3):
        problems.append('Landmark track length or shape does not match the frame count.')
    if session['present'].shape!=(n,2):
        problems.append('Presence masks do not match the frame count.')
    if not np.isfinite(session['timestamps_ms']).all() or n<2 or np.any(np.diff(session['timestamps_ms'])<=0):
        problems.append('Capture timestamps must be finite and strictly increase.')
    if not np.isfinite(session['right']).all() or not np.isfinite(session['left']).all():
        problems.append('Landmark tracks contain non-finite values.')
    if not isinstance(session.get('consent'),dict) or session['consent'].get('granted') is not True:
        problems.append('Session has no consent grant; consented recordings only may be scored.')
    if problems:
        raise SignSessionError(' '.join(problems))
    session['directory']=directory
    return session


def read_annotation(directory,*,draft=False):
    directory=Path(directory)
    annotation=json.loads((directory/(DRAFT_FILE if draft else ANNOTATION_FILE)).read_text(encoding='utf-8'))
    if annotation.get('schema')!=ANNOTATION_SCHEMA:
        raise SignSessionError(f'Unsupported annotation schema {annotation.get("schema")!r}; expected {ANNOTATION_SCHEMA}.')
    return annotation


def write_annotation(directory,annotation,*,draft=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    annotation.setdefault('schema',ANNOTATION_SCHEMA)
    (directory/(DRAFT_FILE if draft else ANNOTATION_FILE)).write_text(
        json.dumps(annotation,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return annotation


def synthetic_recording(session):
    """True when the recording itself is scripted, whatever the annotation claims about its spans."""
    consent=session.get('consent')
    return isinstance(consent,dict) and consent.get('synthetic') is True


def validate(session,annotation):
    """Return (errors, warnings). Errors mean the annotation contradicts itself or the recording."""
    from model.neural import HANDSHAPES,VOCAB
    errors=[]; warnings=[]
    spans=annotation.get('spans')
    if not isinstance(spans,list):
        return ([f'annotation.spans must be a list, got {type(spans).__name__}.'],warnings)
    n=len(session['timestamps_ms'])
    for key in ('session_id','signer_id'):
        if annotation.get(key) and annotation[key]!=session[key]:
            errors.append(f'annotation.{key}={annotation[key]!r} disagrees with the session ({session[key]!r}).')
    provenance=annotation.get('label_provenance')
    if provenance not in KNOWN_PROVENANCES:
        errors.append(f'Unknown label_provenance {provenance!r}; expected one of {sorted(KNOWN_PROVENANCES)}.')
    elif provenance in HUMAN_PROVENANCES and annotation.get('confirmed') is not True:
        errors.append('human-labelled annotations must carry confirmed=true.')
    elif provenance==SUGGESTED_PROVENANCE and annotation.get('confirmed'):
        errors.append('model-suggestion annotations must stay confirmed=false until a human reviews them.')
    elif provenance==SUGGESTED_PROVENANCE:
        warnings.append('DRAFT: these spans are model suggestions, not labels. Confirm them with '
                        'scripts/label_sign_session.py --confirm before quoting any accuracy number.')
    if provenance==SYNTHETIC_PROVENANCE:
        warnings.append('synthetic fixture: no person was recorded, so scores calibrate the harness only.')
    if provenance==TRANSCRIPT_PROVENANCE:
        warnings.append('transcript-derived spans: the published clip transcript is the label, and span timing '
                        'comes from hand presence, so no person read this recording. Held-letter and rejection '
                        'numbers from such a session are a plumbing check, never signer accuracy.')
    if synthetic_recording(session):
        warnings.append('the consent record says this recording is synthetic (scripted landmarks, no person).')
        if provenance in HUMAN_PROVENANCES:
            warnings.append(f'label_provenance={provenance!r} on a synthetic recording: a person confirming what '
                            'a script did is not a signer, so no number scored from this may be quoted as '
                            'human accuracy.')
    if not spans:
        warnings.append('annotation has no spans.')
    letters=set(HANDSHAPES[1:])
    for index,span in enumerate(spans):
        where=f'span {index} ({span.get("kind")})'
        for key in ('kind','start','end'):
            if key not in span:
                errors.append(f'{where}: missing {key}.')
        if span.get('kind') not in SPAN_KINDS:
            errors.append(f'span {index}: unknown kind {span.get("kind")!r}; expected one of {SPAN_KINDS}.')
            continue
        start,end=span.get('start'),span.get('end')
        if isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or not isinstance(end,int):
            errors.append(f'{where}: start/end must be integer frame indices.'); continue
        if not 0<=start<end<=n:
            errors.append(f'{where}: frames [{start},{end}) outside the recorded range [0,{n}).'); continue
        kind=span['kind']; letter,text=span.get('letter'),span.get('text')
        if kind in ('held','motion'):
            if letter is None:
                errors.append(f'{where}: needs a letter.')
            elif kind=='held' and letter not in letters:
                hint=' (J and Z are movement letters; annotate them as kind=motion)' if letter in MOTION_LETTERS else ''
                errors.append(f'{where}: letter {letter!r} is not a static handshape letter{hint}.')
            elif kind=='motion' and letter not in MOTION_LETTERS:
                errors.append(f'{where}: motion spans must hold J or Z, got {letter!r}.')
            if text:
                errors.append(f'{where}: {kind} spans must not carry text.')
            if not session['present'][start:end].any(axis=1).all():
                errors.append(f'{where}: {kind} span contains frames without any hand.')
            if kind=='motion' and movement_score(session,start,end)<MOVEMENT_GATE:
                errors.append(f'{where}: motion span shows no landmark movement above the app gate '
                              f'({MOVEMENT_GATE}); that reads as a held letter, not J or Z.')
        elif kind in ('idle','activity','transition'):
            if letter is not None or text:
                errors.append(f'{where}: {kind} spans must not carry letter or text.')
            if kind=='idle' and session['present'][start:end].any():
                errors.append(f'{where}: idle span has a visible hand; annotate unrelated hand use as '
                              'kind=activity so rejection calibration keeps the two apart.')
        elif kind=='spelling':
            if not text:
                errors.append(f'{where}: spelling spans need the intended text.')
            elif any(char not in VOCAB[1:] for char in text):
                errors.append(f'{where}: spelling text contains characters outside the app vocabulary.')
            elif session['present'][start:end].any(axis=1).mean()<.5:
                errors.append(f'{where}: spelling run is missing the hand for over half its frames.')
        for key in span:
            if key not in SPAN_KEYS:
                warnings.append(f'{where}: unknown key {key!r} ignored.')
    for i in range(len(spans)):
        for j in range(i+1,len(spans)):
            a,b=spans[i],spans[j]
            if not isinstance(a.get('start'),int) or not isinstance(b.get('start'),int):
                continue
            if not (a['start']<b['end'] and b['start']<a['end']):
                continue
            kinds={a['kind'],b['kind']}
            if kinds<= {'transition'}:
                continue
            if kinds=={'transition','spelling'}:
                travel=a if a['kind']=='transition' else b
                run=b if travel is a else a
                if run['start']<=travel['start'] and travel['end']<=run['end']:
                    continue
            errors.append(f'contradictory spans {i} and {j}: {a["kind"]} and {b["kind"]} overlap on frames '
                          f'[{max(a["start"],b["start"])},{min(a["end"],b["end"])}).')
    return errors,warnings


def movement_score(session,start,end):
    """Landmark movement ratio for one span, mirroring the live J/Z motion evidence."""
    best=0.0
    dims=np.array((session['width'],session['height']),dtype=np.float32)
    for stream,flag in ((session['right'],0),(session['left'],1)):
        rows=session['present'][start:end,flag]
        if rows.sum()<3:
            continue
        pixels=stream[start:end][rows][:,:,:2]*dims
        scale=np.median(np.linalg.norm(pixels[:,9]-pixels[:,0],axis=-1))+1e-6
        best=max(best,float(np.max(np.ptp(pixels,axis=0))/scale))
    return best


def presence_blocks(session,*,gap_ms=700,max_block_seconds=30):
    """Visible-hand runs: split at frames with no hand, at long detection gaps and at the app limit.

    This uses only the signal the app itself sees, never the human idle spans, so scoring cannot
    leak the annotation into how the recording is cut.
    """
    present=session['present'].any(axis=1); times=session['timestamps_ms']
    blocks=[]; start=None
    for index in range(len(present)+1):
        cut=index==len(present) or not present[index]
        if not cut and start is None:
            start=index
        if cut and start is not None:
            piece=[start]
            for inner in range(start+1,index):
                if times[inner]-times[inner-1]>gap_ms or times[inner]-times[piece[0]]>=max_block_seconds*1000:
                    piece.append(inner)
            piece.append(index)
            blocks.extend((piece[k],piece[k+1]) for k in range(len(piece)-1))
            start=None
    return blocks


def hands_at(session,index):
    """One recorded frame as the landmark records model/live_neural.analyze expects them."""
    from types import SimpleNamespace
    hands=[]
    for slot,side in enumerate(('Right','Left')):
        if not session['present'][index,slot]:
            continue
        stream=session['right'] if slot==0 else session['left']
        hands.append({'hand':side,'landmarks':[SimpleNamespace(x=float(point[0]),y=float(point[1]),z=float(point[2]))
                                                for point in stream[index]]})
    return hands


def spans_for_block(annotation,start,end):
    """Spans assigned to the block holding their first frame (the app starts a segment there)."""
    return sorted((span for span in annotation['spans'] if start<=span['start']<end),
                  key=lambda span:(span['start'],span['end']))


def block_expected(spans):
    """Expected app characters for one segment, in span order, with each character's owning span."""
    text=''; entries=[]
    for span in spans:
        if span['kind'] in ('held','motion'):
            chars=span.get('letter') or ''
        elif span['kind']=='spelling':
            chars=span.get('text') or ''
        else:
            chars=''
        for char in chars:
            text+=char; entries.append((char,span))
    return text,entries


def frame_span_map(annotation,n_frames):
    """For each frame the innermost non-transition span covering it, falling back to travel spans."""
    primary=[None]*n_frames; travel=[None]*n_frames
    for span in annotation['spans']:
        target=travel if span['kind']=='transition' else primary
        for index in range(span['start'],span['end']):
            if target[index] is None or (span['end']-span['start'])<(target[index]['end']-target[index]['start']):
                target[index]=span
    return [primary[index] or travel[index] for index in range(n_frames)]


# Each transition names the letter the hand travels into, which is the shape it already holds.
FIXTURE_PLAN=(('idle',20,''),('held',24,'A'),('transition',6,'B'),('held',24,'B'),('transition',6,'K'),
              ('held',24,'K'),('idle',20,''),('activity',24,''),('idle',12,''),('motion',30,'J'),
              ('idle',12,''),('spelling',18,'HUG'),('idle',12,''),('motion',30,'Z'),('idle',20,''))
# A second scripted session: more held letters, two pairs people actually confuse (N/M, P/R), two held
# spans back to back with no travel between them, a longer spelling run and a longer activity block.
HARD_FIXTURE_PLAN=(('idle',20,''),('held',24,'M'),('transition',6,'N'),('held',24,'N'),('transition',6,'P'),
                   ('held',24,'P'),('transition',6,'R'),('held',24,'R'),('idle',15,''),('held',24,'A'),
                   ('held',24,'B'),('idle',20,''),('activity',30,''),('idle',12,''),('motion',30,'J'),
                   ('idle',12,''),('spelling',15,'HELLO'),('idle',15,''),('motion',30,'Z'),('idle',15,''))
FIXTURE_PLANS=dict(easy=FIXTURE_PLAN,hard=HARD_FIXTURE_PLAN)
FIXTURE_TRAVEL_FRAMES=6


def fixture_pose(letter):
    """Deterministic 21-point normalized hand for one scripted letter ('' is the neutral hand).

    Only used so the harness has scripted landmark tracks with known answers. These poses are not
    anatomically valid hands and must never be read as a model accuracy measurement.
    """
    local=np.random.default_rng(int.from_bytes((letter or 'nothing').encode('utf-8'),'little'))
    curl=local.uniform(0,1,5)  # per-letter finger extension pattern; this is what separates letters
    points=np.zeros((21,2),dtype=np.float32)
    points[0]=(.5,.66); points[5]=(.36,.55); points[9]=(.47,.52); points[13]=(.58,.53); points[17]=(.67,.57)
    tips=np.array(((-.16,-.04),(.02,-.17),(.15,-.17),(.26,-.12),(.33,-.01)),dtype=np.float32)
    if letter=='J':  # pointing index finger, the shape the fixture then hooks sideways
        curl=np.array((0,1,0,0,0),dtype=np.float32)
    elif letter=='Z':  # flat hand the fixture slides back and forth
        curl=np.ones(5,dtype=np.float32)
    spread=(4,8,12,16,20); bases=(1,5,9,13,17)
    for finger in range(5):
        base=points[bases[finger]]
        extended=base+tips[finger]*1.55
        for step in (3,2,1):  # chain joints between the base and the tip
            weight=curl[finger]*(4-step)/4
            points[spread[finger]-step]=base*(1-weight)+extended*weight
        weight=curl[finger]
        points[spread[finger]]=base*(1-weight)+extended*weight
    return points


def _fixture_segments(plan):
    segments=[]
    for kind,frames,label in plan:
        if kind!='spelling':
            segments.append(dict(kind=kind,frames=frames,letter=label if kind in ('held','motion','transition') else None))
            continue
        for position,char in enumerate(label):
            if position:
                segments.append(dict(kind='transition',frames=FIXTURE_TRAVEL_FRAMES,letter=char))
            segments.append(dict(kind='spelling-letter',frames=frames,letter=char))
    return segments


def make_synthetic_session(directory,*,session_id='synthetic-fixture-1',fps=30.0,seed=12,width=640,height=480,
                           noise=.0015,plan=FIXTURE_PLAN):
    """Scripted, fully annotated session: held letters, idle and activity negatives, J/Z motion,
    transitions and one natural spelling run. The capture clock is exactly on the app's 30 Hz grid,
    so replay generates no interpolated ticks. Nothing here was recorded and no number from it is a
    model accuracy claim; it exists so the harness has known answers."""
    rng=np.random.default_rng(seed)
    segments=_fixture_segments(plan)
    n_frames=sum(segment['frames'] for segment in segments)
    right=np.zeros((n_frames,21,3),dtype=np.float32); left=np.zeros_like(right)
    present=np.zeros((n_frames,2),dtype=np.uint8)
    spans=[]; cursor=0; run_start=None; run_text=''
    for position,segment in enumerate(segments):
        kind,frames,letter=segment['kind'],segment['frames'],segment['letter']
        if kind in ('held','motion','spelling-letter','transition','activity'):
            base=fixture_pose('' if kind=='activity' else letter)  # travel already holds the next shape
            walk=np.zeros(2,dtype=np.float32)
            travel=.12 if kind=='transition' else 0.0
            for index in range(cursor,cursor+frames):
                progress=(index-cursor)/frames
                offset=np.zeros(2,dtype=np.float32)
                if kind=='activity':  # unrelated hand activity: a shifting, unlettered pose
                    walk=np.clip(walk+rng.normal(0,.02,2),-.18,.18); offset=offset+walk
                if kind=='motion':  # zigzag travel above the app's local-motion gate
                    offset=offset+np.array([.15*np.sin(2*np.pi*3*progress),.03*progress],dtype=np.float32)
                if travel:
                    offset=offset+np.array([travel,-travel],dtype=np.float32)*progress
                points=base+offset+rng.normal(0,noise,(21,2)).astype(np.float32)
                right[index,:,:2]=points
                right[index,:,2]=np.mean(points,axis=1)*.4  # z is kept for future extraction, unused today
                present[index,0]=1
        if kind=='spelling-letter':
            if run_start is None:
                run_start=cursor
            run_text+=letter
            following=segments[position+1]['kind'] if position+1<len(segments) else 'idle'
            if following!='transition':
                spans.append(dict(kind='spelling',start=run_start,end=cursor+frames,text=run_text))
                run_start=None; run_text=''
        else:
            span=dict(kind=kind,start=cursor,end=cursor+frames)
            if kind in ('held','motion'):
                span['letter']=letter
            spans.append(span)
        cursor+=frames
    timestamps_ms=np.arange(n_frames,dtype=np.float64)*1000/fps
    directory=Path(directory)
    write_session(directory,session_id=session_id,signer_id=f'synthetic-signer-{session_id}',
                  right=right,left=left,present=present,timestamps_ms=timestamps_ms,width=width,height=height,
                  fps=fps,consent={'granted':True,'recorded_at':time.strftime('%Y-%m-%d'),'synthetic':True,
                                   'note':'Synthetic fixture: scripted landmark tracks, no person was '
                                         'recorded and nothing was paid.'},
                  notes='Scripted fixture for held-letter harness self-checks; see scripts/sign_sessions.py '
                        'make_synthetic_session.')
    write_annotation(directory,dict(schema=ANNOTATION_SCHEMA,session_id=session_id,
                                    signer_id=f'synthetic-signer-{session_id}',confirmed=True,
                                    label_provenance=SYNTHETIC_PROVENANCE,labelled_at=time.strftime('%Y-%m-%d'),
                                    labels_by='fixture generator (not a human)',spans=spans))
    return dict(directory=directory,frame_count=n_frames,
                held=[span['letter'] for span in spans if span['kind']=='held'],
                motion=[span['letter'] for span in spans if span['kind']=='motion'],
                spelling=[span['text'] for span in spans if span['kind']=='spelling'],
                spans=len(spans))


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')
