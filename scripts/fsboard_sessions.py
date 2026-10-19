"""Build sign-session bundles from real FSboard test clips, where the clip transcript is the only label.

FSboard's signers consented and were paid, and every clip carries the word they were asked to spell, so a
clip is a known answer for checking the replay plumbing end to end: the visible run is one spelling span
and every frame without a hand detection is idle. Nobody segmented these recordings into spans, and no
one re-checked the timing, so label_provenance is 'fsboard-transcript'. The scorer treats that as not a
human label, so these sessions belong in their own report file and can never raise a human-label number.

The landmark tracks come from the app's own packed FSboard rows (asl-shop-xy-wrist-v3), memory-mapped so
only the chosen clips are touched instead of the whole pack. The packed contract keeps 21 wrist-relative
shape coordinates per hand and no z, so reconstruct_landmarks() inverts shape_features() and writes z as
0. Everything downstream rescales by wrist and hand size, and the round trip is exact for what the live
path reads: scripts/eval_held_letters.py replaying these bundles derives the same 91-column features the
pack stored, which tests/test_held_letter_eval.py checks against real packed rows.

    uv run --extra neural python scripts/fsboard_sessions.py --limit 4
    uv run --extra neural python scripts/eval_held_letters.py --session data/sign-sessions/fsboard-test-08db \\
        --report data/held-letter-fsboard.report.json
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.neural import VOCAB
from scripts.sign_sessions import (ANNOTATION_SCHEMA,TRANSCRIPT_PROVENANCE,SignSessionError,read_session,
                                   write_annotation,write_session)

DEFAULT_PACK=ROOT/'data/openfs-packed-v3'
LETTERS=''.join(VOCAB[1:27])  # A-Z only: digits and punctuation in transcripts are not spelling practice
FSBOARD_CONSENT_NOTE=('FSboard released this corpus for research use; its signers consented to the public '
                      'release and were paid. No person segmented this clip into spans, and nobody verified '
                      'when each letter was signed: the only label is the word the signer was shown.')


def pack_generation(pack=DEFAULT_PACK):
    """Where the current pack generation keeps its arrays, straight from the pack manifest."""
    manifest=json.loads((pack/'dataset.json').read_text(encoding='utf-8'))
    generation=pack/manifest['generation']
    if not (generation/'test_clips.json').is_file():
        raise SignSessionError(f'pack generation {generation} has no test_clips.json')
    return manifest,generation


def clips_of(generation,split):
    return json.loads((generation/f'{split}_clips.json').read_text(encoding='utf-8'))


def visible_run(frames):
    """Frames where either hand slot has a detection, as the app would see them."""
    return (frames[:,44]>0)|(frames[:,89]>0)


def signed_run(frames):
    """First and last frame plus one after it where either hand slot has a detection.

    FSboard clips keep their leading and trailing detection drop-outs, which the app would see as no hand
    at all, so they become idle spans rather than part of the word.
    """
    visible=visible_run(frames)
    if not visible.any():
        return None
    return int(np.argmax(visible)),int(len(visible)-np.argmax(visible[::-1]))


def candidate_clips(clips,frames,*,limit,min_letters,max_letters,max_frames,min_frames_per_letter,min_visible,
                    seed):
    """Deterministically pick short single-word letter clips whose hand stays up through the word.

    Drop-outs inside the signed run are allowed but capped by min_visible: the transcript says what was
    spelled, not which letters sit on either side of a gap, so a clip that loses the hand through half the
    word cannot be scored against its transcript.
    """
    letters=set(LETTERS)
    pool=[clip for clip in clips
          if min_letters<=len(clip['text'])<=max_letters and set(clip['text'])<=letters
          and max_frames>=clip['end']-clip['start']>=len(clip['text'])*min_frames_per_letter]
    order=np.random.default_rng(seed).permutation(len(pool))
    chosen=[]
    for position in order:
        if len(chosen)>=limit:
            break
        clip=pool[int(position)]
        rows=np.asarray(frames[clip['start']:clip['end']])
        run=signed_run(rows)
        if run and visible_run(rows[run[0]:run[1]]).mean()>=min_visible:
            chosen.append(clip)
    return chosen


def reconstruct_landmarks(rows,*,width=640,height=480):
    """Packed 91-column rows back into normalized 21 x 3 tracks and presence masks.

    The pack stores wrist-relative, hand-size-normalized shape coordinates (mirrored for a Left hand) plus
    the wrist position in hand-size units. Adding the wrist back gives points in hand-size units, and
    dividing by one isotropic factor keeps that relationship: the live path rescales by wrist and hand size
    again, so shape_features() and the sequence wrist columns come back identical. The factor is chosen so
    the hand fits inside the frame the session claims.
    """
    rows=np.asarray(rows,dtype=np.float32)
    present=np.zeros((rows.shape[0],2),dtype=np.uint8); built=[]; extents=[]
    for slot,side in enumerate(('Right','Left')):
        base=slot*45
        shape=rows[:,base:base+42].reshape(-1,21,2).copy()
        wrist=rows[:,base+42:base+44].copy()
        detected=rows[:,base+44]>0
        if side=='Left':
            shape[...,0]*=-1  # shape_features() mirrors Left hands again when it reads them
        points=shape+wrist[:,None,:]  # wrist-relative unit coordinates back into one frame
        points[~detected]=0.0
        built.append(points); present[:,slot]=detected
        if detected.any():
            seen=points[detected]
            extents.append(max(float(np.ptp(seen[...,0])),float(np.ptp(seen[...,1]))))
    span=max(extents) if extents else 1.0
    scale=max(span/max(width,height),1e-6)*1.6  # keep the reconstructed hand inside the nominal frame
    right,left=(np.dstack((points/scale,np.zeros((len(rows),21,1),dtype=np.float32))) for points in built)
    return right,left,present


def spans_of(frames,*,text):
    """Idle spans for the hand-free edges of the clip, one spelling span over the signed run."""
    run=signed_run(frames)
    if not run:
        return [dict(kind='idle',start=0,end=len(frames),note='no hand detection anywhere in the clip')]
    first,last=run
    spans=[dict(kind='spelling',start=first,end=last,text=text,
                note='the published clip transcript; nobody marked where each letter starts')]
    if first:
        spans.append(dict(kind='idle',start=0,end=first,note='no hand detection before the word'))
    if last<len(frames):
        spans.append(dict(kind='idle',start=last,end=len(frames),note='no hand detection after the word'))
    return sorted(spans,key=lambda span:(span['start'],span['end']))


def recorded_clock(clip):
    """The FSboard source recording time carried in the clip name, e.g. ..._20220929_164127_..."""
    stamp=re.search(r'_(\d{8})_(\d{6})',f'_{clip["session"]}')
    if not stamp:
        return None
    day,hour=stamp.groups()
    return f'{day[:4]}-{day[4:6]}-{day[6:]}T{hour[:2]}:{hour[2:4]}:{hour[4:]}'


def write_probe_session(directory,clip,frames,*,pack,generation,split,fps=30.0,width=640,height=480):
    """One FSboard test clip as a session bundle plus its transcript-derived annotation."""
    rows=np.asarray(frames[clip['start']:clip['end']])
    right,left,present=reconstruct_landmarks(rows,width=width,height=height)
    times=np.arange(len(rows),dtype=np.float64)*1000/fps
    directory=Path(directory)
    entry=write_session(directory,session_id=clip['name'],signer_id=clip['signer'],right=right,left=left,
                        present=present,timestamps_ms=times,width=width,height=height,fps=fps,
                        recorded_at=recorded_clock(clip),
                        consent={'granted':True,'note':FSBOARD_CONSENT_NOTE},
                        notes=f'FSboard {split} clip {clip["name"]} from pack {generation.name}; source video '
                              f'{clip["session"]}; packed rows {clip["start"]}-{clip["end"]} reconstructed from '
                              '91-column features (no z, nominal capture clock)')
    write_annotation(directory,dict(schema=ANNOTATION_SCHEMA,session_id=clip['name'],signer_id=clip['signer'],
                                    confirmed=True,label_provenance=TRANSCRIPT_PROVENANCE,
                                    labels_by='FSboard clip transcript (not a span annotation)',
                                    label_source=str(generation/f'{split}_clips.json'),
                                    spans=spans_of(rows,text=clip['text'])))
    return entry


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pack',type=Path,default=DEFAULT_PACK,
                        help='packed FSboard dataset directory holding dataset.json')
    parser.add_argument('--split',default='test',choices=('test','validation'),
                        help='which pack split to take clips from; the app is tuned on train-like data only')
    parser.add_argument('--out',type=Path,default=ROOT/'data/sign-sessions',help='where to write the bundles')
    parser.add_argument('--limit',type=int,default=4,help='how many clips to write (keep this small)')
    parser.add_argument('--min-letters',type=int,default=4)
    parser.add_argument('--max-letters',type=int,default=9)
    parser.add_argument('--max-frames',type=int,default=300,help='clip length cap so the replay stays cheap')
    parser.add_argument('--min-frames-per-letter',type=int,default=10,
                        help='a clip shorter than this per signed letter is too fast to replay honestly')
    parser.add_argument('--min-visible',type=float,default=.8,
                        help='smallest share of the signed run that keeps a hand detection')
    parser.add_argument('--seed',type=int,default=7,help='clip selection is deterministic per seed')
    arguments=parser.parse_args()
    if not 1<=arguments.limit<=25:
        parser.error('--limit stays between 1 and 25; this probe is meant to be a handful of clips')
    manifest,generation=pack_generation(arguments.pack)
    clips=clips_of(generation,arguments.split)
    frames=np.lib.format.open_memmap(generation/f'{arguments.split}_frames.npy',mode='r')
    chosen=candidate_clips(clips,frames,limit=arguments.limit,min_letters=arguments.min_letters,
                           max_letters=arguments.max_letters,max_frames=arguments.max_frames,
                           min_frames_per_letter=arguments.min_frames_per_letter,
                           min_visible=arguments.min_visible,seed=arguments.seed)
    if not chosen:
        print(f'no {arguments.split} clip in {generation} matched those bounds'); return 1
    print(f'{arguments.pack.name} ({manifest["contract"]}) generation {generation.name}: '
          f'{len(clips)} {arguments.split} clips, {len(chosen)} chosen')
    for clip in chosen:
        directory=arguments.out/f"fsboard-{arguments.split}-{clip['name']}"
        entry=write_probe_session(directory,clip,frames,pack=arguments.pack,generation=generation,
                                  split=arguments.split)
        print(f'{directory}: clip {clip["name"]} signer {clip["signer"]} text {clip["text"]!r} '
              f'{entry["frame_count"]} frames')
    print(f'{len(chosen)} FSboard {arguments.split} clips written under {arguments.out}. They are not human '
          'span labels: score them into their own report with scripts/eval_held_letters.py --report '
          'data/held-letter-fsboard.report.json, never into the human-label report.')
    return 0


if __name__=='__main__':
    sys.exit(main())
