"""Fail loudly on a malformed or contradictory sign-session annotation.

Exit 0 means the annotation is internally consistent and the recording carries a consent grant. Exit
1 lists every problem found, because a single contradictory span silently changes what an accuracy
number means. Exit 2 means the session itself could not be read. A draft annotation (model
suggestions nobody confirmed) is reported as unusable unless --draft says it is only a draft.

    uv run --extra neural python scripts/validate_sign_session.py data/sign-sessions/person01-day01
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sign_sessions import (HUMAN_PROVENANCES,SUGGESTED_PROVENANCE,SignSessionError,
                                   read_annotation,read_session,synthetic_recording,validate)


def check(directory,*,draft=False,require_human=False):
    """Return (session, annotation, errors, warnings) for one session directory.

    Errors mean the annotation contradicts itself or the recording, or that nobody human has stood
    behind the labels, so nothing scored from it may be quoted.
    """
    session=read_session(directory)
    annotation=read_annotation(directory,draft=draft)
    errors,warnings=validate(session,annotation)
    if annotation.get('label_provenance')==SUGGESTED_PROVENANCE:
        message='these spans are the current model suggesting its own output back; they are not human ' \
                'labels. Confirm them with scripts/label_sign_session.py --confirm before scoring.'
        (warnings if draft else errors).append(message)
    if draft:
        warnings.append('this is the draft file; scripts/label_sign_session.py --confirm writes the '
                        'confirmed annotation.json that scripts/eval_held_letters.py scores.')
    if require_human and (annotation.get('label_provenance') not in HUMAN_PROVENANCES
                          or synthetic_recording(session)):
        errors.append((f'label_provenance={annotation.get("label_provenance")!r} is not a human label.'
                       if annotation.get('label_provenance') not in HUMAN_PROVENANCES else
                       'the recording is scripted (consent.synthetic), so no person signed these frames; '
                       'confirming a script is not a signer.'))
    return session,annotation,errors,warnings


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories',type=Path,nargs='+',help='session directories or a folder holding them')
    parser.add_argument('--draft',action='store_true',help='validate annotation.draft.json instead')
    parser.add_argument('--require-human',action='store_true',help='fail unless a human confirmed the spans')
    args=parser.parse_args()
    targets=[]
    for directory in args.directories:
        if (directory/'session.json').is_file():
            targets.append(directory)
        elif directory.is_dir():
            targets.extend(sorted(child for child in directory.iterdir() if (child/'session.json').is_file()))
        else:
            print(f'FAIL {directory}: no session.json here'); return 2
    if not targets:
        print('FAIL no sessions found'); return 2
    failed=0
    for directory in targets:
        try:
            session,annotation,errors,warnings=check(directory,draft=args.draft,require_human=args.require_human)
        except (SignSessionError,OSError) as error:
            failed+=1; print(f'FAIL {directory}: {error}'); continue
        spans=annotation.get('spans',[])
        counts={}
        for span in spans:
            counts[span.get('kind')]=counts.get(span.get('kind'),0)+1
        print(f'{directory}: {session["frame_count"]} frames at {session["measured_fps"]} fps, '
              f'{len(spans)} spans {counts}, provenance={annotation.get("label_provenance")}, '
              f'confirmed={annotation.get("confirmed") is True}')
        for warning in warnings:
            print(f'  warning: {warning}')
        for error in errors:
            print(f'  ERROR: {error}')
        if errors:
            failed+=1
    if failed:
        print(f'{failed} of {len(targets)} sessions are not usable for held-letter evaluation.')
        return 1
    print(f'{len(targets)} sessions validate cleanly.')
    return 0


if __name__=='__main__':
    sys.exit(main())
