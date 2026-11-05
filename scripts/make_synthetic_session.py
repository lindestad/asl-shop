"""Write a scripted sign session with its annotation, so the harness has answers it can be proved against.

Nothing here was recorded: scripts/sign_sessions.py generates the landmark tracks from a span plan, so
the held letters, idle gaps, activity block, J/Z motion spans and spelling run all have a known answer.
Scores from these sessions show the scorer and the replay path work; they say nothing about how well the
app reads a person, and the session's consent record says synthetic so the scorer refuses to call any
number from it a human label.

    uv run --extra neural python scripts/make_synthetic_session.py --out data/sign-sessions/fixture-easy
    uv run --extra neural python scripts/make_synthetic_session.py --plan hard --out data/sign-sessions/fixture-hard
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sign_sessions import (FIXTURE_PLANS,SignSessionError,make_synthetic_session,read_annotation,
                                   read_session,validate)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',type=Path,required=True,help='session directory to write')
    parser.add_argument('--plan',choices=sorted(FIXTURE_PLANS),default='easy',
                        help='scripted span plan: easy is the small fixture, hard adds confusable pairs')
    parser.add_argument('--session-id',help='defaults to fixture-<plan>')
    parser.add_argument('--seed',type=int,default=12)
    parser.add_argument('--noise',type=float,default=.0015,help='per-frame landmark jitter')
    parser.add_argument('--fps',type=float,default=30.0,help='capture clock grid; the app resamples to 20 Hz')
    arguments=parser.parse_args()
    try:
        info=make_synthetic_session(arguments.out,session_id=arguments.session_id or f'fixture-{arguments.plan}',
                                    seed=arguments.seed,noise=arguments.noise,fps=arguments.fps,
                                    plan=FIXTURE_PLANS[arguments.plan])
    except (SignSessionError,OSError) as error:
        print(f'FAIL {arguments.out}: {error}'); return 2
    session=read_session(arguments.out); annotation=read_annotation(arguments.out)
    errors,warnings=validate(session,annotation)
    print(f'{arguments.out}: {info["frame_count"]} frames, {info["spans"]} spans, '
          f'held={info["held"]}, motion={info["motion"]}, spelling={info["spelling"]}')
    print('  the annotation is the plan that generated the tracks, so it is right by construction and wrong '
          'about the model: it never says what the app would see.')
    for warning in warnings:
        print(f'  note: {warning}')
    for error in errors:
        print(f'  ERROR: {error}')
    return 1 if errors else 0


if __name__=='__main__':
    sys.exit(main())
