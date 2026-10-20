# FSboard streaming autoresearch

Autonomous research loop (after karpathy/autoresearch) for the asl-shop live fingerspelling
recognizer. Goal: **lowest validation CER** for a model the app can run live.

## Setup

- Branch `autoresearch/<tag>` of asl-shop, off `feature/letter-and-fingerspelling`.
- Data: `uv run --with gdown python scripts/download_openfs.py --extract` plus the four FSboard
  metadata JSONs (see `docs/neural-recognition.md`), then `.venv/bin/python autoresearch/prepare.py`
  to pack `data/autoresearch/` (raw MediaPipe hand landmarks, human transcripts only).
- `prepare.py` is fixed: packing, GPU data access, the CER metric, and the app-constraint checks.
- `train.py` is the only file experiments change. Everything is fair game: features, architecture,
  optimizer, augmentation, loss, batch size, decoding inside the model output.

## Rules

- **Metric**: `val_cer` (greedy CTC, signer-disjoint validation split, 12,015 clips). `val_jz_cer`
  (clips containing J or Z) is a secondary check for the motion letters; don't keep a change that
  clearly hurts it.
- **Hard constraints**, enforced by `final_report` (a run that fails them is a crash):
  - streaming: output at frame t may depend only on frames t-H+1..t+L; `history_frames` H <= 300,
    `lookahead_frames` L <= 9 (0.3 s). Features must come from per-frame landmarks the app has
    (pixel x/y, MediaPipe z, handedness slot, presence); nothing computed over the whole clip.
  - CPU: `within_budget: True`, recomputing one H+L window on 4 threads <= 15 ms.
- **Data**: only FSboard landmarks + human transcripts. No OpenFS weights, teacher posteriors,
  refined/clean frame labels or synthetic clips (keeps the result shippable).
- **Test split is not used for decisions.** Score test only for the final chosen checkpoints.
- Time budget: 15 min of training (`TIME_BUDGET`), plus ~1-2 min startup/eval.
- Simplicity: as in the original program. Equal CER with less code is a win.

## Running experiments

One run saturates the H100 (two concurrent runs each get ~half the steps), so experiments run
one at a time. Each variant is a copy
`autoresearch/exp_<name>.py` (gitignored) launched as

    RUN_OUT=autoresearch/runs/<name>.pt .venv/bin/python autoresearch/exp_<name>.py > autoresearch/logs/<name>.log 2>&1

Keep concurrency constant (one run), because the time budget makes step
counts depend on GPU sharing; compare only runs made under the same concurrency. A kept variant is
copied over `train.py` and committed; `results.tsv` (untracked) logs every run:

    commit	val_cer	val_jz_cer	cpu_ms	params_M	status	description

status = keep / discard / crash. Seed noise is measured with repeated baselines; treat
improvements smaller than ~2x that noise as unconfirmed and re-run before keeping.

Read the summary with `grep -A20 '^---' autoresearch/logs/<name>.log`. On crash, read the tail of
the log, fix trivial bugs, otherwise log it as crash and move on.

## Loop

LOOP FOREVER (until the human stops you): look at results, pick the next ideas, launch them in the
free slots, wait, record, keep/discard, commit kept changes on the branch. Don't push.
