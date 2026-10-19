# Synthetic signing-hand data from the released OpenFS generator — 19 October 2026

Three questions, answered on the VPS: whether the released OpenFS diffusion generator can produce
signing-hand landmark sequences that this repo's 91-column contract can actually train on, whether
mixing those sequences into training moves recognition accuracy, and where the licence on the
released checkpoints leaves the work. The third was closed mid-run by the owner (see "What about the
licence?" below), so the interesting answer is the second one. Machine-readable copy:
`docs/synthetic-data-metrics-2026-10-19.json`.

Headline:

- Generation works and is cheap: 62,892 clips (2,917,386 frames) sampled in 19 minutes on the
  V100-SXM3-32GB at a **189 MiB** peak, plus 624 single-letter clips in 25 seconds. That is two
  orders of magnitude inside the two-hour budget, so the corpus size was never the constraint.
- The quality filter is the expensive step. The released recognizer transcribed **48.9%** of word
  clips as the word they were sampled for under a prefix rule, and only 1.3% under a whole-string
  rule, because greedy decoding at `max_len=32` almost never emits EOS on generated poses: it
  reads `CAMIONES`, then appends ` BRING`. CER on the generated corpus is 94.01%, and 381,392
  of its 421,872 errors are insertions.
- Mixing moved accuracy, and the two mixes moved in opposite directions. The word mixes cost
  CER: 2.6% synthetic train frames cost **0.22 points** and 132 exact clips, 5.4% cost 0.27 and
  169. The 386-clip single-letter mix gains 0.14 overall, and 0.32 CER / 20 exact clips on
  J/Z-bearing clips, the only arm that beats both the shipped checkpoint and the data-paired
  retrain. All rows are single runs at seed 42; the table below is the whole evidence.
- The licence question is settled: this is research work, and it fits. The owner ruled on
  2026-10-19 that asl-shop is a research project, not a commercial one, so the OpenFS Non-Commercial
  Research License covers using the released generator, its outputs, and models fit on generated
  data inside this project. Sequences sampled from a checkpoint are still derivative works of it,
  so the guardrail that matters is the commercial boundary, not the research work itself: every
  artefact here declares `research_only`, and a test keeps the generator out of everything the app
  loads. If the project ever takes a commercial direction, these checkpoints and anything fit on
  them have to come out and be retrained from licensed data.

## What was generated, and from what

`scripts/generate_openfs_synthetic.py` imports the released classes instead of reimplementing them
and replays `generate_data_df.py`: `lib.models.diffusion.Diffusion.sampling` (ddim, 50 steps of a
T=1000 cosine schedule) driven by `lib.models.generator_df.EncoderTransformer`, conditioned by the
per-frame character labels `lib.datasets.fingerspelling.WordDataset` builds
(`use_frame_label_clean=true`), with `normalize_torch(..., normalize_value=0.5)` applied to x and y
exactly where upstream applies it. What conditions the sampler is the frame-label sequence, **not**
the released recognizer: the recognizer named in the checkpoint's `.hydra/config.yaml`
(`recognizer_fsboard`) is loaded only to transcribe each clip for the keep/drop filter.

| Pin | Value |
|---|---|
| generator `best_model.pth` sha256 | `4733de5601d5ee25162b0b51b01c10f09f2fbd72f3a7d75272ce950a277e7d5d` |
| conditioner `recognizer_fsboard` sha256 | `0e66c64baa9c4bb1e1299aeac2a4c4668204674677931398640caa4baaffd5d1` |
| upstream commit | `cd29c853dd880816de1166bab2d9c56ff960d5e1` |
| seed / trials | 20261019 / 3 (words), 24 (letters) |

Both digests were pinned on the command line and cross-checked against the checkpoint's own
`PROVENANCE.json`; the run refuses to start without them. Word lists, the alphabet, the sampler
class and method, the dataset class and its frame-label settings, the seed, the trial count and the
stubs installed for `tqdm`/`scipy`/`sklearn` (all three are imported at module level by upstream
and are absent from the VPS venv; each stub raises the moment upstream code calls it) are recorded
in every generation directory's `provenance.json`.

Words come from our own packed FSboard transcripts, never from held-out text: the train-only pool
after excluding every token in the validation and test transcripts is 19,044 words, the pool of
train-only words containing J or Z is 1,920, and 14,796 held-out tokens were excluded. The one
deliberate exception is the 26 single letters, which are the atoms every word is spelled from and
include the two movement letters. Word sets are chosen as a deterministic prefix of one seeded
permutation, so a smaller set is always a subset of a bigger one.

| Run | Words | Trials | Clips | Frames | Seconds | Clips/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot (50 train words) | 50 | 3 | 150 | 6,614 | 3.6 | 41.5 | 152 MiB |
| words (`train` + `jz`) | 20,964 | 3 | 62,892 | 2,917,386 | 1,131.5 | 55.6 | 189 MiB |
| letters (26 letters) | 26 | 24 | 624 | 4,054 | 24.8 | 25.2 | 110 MiB |

## The model-derived quality filter, and why it has two rules

The keep/drop signal is the released recognizer's transcription of each generated clip. No human
looked at any of these sequences, and every kept pack inherits that recognizer's biases. Two rules
are recorded for every generation:

| Rule | words kept | letters kept | What it means |
|---|---:|---:|---|
| exact | 847 (1.3%) | 2 (0.3%) | the whole transcription is the word |
| prefix | 30,764 (48.9%) | 386 (61.9%) | the transcription starts with the word |

Exact is the honest rule and the one that would be used if the goal were "only obviously-correct
clips", but on single letters it keeps almost nothing, and on words it rejects clips the recognizer
clearly read, because greedy decoding rambles after a correct word instead of stopping. The packs
below were built with the prefix rule, and both rates travel with the provenance so a prefix-kept
pack is never mistaken for an exactly-transcribed one. Per-set prefix rates, recomputed against the
set size rather than the whole corpus: train 28,934 of 57,132 (50.6%), jz 1,828 of 5,760 (31.7%).
(The `match_rate_by_group` denominator in the two generation reports written on the VPS divides by
all clips instead of by the set; the generator script has since been fixed.)

Recognition on generated clips, from the same filter pass: words CER 94.01% (11,167 deletions,
29,313 substitutions, 381,392 insertions over 20,964 reference characters), letters CER 1,965.9%
(0 deletions, 133 substitutions, 12,134 insertions over 624 characters). Insertions dominate by
two orders of magnitude; that is decoder behaviour on out-of-distribution poses, not clips missing
half their letters.

## What the 91-column contract can and cannot say about these rows

`scripts/pack_openfs_synthetic.py` maps generated poses into `asl-shop-xy-wrist-v2` with the repo's
own `encode_poses`, and writes dense per-frame letter targets derived from the labels the generator
was sampled with. What the rows genuinely carry, and what they do not, is written into the manifest:

- one signing hand only; the left slot is empty with its presence flag at 0, so no two-handed sign
  and no non-dominant motion;
- no capture geometry: the wrist columns are positions inside the generator's own normalised frame,
  not measured pixels, and no bounding box or image size exists anywhere in the chain;
- no real capture frame rate: the delta column is a nominal 1/30 step we assert, not a timestamp;
- no detection gaps: every generated frame is a detection, which matters because dropped detections
  are the shape of the app's deletion failure;
- no depth (the generated third coordinate is dropped), and no idle or background frames, so frame
  targets are dense where the real pack has blanks and ignored `-100` clips.

J and Z are representable in the sequence task: `VOCAB` carries all 26 letters and index 0 is the
CTC blank, so a movement letter gets a real per-frame target. The static 26-class `HANDSHAPES`
taxonomy deliberately excludes them, which is a separate known hole and is stated in the manifest
rather than papered over.

Frame-index convention: rows sit on the app's gap-preserving timeline. A generated clip has no
detection gaps, so `scripts/train_openfs_recognizer.timeline_indices` is `cumsum(delta)*fps` =
`arange`, which is also the `compressed` convention and the `frame_idx` the generator was sampled
with. The same rows are legal under both offline recipes; the choice is documented, not forced.

Round trip: the packer re-derives the upstream normalisation from its own packed rows with
`normalized_box` (per-axis divisor) and fails the pack unless it reproduces the sampled x,y within
1e-5. Across the three packs the worst error was **9.03e-6** on 30,762 clips, and the largest
deviation of `timeline_indices` from `arange` was 5.16e-6. An isotropic rescale would have hidden
that: the divisor upstream uses is per axis, and most frames have different x and y extents, which
is the bug class fixed last week in `scripts/fsboard_sessions.py::reconstruct_landmarks`.

Merged packs (`--merge-with data/openfs-packed`) keep the real validation and test files as
hardlinks, byte-identical, and append synthetic clips after the real train rows with their offsets
shifted by the real frame count. The real pack's 12,015 validation and 12,521 test clips are the
same bytes in every mixed pack, so all rows below are scored on identical held-out data.

| Mixed pack | Synthetic clips | Synthetic frames | Share of train frames | Quality-filter drops |
|---|---:|---:|---:|---:|
| `openfs-synthetic-packed-mix15k` | 14,999 (10,771 words) | 672,152 | 2.6% | 32,128 filter, 1 coordinate bound |
| `openfs-synthetic-packed-mixall` | 30,762 (15,433 words) | 1,373,466 | 5.4% | 32,128 filter, 2 coordinate bound |
| `openfs-synthetic-packed-letters` | 386 (26 letters) | 2,386 | 0.009% | 238 filter |

## Does mixing synthetic landmarks help?

No for the word mixes, marginally yes for the letter mix. All four retrained rows are data-paired
with each other: same seed 42, same shipped recipe (`train_openfs.py` sequence task, warmstart from
`models/neural-2026-10-12/handshape.pt`, 30 epochs, hidden 192, batch 64, amp), all on the V100, all
scored on the identical 12,521-clip packed test split by `scripts/evaluate_synthetic_mix.py`. The
shipped checkpoint is listed first for orientation; the retrained real-only baseline is the honest
comparator because it was fit on the same VPS, same venv, same seed. That shipped row is also an
agreement check between two scoring paths: `scripts/evaluate_synthetic_mix.py` puts it at 13.541%
CER with 16,819/6,552/6,569 edits and 2,267 exact strings, where `scripts/eval_recognition.py`
scored the same file at 13.542% with 16,803/6,553/6,586 and the same 2,267 exact
(`data/lm-rescoring-vps.report.json`), so the two harnesses differ by 2 edit operations out of
221,100 characters. Four arms plus four test-split evaluations took 3 h 50 m of V100 wall time end
to end, about twelve times the time it takes to generate the data, which tells you where to spend
the next experiment.

| Row | Test CER | Exact strings | Del/Sub/Ins | CER on J/Z clips | Exact on J/Z clips | s/epoch |
|---|---:|---:|---:|---:|---:|---:|
| shipped `neural-2026-10-12` (real data) | 13.54 | 2,267 (18.11%) | 16,819/6,552/6,569 | 14.10 | 12.63% | n/a |
| baseline retrain (real data only) | 13.59 | 2,289 (18.28%) | 17,277/6,399/6,361 | 14.24 | 12.16% | 106.9 |
| + mix15k (2.6% of train frames) | 13.81 | 2,157 (17.23%) | 17,614/6,534/6,387 | 14.54 | 11.92% | 120.4 |
| + mixall (5.4% of train frames) | 13.86 | 2,120 (16.93%) | 17,267/6,607/6,770 | 14.51 | 12.94% | 118.5 |
| + mixletters (0.009%) | **13.45** | **2,289 (18.28%)** | 16,994/6,318/6,436 | **13.92** | **13.73%** | 109.9 |

The word mixes cost accuracy, and the dose-response runs the wrong way: 2.6% synthetic train frames
cost 0.22 points of CER and 132 exact clips, 5.4% cost 0.27 points and 169 exact clips. Deletions do
not move much; the damage shows up as substitutions and, at the larger dose, insertions. Best-epoch
validation CER on the untouched real validation split agrees (14.14% baseline, 14.31% mix15k,
14.48% mixall), so this is not a test-split fluke: the model is being dragged toward the
generator's distribution, and the model-derived prefix filter does not remove enough of it. 48.9%
of word clips pass the prefix rule, but the recognizer's CER on generated clips is still 94%, so
most kept clips are close-but-wrong poses the trainer can only read as more training signal.

The letter mix is the one that moves. 386 clips, 2,386 frames, one part in ten thousand of the
train split, improve overall CER by 0.14 points against the real-only retrain, tie it on exact
strings, and beat the shipped checkpoint on both. Best-epoch validation CER points the same way:
14.11% for the letter mix is the best of the four retrained arms, the real-only retrain is next at
14.14%. The effect is also concentrated where it should be:
on J/Z-bearing clips it gains 0.32 points of CER and 20 exact clips (12.16% to 13.73%) against the
retrain, and it also beats the shipped checkpoint there (12.63%). One plausible mechanism: the 386
clips are dense per-frame letter targets for the two movement letters, which is exactly the part of
the label space the real training data under-covers, and they arrive in a form (gap-free, dense
targets) that adds signal without adding deletion-shaped noise.

Two caveats stay on the record. Each row is one run at one fixed seed, so the overall CER deltas of
a few tenths of a point are near the noise floor; the J/Z exact-clip gains (+20 clips against the
retrain, +14 against the shipped checkpoint) are the more convincing part of the letter result. And
the test split contains exactly one single-letter clip, so the single-letter column of the test
split carries no information. That is a property of the corpus, not of these runs.

## What about the licence?

Settled during the run, and it is a ruling rather than a discovery. `data/external/openfs/source/LICENSE`
and the checkpoint's `PROVENANCE.json` carry the OpenFS Non-Commercial Research License: "Commercial
use of the source code, model checkpoints, or derivative works is strictly prohibited." A sequence
sampled by a released checkpoint is a derivative work of it, so anything generated here inherits that
licence. On 2026-10-19 Daniel ruled that this project is research work, not commercial, and the
`license_note` in all three `data/openfs-checkpoints/*/PROVENANCE.json` files now records that ruling
and its consequence:

> Research-only work. Daniel ruled on 2026-10-19 that this project is research rather than
> commercial, so using this artifact here is inside its license. Keep it and anything fit on it out
> of any build with commercial intent; if that direction changes, retrain from licensed data.

So the answer to the original framing, "does the licence make it unshippable", is: it makes it
non-commercial, and this project is non-commercial, so no. The experiment is inside its licence and
nothing here was held back by it. What the licence does decide is the boundary to keep honest. The
app's shipped recognizers are trained on FSboard data under its own terms, which is a separate
question; research-only checkpoints and models fit on generated data stay out of the packaged app,
and that separation is enforced mechanically rather than by a comment:

- the generation script refuses to import upstream code without `--trusted-source`, and refuses to
  run at all unless both checkpoint digests are pinned;
- every generation directory, every pack manifest and every evaluation report carries
  `research_only: true`, the licence quote, and the generator/conditioner digests;
- every synthetic clip carries `"synthetic": true`, and merged packs say which clips are theirs;
- `tests/test_openfs_synthetic.py::AppRuntimeFirewall` fails if `model/`, `electron/`, `frontend/`
  or `handTracker/` reaches for the generator, the packer or upstream diffusion sampling, and if
  the generator is referenced anywhere outside `scripts/` and the tests.

One gap worth naming, which the `license_note` already states: research-only checkpoints are kept
out of the packaged app by convention plus the firewall test, not by a loader-level guard that could
catch a careless `fs.readFile` of `data/openfs-checkpoints/`. If anyone wants that closed, it is a
small packaging change, and it should be made before any of this data would ever sit near a build.

## Can it help the J/Z hole?

Yes, at small scale, in the task where the hole exists. The sequence task can already *represent*
J and Z: `VOCAB` carries all 26 letters and frame targets are dense. What it lacks is coverage,
since the two movement letters are the ones the real corpus spells least reliably and the shipped
model substitutes or drops most. A 386-clip corpus of just the 26 letters, mixed into the train
split at 0.009% of frames, moves the J/Z-bearing test clips from 14.24% CER / 155 exact to
**13.92% / 175** against the data-paired retrain, and from 14.10% / 161 against the shipped
checkpoint. The overall test CER also improves (13.45% vs 13.59%), so the letters are not paid for
somewhere else in the split, though with the whole-split delta at 0.14 points on a single run, that
part is close to noise.

Three limits on the claim:

- the word mixes show the same letter coverage arriving inside longer words (the 1,920-word `jz`
  pool) does *not* help: mix15k and mixall are worse on J/Z clips than the retrain (14.54/14.51 vs
  14.24). Isolated, clean letter poses help; generated letters embedded in generated words ride in
  with the generator's distributional error.
- the static 26-class `HANDSHAPES` taxonomy still excludes J and Z entirely; nothing here changes
  that, and any fingerspelling-static-classifier fix has to come from data for that task, not from
  these packs.
- the test split has one single-letter clip, so there is no direct held-out measurement of letter
  recognition. The J/Z-bearing-words subset (1,275 clips, 24,844 characters) is the only real
  signal, and a future corpus should add single-letter test clips before more letter experiments.

If this line of work continues, the cheapest follow-up is the one this run already set up: letters
only, more trials per letter (the 24 trials here produced 624 clips, 386 kept), and a small set of
held-out single-letter clips in the test split so the next answer is measured directly instead of
inferred.

## Files written by this experiment

All of it is in the research-only namespaces on the training box. The generated sets, the merged
packs and their reports were written under the run's worktree `data/` directory while they ran, and
have since been moved into `~/asl-data/` with symlinks left at the old paths, so they outlive the
worktree:

- `~/asl-data/openfs-synthetic-pilot`, `-pilot-prefix`, `-words`, `-letters` with `poses.npy`,
  `frame_labels.npy`, `clips.json`, `words/`, `provenance.json` and a sibling `.report.json` each.
  The word run's poses are 725 MB, the letters run's 1.2 MB;
- `~/asl-data/openfs-synthetic-packed-mix15k`, `-mixall`, `-letters` packs plus `.report.json`
  each, as `dataset.json` with `train`/`validation`/`test` clips, frames and targets. The
  validation and test files in all three are hardlinks into `~/asl-data/openfs-packed`, so they
  share those bytes;
- `~/asl-data/models/synthetic-{baseline,mix15k,mixall,mixletters}/sequence.pt` and the matching
  `sequence.report.json`;
- `~/asl-data/openfs-synthetic-{baseline,baseline-shipped,mix15k,mixall,mixletters}.evaluate.report.json`
  for the test-split subsets;
- `~/asl-wt/synthetic-generator/logs/{pilot,gen-words,gen-letters,train-chain}.log`.

The generation, pack, training and evaluation report JSONs are also copied into this checkout's
gitignored `data/` directory, and the headline rows are in `docs/synthetic-data-metrics-2026-10-19.json`.

The packs, the four training arms and the five test-split scorings ran from
`scripts/run_synthetic_training.sh`, committed with this document, one GPU process at a time. The
two generation runs came first:

```bash
ASL_OPENFS_SOURCE=$PWD/data/external/openfs/source .venv/bin/python scripts/generate_openfs_synthetic.py \
  --source data/external/openfs/source --generator data/openfs-checkpoints/generator_df_fsboard \
  --conditioner data/openfs-checkpoints/recognizer_fsboard \
  --generator-sha256 4733de5601d5...e7d5d --conditioner-sha256 0e66c64baa9c...fd5d1 --trusted-source \
  --words-from-pack data/openfs-packed --word-set train jz --quality-rule prefix \
  --train-words 24000 --jz-words 2400 --trials 3 --seed 20261019 \
  --output data/openfs-synthetic-words --report data/openfs-synthetic-words.report.json
# the letters run, same guards, different word set and trial count:
#   --word-set letters --quality-rule prefix --trials 24 \
#   --output data/openfs-synthetic-letters --report data/openfs-synthetic-letters.report.json
```

`bash scripts/run_synthetic_training.sh` then packs (`--limit-clips 15000` for mix15k, no limit for
mixall), trains baseline, mix15k, mixall and mixletters with the shipped recipe (seed 42), evaluates
each on the test split, and finishes by scoring the shipped checkpoint.
