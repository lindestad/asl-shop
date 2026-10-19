# Autoresearch: best fingerspelling sequence model

You are one iteration in an autonomous research loop. One run = one experiment.
State lives in `docs/autoresearch-journal.md` and in this branch's history; your
context starts empty every time. Do exactly one experiment well, then commit and
end.

## Objective

Train a fingerspelling sequence model that beats the current best on the fixed
FSboard test split. Any architecture is fair game: causal TCN, CNN, transform-
er, Mamba-style, whatever you judge best. Metric is character error rate
(Levenshtein ins/sub/del over reference characters), with exact-string accuracy
reported alongside.

Baselines (12,521 complete test strings, 221,100 reference characters):

- causal TCN trained here (835k params): **13.54% CER**, 18.11% exact — the
  number to beat
- released OpenFS recogniser (8.81M params, research-only, never train it):
  6.29% CER, 59.32% exact — an upper reference, not a target

## Fixed data contract

- `data/openfs-packed/train_clips.json` — training; the standard slice is 12,000
  clips, keep it comparable across experiments unless you scale deliberately
- `data/openfs-packed/validation_clips.json` — model selection
- `data/openfs-packed/test_clips.json` — evaluate only; never train or tune on it
- input contract: `model.neural.CONTRACT` v2, 91-value frames, gap-preserving
  timeline; see `docs/neural-recognition.md`

## Rules

1. Read the journal's Current best and the last few entries first, then pick one
   idea that follows from them (not a reroll of a failed direction).
2. Run `scripts/gpu_budget.sh 6144` before any training. If it fails, halve the
   batch size. Training must stay inside ~6 GiB VRAM; the rest of the GPU
   belongs to the inference server and other jobs.
3. Train on train, select on validation. Evaluate on test **only** when the run
   beats current best on validation, and say so in the journal.
4. CER numbers come from the same Levenshtein metric the harness uses; see
   `scripts/train_streaming_student.py` for the reference implementation.
5. Append a journal entry: idea, architecture, parameter count, peak VRAM, wall
   clock, validation CER/exact, and test CER/exact if you ran it. Keep it one
   short block.
6. Commit new scripts or configs on this branch with a short lowercase message.
   Never commit checkpoints or `data/` files.
7. Everything stays under the same non-commercial research terms as the existing
   experiments (OpenFS labels and posteriors are research-only inputs; no
   dictionary, LLM or beam correction inside a run unless the journal entry
   names it explicitly as part of the experiment).

## End of iteration

Commit, then stop. The next iteration picks it up from the journal.
