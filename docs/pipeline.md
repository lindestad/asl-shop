# Recognition pipeline

## Overview

A frame of video enters as a JPEG data URL and leaves as a letter guess plus a
provisional spelling. Three sources can feed it — the live camera, a picked
still image, and a dataset clip that the demo mode plays back as if it *were*
the camera — and all three converge on the same runtime in
`model/live_neural.py`.

```
 camera / still / demo clip (mp4 via demo-media://)
        │  <video> element, 640px JPEG snapshots
        ▼
 frontend/Camera.tsx ── IPC (preload.cjs) ──▶ electron/recognition.mjs
        ▲                                        │ newline-delimited JSON
        │                                        ▼
   Recognition JSON              scripts/recognition_worker.py
                                 MediaPipe hands → contract v2 features
                                 ▼
                        model/live_neural.py (NeuralRuntime)
                        handshape MLP (letter)   causal TCN (text)
                        KNN fallback (letter)    ensemble refiner (finish)
```

The renderer never runs models; the worker is a persistent Python process that
Electron spawns lazily. The static letter path and the fingerspelling path are
independent: a letter answer for the current frame and a running transcript of
the session travel in the same response but come from different models.

## Renderer

`frontend/Camera.tsx` owns one `<video>` element and one analysis loop. While
`running` is true, a `requestVideoFrameCallback` chain fires per displayed
frame; a shared `frameBusy` ref means at most one request is in flight and
excess frames are dropped rather than queued (recognition is allowed to be
slower than 30 Hz, the display never stalls). Each analyzed frame is
downscaled to ≤640 px wide, exported as JPEG at quality 0.85, and sent as
`recognizeFrame(dataUrl, { sessionId, timestampMs, finish })` with
`performance.now()` as the capture time. Results are applied only while the
current `textSession` UUID still matches, so a restart can never be raced by a
late reply from the old session.

A spelling segment ends in one of three ways, all of which send `finish: true`
on the next analyzed frame:

- the user clicks Finish (`finishRequested` ref),
- the hand leaves the frame for over a second and the pending segment actually
  spelled something (see the departure gate below) — auto-finish,
- the 900-frame / 30-second session limit is reached.

`frontend/Scratchpad.tsx` keeps committed notes separate from the live tail:
provisional text renders after the committed text as a ghost tail and is never
editable as part of the notes. When the session's status transitions into
`finished`, the final text (already ensemble-corrected) is committed into the
notes once; the next provisional update replaces the old tail. Editing the
notes while a word is still in flight cannot bake the ghost tail in.

`LandmarkOverlay` draws `result.hands` from the recognition response — the
landmarks the models actually consumed — over the same video frame, so the
skeleton matches what the pipeline saw rather than what the compositor shows.

## Main process and bridge

`electron/main.mjs` exposes the app through three custom schemes:

- `app://bundle/…` serves the built renderer in production (dev loads the Vite
  URL instead),
- `demo-media://clips/<file>.mp4` serves dataset footage during demo replay.
  It resolves only a basename inside `data/external/openfs/demo-clips`, answers
  `Range` requests with real `206`/`416` responses (Chromium's media stack
  rejects whole-file `200`s), and carries `Access-Control-Allow-Origin: *`
  together with the `corsEnabled` scheme privilege and `crossOrigin="anonymous"`
  on the video element, without which drawing the video into a canvas taints
  it and `toDataURL` refuses to export frames.
- CSP (`index.html`) allows `media-src demo-media:` and nothing else exotic.

`electron/preload.cjs` bridges a small surface via `contextBridge`:
`recognize`, `recognizeFrame`, `recognizeLandmarks`, `demoList`, `demoClip`,
plus image open/save helpers. `electron/recognition.mjs` keeps the Python
worker: it spawns `scripts/recognition_worker.py` lazily on first request,
writes one JSON object per line to stdin, and matches replies by id. A dead or
unresponsive worker rejects everything in flight with an actionable message
("Run ./setup.sh…") and gets respawned on the next request; each request has a
60-second deadline.

## Worker

`scripts/recognition_worker.py` decodes the image with OpenCV (≤16 MP), then
detects hands with MediaPipe Hand Landmarker:

- Still images use IMAGE mode and return every hand found.
- Live video uses VIDEO mode. Full palm detection reruns every 250 ms through
  an eight-hand "scan" detector; between scans a `num_hands`-matched tracker
  handles ordinary frames. When the visible hand count changes, the tracker is
  rebuilt at the new capacity — MediaPipe otherwise reruns detection every
  frame whenever fewer hands than `num_hands` are visible.
- A worker-wide monotonic clock keeps timestamps strictly increasing across
  camera restarts and interleaved stills; the session's own capture timestamps
  must also strictly increase, which the runtime enforces.

The static letter answer prefers the neural handshape model. Only when that
model is unavailable does the legacy KNN step in (`data/knn.pkl`, 63-value XYZ
features, needs a ≥60 % neighbor majority, never answers J/Z). Hand *detection*
is MediaPipe's job; the classifiers only ever see landmarks.

`analyze_landmarks` is a side door for the landmark-only demo fallback: it
skips image decode and MediaPipe entirely and feeds reconstructed landmarks
into the same runtime on a 1000×1000 nominal canvas.

## Models and the v2 contract

`model/neural.py` defines contract `asl-shop-xy-wrist-v2`:

- **Static input, 42 values:** wrist-relative pixel XY divided by the
  wrist-to-middle-MCP distance, left hand mirrored.
- **Sequence input, 91 values per frame:** fixed Right and Left slots of 42
  handshape values + 2 wrist coordinates (in hand-size units) + 1 presence
  mask, plus one elapsed-time value (seconds since the previous captured
  frame, clamped to 1 s). Missing hands are zero slots, never interpolated
  guesses.

`HandshapeNet` is an MLP over the 42-value input with 25 classes (24 static
letters plus `nothing`; J and Z are motion letters and belong to the sequence
model). The runtime abstains below a score of 0.8 and whenever more than one
hand is visible, since single-frame classification cannot pick the signing
hand reliably. Scores are not calibrated probabilities.

`FingerspellingNet` is a causal dilated-convolution TCN with CTC supervision,
reading the 91-value stream at a nominal 30 Hz and emitting one token per
frame, which the runtime collapses greedily (skip blank, skip repeat) into
text. Because it is causal, provisional text is available while the hand is
still moving.

## Timeline resampling

Captured frames arrive at whatever rate recognition manages; the model expects
30 Hz. `model/timeline.py` resamples the capture stream onto an exact 30 Hz
grid: endpoints pass through, interior ticks interpolate linearly only when
both neighbors show the same hand present within a 250 ms gap, and anything
across a presence transition or a long gap stays a zero (missing) slot. The
resampler never synthesizes an unobserved hand and never carries a pose
forward through an absence. Offline packing uses the identical grid.

## Runtime session semantics

`NeuralRuntime.analyze` (called for every video frame) maintains one session
keyed by the renderer's `sessionId`:

- A different session id resets everything. Monotonic-timestamp violations are
  a hard error, not a silent drop.
- Reaching 900 captured or model frames, or 30 s, latches `limit_reached`; the
  segment stops accumulating but keeps answering until finished.
- **Departure gate:** when the gap since the last accumulated frame exceeds
  1000 ms, the segment either auto-finishes or is discarded. It finishes only
  when the evidence supports a spelling attempt: ≥6 frames with a hand, ≥500 ms
  of span, and real motion (peak-to-peak hand travel > 0.12 hand-scales).
  Brief flashes and static holds are discarded so ghost segments never reach
  the scratchpad.
- **J/Z gate:** a sequence-model J or Z survives only with local motion
  evidence (≥0.12 hand-scale travel within the preceding second at that
  position). This is a necessary check, not a geometric classifier of the J
  wiggle or Z zigzag. Separately, `gate_text` clears J/Z (and short text) for
  movement-free holds and caps output length by segment duration (~20
  characters/second plus bounds); segments below the evidence thresholds get
  `insufficient-context` / `held-letter` status messages instead of text.

## Finish and refinement

On finish the provisional text passes the same gates, then the final answer can
be recomputed with full context by a bidirectional model:

- **Ensemble refiner** (`model/ensemble_refiner.py`, used when
  `ASL_ENSEMBLE_MODEL_DIR` points at the checkpoint directory): four
  OpenFS-trained recognizers, beam search over the live 91-value timeline,
  fusion as the weighted mean of per-model frame log-posteriors, MAP decode,
  no language model. Output longer than the duration budget allows is rejected
  and the provisional text stands.
- **OpenFS adapter** (`model/openfs_refiner.py`): research-only path into an
  explicitly provided upstream checkout; env-gated and off by default.

Whichever runs, `sequence.method` reports `causal-tcn`, `ensemble`, or
`openfs`, and failures surface as `refinementError` rather than silently
downgrading.

## Demo replay (clip as camera)

Demo mode exists to exercise the whole pipeline without a camera. Two paths:

- **Footage path (preferred):** `scripts/demo_replay.py --list` picks a fixed
  deterministic selection from the packed test split (shortest, median, first
  both-hands clip, longest, plus curated real-word picks that stress the J and
  Z gates). The renderer loads the clip's mp4 from `demo-media://` into the
  *same* `<video>` element the camera uses and the normal camera loop takes
  over — same snapshots, same `recognizeFrame` channel, MediaPipe included.
  Landmarks are detected from the displayed pixels, so overlay alignment is
  correct by construction. When the clip ends, one last snapshot carries
  `finish: true`, which is indistinguishable from a user pressing Finish.
- **Landmark path (fallback):** when a clip's mp4 is missing, replay sends
  reconstructed landmark frames (`recognizeLandmarks`) at nominal 30 Hz spacing
  through the worker's side door — the same downstream runtime, minus MediaPipe
  detection.

Reconstruction detail: the packed timeline stores model frames, and
`shape_features` inverts exactly (the x mirror is its own inverse), so
replayed landmarks regenerate the 91-value stream the model was scored on.
That is what the packed 91 % numbers transfer through demo mode.

## Where to test what

| layer | test |
| --- | --- |
| renderer + worker, browser mock bridge | `tests/app.spec.ts` (Playwright, web) |
| Electron + real Python worker (image, camera fake, demo replay) | `tests/desktop.spec.ts` |
| runtime/session semantics | `tests/test_neural.py`, `tests/test_live_neural.py` |
| resampler, contract inverse | `tests/test_timeline.py`, `tests/test_demo_replay.py` |

`tests/desktop.spec.ts` demo-replay test is the end-to-end guard on the
demo-media plumbing: a broken range handler or a tainted canvas rejects every
frame, and the test's finish assertion can only pass if real recognition ran.
