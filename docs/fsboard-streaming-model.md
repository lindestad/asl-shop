# Streaming FSboard model in the app (2026-10-21)

The live fingerspelling text now comes from the autoresearch model (`autoresearch/train.py`, run
`bs40k`, commit `d13d9b2`), exported to `assets/models/fsboard-stream.pt` (6.6 MB, float16).

| | Shipped causal TCN | Streaming FSboard model |
|---|---|---|
| Validation CER (greedy, 12,015 clips) | – | **5.95%** |
| Test CER (greedy, 12,521 clips) | 13.54% | 6.12% for the previous run (`bs80k`); `bs40k` not scored on test yet |
| Latency | – | 0.3 s lookahead (9 frames), 3.7 s history |
| CPU per frame (4 threads) | – | ~5.5 ms |

Trained on FSboard landmarks and human transcripts only (no OpenFS weights, teacher outputs or
synthetic clips). J and Z are learned from motion, so the old motion gate is not applied to it.

## Try it

    npm install
    npm run dev

The worker picks the model up automatically: `data/sequence.pt` if present, otherwise
`assets/models/fsboard-stream.pt` (`ASL_SEQUENCE_MODEL` still overrides both). The Python
environment needs the neural extra (`uv sync --extra neural`). Under the spelled text the app shows
"Strømmende FSboard-modell · 0,3 s forsinkelse" when this model is active. Spell with one hand in
view; text appears about 0.3 s behind the hand and the last letters may still change until the
sequence finishes.

## How it runs

- `model/stream_net.py`: the network, raw landmark input (`x*width`, `y*height`,
  `z*width/2000`, Right/Left slots), 30 Hz resampling (same rules as `model/timeline.py`) and an
  incremental greedy CTC decoder that recomputes only the newest 121-frame window per frame.
- `model/live_neural.py`: uses that path when the sequence checkpoint has contract
  `asl-shop-raw-xyz-v1`; the 91-value path is unchanged for older checkpoints.

## Checks

    .venv/bin/python -m unittest tests.test_stream_net
    .venv/bin/python autoresearch/tools/export_app.py autoresearch/runs/bs40k.pt assets/models/fsboard-stream.pt
    .venv/bin/python autoresearch/tools/replay_runtime.py assets/models/fsboard-stream.pt --clips 300

`export_app.py` re-scores the float16 export on validation (5.950% vs 5.951% in float32).
`replay_runtime.py` feeds validation clips frame by frame through the live runtime as a
1920x1080 camera would: 4.69% CER on 300 clips vs 4.67% offline on the same clips, 5.4 ms per frame.

Not yet checked here: the Electron UI itself (no Node on the training machine) and a real webcam.
`build-backend.mjs` still bundles only `data/*.pt`; copy the model to `data/sequence.pt` before
packaging.
