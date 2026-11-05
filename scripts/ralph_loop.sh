#!/usr/bin/env bash
# Autonomous research loop for docs/autoresearch-goal.md. Meant to run on the
# GPU box (screen/setsid). Each pass hands the agent the same goal with a fresh
# context; memory is the journal plus this branch. Stop by creating
# docs/autoresearch-stop, or set RALPH_MAX for a fixed number of passes.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${RALPH_MODEL:-strata/qwen3.8-flash-next-iq3_s}"
MAX_ITERS="${RALPH_MAX:-0}"            # 0 = keep going until stopped
GOAL="docs/autoresearch-goal.md"
LOG_DIR="logs/ralph"
export PATH="$HOME/.opencode/bin:$PATH"

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode not on PATH (looked in ~/.opencode/bin too)"
  exit 1
fi

if [ -f docs/autoresearch-stop ]; then
  echo "docs/autoresearch-stop exists; delete it to start the loop"
  exit 0
fi
mkdir -p "$LOG_DIR"

i=0
while [ ! -f docs/autoresearch-stop ]; do
  i=$((i + 1))
  log="$LOG_DIR/iter-$(printf '%03d' "$i")-$(date +%Y%m%d-%H%M%S).log"
  echo "iteration $i -> $log"
  opencode run --model "$MODEL" "$(cat "$GOAL")" >"$log" 2>&1
  sleep 20
  if [ "$MAX_ITERS" -gt 0 ] && [ "$i" -ge "$MAX_ITERS" ]; then
    echo "reached RALPH_MAX=$MAX_ITERS"
    break
  fi
done
