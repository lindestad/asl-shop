#!/usr/bin/env bash
# Runs experiments named in autoresearch/queue.txt one at a time (first line first), removing each
# line as it starts. Waits for any running experiment first. Stops when the queue is empty.
cd "$(dirname "$0")/../.."
while true; do
  while pgrep -f "python autoresearch/(exp_|train)" >/dev/null; do sleep 15; done
  name=$(head -n1 autoresearch/queue.txt 2>/dev/null)
  [ -z "$name" ] && exit 0
  sed -i '1d' autoresearch/queue.txt
  echo "$(date -u +%H:%M) start $name" >> autoresearch/logs/queue.log
  RUN_OUT=autoresearch/runs/$name.pt .venv/bin/python autoresearch/exp_$name.py > autoresearch/logs/$name.log 2>&1
  echo "$(date -u +%H:%M) done $name $(grep -E '^val_cer' autoresearch/logs/$name.log | awk '{print $2}')" >> autoresearch/logs/queue.log
done
