#!/usr/bin/env bash
# Usage: scripts/gpu_budget.sh [need-mib]
# Exit 0 when at least need-mib of GPU memory is free (default 6 GiB; the rest
# of the card is reserved for the inference server and other jobs).
need="${1:-6144}"
free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
echo "free ${free} MiB, need ${need} MiB"
[ "${free%.*}" -ge "$need" ]
