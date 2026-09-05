#!/usr/bin/env bash
set -euo pipefail
# Use as the torchrun worker entrypoint; each rank captures its own CUDA/NVTX report.
# Example: torchrun --standalone --nproc-per-node=2 --no-python scripts/nsys_rank.sh configs/profile-gpu.json runs/nsys
config=${1:?configuration path required}
out=${2:?shared output directory required}
mkdir -p "$out"
exec nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output="$out/nsys-rank${RANK:-0}" \
  python -m tailtrace train --config "$config" --out "$out"
