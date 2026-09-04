#!/usr/bin/env bash
#SBATCH --job-name=tailtrace
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=tailtrace-%j.log
set -euo pipefail
# Submit from the checkout on a shared filesystem after installing the train extra.
# Set GPUs_PER_NODE to match the requested allocation.
GPUs_PER_NODE=${GPUs_PER_NODE:-4}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TAILTRACE_CONFIG=${TAILTRACE_CONFIG:-configs/gpu-ddp.json}
export TAILTRACE_OUT=${TAILTRACE_OUT:-runs/slurm-$SLURM_JOB_ID}
export GPUs_PER_NODE
srun bash -c 'exec python -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --nproc-per-node="$GPUs_PER_NODE" \
  --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m tailtrace train --config "$TAILTRACE_CONFIG" --out "$TAILTRACE_OUT"'
