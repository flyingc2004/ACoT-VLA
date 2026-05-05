#!/usr/bin/env bash
set -euo pipefail

cart_num="${CUDA_VISIBLE_DEVICES:-0}"
port="${PORT:-8999}"
routing_json="${CHECKPOINT_ROUTING_JSON:-yrm/checkpoint_routing.example.json}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${cart_num}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --env G2SIM \
  --port "${port}" \
  --checkpoint-routing "${routing_json}"
