cart_num=0,1
port=8999

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py --env G2SIM --port ${port} --checkpoint_routing yrm/checkpoint_routing_five_vs_baseline.json
