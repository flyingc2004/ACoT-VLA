export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DEBUG_MODE=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1

CONFIG_NAME=acot_icra_simulation_challenge_reasoning_to_action
EXP_NAME=test

uv run python scripts/train.py $CONFIG_NAME --overwrite --exp-name=$EXP_NAME