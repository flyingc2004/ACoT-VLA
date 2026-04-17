export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DEBUG_MODE=false
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG_NAME=acot_icra_simulation_challenge_reasoning_to_action
EXP_NAME=all_tasks

uv run python scripts/train.py $CONFIG_NAME --overwrite --exp-name=$EXP_NAME