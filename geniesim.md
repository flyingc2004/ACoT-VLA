2.2 Download
Genie Sim

git clone https://github.com/AgibotTech/genie_sim.git
Copy to clipboardErrorCopied
Genie Sim Assets

Download Genie Sim Assets and put them into genie_sim/source/geniesim/assets:

# ModelScope
git clone https://www.modelscope.cn/datasets/agibot_world/GenieSimAssets.git --branch rolling genie_sim/source/geniesim/assets

# Hugging Face
git clone https://huggingface.co/datasets/agibot-world/GenieSimAssets --branch rolling genie_sim/source/geniesim/assets
Copy to clipboardErrorCopied
NOTE: Please use the rolling branch to get the latest assets.

2.3 Installation
2.3.1 Docker Container (recommended)
Prepare docker image
NOTE：Genie Sim Benchmark dropped support for curobo since 3.0

# create docker image from dockerfile
cd genie_sim
genie_sim$ docker build -f ./scripts/dockerfile -t registry.agibot.com/genie-sim/open_source:latest .
Copy to clipboardErrorCopied
Launch docker container and run the demo
Make sure GenieSimAssets is downloaded at genie_sim/source/geniesim/assets

# start a new container in repo root
cd genie_sim
genie_sim$ ./scripts/start_gui.sh

# open a new terminal, into container
genie_sim$ ./scripts/into.sh

# inside container, run the demo
/geniesim/main$ geniesim --config source/geniesim/config/s2r_select_color.yaml
Copy to clipboardErrorCopied
2.3.2 (optional) Use Genie Sim as Python Module
NOTE：This is for import geniesim from other workspace,geniesim is tested only in conda python3.11

# prepare conda env
cd genie_sim
genie_sim$ conda create --name geniesim python=3.11

# cd in genie_sim root dir
genie_sim$ conda activate genniesim
genie_sim$ python -m pip install -e ./source
Copy to clipboardErrorCopied
2.3.3 Host Machine
We STRONGLY recommend developers using our tested docker container environment for development
2.3.4 Developer Guide
2.3.4.1 Enable pre-commit hooks for collaboration (optional)
NOTE: Make sure dependencies in requirements.txt are properly installed, the pre-commit hook will be triggered once git commit is involved.

Install and setup pre-commit to enable auto file-formatter, python / json / yaml etc.
# install pre-commit to your python env
genie_sim$ pip install pre-commit

# enable pre-defined pre-commit-hooks within repo
genie_sim$ pre-commit install
Copy to clipboardErrorCopied
Trigger file-formatter for all tracked files
genie_sim$ pre-commit run --all-files
3.5 AgiBot World Challenge Reasoning to Action Tasks (ICRA)
NOTE: The ICRA challenge is based on GenieSim 3.0.3. Please make sure you are using the correct version (latest commit: 8bf3e57a).

3.5.1 Run Baseline Model Inference
Docker Image Preparation
Baseline model checkpoints and inference scripts are integrated into one unified docker image, which can be obtained by following steps:

On the My submission page of the Test Server, click Get Registry Token to obtain the docker login credentials，then pull the image through the following command. (Note that the validity period of the voucher is 1 hour)

docker pull sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest
Copy to clipboardErrorCopied
Start model inference
Start docker container will automatically launch the inference service

docker run -it --network=host  --gpus all -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 {docker container name}
Copy to clipboardErrorCopied
Adjust XLA_PYTHON_CLIENT_MEM_FRACTION according to your machine GPU memory to run both inference and simulation on one machine.

The following info in terminal indicates the successful launch of the Inference service:

INFO:websockets.server:server listening on 0.0.0.0:8999

3.5.2 Run ICRA tasks
Start simulation docker container
genie_sim$ ./scripts/start_gui.sh
Copy to clipboardErrorCopied
After starting inference service, enter the docker container to launch all ICRA tasks

a. Config VLM checker Set OpenAI configs for VLM auto scoring, see Section 3.1.4 for details.

export BASE_URL=xxx
export VL_MODEL=xxx
export API_KEY=xxx
Copy to clipboardErrorCopied
Two tasks, scoop_popcorn and clean_the_desktop are evaluated by VLM while other tasks are evaluated by rules. The absence of VLM configuration will not affect the simulation run, but the evaluation will be missing.

b. Launch ICRA tasks

# Enter simulation container
genie_sim$ ./scripts/into.sh
# Run ICRA tasks with inference service from localhost:8999
/geniesim/main$ ./scripts/run_icra_tasks.sh
# Run ICRA tasks with inference service from other host machine
/geniesim/main$ ./scripts/run_icra_tasks.sh --infer-host xxx.xxx.xxx.xxx:8999
Copy to clipboardErrorCopied
Scores will be automatically collected when all tasks are finished. It can also be triggered by command below
# Use default directory: output/benchmark
/geniesim/main$ python3 scripts/stat_average.py
