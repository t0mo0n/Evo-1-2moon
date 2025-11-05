## Installation

1. Prepare the environment for Evo-1

```bash
# Clone this repo
git clone https://github.com/DorayakiLin/Evo_1_clean.git


# Create a Conda environment
conda create -n Evo1 python=3.10 -y
conda activate Evo1

# Install requirements
cd Evo_1
pip install -r requirements.txt
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation

```

## Simulation Benchmark

### Meta-World Benchmark

#### 1 Prepare the environment for Meta-World

```bash
# Start a new terminal and create a Conda environment for Meta-World
conda create -n metaworld python=3.10 -y
conda activate metaworld

# Install requirements
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
```

#### 2 Run the weight and code

##### 2.1 Download Model Weight

[Link to Model Weight for Meta-Wolrd](https://huggingface.co/yinxinyuchen/evo1_metaworld/tree/main/step_65000)

##### 2.2 Modify config

Modify the checkpoint dir to where you download the model weight:
[Modify the checkpoint dir](Evo_1/scripts/Evo1_server.py#L149)

Modify the server port (Optional,default 9000):
[Modify the server port](Evo_1/scripts/Evo1_server.py#L152)

Modify the client port (Optional,default 9000):
[Modify the client port](MetaWorld_evaluation/mt50_evo1_client_prompt.py#L40)

#### 3 Run the simulation evaluation

```bash
# Start Evo-1 server (In terminal 1)
conda activate Evo1
cd Evo_1
python scripts/evo1_server.py

# Start Meta-World client (In terminal 2)
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py

```

### LIBERO Benchmark

#### 1 Prepare the environment for LIBERO

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

#### 2 Run the weight and code

##### 2.1 Download Model Weight

[Link to Model Weight for LIBERO](https://huggingface.co/liujiting/evo1_libero/tree/main)

##### 2.2 Modify config

Evo_1_clean/miravla/scripts/evo1_server_json.py

Modify the checkpoint dir to where you download the model weight:
[Modify the checkpoint dir](miravla/scripts/evo1_server_json.py#L149)

#### 3 Run the simulation evaluation

```bash
# Start Evo-1 server (In terminal 1)
conda activate Evo1
cd Evo_1
python scripts/evo1_server_json.py

# Start LIBERO client (In terminal 2)
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py

```

## Training on Your Own Dataset

#### 1 Prepare your dataset

We support lerobot v2.1 format, please convert your data to this format.

We use MetaWorld Dataset as an example

```bash
cd Evo1_training_dataset/

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/DorayakiLin/metaworld_dataset_v2.1

cd metaworld_dataset_v2.1/

git lfs pull
```

#### 2 Modify config

#### 2.1 Modify config.yaml

You need to modify the config.yaml[config.yaml](Evo_1/dataset/config.yaml)

#### 2.2 Set the cache path

You need to change the cache path [cache_dir](Evo_1/dataset/lerobot_dataset_pretrain_mp.py)

#### 3 Start Training

We only train the integration module and action expert in stage 1.

If you are training with multiple GPU, set --num_processes to the GPU number

#### 3.1 Stage 1

```bash
cd Evo_1/

accelerate launch  --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_dataset_v2.1_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50  --finetune_action_head --disable_wandb  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24  --save_dir /home/dell/code/lintao/Evo1_700m_clean/checkpoints/Evo1_metaworld_dataset_v2.1_stage1
```

#### 3.1 Stage 2

We unfreeze the VLM in stage 2.

```bash
cd Evo_1/

accelerate launch  --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_dataset_v2.1_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24  --save_dir /home/dell/code/lintao/Evo1_700m_clean/checkpoints/Evo1_metaworld_dataset_v2.1_stage2
--resume --resume_pretrain --resume_path /home/dell/code/lintao/Evo1_700m_clean/checkpoints/Evo1_metaworld_dataset_v2.1_stage2/step_5000
```
