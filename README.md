# Using Attention in HRL

Framework for training options with different attention mechanism and using them to solve downstream tasks.

## Requirements
GPU required

```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment and install remaining dependencies. (e.g. sub-module gym_minigrid which is a modified version of [MiniGrid](https://github.com/maximecb/gym-minigrid) )
```
conda activate affenv
cd gym-minigrid
pip install -e .
cd ../
pip install -e .
```

## Instructions
In order to train options and IC_net follow these steps:

```
1. Configure desired environment - number of task and objects per task in file config/op_ic_net.yaml. E.g:
  env_args:
    task_size: 3
    num_tasks: 4

2. Configure desired type of attention (between "affordance", "interest", "nan") - in file config/op_ic_net.yaml. E.g. 
main:
  attention: "affordance" 

3. Train by running command
liftoff train_main.py configs/op_ic_net.yaml
```

Once a pre-trained option checkpoint exists a HRL agent can be trained to solve the downstream task (for the same environment the options were trained on). Follow these steps in order to train an HRL-Agent with different types of attentions:

```
1. Configure checkpoint (experiment config file and options_model_id) for pre-trained Options and IC_net - in file configs/hrl-agent.yaml. E.g: 

main:
  options_model_cfg: "results/op_aff_4x3/0000_multiobj/0/cfg.yaml"
  options_model_id: -1  # Last checkpoint will be used

2. Configure type of attention for training the HRL-agent (between "affordance", "interest", "nan") - in file configs/hrl-agent.yaml. E.g:
main:
  modulate_policy: affordance

3. Train HRL-agent by running command
liftoff train_mtop_ppo.py configs/hrl-agent.yaml

```

Both training scrips produce results in the `results` folder, where all the outputs are going to be stored including train/eval logs, checkpoints. Live plotting is integrated using services from Wandb (plotting has to be enabled in the config file `main:plot` and user logged in Wandb or user login api key in the file `.wandb_key`). 

The console output is also available in a form:
- Option Pre-training e.g.:
```
U 11 | F 022528 | FPS 0024 | D 402 | rR:u, 0.03 | F:u, 41.77 | tL:u 0.00 | tPL:u 6.47 | tNL:u 0.00 | t 52 | aff_loss 0.0570 | aff 2.8628 | NOaff 0.0159 | ic 0.0312 | cnt_ic 1.0000 | oe 2.4464 | oic0 0.0000 | oic1 0.0000 | oic2 0.0000 | oic3 0.0000 | oPic0 0.0000 | oPic1 0.0000 | oPic2 0.0000 | oPic3 0.0000 | icB 0.0208 | PicB 0.1429 | icND 0.0192
```

Some of the  training entries decodes as
```
F - number of frames (steps in the env)
tL - termination loss
aff_loss - IC_net loss
cnt_ic - Intent completion per training batch 
oicN - Intent completion fraction for each option N out of Total option N sampled
oPicN - Intent completion fraction for each option N out of affordable ones
PicB - Intent completion average over all options out of affordable ones
```

- HRL-agent training

```
U 1 | F 4555192.0 | FPS 21767 | D 209 | rR:u, 0.00 | F:u, 8.11 | e:u, 2.48 | v:u 0.00 | pL:u 0.01 | vL:u 0.00 | g:u 0.01 | TrR:u, 0.00
```
Some of the  training entries decodes as
```
F - number of frames (steps in the env offseted by the number of pre-training steps)
rR - Accumulated episode reward average
TrR - Average episode success rate
```

## Framework structure

The code is organised as follows:
 
 - `agents/` - implementation of agents (e.g. training options and IC_net `multistep_affordance.py`; hrl-agent PPO `ppo_smdp.py` )
 - `configs/`  - config files for training agents
 - `gym-minigrid/` - sub-module - Minigrid envs
 - `models/` - Neural network modules (e.g options with IC_net `aff_multistep.py` and CNN backbone `extractor_cnn_v2.py`)
 - `utils/` - Scripts for e.g.: running envs in parallel, preprocessing observations,  gym wrappers, data structures, logging modules 
 - `train_main.py` - Train Options with IC_net
 - `train_mtop_ppo.py` - Train HRL-agent


## Acknowledgements
We used [PyTorch](https://pytorch.org/) as a machine learning framework.

We used [liftoff](https://github.com/tudor-berariu/liftoff) for experiment management.

We used [wandb](https://wandb.ai/site) for plotting.

We used [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) adapted for training our agents.

We used [MiniGrid](https://github.com/maximecb/gym-minigrid) to create our environment.


