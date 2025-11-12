# PND Adam Isaac Gym Environment
Public pnd humanoid robot Adam training env based on legged gym
### Installation ###
1. Create a new python virtual env with python 3.8 

```
conda create -n pndrobot python=3.8
conda activate pndrobot
```
2. Install pytorch 1.13 with cuda-11.6:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
3. Install Isaac Gym
- Download the Isaac Gym Preview3 from https://developer.nvidia.com/isaac-gym
```
cd isaacgym/python && pip install -e .
```
- Try running an example `cd examples && python 1080_balls_of_solitude.py`
- For troubleshooting check docs (`isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
```
cd rsl_rl && pip install -e .
```
5. Install legged_gym
```
cd pnd_humanoid_robot && pip install -e .
```
6. Install other support
```
pip install numpy==1.23.5 tensorboard opencv-python
```

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one conatianing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.


### Usage ###
1. Train a policy

```
cd pnd_humanoid_robot/pnd_humanoid_robot_gym/scripts
python train.py --task=adam
```
   -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
   -  To run headless (no rendering) add `--headless`.
   -  The trained policy is saved in `pnd_humanoid_robot/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
   -  The following command line arguments override the values set in the config files:
   - --task TASK: Task name.
   - --resume:   Resume training from a checkpoint
   - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
   - --run_name RUN_NAME:  Name of the run.
   - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
   - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
   - --num_envs NUM_ENVS:  Number of environments to create.
   - --seed SEED:  Random seed.
   - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
   -  To run headless (no rendering) add `--headless`.
   -  To resume from a previous policly, set `resume=True` and set `load_run` and `checkpoint` in the train config
   ```pnd_humanoid_robot/pnd_humanoid_robot_gym/envs/pnd_humanoid_robot/pnd_humanoid_robot_adam_config.py```.  
   -  To see the performance and reward during training: `tensorboard --logdir=./ --bind_all`

2. Play a trained policy 
```
python play.py --task=adam
```
   - By default the loaded policy is the last model of the last run of the experiment folder.
   - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config
   ````pnd_humanoid_robot/pnd_humanoid_robot_gym/envs/pnd_humanoid_robot/pnd_humanoid_robot_adam_config.py```.


### Acknowledgement
- https://github.com/leggedrobotics/legged_gym
  
- https://github.com/leggedrobotics/rsl_rl

- https://arxiv.org/abs/2109.11978

