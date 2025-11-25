from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_config import AdamLite12dofRoughCfg, AdamLite12dofRoughCfgPPO
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_env import AdamLite12dofRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "adam_lite_12dof", AdamLite12dofRobot, AdamLite12dofRoughCfg(), AdamLite12dofRoughCfgPPO())
