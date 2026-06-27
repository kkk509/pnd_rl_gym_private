from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_config import AdamLite12dofRoughCfg, AdamLite12dofRoughCfgPPO
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_env import AdamLite12dofRobot
from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_config import AdamPro12dofRoughCfg, AdamPro12dofRoughCfgPPO
from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_env import AdamPro12dofRobot

from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_hybrid_env import (
    AdamPro12dofHybridRobot,
)
from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_hybrid_config import (
    AdamPro12dofHybridCfg,
    AdamPro12dofHybridCfgPPO,
)

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "adam_lite_12dof", AdamLite12dofRobot, AdamLite12dofRoughCfg(), AdamLite12dofRoughCfgPPO())
task_registry.register( "adam_pro_12dof", AdamPro12dofRobot, AdamPro12dofRoughCfg(), AdamPro12dofRoughCfgPPO())
task_registry.register( "adam_pro_12dof_hybrid", AdamPro12dofHybridRobot, AdamPro12dofHybridCfg(), AdamPro12dofHybridCfgPPO(),)