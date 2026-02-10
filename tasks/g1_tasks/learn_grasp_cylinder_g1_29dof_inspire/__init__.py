# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Learn-to-Grasp task: RL trains finger control while arm is scripted.
Policy learns grasping with right hand using force and position feedback.
"""

import gymnasium as gym

from . import learn_grasp_env_cfg


gym.register(
    id="Isaac-Learn-Grasp-Cylinder-G129-Inspire",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": learn_grasp_env_cfg.LearnGraspCylinderEnvCfg,
    },
    disable_env_checker=True,
)

# Play config with 1 environment
gym.register(
    id="Isaac-Learn-Grasp-Cylinder-G129-Inspire-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": learn_grasp_env_cfg.LearnGraspCylinderEnvCfg_PLAY,
    },
    disable_env_checker=True,
)
