# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Learn-to-Grasp Environment Configuration

This environment trains finger grasping with RL while arm movement is scripted.
The policy controls only the right hand fingers (6 DOF) and learns to:
1. Grasp the cylinder when the arm reaches it
2. Maintain grip during lift and transport
3. Release at the target bin

Observations: finger positions, contact forces, cylinder pose, task phase
Actions: right hand finger joint positions (6 DOF)
"""

from __future__ import annotations

import sys
import os
import importlib.util

# Setup importlib helper BEFORE any relative imports
def _import_module_directly(module_name: str, file_path: str):
    """Import a module directly from file path without triggering parent __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get paths
_this_dir = os.path.dirname(os.path.abspath(__file__))
_tasks_dir = os.path.dirname(os.path.dirname(_this_dir))

# Import local mdp submodules directly (avoid relative import issues)
# Order matters: observations first, then events, then rewards/terminations (which depend on events)
_mdp_observations = _import_module_directly(
    "learn_grasp_mdp_observations",
    os.path.join(_this_dir, "mdp", "observations.py")
)
_mdp_events = _import_module_directly(
    "learn_grasp_mdp_events",
    os.path.join(_this_dir, "mdp", "events.py")
)
_mdp_rewards = _import_module_directly(
    "learn_grasp_mdp_rewards",
    os.path.join(_this_dir, "mdp", "rewards.py")
)
_mdp_terminations = _import_module_directly(
    "learn_grasp_mdp_terminations",
    os.path.join(_this_dir, "mdp", "terminations.py")
)

# Create a namespace object that contains all mdp exports
class _MdpNamespace:
    pass

mdp = _MdpNamespace()

# Copy all exports from each submodule
for _mod in [_mdp_observations, _mdp_rewards, _mdp_events, _mdp_terminations]:
    for _name in dir(_mod):
        if not _name.startswith('_'):
            setattr(mdp, _name, getattr(_mod, _name))

# Import common configurations directly (avoid tasks/__init__.py auto-importer)
_robot_configs = _import_module_directly(
    "robot_configs",
    os.path.join(_tasks_dir, "common_config", "robot_configs.py")
)
_camera_configs = _import_module_directly(
    "camera_configs",
    os.path.join(_tasks_dir, "common_config", "camera_configs.py")
)
_contact_sensor_configs = _import_module_directly(
    "contact_sensor_configs",
    os.path.join(_tasks_dir, "common_config", "contact_sensor_configs.py")
)

# Create a fake tasks.common_config module so base_scene_pickplace_cylindercfg.py can import from it
# This prevents triggering the tasks/__init__.py auto-importer
import types
_fake_common_config = types.ModuleType("tasks.common_config")
_fake_common_config.CameraBaseCfg = _camera_configs.CameraBaseCfg
_fake_common_config.CameraPresets = _camera_configs.CameraPresets
_fake_common_config.RobotBaseCfg = _robot_configs.RobotBaseCfg
_fake_common_config.G1RobotPresets = _robot_configs.G1RobotPresets
_fake_common_config.InspireHandContactSensorCfg = _contact_sensor_configs.InspireHandContactSensorCfg
sys.modules["tasks.common_config"] = _fake_common_config

# Also need a fake tasks module to prevent import errors
if "tasks" not in sys.modules:
    _fake_tasks = types.ModuleType("tasks")
    _fake_tasks.common_config = _fake_common_config
    sys.modules["tasks"] = _fake_tasks

_scene_cfg = _import_module_directly(
    "base_scene_pickplace_cylindercfg",
    os.path.join(_tasks_dir, "common_scene", "base_scene_pickplace_cylindercfg.py")
)

G1RobotPresets = _robot_configs.G1RobotPresets
CameraPresets = _camera_configs.CameraPresets
InspireHandContactSensorCfg = _contact_sensor_configs.InspireHandContactSensorCfg
TableCylinderSceneCfg = _scene_cfg.TableCylinderSceneCfg

# Now safe to import Isaac modules
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


# Right hand finger joint names for action space
RIGHT_HAND_JOINTS = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]


##
# Scene Configuration
##

@configclass
class LearnGraspSceneCfg(TableCylinderSceneCfg):
    """Scene configuration for learn-to-grasp task.

    Inherits table and cylinder from base scene.
    Adds robot with Inspire hand and contact sensors.
    """

    # Robot: G1 with Inspire hand, fixed base
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # Contact sensors for force feedback (FORCE_ACT register 1582)
    contact_forces = InspireHandContactSensorCfg.all_fingers()

    # Disable world camera from base scene (not needed for RL training)
    world_camera = None

    # Optional: wrist camera (disabled by default for faster training)
    # right_wrist_camera = CameraPresets.right_inspire_wrist_camera()


##
# Action Configuration
##

@configclass
class ActionsCfg:
    """Action configuration - policy controls only right hand fingers.

    The arm is controlled by scripted events, not the policy.
    This simplifies the learning problem to just grasping.
    """

    # Policy outputs 6 joint position targets for right hand
    right_hand = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_JOINTS,
        scale=1.0,
        use_default_offset=True,
    )


##
# Observation Configuration
##

@configclass
class ObservationsCfg:
    """Observation configuration for grasping policy.

    Includes finger state, contact forces, and cylinder information.
    No camera observations for faster training.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        # Right hand finger joint positions (6 values)
        finger_pos = ObsTerm(func=mdp.get_right_hand_joint_pos)

        # Right hand finger joint velocities (6 values)
        finger_vel = ObsTerm(func=mdp.get_right_hand_joint_vel)

        # Contact forces on each finger (6 values)
        contact_forces = ObsTerm(func=mdp.get_right_hand_contact_forces)

        # Cylinder position relative to palm (3 values)
        cylinder_rel_pos = ObsTerm(func=mdp.get_cylinder_relative_to_palm)

        # Cylinder height (1 value)
        cylinder_height = ObsTerm(func=mdp.get_cylinder_height)

        # Cylinder velocity (3 values)
        cylinder_vel = ObsTerm(func=mdp.get_cylinder_velocity)

        # Current task phase as scalar (1 value)
        task_phase = ObsTerm(func=mdp.get_task_phase_scalar)

        # Right palm world position (3 values) - for context
        palm_pos = ObsTerm(func=mdp.get_right_palm_position)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # Concatenate into single vector

    policy: PolicyCfg = PolicyCfg()


##
# Event Configuration
##

@configclass
class EventsCfg:
    """Event configuration for scripted arm control and resets."""

    # Reset scene to default on episode start
    reset_scene = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
    )

    # Reset task phase counter
    reset_phase = EventTermCfg(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    # Reset grasp latch
    reset_grasp = EventTermCfg(
        func=mdp.reset_grasp_latch,
        mode="reset",
    )

    # Scripted arm control - runs every step
    arm_control = EventTermCfg(
        func=mdp.scripted_arm_control,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # Every step
        params={
            "interpolation_speed": 0.08,
        },
    )

    # Random cylinder position variation on reset
    reset_object = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.03, 0.03],
                "y": [-0.03, 0.03],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


##
# Reward Configuration
##

@configclass
class RewardsCfg:
    """Reward configuration for grasping task."""

    # Reward for finger-cylinder contact during grasp phases
    contact_reward = RewTerm(
        func=mdp.finger_cylinder_contact_reward,
        weight=1.0,
        params={
            "min_force": 0.2,
            "max_force": 8.0,
        },
    )

    # Penalty for cylinder instability (fast movement)
    velocity_penalty = RewTerm(
        func=mdp.cylinder_velocity_penalty,
        weight=-0.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "max_velocity": 0.4,
        },
    )

    # Reward for lifting cylinder
    lift_reward = RewTerm(
        func=mdp.cylinder_lift_reward,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "initial_height": 0.84,
            "target_height": 1.05,
        },
    )

    # Bonus for cylinder in bin
    in_bin_reward = RewTerm(
        func=mdp.cylinder_in_bin_reward,
        weight=3.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "bin_center_x": 0.3,
            "bin_center_y": -0.2,
            "bin_height": 0.85,
            "radius": 0.15,
        },
    )

    # Large bonus for successful completion
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=10.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "bin_center_x": 0.3,
            "bin_center_y": -0.2,
            "bin_height": 0.85,
            "radius": 0.15,
        },
    )

    # Penalty for excessive force (crushing)
    force_penalty = RewTerm(
        func=mdp.excessive_force_penalty,
        weight=-0.3,
        params={
            "force_threshold": 15.0,
        },
    )

    # Small reward during approach for keeping fingers ready
    approach_reward = RewTerm(
        func=mdp.approach_phase_reward,
        weight=0.1,
    )


##
# Termination Configuration
##

@configclass
class TerminationsCfg:
    """Termination conditions for the task."""

    # Standard timeout
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Success: cylinder placed in bin
    success = DoneTerm(
        func=mdp.cylinder_in_bin,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "bin_center_x": 0.3,
            "bin_center_y": -0.2,
            "bin_height": 0.85,
            "radius": 0.15,
        },
    )

    # Failure: cylinder dropped below table
    dropped = DoneTerm(
        func=mdp.cylinder_dropped,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "min_height": 0.5,
        },
    )

    # Failure: cylinder out of workspace
    out_of_bounds = DoneTerm(
        func=mdp.cylinder_out_of_bounds,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "x_range": (-1.0, 1.0),
            "y_range": (-0.5, 1.5),
        },
    )


##
# Main Environment Configuration
##

@configclass
class LearnGraspCylinderEnvCfg(ManagerBasedRLEnvCfg):
    """Learn-to-Grasp environment configuration.

    Policy learns to control right hand fingers for grasping.
    Arm movement is scripted through task phases.
    """

    # Scene with robot, table, and cylinder
    scene: LearnGraspSceneCfg = LearnGraspSceneCfg(
        num_envs=256,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # No commands needed (phases handled by events)
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization settings."""
        # Physics settings
        self.sim.dt = 0.005  # 200 Hz physics
        self.decimation = 4  # 50 Hz control
        self.sim.render_interval = self.decimation

        # Episode length: 10 seconds (covers all phases)
        self.episode_length_s = 10.0

        # Physics parameters
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Contact sensor update
        self.scene.contact_forces.update_period = self.sim.dt

        # Friction settings for grasping
        self.sim.physics_material.static_friction = 1.2
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"


@configclass
class LearnGraspCylinderEnvCfg_PLAY(LearnGraspCylinderEnvCfg):
    """Play/evaluation configuration with 1 environment."""

    scene: LearnGraspSceneCfg = LearnGraspSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        # Render every step for smooth visualization
        self.sim.render_interval = 1
