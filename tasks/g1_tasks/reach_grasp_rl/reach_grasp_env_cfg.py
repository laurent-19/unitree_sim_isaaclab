# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Environment config for G1 right arm reach and grasp task.

Uses IK-based actions to reach a goal, hardcoded grasp, then lift.
Based on examples-issac-lab/point_nav_env_cfg.py patterns.
"""

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CommandTermCfg as CommandTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import robot config (loaded via direct module import in train script to avoid auto-importer)
from tasks.common_config import G1RobotPresets

# Import MDP helpers (loaded via direct module import in train script)
from . import reach_mdp


# Constants - Right arm joints (4 DOF for IK)
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

EE_BODY_NAME = "R_thumb_proximal"  # Fingertip as EE for grasping
SHOULDER_BODY_NAME = "right_shoulder_pitch_link"  # For reachability checking
EE_COMMAND_NAME = "ee_pose"

# Reachable shell around shoulder (meters)
# r_max should be slightly smaller than real shoulder->palm reach
REACH_R_MIN = 0.12
REACH_R_MAX = 0.48
GOAL_MAX_TRIES = 80

# Joints to lock (everything except right arm)
LOCKED_JOINTS = [
    "waist_.*",
    "left_hip_.*", "left_knee_.*", "left_ankle_.*",
    "right_hip_.*", "right_knee_.*", "right_ankle_.*",
    "left_shoulder_.*", "left_elbow_.*",
    "left_wrist_.*",
]

# Right hand joints for grasping
RIGHT_HAND_JOINTS = [
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

HIGH_FRICTION_MAT = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=0.0,
)


# -----------------------------------------------------------------------------
# Scene (following example setup)
# -----------------------------------------------------------------------------

@configclass
class G1ReachGraspSceneCfg(InteractiveSceneCfg):
    """Scene for right arm reach and grasp.

    Setup matches examples-issac-lab/point_nav_env_cfg.py but for right arm.
    Robot is rotated 90 degrees so it faces +Y direction.
    Goal coordinates are in robot root frame.
    """

    # Ground + light
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0)),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/Light/Key",
        spawn=sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.92), angle=2.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
    )

    # Table (fixed/kinematic) - using packing table from Isaac assets
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Table positioned in front of robot (robot faces +Y after rotation)
            pos=(1.0, 0.1, -0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Grasp cylinder - placed at fixed position on table
    # Table surface is at z≈1.0 (world), cylinder half-height is 0.1, so center at z≈1.1
    # Position: in front of robot, reachable by right arm
    grasp_cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.2,
            physics_material=HIGH_FRICTION_MAT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Fixed position on table - robot at (1.1, 0.5, 0.8) rotated -90° about Z
            # Cylinder placed within reach: world (0.95, 0.15, 1.0)
            # This is ~35cm forward, ~35cm to the right in robot frame
            pos=(0.95, 0.15, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Robot - G1 with Inspire hand, fixed base
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # Position robot (matching example setup)
    # Robot at (1.1, 0.5, 0.8) rotated 90 degrees around Z (facing +Y)
    # Mirrored from example: example has robot at y=-0.5 for left arm
    # For right arm, we put robot at y=+0.5 so right arm faces the table
    robot.init_state.pos = (1.1, 0.5, 0.8)
    robot.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)  # wxyz: 90 deg rotation, right arm faces table


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # CRITICAL: Ensure goals are kinematically reachable
    # Workspace is in robot root frame (x=forward, y=left, z=up)
    # For right arm: y should be negative (right side of robot)
    fix_unreachable_goal_on_reset = EventTerm(
        func=reach_mdp.ensure_goal_reachable_box_sphere,
        mode="reset",
        params={
            "command_name": EE_COMMAND_NAME,
            "shoulder_cfg": SceneEntityCfg("robot", body_names=[SHOULDER_BODY_NAME]),
            # Right arm workspace - matching fixed cylinder position
            "x_range": (0.30, 0.40),    # Forward from robot (~0.35)
            "y_range": (-0.20, -0.10),  # Right side (~-0.15)
            "z_range": (0.15, 0.25),    # Above robot base (~0.2)
            "r_min": REACH_R_MIN,
            "r_max": REACH_R_MAX,
            "max_tries": GOAL_MAX_TRIES,
        },
    )

    # Lock non-right-arm joints
    lock_other_joints = EventTerm(
        func=reach_mdp.hold_joints_at_default_targets,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOCKED_JOINTS)},
    )

    reset_dwell_steps = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_ee_dwell_steps", "dtype": "int"},
    )

    # Cylinder is now at a fixed position - no dynamic placement needed
    # The cylinder is placed at (0.95, 0.15, 1.0) in world frame at scene init
    # Goal command will point to this fixed position

    reset_ee_still = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_ee_still_steps", "dtype": "int"},
    )

    # Hardcoded grasp: open until reach/stall, then close
    hold_hand_open_then_close = EventTerm(
        func=reach_mdp.hold_hand_open_then_close_on_reach,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "hand_joint_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS),
            "open_pos": {
                "R_index_proximal_joint": 0.0,
                "R_index_intermediate_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_middle_intermediate_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_ring_intermediate_joint": 0.0,
                "R_pinky_proximal_joint": 0.0,
                "R_pinky_intermediate_joint": 0.0,
                "R_thumb_proximal_yaw_joint": 45.0,
                "R_thumb_proximal_pitch_joint": 10.0,
                "R_thumb_intermediate_joint": 0.0,
                "R_thumb_distal_joint": 0.0,
            },
            "close_pos": {
                "R_index_proximal_joint": 85.0,
                "R_index_intermediate_joint": 95.0,
                "R_middle_proximal_joint": 85.0,
                "R_middle_intermediate_joint": 95.0,
                "R_ring_proximal_joint": 85.0,
                "R_ring_intermediate_joint": 95.0,
                "R_pinky_proximal_joint": 85.0,
                "R_pinky_intermediate_joint": 95.0,
                "R_thumb_proximal_yaw_joint": 45.0,
                "R_thumb_proximal_pitch_joint": 10.0,
                "R_thumb_intermediate_joint": 65.0,
                "R_thumb_distal_joint": 75.0,
            },
            "close_ramp_steps": 30,
            "close_pos_is_degrees": True,
            "command_name": EE_COMMAND_NAME,
            "reach_threshold": 0.05,
            "latch_buffer_name": "_hand_closed",
            "stop_only": True,
            "min_steps_before_stall": 25,
            "stall_vel_threshold": 0.02,
            "stall_steps": 8,
        },
    )

    reset_hand_closed = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_hand_closed", "dtype": "bool"},
    )

    reset_hand_close_ramp = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_hand_close_ramp", "dtype": "int"},
    )

    reset_ee_total_steps = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_ee_total_steps", "dtype": "int"},
    )

    reset_ee_prev_pos = EventTerm(
        func=reach_mdp.reset_ee_prev_pos_buffer,
        mode="reset",
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "buffer_name": "_ee_prev_pos_b",
        },
    )

    # Switch goal after grasp (for lifting)
    # Target position is in robot root frame
    switch_goal_after_close = EventTerm(
        func=reach_mdp.switch_goal_position_after_hand_close,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": EE_COMMAND_NAME,
            "target_pos": (0.3, -0.1, 0.25),  # Lift position in root frame (forward, right, up)
            "latch_buffer_name": "_hand_closed",
            "wait_for_close_ramp": True,
            "close_ramp_buffer_name": "_hand_close_ramp",
            "close_ramp_steps": 30,
            "once": True,
            "switched_buffer_name": "_goal_switched",
        },
    )

    reset_goal_switched = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_goal_switched", "dtype": "bool"},
    )

    # Open hand when reaching the second goal (release above bin)
    open_hand_after_second_goal = EventTerm(
        func=reach_mdp.open_hand_once_when_at_current_goal,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "hand_joint_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS),
            "command_name": EE_COMMAND_NAME,
            "goal_switched_buffer_name": "_goal_switched",
            "reach_threshold": 0.06,
            "open_pos": {
                "R_index_proximal_joint": 0.0,
                "R_index_intermediate_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_middle_intermediate_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_ring_intermediate_joint": 0.0,
                "R_pinky_proximal_joint": 0.0,
                "R_pinky_intermediate_joint": 0.0,
                "R_thumb_proximal_yaw_joint": 45.0,
                "R_thumb_proximal_pitch_joint": 10.0,
                "R_thumb_intermediate_joint": 0.0,
                "R_thumb_distal_joint": 0.0,
            },
            "open_pos_is_degrees": True,
            "open_ramp_steps": 20,
            "opened_buffer_name": "_hand_opened_drop",
            "ramp_buffer_name": "_hand_open_ramp",
            "start_buffer_name": "_hand_open_start",
        },
    )

    # Reset release buffers
    reset_hand_opened_drop = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_hand_opened_drop", "dtype": "bool"},
    )

    reset_hand_open_ramp = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_hand_open_ramp", "dtype": "int"},
    )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

@configclass
class CommandsCfg:
    # Goal pose in robot root frame - fixed to match cylinder position
    # Cylinder at world (0.95, 0.15, 1.0), robot at (1.1, 0.5, 0.8) rotated -90° about Z
    # In robot root frame: (0.35, -0.15, 0.2) approximately
    # Add small randomization to encourage generalization
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        resampling_time_range=(100.0, 100.0),  # Once per episode
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Fixed goal matching cylinder position with small range
            pos_x=(0.33, 0.37),     # Forward from robot (~0.35)
            pos_y=(-0.17, -0.13),   # Right side (~-0.15)
            pos_z=(0.18, 0.22),     # Above robot base (~0.2)
            # Fixed orientation
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    # IK-based task-space control
    # Policy outputs (dx, dy, dz) and IK turns it into joint targets
    arm_ik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        body_name=EE_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.07},
        ),
        scale=0.006,  # Meters per action unit (from example)
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # Goal pose in robot root frame: [x, y, z, qw, qx, qy, qz]
        goal_pose = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": EE_COMMAND_NAME}
        )

        # EE position in robot root frame: [x, y, z]
        ee_pos = ObsTerm(
            func=reach_mdp.body_pos_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME])},
        )

        # Arm joint state
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)}
        )

        # Last action
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "arm_ik"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # Success bonus when within threshold
    success_bonus = RewTerm(
        func=reach_mdp.success_bonus,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "threshold": 0.05,
        },
    )

    # Position error cost (main shaping)
    ee_pos_cost = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
        },
    )

    # Bounded tracking reward
    ee_pos_track = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "std": 0.12,
        },
    )

    # Smoothness penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# -----------------------------------------------------------------------------
# Env Config
# -----------------------------------------------------------------------------

@configclass
class G1ReachGraspEnvCfg(ManagerBasedRLEnvCfg):
    """RL env for right arm reach and grasp."""

    scene: G1ReachGraspSceneCfg = G1ReachGraspSceneCfg(num_envs=256, env_spacing=2.5)
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # Physics
        self.sim.dt = 0.01
        self.decimation = 2
        self.sim.render_interval = self.decimation

        # Episode
        self.episode_length_s = 6.0

        # Viewer (matching example setup - looking at robot from the side)
        self.viewer.lookat = (1.1, 0.5, 0.85)
        self.viewer.eye = (2.6, 0.5, 1.6)


@configclass
class G1ReachGraspEnvCfg_PLAY(G1ReachGraspEnvCfg):
    """Play config with 1 env."""

    scene: G1ReachGraspSceneCfg = G1ReachGraspSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = 1
