# point_nav_env_cfg.py
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
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

# Unitree G1
from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip

# Your custom helpers (we replace reach_mdp.py below)
import reach_mdp

HIGH_FRICTION_MAT = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=0.0,
)

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

ARM_JOINTS = LEFT_ARM_JOINTS

EE_BODY_NAME = "left_palm_link"

# Sphere center (in robot root frame) = shoulder link
SHOULDER_BODY_NAME = "left_shoulder_pitch_link"  # <-- verify this name exists in your G1 asset

# Reachable “shell” around shoulder (meters)
# r_max should be slightly smaller than your real shoulder->palm reach to keep goals solvable.
REACH_R_MIN = 0.12
REACH_R_MAX = 0.48

# Rejection sampling tries (per env) if a goal is invalid
GOAL_MAX_TRIES = 80

EE_COMMAND_NAME = "ee_pose"


# Everything we want frozen (regex expressions are allowed here) :contentReference[oaicite:2]{index=2}
LOCKED_JOINTS = [
    # torso / waist (common name variants)
    "torso_.*",
    "waist_.*",
    "trunk_.*",
    "pelvis_.*",

    # legs (Unitree-ish naming + common variants)
    "left_hip_.*", "left_knee_.*", "left_ankle_.*",
    "right_hip_.*", "right_knee_.*", "right_ankle_.*",
    ".*_thigh_.*", ".*_calf_.*", ".*_foot_.*",

    # optional: lock the whole right side + head so literally only left arm can move 
    "head_.*",
    "neck_.*",
]


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

@configclass
class G1ReachPoseSceneCfg(InteractiveSceneCfg):
    """Scene for 6D reach-to-pose with left hand."""
    # Ground + light
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # --- lighting ---
    dome_light = AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0)),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/Light/Key",
        spawn=sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 0.98, 0.92), angle=2.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
    )

    # --- table (fixed/kinematic) ---
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
            pos=(1.0, 0.1, -0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    grasp_cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.2,
            physics_material_path="/World/PhysicsMaterials/HighFriction",
            physics_material=HIGH_FRICTION_MAT,
            activate_contact_sensors=True,   # <-- ADD THIS
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

        # --- basket / bin (fixed) ---
    klt_bin: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/KLTBin",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,   # keep the bin fixed in place
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # TODO: tune this so it sits nicely on the table top
            # start near the table and adjust Z until it's resting on the surface visually
            pos=(1.1, -0.1, 0.78),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


    # Robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    robot.spawn.articulation_props.fix_root_link = True


    # your pose
    robot.init_state.pos = (1.1, -0.5, 0.71)
    robot.init_state.rot = (0.7071, 0.0, 0.0, 0.7071)  # wxyz

    left_palm_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_palm_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table", "{ENV_REGEX_NS}/Table/.*"],
    )

    # finger ↔ table contact sensors
    left_zero_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_zero_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_one_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_one_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_two_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_two_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_three_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_three_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_four_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_four_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_five_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_five_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    left_six_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_six_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )


    left_palm_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_palm_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder", "{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_zero_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_zero_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_one_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_one_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_two_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_two_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_three_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_three_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_four_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_four_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_five_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_five_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )

    left_six_cyl_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_six_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/GraspCylinder/.*"],
    )
    

# -----------------------------------------------------------------------------
# Events: hard-lock non-left-arm joints by snapping them to default each step
# -----------------------------------------------------------------------------

@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # NEW: ensure the sampled ee_pose goal is actually reachable
    fix_unreachable_goal_on_reset = EventTerm(
        func=reach_mdp.ensure_goal_reachable_box_sphere,
        mode="reset",
        params={
            "command_name": EE_COMMAND_NAME,
            "shoulder_cfg": SceneEntityCfg("robot", body_names=[SHOULDER_BODY_NAME]),
            "x_range": (0.3, 0.3),
            "y_range": (0.1, 0.2),
            "z_range": (0.05, 0.05),
            "r_min": REACH_R_MIN,
            "r_max": REACH_R_MAX,
            "max_tries": GOAL_MAX_TRIES,
        },
    )

    

    lock_non_left_arm = EventTerm(
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

    place_grasp_cylinder = EventTerm(
        func=reach_mdp.place_object_near_ee_goal_once,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "robot_cfg": SceneEntityCfg("robot"),
            "command_name": EE_COMMAND_NAME,
            "offset_root": (0.06, -0.03, 0.0),
            "placed_buffer_name": "_cyl_placed",
        },
    )


    reset_cyl_placed = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_cyl_placed", "dtype": "bool"},
    )

    reset_ee_still = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_ee_still_steps", "dtype": "int"},
    )

    # Hold hand joints open until we reach the EE goal, then close (latched)
    hold_hand_open_then_close = EventTerm(
        func=reach_mdp.hold_hand_open_then_close_on_reach,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "hand_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_zero_joint",
                    "left_one_joint",
                    "left_two_joint",
                    "left_three_joint",
                    "left_four_joint",
                    "left_five_joint",
                    "left_six_joint",
                ],
            ),
            "open_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "close_pos": (0.0, 50.0, 80.0, -65.0, -100.0, -65.0, -100.0),
            "close_ramp_steps": 30,
            "close_pos_is_degrees": True,
            "command_name": EE_COMMAND_NAME,
            "reach_threshold": 0.05,
            "close_pos_is_degrees": True,
            "latch_buffer_name": "_hand_closed",
            "stop_only": True,
            "min_steps_before_stall": 25,     # ~0.5s (given your step_dt ~0.02)
            "stall_vel_threshold": 0.02,      # bump up if it's jittering and never "stops"
            "stall_steps": 8,                 # how many consecutive steps to count as stopped
        },
    )

    # Reset the hand latch + ramp each episode
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

    # Reset the step counter used by the same hand helper
    reset_ee_total_steps = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_ee_total_steps", "dtype": "int"},
    )

    # Reset prev EE position buffer (3D)
    reset_ee_prev_pos = EventTerm(
        func=reach_mdp.reset_ee_prev_pos_buffer,
        mode="reset",
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "buffer_name": "_ee_prev_pos_b",
        },
    )

    switch_goal_after_close = EventTerm(
        func=reach_mdp.switch_goal_position_after_hand_close,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": EE_COMMAND_NAME,
            "target_pos": (0.3, 0.05, 0.25),

            # Only AFTER the hand is actually closed
            "latch_buffer_name": "_hand_closed",
            "wait_for_close_ramp": True,
            "close_ramp_buffer_name": "_hand_close_ramp",
            "close_ramp_steps": 30,   # must match your close_ramp_steps in the hand close event

            # Switch once per episode
            "once": True,
            "switched_buffer_name": "_goal_switched",
        },
    )

        # Reset the "goal switched" latch each episode
    reset_goal_switched = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_goal_switched", "dtype": "bool"},
    )


    open_hand_after_second_goal = EventTerm(
        func=reach_mdp.open_hand_once_when_at_current_goal,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "hand_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_zero_joint",
                    "left_one_joint",
                    "left_two_joint",
                    "left_three_joint",
                    "left_four_joint",
                    "left_five_joint",
                    "left_six_joint",
                ],
            ),
            "command_name": EE_COMMAND_NAME,

            # only open once we're on goal #2
            "goal_switched_buffer_name": "_goal_switched",

            # open when we reach the second goal
            "reach_threshold": 0.06,

            # targets (same open you already use)
            "open_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "open_pos_is_degrees": True,

            # smooth opening
            "open_ramp_steps": 20,

            # buffers
            "opened_buffer_name": "_hand_opened_drop",
            "ramp_buffer_name": "_hand_open_ramp",
            "start_buffer_name": "_hand_open_start",
        },
    )

    reset_goal_switched = EventTerm(
        func=reach_mdp.reset_named_buffer,
        mode="reset",
        params={"buffer_name": "_goal_switched", "dtype": "bool"},
    ) 

    # --- open_hand_after_second_goal: reset its latches each episode ---
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

    reset_open_hand_after_second_goal_on_episode_start = EventTerm(
        func=reach_mdp.reset_open_hand_after_second_goal_buffers_on_episode_start,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "hand_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_zero_joint","left_one_joint","left_two_joint",
                    "left_three_joint","left_four_joint","left_five_joint","left_six_joint",
                ],
            ),
            "goal_switched_buffer_name": "_goal_switched",
            "opened_buffer_name": "_hand_opened_drop",
            "ramp_buffer_name": "_hand_open_ramp",
            "start_buffer_name": "_hand_open_start",
        },
    )


    reset_hand_open_start = EventTerm(
        func=reach_mdp.reset_hand_open_start_buffer,
        mode="reset",
        params={
            "hand_joint_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_zero_joint","left_one_joint","left_two_joint",
                    "left_three_joint","left_four_joint","left_five_joint","left_six_joint",
                ],
            ),
            "buffer_name": "_hand_open_start",
        },
    )
 







# -----------------------------------------------------------------------------
# Commands: sample a goal pose (pos + orientation) in robot base frame
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    # Sample a random end-effector pose goal (expressed in the robot base frame).
    # This is the "target point" the hand should reach.
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        resampling_time_range=(100.0, 100.0),  # effectively once per episode (episode_length_s=6.0)
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Tune these to your G1's reachable workspace.
            pos_x=(0.3, 0.3),
            pos_y=(0.0, 0.3),
            pos_z=(0.05, 0.2),
            # Keep orientation fixed for now (reaching a point). You can widen later.
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    


# -----------------------------------------------------------------------------
# Actions: RL outputs delta pose; Diff-IK turns it into joint targets
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    # Task-space action: policy outputs (dx, dy, dz) and the controller turns it into arm joint targets.
    # This is vastly easier than learning raw joint-space reaching.
    arm_ik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=EE_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.07},  # try 0.03 → 0.1
        ),
        scale=0.006,  # meters per action unit (tune: 0.05–0.15)
    )

    


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Observation terms for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Goal (in base frame): [x, y, z, qw, qx, qy, qz]
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": EE_COMMAND_NAME})

        # End-effector position in the robot base frame: [x, y, z]
        ee_pos = ObsTerm(
            func=reach_mdp.body_pos_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME])},
        )

        # Arm joint state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})

        # Last action
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_ik"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

        
    

# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    # Big-ish terminal reward (doesn't need to be insane if your per-step is a cost)
    success_bonus = RewTerm(
        func=reach_mdp.success_bonus,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "threshold": 0.05,
        },
    )

    # Main shaping as a COST (negative): every step you haven't finished hurts a bit
    ee_pos_cost = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
        },
    )

    # Keep a *small* bounded shaping reward if you want (optional)
    ee_pos_track = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "std": 0.12,
        },
    )

    # Make jitter actually expensive
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-4)

    # Re-introduce smoothness (these are your anti-flail seatbelts)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )

    table_collision = RewTerm(
        func=reach_mdp.contact_force_reward_any,
        weight=-2.0,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("left_palm_table_contact"),
                SceneEntityCfg("left_zero_table_contact"),
                SceneEntityCfg("left_one_table_contact"),
                SceneEntityCfg("left_two_table_contact"),
                SceneEntityCfg("left_three_table_contact"),
                SceneEntityCfg("left_four_table_contact"),
                SceneEntityCfg("left_five_table_contact"),
                SceneEntityCfg("left_six_table_contact"),
            ],
            "min_force": 2.0,
            "max_force": 30.0,
        },
    ) 

    cyl_push_penalty = RewTerm(
        func=reach_mdp.contact_force_reward_any,
        weight=-3.0,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("left_palm_cyl_contact"),
                SceneEntityCfg("left_zero_cyl_contact"),
                SceneEntityCfg("left_one_cyl_contact"),
                SceneEntityCfg("left_two_cyl_contact"),
                SceneEntityCfg("left_three_cyl_contact"),
                SceneEntityCfg("left_four_cyl_contact"),
                SceneEntityCfg("left_five_cyl_contact"),
                SceneEntityCfg("left_six_cyl_contact"),
            ],
            "min_force": 0.5,
            "max_force": 20.0,
        },
    )

    cyl_upright_penalty = RewTerm(
        func=reach_mdp.object_upright_penalty_gated,
        weight=-4.0,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "pos_std": 0.18,
            "tilt_std": 0.25,
        },
    )



# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    # Standard time-out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success termination: end episode when palm is close enough to the goal
    """ success = DoneTerm(
        func=reach_mdp.reached_and_still_dwell,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "command_name": EE_COMMAND_NAME,
            "pos_threshold": 0.05,
            "speed_threshold": 0.03,
            "dwell_steps": 10,
            "buffer_name": "_ee_dwell_steps",
        },
    ) """


    """ too_fast = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 8.0,  # start 6–10 rad/s for arm joints
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )  """

    """ out_of_pos_limits = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
    ) """

    effort_clip = DoneTerm(
        func=mdp.joint_effort_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )

    table_smash = DoneTerm(
        func=reach_mdp.contact_force_exceeds_any,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("left_palm_table_contact"),
                SceneEntityCfg("left_zero_table_contact"),
                SceneEntityCfg("left_one_table_contact"),
                SceneEntityCfg("left_two_table_contact"),
                SceneEntityCfg("left_three_table_contact"),
                SceneEntityCfg("left_four_table_contact"),
                SceneEntityCfg("left_five_table_contact"),
                SceneEntityCfg("left_six_table_contact"),
            ],
            "force_thresh": 80.0,
        },
    )

    """ cyl_tipped = DoneTerm(
        func=reach_mdp.object_tipped,
        params={
            "object_cfg": SceneEntityCfg("grasp_cylinder"),
            "cos_thresh": 0.93,  # ~21 degrees
        },
    ) """



    

    


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------

@configclass
class G1ReachPoseEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL env for 6D pose reaching with the left hand."""
    scene: G1ReachPoseSceneCfg = G1ReachPoseSceneCfg(num_envs=264, env_spacing=2.5)
    events: EventsCfg = EventsCfg()   # <-- ADD THIS LINE
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
        else:
            self.scene.robot.spawn.articulation_props.fix_root_link = True
        # physics + stepping
        self.sim.dt = 0.01
        self.decimation = 2
        self.sim.render_interval = self.decimation

        # episode horizon
        self.episode_length_s = 6.0

        # viewer
        self.viewer.lookat = (1.1, -0.5, 0.85)
        self.viewer.eye    = (2.6, -0.5, 1.6)



@configclass
class G1ReachPoseEnvCfg_PLAY(G1ReachPoseEnvCfg):
    """Play config: 1 env + rendering enabled."""
    scene: G1ReachPoseSceneCfg = G1ReachPoseSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # Render every physics step so you can see it smoothly
        self.sim.render_interval = 1

        # (Optional) make the camera nicer
        self.viewer.lookat = (1.1, -0.5, 0.85)
        self.viewer.eye    = (2.6, -0.5, 1.6)
