# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Scripted action provider for autonomous arm grasp demonstration.

Uses IK-based approach similar to the RL training examples, but with
hardcoded waypoints instead of policy outputs.
"""

from action_provider.action_base import ActionProvider
from typing import Optional
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms


class ScriptedActionProvider(ActionProvider):
    """Action provider using IK-based control to reach and grasp objects."""

    # Right arm joint names (for IK)
    RIGHT_ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # Right hand joints for grasping (Inspire hand)
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

    # Hand open positions (radians)
    HAND_OPEN = {
        "R_index_proximal_joint": 0.0,
        "R_index_intermediate_joint": 0.0,
        "R_middle_proximal_joint": 0.0,
        "R_middle_intermediate_joint": 0.0,
        "R_ring_proximal_joint": 0.0,
        "R_ring_intermediate_joint": 0.0,
        "R_pinky_proximal_joint": 0.0,
        "R_pinky_intermediate_joint": 0.0,
        "R_thumb_proximal_yaw_joint": 0.8,  # ~45 degrees
        "R_thumb_proximal_pitch_joint": 0.2,
        "R_thumb_intermediate_joint": 0.0,
        "R_thumb_distal_joint": 0.0,
    }

    # Hand closed positions (radians) for cylinder grasp
    HAND_CLOSED = {
        "R_index_proximal_joint": 1.5,
        "R_index_intermediate_joint": 1.7,
        "R_middle_proximal_joint": 1.5,
        "R_middle_intermediate_joint": 1.7,
        "R_ring_proximal_joint": 1.5,
        "R_ring_intermediate_joint": 1.7,
        "R_pinky_proximal_joint": 1.5,
        "R_pinky_intermediate_joint": 1.7,
        "R_thumb_proximal_yaw_joint": 0.8,
        "R_thumb_proximal_pitch_joint": 0.2,
        "R_thumb_intermediate_joint": 1.1,
        "R_thumb_distal_joint": 1.3,
    }

    # EE body name
    EE_BODY_NAME = "right_wrist_yaw_link"

    def __init__(self, env, args_cli):
        super().__init__("ScriptedActionProvider")
        self.env = env
        self.args_cli = args_cli
        self.device = env.device

        # Control parameters
        self.reach_threshold = 0.03  # 3cm to consider "reached"
        self.grasp_threshold = 0.05  # 5cm to start closing hand
        self.ik_scale = 0.05  # IK step size (bigger = faster movement)
        self.close_ramp_steps = 20  # Steps to ramp hand closed
        self.lift_height = 0.15  # Lift 15cm

        # State tracking
        self.step_count = 0
        self.hand_closing = False
        self.hand_close_step = 0
        self.lifting = False
        self.lift_target = None

        # Grasp offset from object center (hand approaches from above)
        self.grasp_offset = torch.tensor([0.0, 0.0, 0.05], device=self.device)  # 5cm above

        # Setup
        self._setup_joint_mapping()
        self._setup_ik_controller()

        print(f"\n{'='*60}")
        print("SCRIPTED GRASP PROVIDER INITIALIZED")
        print("="*60)
        print(f"  Reach threshold: {self.reach_threshold}m")
        print(f"  IK scale: {self.ik_scale}")
        print(f"  Lift height: {self.lift_height}m")
        print("="*60 + "\n")

    def _setup_joint_mapping(self):
        """Setup joint index mappings."""
        robot = self.env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        self.joint_to_idx = {name: i for i, name in enumerate(joint_names)}

        # Default positions
        self.default_pos = robot.data.default_joint_pos[0].clone()

        # Arm joint indices
        self.arm_joint_ids = []
        for name in self.RIGHT_ARM_JOINTS:
            if name in self.joint_to_idx:
                self.arm_joint_ids.append(self.joint_to_idx[name])
        self.arm_joint_ids_t = torch.tensor(self.arm_joint_ids, dtype=torch.long, device=self.device)

        # Hand joint indices and targets
        self.hand_joint_ids = []
        self.hand_open_targets = []
        self.hand_closed_targets = []
        for name in self.RIGHT_HAND_JOINTS:
            if name in self.joint_to_idx:
                self.hand_joint_ids.append(self.joint_to_idx[name])
                self.hand_open_targets.append(self.HAND_OPEN.get(name, 0.0))
                self.hand_closed_targets.append(self.HAND_CLOSED.get(name, 0.0))

        self.hand_joint_ids_t = torch.tensor(self.hand_joint_ids, dtype=torch.long, device=self.device)
        self.hand_open_t = torch.tensor(self.hand_open_targets, dtype=torch.float32, device=self.device)
        self.hand_closed_t = torch.tensor(self.hand_closed_targets, dtype=torch.float32, device=self.device)

        # Find EE body index
        body_names = list(robot.data.body_names)
        self.ee_body_idx = None
        for i, name in enumerate(body_names):
            if self.EE_BODY_NAME in name.lower():
                self.ee_body_idx = i
                break

        print(f"[{self.name}] Arm joints: {len(self.arm_joint_ids)}, Hand joints: {len(self.hand_joint_ids)}")
        print(f"[{self.name}] EE body: {self.EE_BODY_NAME} at index {self.ee_body_idx}")

    def _setup_ik_controller(self):
        """Setup IK controller for position-based control."""
        # Use relative mode like the examples - we'll feed delta positions
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,  # Delta position mode
            ik_method="dls",
            ik_params={"lambda_val": 0.07},
        )
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=self.device)
        print(f"[{self.name}] IK controller initialized (relative position mode)")

    def _get_ee_pose_base(self):
        """Get EE pose in robot base frame."""
        if self.ee_body_idx is None:
            return None, None

        robot = self.env.scene["robot"]
        ee_pose_w = robot.data.body_pose_w[:, self.ee_body_idx]
        ee_pos_w, ee_quat_w = ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]

        root_pose_w = robot.data.root_pose_w
        root_pos_w, root_quat_w = root_pose_w[:, 0:3], root_pose_w[:, 3:7]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        return ee_pos_b, ee_quat_b

    def _get_object_pos_base(self):
        """Get object position in robot base frame."""
        obj = self.env.scene["object"]
        robot = self.env.scene["robot"]

        obj_pos_w = obj.data.root_pos_w
        obj_quat_w = obj.data.root_quat_w

        root_pos_w = robot.data.root_pose_w[:, 0:3]
        root_quat_w = robot.data.root_pose_w[:, 3:7]

        obj_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, obj_pos_w, obj_quat_w
        )
        return obj_pos_b

    def _compute_ik_delta(self, delta_pos):
        """Compute joint position changes for a delta EE position."""
        robot = self.env.scene["robot"]

        ee_pos_b, ee_quat_b = self._get_ee_pose_base()
        if ee_pos_b is None:
            return None

        # Get Jacobian
        jacobian_full = robot.root_physx_view.get_jacobians()

        if robot.is_fixed_base:
            ee_jacobi_idx = self.ee_body_idx - 1
        else:
            ee_jacobi_idx = self.ee_body_idx

        # Position-only Jacobian (3 rows)
        jacobian = jacobian_full[:, ee_jacobi_idx, 0:3, self.arm_joint_ids_t]

        # Current joint positions
        joint_pos = robot.data.joint_pos[:, self.arm_joint_ids_t]

        # Set command (delta position in relative mode requires current EE pos)
        self.ik_controller.set_command(delta_pos, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        # Compute new joint positions
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        return joint_pos_des

    def _smoothstep(self, t):
        """Smooth interpolation."""
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    def get_action(self, env) -> Optional[torch.Tensor]:
        """Generate IK-based action to reach and grasp object."""
        robot = self.env.scene["robot"]

        # Start with default positions
        action = self.default_pos.clone()

        # Get current EE and object positions
        ee_pos_b, ee_quat_b = self._get_ee_pose_base()
        obj_pos_b = self._get_object_pos_base()

        if ee_pos_b is None or obj_pos_b is None:
            self.step_count += 1
            return action.unsqueeze(0)

        # Compute target (object + offset for grasp approach)
        if self.lifting and self.lift_target is not None:
            target_pos = self.lift_target
        else:
            target_pos = obj_pos_b + self.grasp_offset

        # Compute distance to target
        dist = torch.norm(ee_pos_b - target_pos).item()

        # Log at start and periodically
        if self.step_count == 0:
            print(f"\n{'='*60}")
            print("STARTING SCRIPTED GRASP")
            print("="*60)
            print(f"  EE (base):  [{ee_pos_b[0,0]:.3f}, {ee_pos_b[0,1]:.3f}, {ee_pos_b[0,2]:.3f}]")
            print(f"  Obj (base): [{obj_pos_b[0,0]:.3f}, {obj_pos_b[0,1]:.3f}, {obj_pos_b[0,2]:.3f}]")
            print(f"  Target:     [{target_pos[0,0]:.3f}, {target_pos[0,1]:.3f}, {target_pos[0,2]:.3f}]")
            print(f"  Distance:   {dist:.3f}m")
            print("="*60 + "\n")
            self.ik_controller.reset()

        # --- Phase 1: Reach toward target using IK ---
        if not self.lifting:
            # Compute direction to target
            direction = target_pos - ee_pos_b
            direction_norm = torch.norm(direction)

            if direction_norm > 0.001:
                # Normalize and scale
                delta = (direction / direction_norm) * self.ik_scale
                # Clamp to avoid overshooting
                delta = torch.clamp(delta, -self.ik_scale, self.ik_scale)
            else:
                delta = torch.zeros_like(direction)

            # Compute IK
            joint_pos_des = self._compute_ik_delta(delta)
            if joint_pos_des is not None:
                action[self.arm_joint_ids_t] = joint_pos_des[0]

        # --- Phase 2: Close hand when near object ---
        if dist < self.grasp_threshold and not self.hand_closing and not self.lifting:
            self.hand_closing = True
            self.hand_close_step = 0
            print(f"\n{'='*60}")
            print("CLOSING HAND - Near object!")
            print(f"  Distance: {dist:.3f}m")
            print("="*60 + "\n")

        if self.hand_closing:
            self.hand_close_step += 1
            alpha = self._smoothstep(self.hand_close_step / self.close_ramp_steps)
            hand_pos = self.hand_open_t + alpha * (self.hand_closed_t - self.hand_open_t)
            action[self.hand_joint_ids_t] = hand_pos

            # Start lifting after hand is mostly closed
            if self.hand_close_step >= self.close_ramp_steps and not self.lifting:
                self.lifting = True
                self.lift_target = ee_pos_b.clone()
                self.lift_target[:, 2] += self.lift_height
                print(f"\n{'='*60}")
                print("LIFTING - Hand closed!")
                print(f"  Lift target Z: {self.lift_target[0,2]:.3f}")
                print("="*60 + "\n")
        else:
            # Keep hand open
            action[self.hand_joint_ids_t] = self.hand_open_t

        # --- Phase 3: Lift using IK ---
        if self.lifting:
            # Keep hand closed
            action[self.hand_joint_ids_t] = self.hand_closed_t

            # Move toward lift target
            direction = self.lift_target - ee_pos_b
            direction_norm = torch.norm(direction)

            if direction_norm > 0.001:
                delta = (direction / direction_norm) * self.ik_scale
                delta = torch.clamp(delta, -self.ik_scale, self.ik_scale)
            else:
                delta = torch.zeros_like(direction)

            joint_pos_des = self._compute_ik_delta(delta)
            if joint_pos_des is not None:
                action[self.arm_joint_ids_t] = joint_pos_des[0]

        self.step_count += 1

        # Periodic logging
        if self.step_count % 100 == 0:
            phase = "lifting" if self.lifting else ("closing" if self.hand_closing else "reaching")
            print(f"[Step {self.step_count}] {phase}, dist={dist:.3f}, ee_z={ee_pos_b[0,2]:.3f}")

        return action.unsqueeze(0)

    def reset(self):
        """Reset for new episode."""
        self.step_count = 0
        self.hand_closing = False
        self.hand_close_step = 0
        self.lifting = False
        self.lift_target = None
        if hasattr(self, 'ik_controller'):
            self.ik_controller.reset()
        print(f"[{self.name}] Reset")

    def cleanup(self):
        """Cleanup."""
        print(f"[{self.name}] Cleanup complete")
