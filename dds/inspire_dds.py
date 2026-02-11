# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Inspire Hand DDS communication class using native Inspire IDL types.

Publishes hand state data and subscribes to control commands using the same
message types as the real Inspire hand SDK for compatibility.
"""

import sys
import os
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
import numpy as np

# Add inspire SDK to path for IDL import
_inspire_sdk_paths = [
    "/home/code/inspire_hand_ws/inspire_hand_sdk/inspire_sdkpy",  # Container path
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "inspire_hand_ws", "inspire_hand_sdk", "inspire_sdkpy"
    )
]
for _inspire_sdk_path in _inspire_sdk_paths:
    if os.path.exists(_inspire_sdk_path) and _inspire_sdk_path not in sys.path:
        sys.path.insert(0, _inspire_sdk_path)
        break

try:
    from inspire_dds._inspire_hand_state import inspire_hand_state
    from inspire_dds._inspire_hand_ctrl import inspire_hand_ctrl
    _IDL_AVAILABLE = True
    print(f"[inspire_dds] IDL imported successfully from {sys.path[0]}")
except ImportError as e:
    _IDL_AVAILABLE = False
    print(f"[inspire_dds] Warning: Inspire IDL not available: {e}")
    print(f"[inspire_dds] Python path: {sys.path[:3]}")


class InspireDDS(DDSObject):
    """Inspire Hand DDS communication class using native IDL types.

    Features:
    - Publish hand state to DDS (rt/inspire_hand/state/{l,r})
    - Receive control commands (rt/inspire_hand/ctrl/{l,r})
    - Compatible with real Inspire hand SDK examples
    """

    # Joint angle ranges for normalization (radians)
    # From documentation: fingers 20°-176°, thumb bend -13°-70°, thumb rot 90°-165°
    # Indices 0-3: finger joints (pinky, ring, middle, index)
    # Index 4: thumb bend, Index 5: thumb rotation
    JOINT_RANGES = {
        'finger': (0.0, 1.7),      # indices 0,1,2,3 (~20°-176° mapped)
        'thumb_rot': (0.0, 0.5),   # index 5 (thumb rotation)
        'thumb_flex': (-0.1, 1.3), # index 4 (thumb bend)
    }

    # Register documentation (RH56DFTP User Manual):
    # - ANGLE_ACT (1546-1557): Actual angle 0-1000, read-only
    # - FORCE_ACT (1582-1593): Force sensor reading in grams (-4000 to 4000), read-only
    # - CURRENT (1594-1605): Actuator current in mA (0-2000), read-only
    # - TEMP (1618-1623): Actuator temperature in °C (0-100), read-only

    def __init__(self, node_name: str = "inspire", lr: str = 'r'):
        """Initialize the Inspire Hand DDS node.

        Args:
            node_name: Name identifier for this DDS node
            lr: Hand side, 'l' for left or 'r' for right
        """
        if hasattr(self, '_initialized'):
            return

        super().__init__()
        self.node_name = node_name
        self.lr = lr.lower()
        if self.lr not in ('l', 'r'):
            raise ValueError(f"lr must be 'l' or 'r', got '{lr}'")

        self._initialized = True

        # Setup shared memory with side-specific names
        self.setup_shared_memory(
            input_shm_name=f"isaac_inspire_state_{self.lr}",
            input_size=1024,
            output_shm_name=f"isaac_inspire_cmd_{self.lr}",
            output_size=1024,
        )

        print(f"[{self.node_name}] Inspire Hand DDS node initialized (side={self.lr})")

    def setup_publisher(self) -> bool:
        """Setup the publisher for inspire_hand_state."""
        if not _IDL_AVAILABLE:
            print(f"[{self.node_name}] Cannot setup publisher: IDL not available")
            return False

        try:
            topic = f"rt/inspire_hand/state/{self.lr}"
            self.publisher = ChannelPublisher(topic, inspire_hand_state)
            self.publisher.Init()

            print(f"[{self.node_name}] State publisher initialized on {topic}")
            return True
        except Exception as e:
            print(f"[{self.node_name}] State publisher initialization failed: {e}")
            return False

    def setup_subscriber(self) -> bool:
        """Setup the subscriber for inspire_hand_ctrl."""
        if not _IDL_AVAILABLE:
            print(f"[{self.node_name}] Cannot setup subscriber: IDL not available")
            return False

        try:
            topic = f"rt/inspire_hand/ctrl/{self.lr}"
            self.subscriber = ChannelSubscriber(topic, inspire_hand_ctrl)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 32)

            print(f"[{self.node_name}] Control subscriber initialized on {topic}")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Control subscriber initialization failed: {e}")
            return False

    def _get_joint_range(self, idx: int) -> tuple:
        """Get the joint angle range for a given joint index (0-5)."""
        if idx in [0, 1, 2, 3]:
            return self.JOINT_RANGES['finger']
        elif idx == 4:
            return self.JOINT_RANGES['thumb_rot']
        else:  # idx == 5
            return self.JOINT_RANGES['thumb_flex']

    def _radians_to_inspire(self, rad_val: float, idx: int) -> int:
        """Convert radians to Inspire 0-1000 scale.

        Higher radians (more closed) -> lower Inspire value
        """
        min_val, max_val = self._get_joint_range(idx)
        normalized = np.clip((max_val - rad_val) / (max_val - min_val), 0.0, 1.0)
        return int(normalized * 1000)

    def _inspire_to_radians(self, inspire_val: int, idx: int) -> float:
        """Convert Inspire 0-1000 scale to radians.

        Higher Inspire value (more open) -> lower radians
        """
        min_val, max_val = self._get_joint_range(idx)
        normalized = np.clip(inspire_val / 1000.0, 0.0, 1.0)
        return (1.0 - normalized) * (max_val - min_val) + min_val

    def dds_publisher(self) -> Any:
        """Convert Isaac Lab state to inspire_hand_state and publish.

        Expected input data format from shared memory:
        {
            "positions": [6 joint positions in radians],
            "velocities": [6 joint velocities],
            "torques": [6 joint torques]
        }
        """
        if not _IDL_AVAILABLE:
            return
        if not hasattr(self, 'publisher'):
            return

        try:
            data = self.input_shm.read_data()
            if data is None:
                return

            if not all(key in data for key in ["positions", "velocities", "torques"]):
                return

            positions = data["positions"]
            velocities = data.get("velocities", [0.0] * 6)
            torques = data.get("torques", [0.0] * 6)
            contact_forces = data.get("contact_forces", None)  # Contact forces in Newtons

            # Build inspire_hand_state message
            pos_act = []
            angle_act = []
            force_act = []
            current = []
            err = []
            status = []
            temperature = []

            for i in range(min(6, len(positions))):
                # Convert position to 0-1000 scale
                inspire_val = self._radians_to_inspire(float(positions[i]), i)
                pos_act.append(inspire_val)
                angle_act.append(inspire_val)

                # FORCE_ACT: Force in grams, range -4000 to 4000 (from documentation)
                if contact_forces is not None and i < len(contact_forces):
                    # Use contact force from physics simulation
                    # Convert Newtons to grams: 1 N = 101.97 g ≈ 102 g
                    force_newtons = float(contact_forces[i])
                    force_grams = int(force_newtons * 102)
                    force_val = int(np.clip(force_grams, -4000, 4000))
                else:
                    # Fallback: estimate from torque (less accurate)
                    force_val = int(np.clip(float(torques[i]) * 500, -4000, 4000))
                force_act.append(force_val)

                # CURRENT: Actuator current in mA (0-2000), measured by real hand electronics
                # Cannot simulate - set to 0 (N/A)
                current.append(0)

                # Error and status flags
                err.append(0)
                status.append(0)

                # TEMP: Actuator temperature in °C (0-100), measured by real hand sensors
                # Cannot simulate - set to 0 (N/A)
                temperature.append(0)

            # Pad to 6 elements if needed
            while len(pos_act) < 6:
                pos_act.append(0)
                angle_act.append(0)
                force_act.append(0)
                current.append(0)
                err.append(0)
                status.append(0)
                temperature.append(25)

            msg = inspire_hand_state(
                pos_act=pos_act,
                angle_act=angle_act,
                force_act=force_act,
                current=current,
                err=err,
                status=status,
                temperature=temperature,
            )

            self.publisher.Write(msg)

        except Exception as e:
            print(f"[{self.node_name}] Error in dds_publisher: {e}")

    def dds_subscriber(self, msg: 'inspire_hand_ctrl', datatype: str = None) -> Dict[str, Any]:
        """Convert inspire_hand_ctrl to Isaac Lab commands.

        Writes to shared memory:
        {
            "positions": [6 joint position targets in radians],
            "velocities": [6 velocity limits],
            "torques": [6 torque limits],
            "kp": [6 position gains],
            "kd": [6 velocity gains],
            "mode": control mode bitfield
        }
        """
        try:
            cmd_data = {
                "positions": [],
                "velocities": [],
                "torques": [],
                "kp": [],
                "kd": [],
                "mode": int(msg.mode)
            }

            # Convert angle_set (0-1000) to radians
            for i in range(min(6, len(msg.angle_set))):
                joint_angle = self._inspire_to_radians(int(msg.angle_set[i]), i)
                cmd_data["positions"].append(joint_angle)

                # Map speed_set to velocity limit
                speed_limit = float(msg.speed_set[i]) / 1000.0 * 2.0 if i < len(msg.speed_set) else 1.0
                cmd_data["velocities"].append(speed_limit)

                # Map force_set to torque limit
                force_limit = float(msg.force_set[i]) / 1000.0 * 1.0 if i < len(msg.force_set) else 0.5
                cmd_data["torques"].append(force_limit)

                # Default gains
                cmd_data["kp"].append(100.0)
                cmd_data["kd"].append(10.0)

            self.output_shm.write_data(cmd_data)

        except Exception as e:
            print(f"[{self.node_name}] Error in dds_subscriber: {e}")

    def get_inspire_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the hand control command from shared memory.

        Returns:
            Dict: the hand command, or None if no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def write_inspire_state(self, positions, velocities, torques, contact_forces=None):
        """Write hand state to shared memory for publishing.

        Args:
            positions: joint positions (list or tensor, 6 values in radians)
            velocities: joint velocities (list or tensor)
            torques: joint torques (list or tensor)
            contact_forces: contact forces in Newtons (list or tensor, 6 values)
                           Used for force_act field. If None, falls back to torque estimate.
        """
        try:
            inspire_hand_data = {
                "positions": positions.tolist() if hasattr(positions, 'tolist') else list(positions),
                "velocities": velocities.tolist() if hasattr(velocities, 'tolist') else list(velocities),
                "torques": torques.tolist() if hasattr(torques, 'tolist') else list(torques),
            }

            # Add contact forces if provided
            if contact_forces is not None:
                inspire_hand_data["contact_forces"] = (
                    contact_forces.tolist() if hasattr(contact_forces, 'tolist') else list(contact_forces)
                )

            if self.input_shm:
                self.input_shm.write_data(inspire_hand_data)

        except Exception as e:
            print(f"[{self.node_name}] Error writing inspire hand state: {e}")
