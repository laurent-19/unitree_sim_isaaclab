# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Inspire Hand Tactile DDS communication class.

Publishes tactile sensor data matching the real inspire_hand_touch format
to the rt/inspire/touch DDS topic.
"""

import sys
import os
from typing import Any, Dict, Optional

from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher

# Add inspire SDK to path for IDL import
# Try absolute path first (for Docker container), then relative
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
    from inspire_dds._inspire_hand_touch import inspire_hand_touch
    _IDL_AVAILABLE = True
    print(f"[inspire_touch_dds] ✓ IDL imported successfully from {sys.path[0]}")
except ImportError as e:
    _IDL_AVAILABLE = False
    print(f"[inspire_touch_dds] ✗ Warning: inspire_hand_touch IDL not available: {e}")
    print(f"[inspire_touch_dds] Python path: {sys.path[:3]}")


class InspireTouchDDS(DDSObject):
    """Publishes tactile data matching real inspire_hand_touch format.

    This class receives tactile data via shared memory from Isaac Lab
    and publishes it to the DDS topic rt/inspire_hand/touch/{l,r} in the same
    format as the real RH56DFTP hand's tactile sensors.
    """

    def __init__(self, node_name: str = "inspire_touch", lr: str = 'r'):
        """Initialize the Inspire Touch DDS node.

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

        # Setup shared memory for receiving tactile data from Isaac Lab
        self.setup_shared_memory(
            input_shm_name=f"isaac_inspire_touch_{self.lr}",
            input_size=8192,  # Sufficient for all taxel data
            output_shm_name=None,
            output_size=0,
            inputshm_flag=True,
            outputshm_flag=False,
        )

        print(f"[{self.node_name}] Inspire Touch DDS node initialized (side={self.lr})")

    def setup_publisher(self) -> bool:
        """Setup the DDS publisher for tactile data."""
        if not _IDL_AVAILABLE:
            print(f"[{self.node_name}] Cannot setup publisher: IDL not available")
            return False

        try:
            topic = f"rt/inspire_hand/touch/{self.lr}"
            self.publisher = ChannelPublisher(topic, inspire_hand_touch)
            self.publisher.Init()
            print(f"[{self.node_name}] Tactile publisher initialized on {topic}")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Failed to initialize publisher: {e}")
            return False

    def setup_subscriber(self) -> bool:
        """Setup subscriber (not used - output only)."""
        return True

    def dds_subscriber(self, msg: Any, datatype: str = None) -> None:
        """Process subscribe data (not used - output only)."""
        pass

    def dds_publisher(self) -> None:
        """Read tactile data from shared memory and publish to DDS."""
        if not _IDL_AVAILABLE:
            return
        if not hasattr(self, 'publisher'):
            return

        try:
            data = self.input_shm.read_data()
            if data is None:
                return

            msg = self._build_dds_message(data)
            if msg is not None:
                self.publisher.Write(msg)
                # Log first successful publish
                if not hasattr(self, '_first_publish_logged'):
                    print(f"[{self.node_name}] First tactile message published")
                    self._first_publish_logged = True

        except Exception as e:
            print(f"[{self.node_name}] Error in dds_publisher: {e}")

    def _build_dds_message(self, data: Dict[str, Any]) -> Optional[inspire_hand_touch]:
        """Build inspire_hand_touch message from tactile data dictionary.

        Args:
            data: Dictionary with keys matching IDL field names:
                  fingerone_tip_touch, fingerone_top_touch, fingerone_palm_touch,
                  fingertwo_*, fingerthree_*, fingerfour_*,
                  fingerfive_tip_touch, fingerfive_top_touch,
                  fingerfive_middle_touch, fingerfive_palm_touch,
                  palm_touch

        Returns:
            Populated inspire_hand_touch message or None if invalid data
        """
        if not _IDL_AVAILABLE:
            return None

        try:
            msg = inspire_hand_touch(
                fingerone_tip_touch=data.get("fingerone_tip_touch", [0] * 9),
                fingerone_top_touch=data.get("fingerone_top_touch", [0] * 96),
                fingerone_palm_touch=data.get("fingerone_palm_touch", [0] * 80),
                fingertwo_tip_touch=data.get("fingertwo_tip_touch", [0] * 9),
                fingertwo_top_touch=data.get("fingertwo_top_touch", [0] * 96),
                fingertwo_palm_touch=data.get("fingertwo_palm_touch", [0] * 80),
                fingerthree_tip_touch=data.get("fingerthree_tip_touch", [0] * 9),
                fingerthree_top_touch=data.get("fingerthree_top_touch", [0] * 96),
                fingerthree_palm_touch=data.get("fingerthree_palm_touch", [0] * 80),
                fingerfour_tip_touch=data.get("fingerfour_tip_touch", [0] * 9),
                fingerfour_top_touch=data.get("fingerfour_top_touch", [0] * 96),
                fingerfour_palm_touch=data.get("fingerfour_palm_touch", [0] * 80),
                fingerfive_tip_touch=data.get("fingerfive_tip_touch", [0] * 9),
                fingerfive_top_touch=data.get("fingerfive_top_touch", [0] * 96),
                fingerfive_middle_touch=data.get("fingerfive_middle_touch", [0] * 9),
                fingerfive_palm_touch=data.get("fingerfive_palm_touch", [0] * 96),
                palm_touch=data.get("palm_touch", [0] * 112),
            )
            return msg

        except Exception as e:
            print(f"[{self.node_name}] Error building DDS message: {e}")
            return None

    def write_tactile_data(self, tactile_dict: Dict[str, list]) -> None:
        """Write tactile data to shared memory for publishing.

        This is called by the tactile observation module to send data
        from Isaac Lab to the DDS publisher.

        Args:
            tactile_dict: Dictionary with tactile arrays keyed by IDL field names
        """
        try:
            if self.input_shm:
                self.input_shm.write_data(tactile_dict)
        except Exception as e:
            print(f"[{self.node_name}] Error writing tactile data: {e}")

    def get_tactile_data(self) -> Optional[Dict[str, list]]:
        """Read the current tactile data from shared memory.

        Returns:
            Dictionary of tactile arrays or None if unavailable
        """
        if self.input_shm:
            return self.input_shm.read_data()
        return None
