#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Subscribe to simulated Inspire Hand tactile data from Isaac Lab.

Usage:
    python scripts/subscribe_inspire_touch.py [--network INTERFACE]

Example:
    python scripts/subscribe_inspire_touch.py --network lo
"""

import sys
import os
import time
import signal
import argparse
import numpy as np

# Add inspire SDK to path
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "inspire_hand_ws", "inspire_hand_sdk", "inspire_sdkpy"
))

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

try:
    from inspire_dds._inspire_hand_touch import inspire_hand_touch
except ImportError:
    print("Error: Could not import inspire_hand_touch IDL")
    print("Make sure inspire_hand_ws/inspire_hand_sdk is properly built")
    sys.exit(1)


class TactileSubscriber:
    """Simple subscriber for simulated Inspire hand tactile data."""

    # Finger region sizes matching the IDL
    REGION_SIZES = {
        "fingerone_tip_touch": (3, 3),      # 9
        "fingerone_top_touch": (12, 8),     # 96
        "fingerone_palm_touch": (10, 8),    # 80
        "fingertwo_tip_touch": (3, 3),
        "fingertwo_top_touch": (12, 8),
        "fingertwo_palm_touch": (10, 8),
        "fingerthree_tip_touch": (3, 3),
        "fingerthree_top_touch": (12, 8),
        "fingerthree_palm_touch": (10, 8),
        "fingerfour_tip_touch": (3, 3),
        "fingerfour_top_touch": (12, 8),
        "fingerfour_palm_touch": (10, 8),
        "fingerfive_tip_touch": (3, 3),
        "fingerfive_top_touch": (12, 8),
        "fingerfive_middle_touch": (3, 3),  # 9
        "fingerfive_palm_touch": (12, 8),   # 96
        "palm_touch": (8, 14),              # 112
    }

    def __init__(self, network: str = "lo", topic: str = "rt/inspire/touch"):
        """Initialize the tactile subscriber.

        Args:
            network: Network interface (e.g., 'lo' for loopback, 'eth0')
            topic: DDS topic to subscribe to
        """
        # Use domain ID 1 to match the simulation (see sim_main.py)
        ChannelFactoryInitialize(1, network)

        self.topic = topic
        self.last_msg = None
        self.msg_count = 0
        self.last_time = time.time()

        self.subscriber = ChannelSubscriber(topic, inspire_hand_touch)
        self.subscriber.Init(self._on_message, 10)

        print(f"Subscribed to: {topic}")
        print("Waiting for tactile data...")

    def _on_message(self, msg: inspire_hand_touch):
        """Callback when a new tactile message is received."""
        self.last_msg = msg
        self.msg_count += 1

    def get_data(self) -> dict:
        """Get the latest tactile data as a dictionary of numpy arrays."""
        if self.last_msg is None:
            return None

        data = {}
        for field, shape in self.REGION_SIZES.items():
            values = getattr(self.last_msg, field, [])
            if values:
                data[field] = np.array(values).reshape(shape)
            else:
                data[field] = np.zeros(shape, dtype=np.int16)
        return data

    def print_summary(self):
        """Print a summary of the current tactile state."""
        data = self.get_data()
        if data is None:
            print("No data received yet")
            return

        now = time.time()
        rate = self.msg_count / (now - self.last_time) if now > self.last_time else 0

        print(f"\n{'='*60}")
        print(f"Messages: {self.msg_count} | Rate: {rate:.1f} Hz")
        print(f"{'='*60}")

        # Group by finger
        fingers = [
            ("Little (fingerone)", "fingerone"),
            ("Ring (fingertwo)", "fingertwo"),
            ("Middle (fingerthree)", "fingerthree"),
            ("Index (fingerfour)", "fingerfour"),
            ("Thumb (fingerfive)", "fingerfive"),
        ]

        for name, prefix in fingers:
            tip = data.get(f"{prefix}_tip_touch", np.zeros((3, 3)))
            top = data.get(f"{prefix}_top_touch", np.zeros((12, 8)))

            if prefix == "fingerfive":
                pad = data.get(f"{prefix}_palm_touch", np.zeros((12, 8)))
                mid = data.get(f"{prefix}_middle_touch", np.zeros((3, 3)))
                print(f"{name}: tip={tip.max():4d} top={top.max():4d} mid={mid.max():4d} pad={pad.max():4d}")
            else:
                pad = data.get(f"{prefix}_palm_touch", np.zeros((10, 8)))
                print(f"{name}: tip={tip.max():4d} top={top.max():4d} pad={pad.max():4d}")

        palm = data.get("palm_touch", np.zeros((8, 14)))
        print(f"Palm: max={palm.max():4d}")

    def print_detailed(self, finger: str = "fingerfour"):
        """Print detailed grid for a specific finger."""
        data = self.get_data()
        if data is None:
            return

        tip = data.get(f"{finger}_tip_touch")
        if tip is not None:
            print(f"\n{finger} tip (3x3):")
            print(tip)


def main():
    parser = argparse.ArgumentParser(description="Subscribe to Inspire hand tactile data")
    parser.add_argument("--network", default="enp0s31f6", help="Network interface (default: enp0s31f6, use 'lo' for same container)")
    parser.add_argument("--topic", default="rt/inspire/touch", help="DDS topic")
    parser.add_argument("--rate", type=float, default=2.0, help="Print rate in Hz")
    parser.add_argument("--detailed", action="store_true", help="Show detailed grid data")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    subscriber = TactileSubscriber(network=args.network, topic=args.topic)

    interval = 1.0 / args.rate
    try:
        while True:
            time.sleep(interval)
            subscriber.print_summary()
            if args.detailed:
                subscriber.print_detailed()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
