#!/usr/bin/env python3
"""
Quick DDS monitor to check what topics are active and receive messages.

Usage:
    python scripts/dds_monitor.py [--domain DOMAIN] [--interface IFACE]
"""
import sys
import os
import time
import argparse

# Add inspire SDK to path
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "inspire_hand_ws", "inspire_hand_sdk", "inspire_sdkpy"
))

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import String_

try:
    from inspire_dds._inspire_hand_touch import inspire_hand_touch
    HAS_INSPIRE_TOUCH = True
except ImportError:
    HAS_INSPIRE_TOUCH = False
    print("Warning: inspire_hand_touch IDL not available")


class TopicMonitor:
    def __init__(self):
        self.msg_counts = {}
        self.last_msg = {}
    
    def make_callback(self, topic_name):
        def cb(msg):
            self.msg_counts[topic_name] = self.msg_counts.get(topic_name, 0) + 1
            self.last_msg[topic_name] = msg
        return cb

    def print_status(self):
        print("\n" + "="*60)
        print(f"DDS Topic Status - {time.strftime('%H:%M:%S')}")
        print("="*60)
        if not self.msg_counts:
            print("No messages received on any topic")
        else:
            for topic, count in sorted(self.msg_counts.items()):
                print(f"  {topic}: {count} messages")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Monitor DDS topics")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain ID")
    parser.add_argument("--interface", default="", help="Network interface")
    args = parser.parse_args()

    print(f"Initializing DDS on domain {args.domain}")
    if args.interface:
        print(f"Using network interface: {args.interface}")
        ChannelFactoryInitialize(args.domain, args.interface)
    else:
        ChannelFactoryInitialize(args.domain)

    monitor = TopicMonitor()
    subscribers = []

    # Subscribe to known topics
    topics_to_monitor = [
        ("rt/inspire/touch", inspire_hand_touch if HAS_INSPIRE_TOUCH else None),
        ("rt/inspire/state", None),  # Would need MotorStates_ IDL
        ("rt/inspire/cmd", None),    # Would need MotorCmds_ IDL
        ("rt/reset_pose/cmd", String_),
        ("rt/lowstate", LowState_),
    ]

    for topic, msg_type in topics_to_monitor:
        if msg_type is not None:
            try:
                sub = ChannelSubscriber(topic, msg_type)
                sub.Init(monitor.make_callback(topic), 10)
                subscribers.append(sub)
                print(f"Subscribed to: {topic}")
            except Exception as e:
                print(f"Failed to subscribe to {topic}: {e}")

    print("\nMonitoring DDS traffic... (Ctrl+C to stop)")
    print("If simulation is running, you should see message counts increase.\n")

    try:
        while True:
            time.sleep(2)
            monitor.print_status()
    except KeyboardInterrupt:
        print("\nStopped monitoring")


if __name__ == "__main__":
    main()
