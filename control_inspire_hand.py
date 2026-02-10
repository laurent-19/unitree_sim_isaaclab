#!/usr/bin/env python3
"""
Control Inspire Hand via DDS
Run this script outside the container to send commands to the Inspire hand in simulation

Supports:
- Position control for all 12 motors
- Force feedback from contact sensors (FORCE_ACT register 1582)
"""

import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

# Finger names for display (maps to FORCE_ACT register 1582)
FINGER_NAMES = [
    "L_pinky", "L_ring", "L_middle", "L_index", "L_thumb_pitch", "L_thumb_yaw",
    "R_pinky", "R_ring", "R_middle", "R_index", "R_thumb_pitch", "R_thumb_yaw",
]

# Force sensor finger names (6 per hand)
FORCE_SENSOR_NAMES = [
    "L_index", "L_middle", "L_ring", "L_pinky", "L_thumb", "L_palm",
    "R_index", "R_middle", "R_ring", "R_pinky", "R_thumb", "R_palm",
]

class InspireHandController:
    def __init__(self):
        # Initialize DDS domain
        ChannelFactoryInitialize(1)  # Domain ID 1 (same as simulation)
        
        # Create command publisher
        self.cmd_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_pub.Init()
        
        # Create state subscriber
        self.state_sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self.state_sub.Init(self.state_callback, 10)
        
        self.latest_state = None
        self.contact_detected = False
        print("[InspireController] Initialized")
        print("[InspireController] Publishing to: rt/inspire/cmd")
        print("[InspireController] Subscribing to: rt/inspire/state")
        print("[InspireController] Force feedback enabled (FORCE_ACT register 1582)")
    
    def state_callback(self, msg):
        """Callback for state updates"""
        self.latest_state = msg
        
        # Check for contact forces (stored in temperature field, scaled by 100)
        if msg and msg.states:
            forces = [s.temperature / 100.0 for s in msg.states]
            self.contact_detected = any(f > 0.1 for f in forces)
    
    def get_contact_forces(self):
        """Get contact force values for each finger.
        
        Returns:
            list: Force values for 12 sensors (6 per hand), or None if no state
            
        Note: Maps to FORCE_ACT register (1582) on real Inspire hand.
        Force values are in Newtons (approximately).
        """
        if self.latest_state and self.latest_state.states:
            # Force is stored in temperature field, scaled by 100
            return [s.temperature / 100.0 for s in self.latest_state.states]
        return None
    
    def send_command(self, positions, kp=50.0, kd=5.0):
        """
        Send command to all 12 motors
        
        Args:
            positions: list of 12 normalized positions [0-1]
            kp: position gain (default: 50.0)
            kd: damping gain (default: 5.0)
        """
        cmd = MotorCmds_()
        cmd.cmds = []
        
        for i in range(12):
            motor_cmd = unitree_go_msg_dds__MotorCmd_()
            motor_cmd.q = float(positions[i])  # Normalized position [0-1]
            motor_cmd.dq = 0.0
            motor_cmd.tau = 0.0
            motor_cmd.kp = float(kp)
            motor_cmd.kd = float(kd)
            cmd.cmds.append(motor_cmd)
        
        self.cmd_pub.Write(cmd)
    
    def print_state(self, show_forces=True):
        """Print current state of all motors and force sensors"""
        if self.latest_state:
            print("\n=== Inspire Hand State ===")
            print(f"{'Motor':<20} {'Pos':>8} {'Vel':>8} {'Torque':>8}", end="")
            if show_forces:
                print(f" {'Force':>8}")
            else:
                print()
            print("-" * 60)
            
            for i, state in enumerate(self.latest_state.states):
                name = FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"Motor {i}"
                force = state.temperature / 100.0  # Force stored in temperature field
                print(f"{name:<20} {state.q:>8.3f} {state.dq:>8.3f} {state.tau_est:>8.3f}", end="")
                if show_forces:
                    force_bar = "█" * min(int(force * 10), 20)  # Visual bar
                    print(f" {force:>8.2f} {force_bar}")
                else:
                    print()
            
            if show_forces and self.contact_detected:
                print("\n⚠️  CONTACT DETECTED!")
        else:
            print("No state received yet")
    
    def print_forces_only(self):
        """Print only the force sensor values"""
        forces = self.get_contact_forces()
        if forces:
            print("\n=== Contact Forces (FORCE_ACT) ===")
            print("Left Hand:")
            for i in range(6):
                force_bar = "█" * min(int(forces[i] * 10), 30)
                print(f"  {FORCE_SENSOR_NAMES[i]:<12}: {forces[i]:>6.2f} N  {force_bar}")
            print("\nRight Hand:")
            for i in range(6, 12):
                force_bar = "█" * min(int(forces[i] * 10), 30)
                print(f"  {FORCE_SENSOR_NAMES[i]:<12}: {forces[i]:>6.2f} N  {force_bar}")
            print(f"\nTotal force: {sum(forces):.2f} N")
        else:
            print("No force data received yet")

def demo_open_close():
    """Demo: Open and close the hand with force feedback"""
    controller = InspireHandController()
    
    print("\n[Demo] Open and close hand (with force feedback)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Close hand (all motors to 1.0)
            print("Closing hand...")
            controller.send_command([1.0] * 12, kp=50.0, kd=5.0)
            time.sleep(2)
            controller.print_state(show_forces=True)
            
            # Open hand (all motors to 0.0)
            print("\nOpening hand...")
            controller.send_command([0.0] * 12, kp=50.0, kd=5.0)
            time.sleep(2)
            controller.print_state(show_forces=True)
            
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")

def demo_wave():
    """Demo: Wave fingers one by one"""
    controller = InspireHandController()
    
    print("\n[Demo] Wave fingers (with force feedback)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Wave each finger
            for i in range(12):
                positions = [0.0] * 12
                positions[i] = 1.0  # Close one finger
                print(f"Moving {FINGER_NAMES[i]}")
                controller.send_command(positions, kp=50.0, kd=5.0)
                time.sleep(0.3)
                controller.print_forces_only()
            
            # Reset all
            controller.send_command([0.0] * 12)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")

def demo_force_monitor():
    """Demo: Continuously monitor force sensors"""
    controller = InspireHandController()
    
    print("\n[Demo] Force sensor monitor")
    print("Displaying real-time contact forces (FORCE_ACT register 1582)")
    print("Press Ctrl+C to stop\n")
    
    # Start with hand partially closed
    controller.send_command([0.5] * 12, kp=50.0, kd=5.0)
    
    try:
        while True:
            # Clear screen and print forces
            print("\033[2J\033[H")  # Clear screen
            print("=== Real-time Force Monitor ===")
            print("(Move objects to see force changes)\n")
            controller.print_forces_only()
            
            if controller.contact_detected:
                print("\n🔴 CONTACT DETECTED - Gripping object!")
            else:
                print("\n🟢 No significant contact")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")

def demo_force_grip():
    """Demo: Force-controlled grip - stop when force threshold reached"""
    controller = InspireHandController()
    
    print("\n[Demo] Force-controlled grip")
    print("Hand will close until force threshold is reached")
    print("Press Ctrl+C to stop\n")
    
    FORCE_THRESHOLD = 0.5  # Newtons
    
    try:
        while True:
            # Gradually close hand
            for grip in np.arange(0.0, 1.01, 0.05):
                controller.send_command([grip] * 12, kp=50.0, kd=5.0)
                time.sleep(0.1)
                
                forces = controller.get_contact_forces()
                if forces:
                    max_force = max(forces)
                    print(f"Grip: {grip:.2f}, Max force: {max_force:.2f} N")
                    
                    if max_force > FORCE_THRESHOLD:
                        print(f"\n⚠️  Force threshold ({FORCE_THRESHOLD} N) reached!")
                        print("Holding grip...")
                        time.sleep(2)
                        controller.print_forces_only()
                        break
            
            # Release
            print("\nReleasing...")
            controller.send_command([0.0] * 12, kp=50.0, kd=5.0)
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")

def demo_custom():
    """Demo: Custom control with force feedback"""
    controller = InspireHandController()
    
    print("\n[Demo] Custom control (with force feedback)")
    print("Commands:")
    print("  12 values (0.0-1.0): Set motor positions")
    print("  'f' or 'force': Show force sensors only")
    print("  's' or 'state': Show full state")
    print("  'q': Quit")
    print("\nExample: 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5\n")
    
    try:
        while True:
            user_input = input("Enter command: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input in ['f', 'force']:
                controller.print_forces_only()
                continue
            elif user_input in ['s', 'state']:
                controller.print_state(show_forces=True)
                continue
            
            try:
                positions = [float(x.strip()) for x in user_input.split(',')]
                if len(positions) != 12:
                    print(f"Error: Need 12 values, got {len(positions)}")
                    continue
                
                # Clip to [0, 1]
                positions = [np.clip(p, 0.0, 1.0) for p in positions]
                
                controller.send_command(positions)
                time.sleep(0.1)
                controller.print_state(show_forces=True)
                
            except ValueError as e:
                print(f"Error parsing input: {e}")
    
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")

if __name__ == "__main__":
    import sys
    
    print("=== Inspire Hand Controller ===")
    print("(with Force Feedback - FORCE_ACT register 1582)")
    print("\nChoose demo mode:")
    print("1. Open/Close hand")
    print("2. Wave fingers")
    print("3. Force sensor monitor (real-time)")
    print("4. Force-controlled grip")
    print("5. Custom control")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        demo_open_close()
    elif choice == "2":
        demo_wave()
    elif choice == "3":
        demo_force_monitor()
    elif choice == "4":
        demo_force_grip()
    elif choice == "5":
        demo_custom()
    else:
        print("Invalid choice")
        sys.exit(1)
