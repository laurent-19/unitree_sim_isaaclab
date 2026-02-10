# Inspire Hand Tactile DDS Publishing Solution

## Overview

This document summarizes the implementation of tactile sensor data publishing from Isaac Lab simulation to DDS (Data Distribution Service), enabling external systems to receive simulated contact force data from the Inspire Hand (RH56DFTP).

## Problem Statement

1. **Contact sensor error**: Original configuration used `wrist_yaw` links for palm contact sensors, which lack PhysX ContactReportAPI
2. **DDS communication**: Need to publish tactile data from simulation to external subscribers
3. **Cross-container communication**: DDS must work between Docker containers using `--network host`

## Solution Components

### 1. Contact Sensor Configuration Fix

**File**: `tasks/g1_tasks/pick_place_cylinder_g1_29dof_inspire/pickplace_cylinder_g1_29dof_inspire_env_cfg.py`

Changed palm contact sensors from wrist joints to finger proximal links:

```python
# Before (broken)
left_palm_contacts = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw",  # No collision geometry
    ...
)

# After (working)
left_palm_contacts = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/L_.*_proximal",  # 5 proximal links
    ...
)
```

**Result**: 22 contact sensor bodies total (11 per hand)
- 1 fingertip (thumb distal)
- 5 intermediate pads
- 5 proximal (palm proxy)

### 2. DDS Network Interface Configuration

**File**: `dds/dds_master.py`

```python
# Use multicast-capable interface for DDS discovery
network_interface = os.environ.get('DDS_NETWORK_INTERFACE', 'enp0s31f6')
ChannelFactoryInitialize(1, network_interface)
```

**Key points**:
- Loopback (`lo`) is NOT multicast-capable - DDS discovery won't work
- Physical interface (`enp0s31f6`) required for proper DDS operation
- Domain ID must match between publisher and subscriber (using domain 1)

### 3. Tactile DDS Publisher

**File**: `dds/inspire_touch_dds.py`

- Publishes to topic: `rt/inspire/touch`
- Message type: `inspire_hand_touch` IDL from inspire_hand_sdk
- Uses shared memory for inter-thread communication

### 4. Tactile Observation Module

**File**: `tasks/common_observations/inspire_tactile.py`

Key fixes:
- Retry logic for DDS object acquisition (objects may not be registered when observations are first computed)
- Rate limiting (20ms interval = 50Hz publish rate)

```python
def _get_touch_dds():
    # Always try to get DDS if we don't have it yet (retry logic)
    if _tactile_cache["dds"] is None:
        _tactile_cache["dds"] = dds_manager.get_object("inspire_touch")
```

### 5. IDL Import Path

**File**: `dds/inspire_touch_dds.py`

Added multiple path attempts for Docker compatibility:
```python
_inspire_sdk_paths = [
    "/home/code/inspire_hand_ws/inspire_hand_sdk/inspire_sdkpy",  # Container path
    os.path.join(..., "inspire_hand_ws", "inspire_hand_sdk", "inspire_sdkpy")
]
```

## Usage

### Running the Simulation (Publisher)

```bash
# Start Docker container
./run_docker.sh

# Inside container, run simulation
./run_inspire_task.sh cylinder
```

### Subscribing to Tactile Data

**Inside the same container:**
```bash
docker exec -it <container_name> bash
source /opt/conda/etc/profile.d/conda.sh
conda activate unitree_sim_env
cd /home/code/unitree_sim_isaaclab
python scripts/subscribe_inspire_touch.py --network enp0s31f6
```

**From host or another container:**
```bash
python scripts/subscribe_inspire_touch.py --network enp0s31f6
```

### Quick Test Script

```python
import sys
sys.path.insert(0, "../inspire_hand_ws/inspire_hand_sdk/inspire_sdkpy")
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from inspire_dds._inspire_hand_touch import inspire_hand_touch

ChannelFactoryInitialize(1, "enp0s31f6")

def callback(msg):
    print(f"Received tactile data!")

sub = ChannelSubscriber("rt/inspire/touch", inspire_hand_touch)
sub.Init(callback, 10)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Isaac Lab Simulation                        │
│                                                                 │
│  Contact Sensors (PhysX)                                        │
│  ├── left_fingertip_contacts  (1 body: thumb_distal)           │
│  ├── left_finger_pad_contacts (5 bodies: *_intermediate)       │
│  ├── left_palm_contacts       (5 bodies: *_proximal)           │
│  ├── right_fingertip_contacts (1 body: thumb_distal)           │
│  ├── right_finger_pad_contacts(5 bodies: *_intermediate)       │
│  └── right_palm_contacts      (5 bodies: *_proximal)           │
│                          │                                      │
│                          ▼                                      │
│  get_inspire_tactile_state() → sensor_data dict                │
│                          │                                      │
│                          ▼                                      │
│  _publish_tactile_to_dds() → Shared Memory                     │
│                          │                                      │
│                          ▼                                      │
│  InspireTouchDDS.dds_publisher() → DDS Topic                   │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼ rt/inspire/touch (Domain 1)
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                    DDS Subscriber                               │
│  ChannelSubscriber("rt/inspire/touch", inspire_hand_touch)     │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### No DDS entities discovered
- Ensure using multicast-capable interface (not `lo`)
- Check domain ID matches (should be 1)
- Verify `--network host` in Docker run command

### "Publisher failed to start (Timeout)"
- Network interface may not be accessible
- Check interface is UP: `cat /sys/class/net/enp0s31f6/operstate`

### Contact sensor errors
- Verify prim_path matches actual robot URDF links
- Links must have collision geometry with PhysxContactReportAPI

### DDS object not found
- Observation module queries DDS before objects are registered
- Fixed with retry logic in `_get_touch_dds()`

## Configuration Summary

| Setting | Value |
|---------|-------|
| DDS Domain ID | 1 |
| Network Interface | enp0s31f6 |
| Topic Name | rt/inspire/touch |
| Message Type | inspire_hand_touch |
| Publish Rate | 50 Hz (20ms interval) |
| Contact Sensors | 22 bodies (11 per hand) |

## Files Modified

1. `tasks/g1_tasks/pick_place_cylinder_g1_29dof_inspire/pickplace_cylinder_g1_29dof_inspire_env_cfg.py` - Contact sensor paths
2. `dds/dds_master.py` - Network interface configuration
3. `dds/inspire_touch_dds.py` - IDL import paths, debug logging
4. `tasks/common_observations/inspire_tactile.py` - DDS retry logic, debug logging
5. `scripts/subscribe_inspire_touch.py` - Domain ID and network interface
6. `run_docker.sh` - Volume mount for inspire_hand_ws
