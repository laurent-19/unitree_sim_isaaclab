# Inspire Hand DDS Data Documentation

This document explains how the DDS messages for the Inspire Hand are computed in the Isaac Lab simulation.

## Overview

The simulation publishes three types of DDS messages and subscribes to one:

| Topic | Direction | Message Type | Purpose |
|-------|-----------|--------------|---------|
| `rt/inspire_hand/state/{l,r}` | Publish | `inspire_hand_state` | Hand joint state feedback |
| `rt/inspire_hand/touch/{l,r}` | Publish | `inspire_hand_touch` | Tactile sensor data |
| `rt/inspire_hand/ctrl/{l,r}` | Subscribe | `inspire_hand_ctrl` | Hand control commands |

---

## 1. State Data (`inspire_hand_state`)

**Topic:** `rt/inspire_hand/state/r` (right) or `rt/inspire_hand/state/l` (left)

**Source Files:**
- `tasks/common_observations/inspire_state.py` - Extracts data from simulation
- `dds/inspire_dds.py` - Converts and publishes via DDS

### Fields

| Field | Type | Range | Real Hand Source | Simulation Source |
|-------|------|-------|------------------|-------------------|
| `pos_act[6]` | short | 0-2000 | Linear actuator position | Joint position (radians → 0-1000 scale) |
| `angle_act[6]` | short | 0-1000 | Joint encoder | Joint position (radians → 0-1000 scale) |
| `force_act[6]` | short | -4000 to 4000 | Force sensor (grams) | Contact force magnitude (N → grams) |
| `current[6]` | short | 0-2000 | Motor current (mA) | **0 (N/A)** - Cannot simulate |
| `err[6]` | byte | 0-255 | Error codes | 0 (no errors) |
| `status[6]` | byte | 0-255 | Status flags | 0 (normal) |
| `temperature[6]` | byte | 0-100 | Motor temp (°C) | **0 (N/A)** - Cannot simulate |

### Finger Order (indices 0-5)

| Index | Real Hand | Simulation Joint |
|-------|-----------|------------------|
| 0 | Little finger (pinky) | `L/R_pinky_proximal_joint` |
| 1 | Ring finger | `L/R_ring_proximal_joint` |
| 2 | Middle finger | `L/R_middle_proximal_joint` |
| 3 | Index finger | `L/R_index_proximal_joint` |
| 4 | Thumb bend | `L/R_thumb_proximal_pitch_joint` |
| 5 | Thumb rotation | `L/R_thumb_proximal_yaw_joint` |

### Computation Details

#### `angle_act` / `pos_act` - Position Feedback

```python
# Source: Isaac Lab joint state
joint_pos = env.scene["robot"].data.joint_pos  # radians

# Conversion: radians → 0-1000 scale
# 1000 = fully open, 0 = fully closed
def radians_to_inspire(rad_val, joint_idx):
    # Joint ranges (approximate)
    # Fingers: 0.0 to 1.7 radians
    # Thumb rotation: 0.0 to 0.5 radians
    # Thumb bend: -0.1 to 1.3 radians

    min_rad, max_rad = get_joint_range(joint_idx)
    normalized = (max_rad - rad_val) / (max_rad - min_rad)
    return int(normalized * 1000)  # 0-1000
```

#### `force_act` - Force Feedback

```python
# Source: Isaac Lab ContactSensor
net_forces = env.scene["right_fingertip_contacts"].data.net_forces_w
# Shape: (num_envs, num_bodies, 3) - 3D force vectors in Newtons

# Per-finger force magnitude
force_newtons = torch.norm(net_forces[:, finger_idx, :], dim=-1)

# Convert Newtons to grams (real hand uses grams)
# 1 Newton ≈ 102 grams
force_grams = int(force_newtons * 102)
force_act = clip(force_grams, -4000, 4000)
```

**Note:** The real hand has dedicated force sensors on each finger. The simulation estimates force from contact sensor readings, which may differ from real hardware behavior.

---

## 2. Touch Data (`inspire_hand_touch`)

**Topic:** `rt/inspire_hand/touch/r` (right) or `rt/inspire_hand/touch/l` (left)

**Source Files:**
- `tasks/common_observations/inspire_tactile.py` - Extracts contact forces
- `tasks/common_observations/tactile_mapping.py` - Converts to taxel grids
- `dds/inspire_touch_dds.py` - Publishes via DDS

### Fields

Each finger has multiple tactile regions:

| Field | Grid Size | Taxels | Description |
|-------|-----------|--------|-------------|
| `fingerX_tip_touch` | 3×3 | 9 | Fingertip (distal) |
| `fingerX_top_touch` | 12×8 | 96 | Nail/top of finger |
| `fingerX_palm_touch` | 10×8 | 80 | Finger pad (intermediate) |
| `palm_touch` | 8×14 | 112 | Palm area |

Where X = `one` (pinky), `two` (ring), `three` (middle), `four` (index), `five` (thumb)

### Taxel Values

- **Type:** 16-bit integer
- **Range:** 0-4095
- **Unit:** Pressure (arbitrary units)

### Computation Details

#### Step 1: Get Contact Force from Physics

```python
# Isaac Lab ContactSensor reports 3D force when finger touches object
net_forces = env.scene["right_fingertip_contacts"].data.net_forces_w
tip_force = net_forces[0, finger_idx]  # [Fx, Fy, Fz] in Newtons
```

#### Step 2: Convert to Taxel Grid

```python
def force_to_taxel_grid(force_vector, grid_shape):
    # 1. Compute force magnitude
    magnitude = np.linalg.norm(force_vector)  # Newtons

    # 2. Scale to taxel range (0-4095)
    scaled = min(int(magnitude * 1000), 4095)

    # 3. Create Gaussian pressure distribution
    # Simulates how pressure spreads across sensor surface
    kernel = gaussian_2d(grid_shape.rows, grid_shape.cols)

    # 4. Shift based on shear force direction
    shear_x = force_vector[0] / magnitude
    shear_y = force_vector[1] / magnitude
    shift_x = int(shear_x * grid_shape.cols / 4)
    shift_y = int(shear_y * grid_shape.rows / 4)
    kernel = np.roll(kernel, (shift_y, shift_x))

    # 5. Apply scaling
    taxel_grid = (kernel * scaled).astype(int)
    return np.clip(taxel_grid, 0, 4095)
```

#### Step 3: Flatten for DDS

```python
# Fingers: row-major order
tip_touch = taxel_grid.flatten().tolist()

# Palm: column-major from bottom row
palm_touch = grid[::-1, :].T.flatten().tolist()
```

### Limitation: Contact Position

**Important:** The simulation does NOT accurately represent WHERE on the finger surface contact occurs.

- Isaac Lab's ContactSensor only provides total force magnitude, not contact position
- The taxel distribution is always centered with slight shift based on shear direction
- Real tactile sensors would show pressure concentrated at actual contact location

```
Real sensor (contact on left):    Simulation (centered):
┌───┬───┬───┐                     ┌───┬───┬───┐
│███│   │   │                     │ ▪ │ ▪ │ ▪ │
├───┼───┼───┤                     ├───┼───┼───┤
│███│   │   │  → simulated as →   │ ▪ │███│ ▪ │
├───┼───┼───┤                     ├───┼───┼───┤
│███│   │   │                     │ ▪ │ ▪ │ ▪ │
└───┴───┴───┘                     └───┴───┴───┘
```

---

## 3. Control Commands (`inspire_hand_ctrl`)

**Topic:** `rt/inspire_hand/ctrl/r` (right) or `rt/inspire_hand/ctrl/l` (left)

**Direction:** Subscribe (receive commands from external controller)

**Source Files:**
- `dds/inspire_dds.py` - Receives commands via DDS
- `action_provider/action_provider_dds.py` - Applies to simulation

### Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `angle_set[6]` | short | 0-1000 | Target angle (1000=open, 0=closed) |
| `speed_set[6]` | short | 0-1000 | Movement speed |
| `force_set[6]` | short | 0-3000 | Force limit (grams) |
| `mode` | byte | 0-15 | Control mode bitfield |

### Control Mode Bits

```
bit 0: Angle control enabled
bit 1: Position control enabled
bit 2: Force control enabled
bit 3: Speed control enabled

Examples:
mode = 0b0001 (1)  → Angle control only
mode = 0b0101 (5)  → Angle + Force control
mode = 0b1111 (15) → All controls enabled
```

### Computation: Command → Simulation

```python
# Receive inspire_hand_ctrl message
def dds_subscriber(msg):
    # Convert 0-1000 scale to radians
    for i in range(6):
        inspire_val = msg.angle_set[i]  # 0-1000

        # 1000 = open (low radians), 0 = closed (high radians)
        min_rad, max_rad = get_joint_range(i)
        normalized = inspire_val / 1000.0
        target_radians = (1.0 - normalized) * (max_rad - min_rad) + min_rad

        # Write to shared memory for simulation to read
        cmd_data["positions"].append(target_radians)
```

### Action Application

```python
# action_provider_dds.py reads shared memory and applies to simulation
def get_action(self, env):
    cmd = self.inspire_dds_r.get_inspire_hand_command()

    if cmd:
        # Apply to finger joints
        for i, pos in enumerate(cmd["positions"]):
            joint_idx = finger_joint_map[i]
            action[joint_idx] = pos

    return action
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ISAAC LAB SIMULATION                            │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │   Robot Model   │    │ Contact Sensors │    │  Joint State   │  │
│  │  (Inspire Hand) │    │ (fingertips)    │    │  (positions)   │  │
│  └────────┬────────┘    └────────┬────────┘    └───────┬────────┘  │
│           │                      │                      │           │
│           ▼                      ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    inspire_state.py                          │  │
│  │  - Extract joint positions (radians)                         │  │
│  │  - Extract contact forces (Newtons)                          │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    inspire_dds.py                            │  │
│  │  - Convert radians → 0-1000 (angle_act)                      │  │
│  │  - Convert Newtons → grams (force_act)                       │  │
│  │  - Set current/temp = 0 (N/A)                                │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      DDS NETWORK        │
                    │                         │
                    │  rt/inspire_hand/state  │──────► External
                    │  rt/inspire_hand/touch  │──────► Subscriber
                    │  rt/inspire_hand/ctrl   │◄────── External
                    │                         │        Publisher
                    └─────────────────────────┘
```

---

## Contact Sensor Configuration

The simulation uses these contact sensors (defined in `pickplace_cylinder_g1_29dof_inspire_env_cfg.py`):

| Sensor Name | Prim Path | Bodies Found |
|-------------|-----------|--------------|
| `left_fingertip_contacts` | `L_(index\|middle\|ring\|pinky)_intermediate` | 4 finger tips |
| `left_thumb_contacts` | `L_thumb_distal` | 1 thumb tip |
| `left_finger_pad_contacts` | `L_(index\|middle\|ring\|pinky)_proximal` | 4 finger pads |
| `left_palm_contacts` | `L_thumb_proximal` | 1 palm proxy |
| (same for right hand with `R_` prefix) | | |

**Note:** The Inspire hand model has:
- **Fingers (index, middle, ring, pinky):** proximal + intermediate joints only (no distal)
- **Thumb:** proximal + intermediate + distal joints

So fingertips for non-thumb fingers are the `intermediate` links, while thumb tip is the `distal` link.

---

## Limitations Summary

| Aspect | Real Hand | Simulation | Accuracy |
|--------|-----------|------------|----------|
| Joint position | Encoder | Physics engine | ✅ Accurate |
| Joint velocity | Encoder | Physics engine | ✅ Accurate |
| Force magnitude | Force sensor | Contact sensor | ⚠️ Approximate |
| Motor current | Current sensor | N/A | ❌ Not simulated |
| Motor temperature | Temp sensor | N/A | ❌ Not simulated |
| Tactile position | Taxel array | Centered Gaussian | ❌ Not accurate |
| Tactile magnitude | Taxel array | Contact force | ⚠️ Approximate |
