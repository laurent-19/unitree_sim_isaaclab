# Hand Grasp RL Task - Stage 2

## Overview

The **Hand Grasp RL Task** trains a policy to close the Inspire Hand fingers to grasp an object and lift it. The arm is frozen at a pre-grasp position, and only the hand joints are learned. This is Stage 2 of the two-stage grasp learning pipeline.

## Task Description

| Property | Value |
|----------|-------|
| Task Name | `Isaac-G1-HandGrasp-v0` |
| Action Space | 6 DOF (hand proximal joints) |
| Observation Space | ~40 dimensions |
| Episode Length | 4 seconds |
| Control Type | Joint Position |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   HAND GRASP RL TASK                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Observation (40 dim)          Action (6 dim)          │
│   ┌──────────────────┐          ┌───────────────────┐   │
│   │ Hand joint pos(6)│          │ Index proximal    │   │
│   │ Hand joint vel(6)│          │ Middle proximal   │   │
│   │ Contact forces(15)│ ──────► │ Ring proximal     │   │
│   │ Object pose (7)  │  Policy  │ Pinky proximal    │   │
│   │ Last action (6)  │          │ Thumb pitch       │   │
│   └──────────────────┘          │ Thumb yaw         │   │
│                                 └───────────────────┘   │
│                                         │               │
│   ┌─────────────────┐                   │               │
│   │   ARM FROZEN    │                   ▼               │
│   │  (Pre-grasp)    │          ┌───────────────────┐   │
│   └─────────────────┘          │  Hand Joints (6)  │   │
│                                └───────────────────┘   │
│                                                         │
│   ┌─────────────────┐          ┌───────────────────┐   │
│   │ Contact Sensors │ ◄─────── │  Grasp Cylinder   │   │
│   │  (Fingertips)   │          │     (Object)      │   │
│   └─────────────────┘          └───────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Observations

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Hand joint positions | 6 | Proximal joint angles |
| Hand joint velocities | 6 | Proximal joint angular velocities |
| Contact forces | 15 | 5 fingertips × 3D force vectors |
| Object pose | 7 | Position (3) + quaternion (4) relative to hand |
| Last action | 6 | Previous action for temporal consistency |

**Total: ~40 dimensions**

### Contact Force Layout

```
Contact Forces (15 dim):
├── Thumb (0-2):    [fx, fy, fz]
├── Index (3-5):    [fx, fy, fz]
├── Middle (6-8):   [fx, fy, fz]
├── Ring (9-11):    [fx, fy, fz]
└── Pinky (12-14):  [fx, fy, fz]
```

## Actions

The policy directly controls 6 proximal hand joints:

| Action | Joint | Description |
|--------|-------|-------------|
| a[0] | `R_index_proximal_joint` | Index finger flexion |
| a[1] | `R_middle_proximal_joint` | Middle finger flexion |
| a[2] | `R_ring_proximal_joint` | Ring finger flexion |
| a[3] | `R_pinky_proximal_joint` | Pinky finger flexion |
| a[4] | `R_thumb_proximal_pitch_joint` | Thumb flexion |
| a[5] | `R_thumb_proximal_yaw_joint` | Thumb opposition |

**Note**: Intermediate joints are coupled and follow proximal joints.

## Rewards

| Reward Term | Weight | Description |
|-------------|--------|-------------|
| `finger_contact` | +0.5 | Any finger touching object |
| `multi_finger` | +1.0 | 3+ fingers in contact |
| `lift_progress` | +2.0 | Continuous reward for lifting (0-1) |
| `grasp_success` | +5.0 | Object lifted 5cm and held for 30 steps |
| `drop_penalty` | -3.0 | Object falls below 0.7m |
| `action_rate` | -0.01 | Smooth action penalty |

### Reward Progression

```
Episode Timeline:
├── Phase 1: Contact
│   └── finger_contact + multi_finger rewards
├── Phase 2: Lift
│   └── lift_progress reward scales 0→1
└── Phase 3: Hold
    └── grasp_success bonus after stable hold
```

## Terminations

| Condition | Type | Description |
|-----------|------|-------------|
| Time out | Truncation | Episode ends after 4 seconds |
| Object dropped | Termination | Object falls below 0.6m |
| Grasp success | Termination | Object lifted 10cm and held 50 steps |

## Scene Setup

```
Robot Position: (0.0, 0.0, 0.8)

Pre-grasp Arm Pose:
  right_shoulder_pitch: 0.3 rad
  right_shoulder_roll: -0.2 rad
  right_shoulder_yaw: 0.0 rad
  right_elbow: 0.5 rad

Object (Cylinder):
  Radius: 2cm
  Height: 12cm
  Mass: 80g
  Initial Position: Near hand (placed by event)
  Friction: 2.0 (high)

Contact Sensors:
  - R_index_intermediate
  - R_middle_intermediate
  - R_ring_intermediate
  - R_pinky_intermediate
  - R_thumb_distal
```

## Object Placement

The cylinder is spawned near the hand with slight randomization:

```python
# Base offset from hand (EE body)
offset_local = (0.08, 0.0, -0.02)  # meters

# Randomization range
x: [-0.02, 0.02]
y: [-0.02, 0.02]
z: [-0.01, 0.01]
```

## Training

### Command

```bash
python train_hand_grasp_rl.py --num_envs 512 --max_iterations 2000
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| PPO epochs | 5 |
| Mini-batches | 4 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Entropy coef | 0.01 |

### Network Architecture

| Component | Dimensions |
|-----------|------------|
| Actor hidden | [128, 128] |
| Critic hidden | [128, 128] |
| Activation | ELU |
| Init noise std | 0.5 |

**Note**: Smaller network than reach task (simpler action space).

## Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Contact rate | > 90% | Episodes with finger contact |
| Multi-finger rate | > 70% | Episodes with 3+ fingers |
| Lift success | > 50% | Object lifted 5cm |
| Stable grasp | > 30% | Object held for 30 steps |
| Training time | ~20 min | 2000 iterations on RTX 3090 |

## Output

After training, the checkpoint is saved to:
```
logs/g1_hand_grasp/model_XXXX.pt
```

## Evaluation

```bash
# Evaluate single policy
python play_hand_grasp_rl.py \
    --checkpoint logs/g1_hand_grasp/model_2000.pt \
    --num_episodes 10

# Output metrics:
#   - Success rate
#   - Drop rate
#   - Mean reward
```

## Files

| File | Description |
|------|-------------|
| `tasks/g1_tasks/hand_grasp_rl/__init__.py` | Gym registration |
| `tasks/g1_tasks/hand_grasp_rl/hand_grasp_env_cfg.py` | Environment config |
| `tasks/g1_tasks/hand_grasp_rl/grasp_mdp.py` | MDP functions |
| `train_hand_grasp_rl.py` | Training script |
| `play_hand_grasp_rl.py` | Evaluation script |

## Two-Stage Pipeline Integration

### Stage 1 → Stage 2 Transition

```
┌──────────────┐         ┌──────────────┐
│  STAGE 1     │         │  STAGE 2     │
│  Reach       │ ──────► │  Grasp       │
│  (Arm IK)    │         │  (Hand)      │
└──────────────┘         └──────────────┘
     │                         │
     ▼                         ▼
 model_reach.pt           model_grasp.pt
```

### Combined Pipeline

```bash
python play_combined_pipeline.py \
    --reach_checkpoint logs/g1_reach_grasp/model_2000.pt \
    --grasp_checkpoint logs/g1_hand_grasp/model_2000.pt
```

### Pipeline Logic

```python
while not done:
    if phase == REACH:
        # Use reach policy for arm IK
        action = reach_policy(reach_obs)
        if distance_to_goal < threshold:
            phase = GRASP

    elif phase == GRASP:
        # Use grasp policy for hand joints
        action = grasp_policy(grasp_obs)
```

## Curriculum Ideas (Future)

1. **Object Size**: Start with larger objects, decrease radius
2. **Object Position**: Start centered, add randomization
3. **Lift Height**: Start with low targets, increase
4. **Grasp Time**: Start with short holds, extend duration

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No contact | Object too far | Adjust `offset_local` |
| Unstable grasp | Low friction | Increase `physics_material` friction |
| Fingers slip | Fast closing | Reduce `action_rate` penalty |
| Object flies | High forces | Lower contact sensor threshold |

### Debug Visualization

Enable contact sensor debug visualization:
```python
fingertip_contacts = ContactSensorCfg(
    ...
    debug_vis=True,  # Show contact forces
)
```
