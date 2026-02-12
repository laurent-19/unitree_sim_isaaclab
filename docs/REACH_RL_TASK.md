# Reach RL Task - Stage 1

## Overview

The **Reach RL Task** trains a policy to move the G1 robot's right arm end-effector (EE) to a goal position in 3D space. This is Stage 1 of the two-stage grasp learning pipeline.

## Task Description

| Property | Value |
|----------|-------|
| Task Name | `Isaac-G1-ReachGrasp-v0` |
| Action Space | 3 DOF (IK: dx, dy, dz) |
| Observation Space | ~21 dimensions |
| Episode Length | 6 seconds |
| Control Type | Differential IK |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REACH RL TASK                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Observation (21 dim)          Action (3 dim)          │
│   ┌─────────────────┐           ┌─────────────┐         │
│   │ Goal pose (7)   │           │ dx          │         │
│   │ EE position (3) │  ──────►  │ dy          │         │
│   │ Joint pos (4)   │  Policy   │ dz          │         │
│   │ Joint vel (4)   │           └─────────────┘         │
│   │ Last action (3) │                 │                 │
│   └─────────────────┘                 │                 │
│                                       ▼                 │
│                              ┌─────────────────┐        │
│                              │ Differential IK │        │
│                              │   Controller    │        │
│                              └────────┬────────┘        │
│                                       │                 │
│                                       ▼                 │
│                              ┌─────────────────┐        │
│                              │ Right Arm Joints│        │
│                              │ (4 DOF)         │        │
│                              └─────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Observations

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Goal pose | 7 | Target position (x,y,z) + orientation (qw,qx,qy,qz) in robot root frame |
| EE position | 3 | Current end-effector position in robot root frame |
| Joint positions | 4 | Right arm joint angles (relative to default) |
| Joint velocities | 4 | Right arm joint angular velocities |
| Last action | 3 | Previous action for temporal consistency |

**Total: ~21 dimensions**

## Actions

The policy outputs 3D position deltas which are converted to joint targets via Differential IK:

| Action | Range | Description |
|--------|-------|-------------|
| dx | [-1, 1] | Forward/backward movement |
| dy | [-1, 1] | Left/right movement |
| dz | [-1, 1] | Up/down movement |

**Scale**: 0.006 meters per action unit

## Rewards

| Reward Term | Weight | Description |
|-------------|--------|-------------|
| `success_bonus` | +2.0 | Binary reward when EE is within 5cm of goal |
| `ee_pos_cost` | -2.0 | L2 distance to goal (main shaping signal) |
| `ee_pos_track` | +0.25 | Bounded tracking reward using tanh |
| `action_rate` | -5e-4 | Penalize rapid action changes |
| `joint_vel` | -5e-4 | Penalize high joint velocities |
| `joint_acc` | -2e-6 | Penalize joint accelerations |

## Terminations

| Condition | Type | Description |
|-----------|------|-------------|
| Time out | Truncation | Episode ends after 6 seconds |

## Scene Setup

```
Robot Position: (1.1, 0.5, 0.8)
Robot Rotation: 90° around Z-axis (facing +Y)

Workspace (in robot root frame):
  X: [0.3, 0.4]   - Forward from robot
  Y: [-0.3, 0.0]  - Right side (right arm)
  Z: [0.05, 0.3]  - Above robot base

Reachability Shell:
  R_min: 0.12m
  R_max: 0.48m
```

## Right Arm Joints

| Joint Name | Description |
|------------|-------------|
| `right_shoulder_pitch_joint` | Shoulder forward/backward |
| `right_shoulder_roll_joint` | Shoulder in/out |
| `right_shoulder_yaw_joint` | Shoulder rotation |
| `right_elbow_joint` | Elbow flexion |

## Training

### Command

```bash
python train_reach_grasp.py --num_envs 256 --max_iterations 2000
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
| Actor hidden | [256, 256] |
| Critic hidden | [256, 256] |
| Activation | ELU |
| Init noise std | 1.0 |

## Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Position error | < 5cm | Distance from EE to goal |
| Success rate | > 80% | Episodes reaching goal |
| Training time | ~30 min | 2000 iterations on RTX 3090 |

## Output

After training, the checkpoint is saved to:
```
logs/g1_reach_grasp/model_XXXX.pt
```

This checkpoint is used as Stage 1 for the combined pipeline.

## Key Events

### Goal Validation
Goals are validated to ensure kinematic reachability:
- Must be within workspace box
- Must be within spherical shell around shoulder

### Hand Control
During reach training, the hand is controlled by events (not learned):
- **Open phase**: Hand stays open while reaching
- **Close phase**: Hand closes after reaching goal or stalling

### Object Placement
A cylinder is placed near the goal position for visual feedback and grasp preparation.

## Files

| File | Description |
|------|-------------|
| `tasks/g1_tasks/reach_grasp_rl/__init__.py` | Gym registration |
| `tasks/g1_tasks/reach_grasp_rl/reach_grasp_env_cfg.py` | Environment config |
| `tasks/g1_tasks/reach_grasp_rl/reach_mdp.py` | MDP helper functions |
| `train_reach_grasp.py` | Training script |

## Next Steps

After training the reach policy:
1. Evaluate with `play_reach_grasp.py`
2. Proceed to Stage 2: Hand Grasp Training
3. Combine both policies with `play_combined_pipeline.py`
