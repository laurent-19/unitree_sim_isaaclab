#!/bin/bash
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

# ============================================================================
# Tactile-Guided Hand Grasping Experiments
# ============================================================================
#
# This script runs the experiments for the research paper:
# "Tactile-Guided Grasping for Dexterous Hands: Sim-to-Real with the Inspire Hand"
#
# Usage:
#   ./run_experiments.sh train_tactile      # Train with tactile
#   ./run_experiments.sh train_no_tactile   # Train without tactile (ablation)
#   ./run_experiments.sh train_all          # Train both variants
#   ./run_experiments.sh eval_tactile       # Evaluate tactile policy
#   ./run_experiments.sh eval_no_tactile    # Evaluate no-tactile policy
#   ./run_experiments.sh eval_all           # Evaluate all policies
#   ./run_experiments.sh plot               # Generate paper figures
#   ./run_experiments.sh full_pipeline      # Run everything
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
FIGURE_DIR="${SCRIPT_DIR}/figures"
RESULTS_DIR="${SCRIPT_DIR}/results"
NUM_ENVS=4096
MAX_ITERATIONS=3000
EVAL_EPISODES=100

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directories
mkdir -p "${LOG_DIR}" "${FIGURE_DIR}" "${RESULTS_DIR}"

# ============================================================================
# Training Functions
# ============================================================================

train_tactile() {
    log_info "Training TACTILE policy..."
    python train_hand_grasp.py \
        --max_iterations ${MAX_ITERATIONS} \
        --num_envs ${NUM_ENVS} \
        --logdir "${LOG_DIR}/hand_grasp_tactile" \
        --seed 42 \
        "$@"
    log_info "Tactile training complete!"
}

train_no_tactile() {
    log_info "Training NO-TACTILE policy..."
    python train_hand_grasp.py \
        --no_tactile \
        --max_iterations ${MAX_ITERATIONS} \
        --num_envs ${NUM_ENVS} \
        --logdir "${LOG_DIR}/hand_grasp_no_tactile" \
        --seed 42 \
        "$@"
    log_info "No-tactile training complete!"
}

train_all_seeds() {
    log_info "Training all seeds for statistical significance..."
    for seed in 42 123 456; do
        log_info "Seed ${seed} - Tactile..."
        python train_hand_grasp.py \
            --max_iterations ${MAX_ITERATIONS} \
            --num_envs ${NUM_ENVS} \
            --logdir "${LOG_DIR}/seed_${seed}_tactile" \
            --seed ${seed}

        log_info "Seed ${seed} - No-Tactile..."
        python train_hand_grasp.py \
            --no_tactile \
            --max_iterations ${MAX_ITERATIONS} \
            --num_envs ${NUM_ENVS} \
            --logdir "${LOG_DIR}/seed_${seed}_no_tactile" \
            --seed ${seed}
    done
    log_info "All seeds trained!"
}

# ============================================================================
# Evaluation Functions
# ============================================================================

eval_tactile() {
    local checkpoint="${1:-${LOG_DIR}/hand_grasp_tactile/model_${MAX_ITERATIONS}.pt}"

    if [ ! -f "${checkpoint}" ]; then
        log_error "Checkpoint not found: ${checkpoint}"
        exit 1
    fi

    log_info "Evaluating TACTILE policy: ${checkpoint}"
    python eval_hand_grasp.py \
        --checkpoint "${checkpoint}" \
        --num_episodes ${EVAL_EPISODES} \
        --output "${RESULTS_DIR}/eval_tactile.json"
}

eval_no_tactile() {
    local checkpoint="${1:-${LOG_DIR}/hand_grasp_no_tactile/model_${MAX_ITERATIONS}.pt}"

    if [ ! -f "${checkpoint}" ]; then
        log_error "Checkpoint not found: ${checkpoint}"
        exit 1
    fi

    log_info "Evaluating NO-TACTILE policy: ${checkpoint}"
    python eval_hand_grasp.py \
        --checkpoint "${checkpoint}" \
        --no_tactile \
        --num_episodes ${EVAL_EPISODES} \
        --output "${RESULTS_DIR}/eval_no_tactile.json"
}

eval_all() {
    eval_tactile "$@"
    eval_no_tactile "$@"
    log_info "All evaluations complete! Results in ${RESULTS_DIR}/"
}

# ============================================================================
# Visualization Functions
# ============================================================================

play_tactile() {
    local checkpoint="${1:-${LOG_DIR}/hand_grasp_tactile/model_${MAX_ITERATIONS}.pt}"
    log_info "Playing TACTILE policy..."
    python play_hand_grasp.py --checkpoint "${checkpoint}"
}

play_no_tactile() {
    local checkpoint="${1:-${LOG_DIR}/hand_grasp_no_tactile/model_${MAX_ITERATIONS}.pt}"
    log_info "Playing NO-TACTILE policy..."
    python play_hand_grasp.py --checkpoint "${checkpoint}" --no_tactile
}

# ============================================================================
# Export Functions
# ============================================================================

export_onnx() {
    local checkpoint="${1:-${LOG_DIR}/hand_grasp_tactile/model_${MAX_ITERATIONS}.pt}"
    local output="${2:-policy_tactile.onnx}"

    log_info "Exporting policy to ONNX: ${output}"
    python play_hand_grasp.py \
        --checkpoint "${checkpoint}" \
        --export_onnx "${output}" \
        --headless \
        --max_steps 1
}

# ============================================================================
# Plotting Functions
# ============================================================================

generate_plots() {
    log_info "Generating paper figures..."
    python plot_results.py \
        --log_dir "${LOG_DIR}" \
        --output_dir "${FIGURE_DIR}" \
        --eval_results "${RESULTS_DIR}/eval_tactile.json" "${RESULTS_DIR}/eval_no_tactile.json"
    log_info "Figures saved to ${FIGURE_DIR}/"
}

# ============================================================================
# Test Functions
# ============================================================================

test_arm_position() {
    log_info "Testing arm positioning in simulation..."
    python test_arm_position_sim.py --steps 200
}

test_env() {
    log_info "Testing environment creation..."
    python -c "
from tasks.g1_tasks.hand_grasp_inspire import HandGraspInspireEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
print('Environment import successful!')
"
}

# ============================================================================
# Full Pipeline
# ============================================================================

full_pipeline() {
    log_info "Running full experiment pipeline..."
    log_info "This will take several hours!"

    # Training
    train_tactile
    train_no_tactile

    # Evaluation
    eval_tactile
    eval_no_tactile

    # Generate plots
    generate_plots

    log_info "==================================="
    log_info "Full pipeline complete!"
    log_info "Results: ${RESULTS_DIR}/"
    log_info "Figures: ${FIGURE_DIR}/"
    log_info "==================================="
}

# ============================================================================
# Main
# ============================================================================

show_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train_tactile      Train policy with tactile observations"
    echo "  train_no_tactile   Train policy without tactile (ablation)"
    echo "  train_all          Train both variants"
    echo "  train_all_seeds    Train both variants with multiple seeds"
    echo ""
    echo "  eval_tactile       Evaluate tactile policy"
    echo "  eval_no_tactile    Evaluate no-tactile policy"
    echo "  eval_all           Evaluate all policies"
    echo ""
    echo "  play_tactile       Visualize tactile policy"
    echo "  play_no_tactile    Visualize no-tactile policy"
    echo ""
    echo "  export_onnx        Export policy to ONNX"
    echo "  generate_plots     Generate paper figures"
    echo ""
    echo "  test_arm_position  Test arm positioning"
    echo "  test_env           Test environment import"
    echo ""
    echo "  full_pipeline      Run complete experiment pipeline"
    echo ""
}

# Parse command
COMMAND="${1:-}"
shift || true

case "${COMMAND}" in
    train_tactile)
        train_tactile "$@"
        ;;
    train_no_tactile)
        train_no_tactile "$@"
        ;;
    train_all)
        train_tactile "$@"
        train_no_tactile "$@"
        ;;
    train_all_seeds)
        train_all_seeds "$@"
        ;;
    eval_tactile)
        eval_tactile "$@"
        ;;
    eval_no_tactile)
        eval_no_tactile "$@"
        ;;
    eval_all)
        eval_all "$@"
        ;;
    play_tactile)
        play_tactile "$@"
        ;;
    play_no_tactile)
        play_no_tactile "$@"
        ;;
    export_onnx)
        export_onnx "$@"
        ;;
    generate_plots|plot)
        generate_plots "$@"
        ;;
    test_arm_position)
        test_arm_position "$@"
        ;;
    test_env)
        test_env "$@"
        ;;
    full_pipeline)
        full_pipeline "$@"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
