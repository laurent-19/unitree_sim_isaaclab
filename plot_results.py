#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Generate paper figures from training and evaluation results.

Creates publication-quality plots for:
1. Learning curves (reward over iterations)
2. Tactile vs no-tactile comparison
3. Ablation study results
4. Real robot evaluation bar charts

Usage:
    # Generate all plots from training logs
    python plot_results.py --log_dir logs/ --output_dir figures/

    # Generate specific plot
    python plot_results.py --log_dir logs/ --plot learning_curves

    # Generate from evaluation results
    python plot_results.py --eval_results results/tactile.json results/no_tactile.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Install with: pip install matplotlib")


def load_tensorboard_data(log_dir: str, tag: str = "reward") -> Dict[str, np.ndarray]:
    """Load data from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard logs
        tag: Tag to extract (e.g., "reward", "loss")

    Returns:
        Dictionary with "steps" and "values" arrays
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WARNING] tensorboard not installed. Using dummy data.")
        return {"steps": np.arange(100), "values": np.random.randn(100).cumsum()}

    # Find event files
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
    if not event_files:
        print(f"[WARNING] No event files found in {log_dir}")
        return {"steps": np.array([]), "values": np.array([])}

    # Load first event file
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    # Get scalar data
    if tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return {"steps": steps, "values": values}

    return {"steps": np.array([]), "values": np.array([])}


def plot_learning_curves(
    tactile_dir: str,
    no_tactile_dir: str,
    output_path: str,
    title: str = "Learning Curves: Tactile vs No-Tactile",
):
    """Plot learning curves comparing tactile and no-tactile training.

    Args:
        tactile_dir: Log directory for tactile training
        no_tactile_dir: Log directory for no-tactile training
        output_path: Path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Load data
    tactile_data = load_tensorboard_data(tactile_dir, "reward")
    no_tactile_data = load_tensorboard_data(no_tactile_dir, "reward")

    # Plot tactile
    if len(tactile_data["steps"]) > 0:
        ax.plot(
            tactile_data["steps"],
            tactile_data["values"],
            label="Tactile (Ours)",
            color="#2ecc71",
            linewidth=2,
        )

    # Plot no-tactile
    if len(no_tactile_data["steps"]) > 0:
        ax.plot(
            no_tactile_data["steps"],
            no_tactile_data["values"],
            label="No-Tactile (Baseline)",
            color="#e74c3c",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Training Iterations", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {output_path}")


def plot_success_comparison(
    eval_results: List[Dict],
    output_path: str,
    title: str = "Grasp Success Rate Comparison",
):
    """Plot bar chart comparing success rates.

    Args:
        eval_results: List of evaluation result dictionaries
        output_path: Path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    labels = []
    grasp_rates = []
    lift_rates = []

    for result in eval_results:
        label = "No-Tactile" if result.get("no_tactile", False) else "Tactile (Ours)"
        labels.append(label)
        grasp_rates.append(result.get("grasp_success_rate", 0) * 100)
        lift_rates.append(result.get("lift_success_rate", 0) * 100)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, grasp_rates, width, label="Grasp Success", color="#2ecc71")
    bars2 = ax.bar(x + width/2, lift_rates, width, label="Lift Success", color="#3498db")

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {output_path}")


def plot_ablation_study(
    results: Dict[str, float],
    output_path: str,
    title: str = "Observation Ablation Study",
):
    """Plot ablation study results.

    Args:
        results: Dictionary mapping condition name to success rate
        output_path: Path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    success_rates = [results[c] * 100 for c in conditions]

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f1c40f"]
    bars = ax.barh(conditions, success_rates, color=colors[:len(conditions)])

    ax.set_xlabel("Grasp Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        ax.text(rate + 2, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {output_path}")


def plot_real_robot_results(
    results: Dict[str, Dict],
    output_path: str,
    title: str = "Real Robot Evaluation",
):
    """Plot real robot experiment results.

    Args:
        results: Dictionary mapping condition to {success, total}
        output_path: Path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = list(results.keys())
    success_rates = [results[c]["success"] / results[c]["total"] * 100 for c in conditions]

    colors = ["#2ecc71" if "tactile" in c.lower() else "#e74c3c" for c in conditions]
    bars = ax.bar(conditions, success_rates, color=colors)

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 100)

    # Add success/total labels
    for bar, cond in zip(bars, conditions):
        height = bar.get_height()
        success = results[cond]["success"]
        total = results[cond]["total"]
        ax.annotate(f'{success}/{total}\n({height:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {output_path}")


def generate_latex_table(results: List[Dict], output_path: str):
    """Generate LaTeX table from evaluation results.

    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save the LaTeX file
    """
    header = r"""
\begin{table}[h]
\centering
\caption{Simulation Experiment Results}
\label{tab:sim_results}
\begin{tabular}{lcc}
\toprule
\textbf{Policy} & \textbf{Grasp Success (\%)} & \textbf{Lift Success (\%)} \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    rows = []
    for result in results:
        label = "No-Tactile" if result.get("no_tactile", False) else "Tactile (Ours)"
        grasp = result.get("grasp_success_rate", 0) * 100
        lift = result.get("lift_success_rate", 0) * 100
        rows.append(f"{label} & {grasp:.1f} & {lift:.1f} \\\\")

    content = header + "\n".join(rows) + footer

    with open(output_path, "w") as f:
        f.write(content)
    print(f"[plot] Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--log_dir", type=str, default="logs/",
                        help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Directory to save figures")
    parser.add_argument("--eval_results", type=str, nargs="+", default=None,
                        help="Path(s) to evaluation result JSON files")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["all", "learning_curves", "comparison", "ablation", "real_robot"],
                        help="Which plot to generate")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation results if provided
    eval_results = []
    if args.eval_results:
        for path in args.eval_results:
            with open(path) as f:
                eval_results.append(json.load(f))

    # Generate plots
    if args.plot in ["all", "learning_curves"]:
        tactile_dir = os.path.join(args.log_dir, "hand_grasp_tactile")
        no_tactile_dir = os.path.join(args.log_dir, "hand_grasp_no_tactile")
        if os.path.exists(tactile_dir) and os.path.exists(no_tactile_dir):
            plot_learning_curves(
                tactile_dir,
                no_tactile_dir,
                os.path.join(args.output_dir, "learning_curves.png"),
            )

    if args.plot in ["all", "comparison"]:
        if eval_results:
            plot_success_comparison(
                eval_results,
                os.path.join(args.output_dir, "success_comparison.png"),
            )
            generate_latex_table(
                eval_results,
                os.path.join(args.output_dir, "results_table.tex"),
            )

    if args.plot in ["all", "ablation"]:
        # Example ablation data (replace with actual results)
        ablation_results = {
            "Full (Tactile + Pose)": 0.85,
            "Tactile Only": 0.72,
            "Pose Only": 0.65,
            "Position Only": 0.58,
        }
        plot_ablation_study(
            ablation_results,
            os.path.join(args.output_dir, "ablation_study.png"),
        )

    if args.plot in ["all", "real_robot"]:
        # Example real robot data (replace with actual results)
        real_robot_results = {
            "Tactile Policy": {"success": 16, "total": 20},
            "No-Tactile Policy": {"success": 11, "total": 20},
        }
        plot_real_robot_results(
            real_robot_results,
            os.path.join(args.output_dir, "real_robot_results.png"),
        )

    print(f"[plot] All figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
