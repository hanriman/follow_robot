#!/usr/bin/env python3
"""
Generate publication-quality training curves for Curriculum PPO.
Plots:
1. Training Episodic Return & Unlocked Stage Progression over 400k steps.
2. Per-Stage Evaluation Returns (Stage 1 Open, Stage 2 Corner, Stage 3 U-Turn, Stage 4 Clutter).
"""

import os
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

csv_path = Path("outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/logs/metrics.csv")
output_dir = Path("/Users/hanriman/.gemini/antigravity-cli/brain/1dcba7bc-66da-4699-acec-516ddc9ad429")
output_dir.mkdir(parents=True, exist_ok=True)

steps = []
returns = []
stages = []
unlocked_stages = []

eval_steps = []
eval_rewards = []
s1_rews = []
s2_rews = []
s3_rews = []
s4_rews = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = float(row["step"])
        steps.append(step)
        returns.append(float(row["mean_ep_return"]))
        stages.append(float(row["curriculum_stage"]))
        unlocked_stages.append(float(row["unlocked_stage"]))

        if row.get("eval_reward") and row["eval_reward"].strip():
            eval_steps.append(step)
            eval_rewards.append(float(row["eval_reward"]))
            s1_rews.append(float(row["s1_rew"]))
            s2_rews.append(float(row["s2_rew"]))
            s3_rews.append(float(row["s3_rew"]))
            s4_rews.append(float(row["s4_rew"]))

steps = np.array(steps)
returns = np.array(returns)
unlocked_stages = np.array(unlocked_stages)
eval_steps = np.array(eval_steps)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Colors
c_ret = "#2b5c8f"
c_stage = "#e06c75"
colors = {
    "s1_rew": "#61afef",  # blue
    "s2_rew": "#98c379",  # green
    "s3_rew": "#e5c07b",  # yellow/orange
    "s4_rew": "#c678dd",  # purple
}

# --- Plot 1: Training Return & Unlocked Stage ---
ax1 = axes[0]
ax1_stage = ax1.twinx()

# Smooth return curve using numpy convolve
kernel_size = 5
kernel = np.ones(kernel_size) / kernel_size
smooth_returns = np.convolve(returns, kernel, mode="same")

ax1.plot(steps / 1000, smooth_returns, color=c_ret, linewidth=2.0, label="Training Mean Return (Smoothed)")
ax1.plot(steps / 1000, returns, color=c_ret, alpha=0.25, linewidth=1.0)
ax1.set_ylabel("Episodic Return", color=c_ret, fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=c_ret)

# Unlocked stage step line
ax1_stage.step(steps / 1000, unlocked_stages, color=c_stage, linewidth=2.5, where="post", linestyle="--", label="Unlocked Stage")
ax1_stage.set_ylabel("Curriculum Stage Unlocked", color=c_stage, fontsize=12, fontweight="bold")
ax1_stage.set_yticks([1, 2, 3, 4])
ax1_stage.set_yticklabels(["Stage 1\n(Open)", "Stage 2\n(Corner)", "Stage 3\n(U-Turn)", "Stage 4\n(Clutter)"])
ax1_stage.tick_params(axis="y", labelcolor=c_stage)
ax1_stage.grid(False)

# Graduation milestone markers
milestones = [
    (6.144, "Stage 2 Unlocked", "#98c379"),
    (16.384, "Stage 3 Unlocked", "#e5c07b"),
    (22.528, "Stage 4 Unlocked", "#c678dd"),
]
for x_k, label, col in milestones:
    ax1.axvline(x_k, color=col, linestyle=":", alpha=0.8, linewidth=1.8)
    ax1.text(x_k + 2.0, 150, label, rotation=90, verticalalignment="bottom", color=col, fontweight="bold", fontsize=10)

ax1.set_title("Curriculum PPO Progression: From-Scratch Training (400k Steps)", fontsize=14, fontweight="bold", pad=12)

# --- Plot 2: Per-Stage Evaluation Returns ---
ax2 = axes[1]
ax2.plot(eval_steps / 1000, s1_rews, marker="o", color=colors["s1_rew"], linewidth=2.0, label="Stage 1: Open Pursuit")
ax2.plot(eval_steps / 1000, s2_rews, marker="s", color=colors["s2_rew"], linewidth=2.0, label="Stage 2: Blind Corner")
ax2.plot(eval_steps / 1000, s3_rews, marker="^", color=colors["s3_rew"], linewidth=2.0, label="Stage 3: Corridor U-Turn")
ax2.plot(eval_steps / 1000, s4_rews, marker="D", color=colors["s4_rew"], linewidth=2.0, label="Stage 4: Clutter Arena")

# Highlight best model checkpoint at 40.96k steps
ax2.axvline(40.960, color="#d19a66", linestyle="--", linewidth=2.0, alpha=0.9)
ax2.text(42.5, 950, "★ Best Model (Composite R = 528.4)", color="#d19a66", fontweight="bold", fontsize=11)

ax2.set_xlabel("Environment Steps (Thousands)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Evaluation Return", fontsize=12, fontweight="bold")
ax2.set_title("Periodic Deterministic Evaluation Breakdown Across All 4 Stages", fontsize=13, fontweight="bold", pad=10)
ax2.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plot_path = output_dir / "curriculum_training_curves.png"
plt.savefig(plot_path, dpi=200)
print(f"Saved curriculum training curve to: {plot_path}")
