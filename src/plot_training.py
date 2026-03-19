"""
plot_training.py

Run AFTER training to produce all figures and tables for the report.
Reads from the CSV logs that SB3 writes via the custom callback below.

HOW TO USE:
1. Add LoggingCallback to your train_boat_ppo.py (see bottom of this file)
2. Train — this writes training_log.csv
3. Run: python plot_training.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
RUN_NAME = "fullcsv.csv"
LOG_FILE = os.path.join("./csvs/", RUN_NAME)
OUT_DIR = Path("./reports")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load logs ─────────────────────────────────────────────────────────────

df = pd.read_csv(LOG_FILE)
print(df.columns.tolist())
print(df.head())

# Rolling window for smoothing
WINDOW = 20

def smooth(series, w=WINDOW):
    return series.rolling(w, min_periods=1).mean()

# ── 2. Episode reward over time ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['timestep'], smooth(df['ep_rew_mean']),
        label='Mean episode reward (smoothed)', color='steelblue', lw=2)
ax.fill_between(df['timestep'],
                smooth(df['ep_rew_mean']) - smooth(df['ep_rew_std']),
                smooth(df['ep_rew_mean']) + smooth(df['ep_rew_std']),
                alpha=0.2, color='steelblue', label='±1 std dev')
ax.axhline(0, color='gray', lw=0.8, linestyle='--')
ax.set_xlabel("Environment timesteps")
ax.set_ylabel("Mean episode reward")
ax.set_title("Training Curve — PPO Ice Boat Racer")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_training_curve.png", dpi=150)
print("Saved fig1_training_curve.png")

# ── 3. Checkpoints reached per episode ───────────────────────────────────────

if 'checkpoints_reached' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestep'], smooth(df['checkpoints_reached']),
            color='darkorange', lw=2, label='Avg checkpoints/episode (smoothed)')
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Checkpoints reached")
    ax.set_title("Learning Progress — Checkpoints Reached Per Episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_checkpoints.png", dpi=150)
    print("Saved fig2_checkpoints.png")

# ── 4. Episode length over time ───────────────────────────────────────────────

if 'ep_len_mean' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestep'], smooth(df['ep_len_mean']),
            color='mediumseagreen', lw=2, label='Mean episode length (smoothed)')
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Steps per episode")
    ax.set_title("Episode Length — Longer = More Exploration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_ep_length.png", dpi=150)
    print("Saved fig3_ep_length.png")

# ── 5. Reward component breakdown table ──────────────────────────────────────
# This is a STATIC table you fill in manually based on your reward function.
# Shows every reward component, its range, and its purpose.

reward_table = pd.DataFrame([
    # component, min, max, condition, purpose
    ("Lava penalty",          -50.0,  -50.0, "YPos < 226.5",            "Terminal deterrent — fall in lava"),
    ("Idle pressure",          -1.0,    0.0, "Always, quadratic growth", "Forces progress, prevents stationary farming"),
    ("Alignment (moving)",     0.0,    1.5,  "speed > 0.1",             "Rewards facing checkpoint while in motion"),
    ("Alignment (stationary)", 0.0,    0.2,  "speed ≤ 0.1",             "Weak orientation signal at spawn"),
    ("Dwell bonus",            0.0,    1.0,  "abs_rel<30°, speed>0.1",  "Rewards sustained alignment over twitching"),
    ("Steering correction",   -3.0,   +3.0,  "abs_rel > 15°",           "Outcome-based: rewards abs_rel decreasing"),
    ("Edge penalty",          -float('inf'), 0.0,  "front_edge & speed>0.3",  "Penalises charging at track edges"),
    ("Speed reward",           0.0,          2.0,  "abs_rel<45°, no edge",    "Rewards forward motion when aligned"),
    ("Progress reward",        0.0,   float('inf'), "abs_rel<45°, delta>0",   "Rewards closing distance to checkpoint"),
    ("Checkpoint bonus",      +100.0, +100.0,"dist < 5.0",              "Sparse goal signal"),
    ("Lap complete",          +200.0, +200.0,"all checkpoints done",    "Terminal success signal"),
])
reward_table.columns = ["Component", "Min", "Max", "Condition", "Purpose"]

print("\n=== Reward Component Table ===")
print(reward_table.to_string(index=False))

# Save as CSV for copy-paste into report
reward_table.to_csv(OUT_DIR / "reward_table.csv", index=False)
print("\nSaved reward_table.csv")

# ── 6. Observation space table ────────────────────────────────────────────────

obs_table = pd.DataFrame([
    (0,    "Next checkpoint Δx",          "blocks", "Signed offset to checkpoint 1"),
    (1,    "Next checkpoint Δz",          "blocks", "Signed offset to checkpoint 1"),
    (2,    "Checkpoint+1 Δx",             "blocks", "Look-ahead: checkpoint 2"),
    (3,    "Checkpoint+1 Δz",             "blocks", "Look-ahead: checkpoint 2"),
    (4,    "Checkpoint+2 Δx",             "blocks", "Look-ahead: checkpoint 3"),
    (5,    "Checkpoint+2 Δz",             "blocks", "Look-ahead: checkpoint 3"),
    (6,    "Velocity x (vx)",             "m/tick", "East-West momentum"),
    (7,    "Velocity z (vz)",             "m/tick", "North-South momentum"),
    (8,    "Angular velocity",            "[-1,1]",  "Yaw change/180 — spinout detection"),
    (9,    "cos(rel angle to checkpoint)","—",       "1=aligned, -1=facing away"),
    (10,   "sin(rel angle to checkpoint)","—",       "+=target right, -=target left"),
    (11,   "Front edge sensor",           "binary",  "1=lava within 4 blocks ahead"),
    (12,   "Right edge sensor",           "binary",  "1=lava within 4 blocks right"),
    (13,   "Left edge sensor",            "binary",  "1=lava within 4 blocks left"),
    (14,   "Behind edge sensor",          "binary",  "1=lava within 4 blocks behind"),
])
obs_table.columns = ["Index", "Feature", "Unit", "Description"]
print("\n=== Observation Space Table ===")
print(obs_table.to_string(index=False))
obs_table.to_csv(OUT_DIR / "obs_table.csv", index=False)
print("Saved obs_table.csv")

# ── 7. Action space table ─────────────────────────────────────────────────────

action_table = pd.DataFrame([
    ("[0, 0]", "Coast",         "No throttle, no steer — boat decelerates on ice"),
    ("[1, 0]", "Forward",       "Throttle on, straight ahead"),
    ("[1, 1]", "Forward+Left",  "Throttle on, steer left"),
    ("[1, 2]", "Forward+Right", "Throttle on, steer right"),
])
action_table.columns = ["Action [throttle, steer]", "Name", "Description"]
print("\n=== Action Space Table ===")
print(action_table.to_string(index=False))
action_table.to_csv(OUT_DIR / "action_table.csv", index=False)
print("Saved action_table.csv")

print(f"\nAll outputs in: {OUT_DIR.resolve()}")