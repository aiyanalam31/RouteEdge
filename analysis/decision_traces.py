"""
decision_traces.py

Visualizes routing decisions over time as a timeline — the "same request
routed differently as conditions change" evidence the proposal promises for
the demo video. Plots decision (LOCAL/CLOUD) per frame_id, with the
underlying confidence signals and network delay overlaid so a viewer can
see WHY the router flipped, not just that it did.
"""

import matplotlib.pyplot as plt
import pandas as pd


def load_session(jsonl_path):
    return pd.read_json(jsonl_path, lines=True)


def plot_decision_trace(df, out_path="analysis/decision_trace.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel 1: routing decision over time
    decision_numeric = df["decision"].map({"LOCAL": 0, "CLOUD": 1})
    axes[0].step(df["frame_id"], decision_numeric, where="post", color="#4a90d9")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["LOCAL", "CLOUD"])
    axes[0].set_title("Routing Decision Over Time")

    # Panel 2: local confidence signals
    axes[1].plot(df["frame_id"], df["aspect_margin"], label="aspect_margin")
    axes[1].plot(df["frame_id"], df["mask_noise"], label="mask_noise")
    axes[1].plot(df["frame_id"], df["flicker_rate"], label="flicker_rate")
    axes[1].plot(df["frame_id"], df["brightness"], label="brightness")
    axes[1].set_title("Local Confidence Signals")
    axes[1].legend(loc="upper right", fontsize=8)

    # Panel 3: cost comparison
    axes[2].plot(df["frame_id"], df["local_cost"], label="local_cost", color="#7ee29a")
    cloud_cost_clean = df["cloud_cost"].replace([float("inf")], df["cloud_cost"].replace([float("inf")], None).max())
    axes[2].plot(df["frame_id"], cloud_cost_clean, label="cloud_cost", color="#f0b45a")
    axes[2].set_title("Router Cost Comparison")
    axes[2].set_xlabel("Frame ID")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/session_live.jsonl"
    df = load_session(path)
    plot_decision_trace(df)
