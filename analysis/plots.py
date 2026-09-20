"""
plots.py

Generates the latency/energy-vs-condition plots referenced in the
proposal's evaluation section, from the aggregated CSV produced by
experiments/collect_results.py.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def load_data(csv_path="experiments/results/aggregated.csv"):
    return pd.read_csv(csv_path)


def plot_latency_by_scenario_and_baseline(df, out_path="analysis/latency_comparison.png"):
    """Grouped bar chart: avg latency per (scenario, baseline) pair. This is
    the headline plot for the 'RouteEdge beats fixed baselines' claim."""
    if "scenario" not in df.columns or "baseline" not in df.columns:
        print("Expected 'scenario' and 'baseline' columns — "
              "run experiments/run_baselines.py first.")
        return

    pivot = df.pivot_table(values="latency_ms", index="scenario",
                            columns="baseline", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Average Latency by Scenario and Routing Policy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_energy_by_scenario_and_baseline(df, out_path="analysis/energy_comparison.png"):
    if "energy_estimate_mj" not in df.columns:
        print("No energy_estimate_mj column found.")
        return

    pivot = df.pivot_table(values="energy_estimate_mj", index="scenario",
                            columns="baseline", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Avg estimated energy (mJ)")
    ax.set_title("Average Estimated Energy by Scenario and Routing Policy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_routing_split(df, out_path="analysis/routing_split.png"):
    """Shows the % of frames routed local vs. cloud, per scenario — this is
    the plot that most directly demonstrates 'adaptive behavior, not a
    static comparison', since it should visibly shift across scenarios."""
    if "decision" not in df.columns:
        print("No 'decision' column found.")
        return

    routeedge_df = df[df["baseline"] == "routeedge"] if "baseline" in df.columns else df
    split = routeedge_df.groupby("scenario")["decision"].value_counts(normalize=True).unstack().fillna(0)
    ax = split.plot(kind="bar", stacked=True, figsize=(10, 6), color=["#4d3a1f", "#1f4d2c"])
    ax.set_ylabel("Fraction of frames")
    ax.set_title("RouteEdge Routing Split by Scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    df = load_data()
    plot_latency_by_scenario_and_baseline(df)
    plot_energy_by_scenario_and_baseline(df)
    plot_routing_split(df)
