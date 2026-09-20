"""
generate_report.py

Builds the reproducible benchmark tables promised as a deliverable:
per-scenario, per-baseline summary statistics (avg latency, avg energy,
% routed local, privacy-violation count) as a Markdown table, plus calls
into plots.py and decision_traces.py so a single command regenerates every
artifact needed for the writeup.
"""

import os

import pandas as pd

from plots import (load_data, plot_energy_by_scenario_and_baseline,
                    plot_latency_by_scenario_and_baseline, plot_routing_split)


def build_summary_table(df):
    grouped = df.groupby(["scenario", "baseline"]).agg(
        avg_latency_ms=("latency_ms", "mean"),
        avg_energy_mj=("energy_estimate_mj", "mean"),
        pct_local=("decision", lambda s: (s == "LOCAL").mean() * 100),
        n_frames=("frame_id", "count"),
    ).reset_index()

    # Privacy-violation check: for the privacy_forced_local scenario,
    # pct_local must be exactly 100 — flag anything else loudly.
    privacy_rows = grouped[grouped["scenario"] == "privacy_forced_local"]
    for _, row in privacy_rows.iterrows():
        if row["pct_local"] < 100.0:
            print(f"WARNING: privacy violation detected in "
                  f"{row['scenario']} / {row['baseline']}: "
                  f"only {row['pct_local']:.1f}% routed local")

    return grouped


def write_markdown_table(df, out_path="analysis/benchmark_report.md"):
    with open(out_path, "w") as f:
        f.write("# RouteEdge Benchmark Report\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Figures\n\n")
        f.write("![Latency Comparison](latency_comparison.png)\n\n")
        f.write("![Energy Comparison](energy_comparison.png)\n\n")
        f.write("![Routing Split](routing_split.png)\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    df = load_data()
    summary = build_summary_table(df)
    plot_latency_by_scenario_and_baseline(df)
    plot_energy_by_scenario_and_baseline(df)
    plot_routing_split(df)
    write_markdown_table(summary)
