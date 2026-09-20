"""
collect_results.py

Aggregates all per-scenario, per-baseline JSONL logs produced by
run_baselines.py into a single tidy CSV, ready for analysis/plots.py and
analysis/generate_report.py.
"""

import glob
import json
import os

import pandas as pd


def collect(results_dir="experiments/results", out_csv="experiments/results/aggregated.csv"):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        fname = os.path.basename(path)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_source_file"] = fname
                rows.append(record)

    if not rows:
        print(f"No JSONL records found in {results_dir}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} records to {out_csv}")


if __name__ == "__main__":
    collect()
