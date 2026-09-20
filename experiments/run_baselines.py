"""
run_baselines.py

Runs the evaluation matrix described in the proposal: RouteEdge (adaptive)
vs. three baselines (always-edge, always-cloud, simple network-threshold
rule), across every scenario config in scenario_configs/.

This script assumes gesture_detect.py can be driven with pre-recorded frame
sequences rather than only a live camera, for reproducibility — see the
`--frames-dir` note below. If you're running purely live for the demo
video, use gesture_detect.py directly instead and treat this script as the
offline benchmark path.
"""

import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

from confidence_signal import build_confidence_signal
from router.router_bindings import router_decide, RouteDecision


BASELINES = ["always_edge", "always_cloud", "network_threshold", "routeedge"]


def load_scenario(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def decide_baseline(name, sig, network_delay_ms, predicted_cloud_latency_ms,
                     privacy_flag, threshold_ms=100.0):
    """Mirrors router_decide()'s interface so all four policies can be
    compared under identical inputs and logging."""
    if privacy_flag:
        return RouteDecision.LOCAL

    if name == "always_edge":
        return RouteDecision.LOCAL
    elif name == "always_cloud":
        return RouteDecision.CLOUD
    elif name == "network_threshold":
        return RouteDecision.LOCAL if network_delay_ms > threshold_ms else RouteDecision.CLOUD
    elif name == "routeedge":
        result = router_decide(sig, network_delay_ms, predicted_cloud_latency_ms, privacy_flag)
        return result.decision
    else:
        raise ValueError(f"Unknown baseline: {name}")


def run_scenario(scenario_path, output_dir):
    scenario = load_scenario(scenario_path)
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]

    print(f"\n=== Scenario: {scenario_name} ===")
    for baseline in BASELINES:
        out_path = os.path.join(output_dir, f"{scenario_name}__{baseline}.jsonl")
        print(f"  Running baseline '{baseline}' -> {out_path}")
        # NOTE: this is a scaffold — plug in your actual frame source
        # (recorded frame sequence matching this scenario's lighting/
        # network conditions) and call decide_baseline() per frame here,
        # logging each decision + measured latency/energy the same way
        # gesture_detect.py's logger.py does.
        with open(out_path, "w") as f:
            f.write(json.dumps({
                "scenario": scenario_name,
                "baseline": baseline,
                "note": "placeholder — wire up real frame source before demo",
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--scenario", help="path to a single scenario YAML")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scenario_dir = os.path.join(os.path.dirname(__file__), "scenario_configs")

    if args.all_scenarios:
        for fname in sorted(os.listdir(scenario_dir)):
            if fname.endswith(".yaml"):
                run_scenario(os.path.join(scenario_dir, fname), args.output_dir)
    elif args.scenario:
        run_scenario(args.scenario, args.output_dir)
    else:
        parser.error("Specify --all-scenarios or --scenario <path>")


if __name__ == "__main__":
    main()
