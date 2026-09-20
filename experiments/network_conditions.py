"""
network_conditions.py

Wraps `tc netem` to inject controllable, repeatable latency/bandwidth
conditions on the network interface used to reach the cloud server. This is
what makes RouteEdge's "adaptive behavior, not a static device comparison"
claim testable: the same gesture, run under different tc netem profiles,
should produce different routing decisions from the same scoring function.

Requires root (or CAP_NET_ADMIN) to run `tc` commands. Intended to run on
whichever machine hosts the Pi<->server link you want to control — if
you're using two containers on one host instead of physical hardware (see
earlier discussion), point IFACE at the virtual bridge/veth interface
between them instead.
"""

import subprocess

PRESETS = {
    "good":   {"delay_ms": 5,   "jitter_ms": 1,  "rate_mbit": 100},
    "medium": {"delay_ms": 50,  "jitter_ms": 10, "rate_mbit": 20},
    "bad":    {"delay_ms": 200, "jitter_ms": 50, "rate_mbit": 2},
}


def apply_condition(iface, preset_name):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                          f"Options: {list(PRESETS)}")
    p = PRESETS[preset_name]

    clear_conditions(iface)  # avoid stacking multiple qdiscs across runs

    cmd = [
        "tc", "qdisc", "add", "dev", iface, "root", "netem",
        "delay", f"{p['delay_ms']}ms", f"{p['jitter_ms']}ms",
        "rate", f"{p['rate_mbit']}mbit",
    ]
    subprocess.run(cmd, check=True)
    print(f"[network_conditions] applied '{preset_name}' preset to {iface}: {p}")


def clear_conditions(iface):
    # Ignore failure — this fails harmlessly if no qdisc is currently set.
    subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"],
                    stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True, help="e.g. eth0, wlan0, veth0")
    parser.add_argument("--preset", choices=list(PRESETS), required=True)
    args = parser.parse_args()

    apply_condition(args.iface, args.preset)
