"""
logger.py

Writes one structured JSON record per frame to a JSONL file (one JSON object
per line — easy to append to, easy to stream, easy to parse later). This is
the data source for:
  - dashboard/app.py (tails the file for the live view)
  - experiments/collect_results.py (aggregates across scenario runs)
  - analysis/*.py (builds plots and decision traces)

Keeping this format simple and flat (rather than nesting everything into
custom objects) makes it trivial to load into pandas later with
`pd.read_json(path, lines=True)`.
"""

import json
import time


class FrameLogger:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def log(self, frame_id, signal, route_result, final_direction, source,
             latency_ms, energy_estimate):
        record = {
            "timestamp":        time.time(),
            "frame_id":         frame_id,
            "has_contour":      signal.has_contour,
            "aspect_margin":    signal.aspect_margin,
            "mask_noise":       signal.mask_noise,
            "flicker_rate":     signal.flicker_rate,
            "brightness":       signal.brightness,
            "decision":         route_result.decision.name,  # "LOCAL" / "CLOUD"
            "local_cost":       route_result.local_cost,
            "cloud_cost":       route_result.cloud_cost,
            "final_direction":  final_direction,
            "source":           source,
            "latency_ms":       latency_ms,
            "energy_estimate_mj": energy_estimate,
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self):
        self._fh.close()
