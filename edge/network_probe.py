"""
network_probe.py

Maintains a rolling estimate of network conditions between the Pi and the
cloud server, so the router isn't measuring fresh on every single frame
just to feed itself a number. Two sources of truth are blended:

1. Periodic lightweight pings (cheap, always available, measures raw
   round-trip time even when no cloud requests are happening).
2. A rolling average of ACTUAL cloud-request latencies, logged by
   cloud_client.py after every real request — this captures server queueing
   delay that a plain ping wouldn't see.

This is also the natural place to hook in for experiments/network_conditions.py,
which uses `tc netem` to shape the interface this class measures against —
those experiments don't need to touch this file at all, they just change
what it observes.
"""

import subprocess
import time
from collections import deque


class NetworkProbe:
    def __init__(self, server_ip, port, ping_interval_s=2.0,
                 history_len=20):
        self.server_ip = server_ip
        self.port = port
        self.ping_interval_s = ping_interval_s
        self._ping_history = deque(maxlen=history_len)
        self._cloud_latency_history = deque(maxlen=history_len)
        self._last_ping_time = 0.0
        self._last_delay_ms = 50.0  # sane default before first measurement

    def _do_ping(self):
        """One lightweight ICMP ping. Falls back to the last known value on
        failure (e.g. ICMP blocked) rather than raising, since routing
        decisions must never crash the camera loop."""
        try:
            out = subprocess.run(
                ["ping", "-c", "1", "-W", "1", self.server_ip],
                capture_output=True, text=True, timeout=2,
            )
            for line in out.stdout.splitlines():
                if "time=" in line:
                    time_part = line.split("time=")[1].split()[0]
                    return float(time_part)
        except Exception:
            pass
        return None

    def current_delay_ms(self):
        """Rolling round-trip estimate, refreshed at most every
        ping_interval_s seconds to avoid pinging on every frame."""
        now = time.time()
        if now - self._last_ping_time >= self.ping_interval_s:
            delay = self._do_ping()
            self._last_ping_time = now
            if delay is not None:
                self._ping_history.append(delay)
                self._last_delay_ms = delay

        if self._ping_history:
            return sum(self._ping_history) / len(self._ping_history)
        return self._last_delay_ms

    def record_cloud_latency(self, latency_ms):
        """Called by cloud_client.py after every real cloud request, so the
        rolling estimate reflects actual server-side behavior (queueing,
        model inference time), not just raw network RTT."""
        self._cloud_latency_history.append(latency_ms)

    def predicted_cloud_latency_ms(self, default_ms=80.0):
        """Estimate of end-to-end cloud processing time (excluding network
        transit, which is accounted for separately via current_delay_ms).
        Falls back to a config-driven default until enough real
        measurements have accumulated."""
        if self._cloud_latency_history:
            return sum(self._cloud_latency_history) / len(self._cloud_latency_history)
        return default_ms
