"""
load_agent.py

Tracks the cloud server's current load (concurrent connections relative to
capacity) and can periodically report it. In this reference implementation
the Pi's router doesn't yet consume this directly (its router_decide() call
uses network_delay_ms and predicted_cloud_latency_ms, both derived from
observed round-trip behavior on the Pi side) — but this is the natural hook
point for extending the router to react to genuine server-side congestion
rather than only network conditions.

Simplest integration path: have this agent periodically push its current
load figure to the Pi over a lightweight side-channel (or have the Pi poll
a small HTTP/JSON status endpoint added to server.py), and fold that into
NetworkProbe.predicted_cloud_latency_ms() as an additional weighted term.
"""

import threading
import time


class LoadAgent:
    def __init__(self, max_concurrent=8, report_interval_s=5.0):
        self.max_concurrent = max_concurrent
        self.report_interval_s = report_interval_s
        self._active_connections = 0
        self._lock = threading.Lock()

    def connection_started(self):
        with self._lock:
            self._active_connections += 1

    def connection_finished(self):
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def current_load_fraction(self):
        with self._lock:
            return self._active_connections / self.max_concurrent

    def start_reporting_thread(self):
        def _loop():
            while True:
                load = self.current_load_fraction()
                print(f"[load_agent] active={self._active_connections} "
                      f"load_fraction={load:.2f}")
                time.sleep(self.report_interval_s)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
