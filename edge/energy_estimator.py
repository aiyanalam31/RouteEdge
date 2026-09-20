"""
energy_estimator.py

CPU-utilization-based power PROXY for the Pi side of the evaluation. This is
explicitly a proxy, not a measured value — there is no physical power meter
in this build. If a USB power meter or an INA219 sensor is later wired in,
swap the `sample()` implementation to read real wattage and keep the same
interface so nothing else in the codebase needs to change.

Model: estimated_energy_mJ = cpu_utilization * elapsed_time_s * assumed_TDP_W * 1000
assumed_TDP_W is a rough per-core draw figure for the Pi 5's Cortex-A76 under
load; calibrate this against published Pi 5 power figures or your own
smart-plug measurement before trusting absolute numbers. Relative comparisons
across routing conditions (local vs. cloud, high vs. low load) are more
defensible than absolute joule counts from this proxy.
"""

import time

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


ASSUMED_TDP_W_PER_CORE = 1.75  # rough figure for Pi 5 Cortex-A76 under load;
                                # calibrate against your own measurements


class EnergyEstimator:
    def __init__(self):
        self._last_sample_time = time.time()
        if _HAVE_PSUTIL:
            psutil.cpu_percent(interval=None)  # prime the internal counter

    def sample(self):
        """
        Returns an estimated energy cost (in millijoules) for the time
        elapsed since the last call to sample(). Call this once per frame,
        right after the routing decision, so the logged energy figure
        corresponds to that frame's processing window.
        """
        now = time.time()
        elapsed_s = now - self._last_sample_time
        self._last_sample_time = now

        if _HAVE_PSUTIL:
            cpu_pct = psutil.cpu_percent(interval=None) / 100.0
        else:
            # No psutil available (e.g. running this file's tests off-Pi) —
            # fall back to a fixed mid-range utilization assumption so the
            # pipeline still produces a number rather than crashing.
            cpu_pct = 0.5

        num_cores = 4  # Pi 5 quad-core
        estimated_mj = cpu_pct * elapsed_s * ASSUMED_TDP_W_PER_CORE * num_cores * 1000.0
        return estimated_mj
