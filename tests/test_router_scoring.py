"""
test_router_scoring.py

Unit tests for router_decide() in router/router.c, exercised through
router_bindings.py's ctypes wrapper. Build librouter.so first:

    cd edge/router && make

Run with:
    python -m pytest tests/test_router_scoring.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

import pytest

from confidence_signal import ConfidenceSignal
from router.router_bindings import RouteDecision, router_decide


def confident_local_signal():
    """A clean, confident local detection — contour found, aspect ratio far
    from the threshold, quiet mask, stable history, good lighting."""
    return ConfidenceSignal(
        has_contour=True, aspect_margin=0.9, mask_noise=0.05,
        flicker_rate=0.0, brightness=0.7,
    )


def poor_local_signal():
    """No contour at all — the worst possible local signal."""
    return ConfidenceSignal(
        has_contour=False, aspect_margin=0.0, mask_noise=0.9,
        flicker_rate=0.8, brightness=0.15,
    )


def ambiguous_local_signal():
    """Contour found, but aspect ratio right at the threshold (a roughly
    square blob) — this is the case test_confidence_signal.py's docstring
    flags as the one most worth explicitly testing."""
    return ConfidenceSignal(
        has_contour=True, aspect_margin=0.02, mask_noise=0.1,
        flicker_rate=0.1, brightness=0.6,
    )


class TestRouterDecide:
    def test_confident_local_and_good_network_stays_local(self):
        result = router_decide(confident_local_signal(),
                                network_delay_ms=10.0,
                                predicted_cloud_latency_ms=30.0,
                                privacy_flag=False)
        assert result.decision == RouteDecision.LOCAL

    def test_poor_local_and_good_network_escalates_to_cloud(self):
        result = router_decide(poor_local_signal(),
                                network_delay_ms=10.0,
                                predicted_cloud_latency_ms=30.0,
                                privacy_flag=False)
        assert result.decision == RouteDecision.CLOUD

    def test_poor_local_but_bad_network_may_still_stay_local(self):
        """Even a poor local signal shouldn't automatically escalate if the
        network is bad enough — the router should weigh both costs, not
        just react to local confidence alone."""
        result = router_decide(poor_local_signal(),
                                network_delay_ms=500.0,
                                predicted_cloud_latency_ms=300.0,
                                privacy_flag=False)
        # Not asserting a specific outcome here since it depends on tuned
        # weights, but the two costs should reflect the tradeoff:
        assert result.cloud_cost > 1.0  # bad network should be visibly costly

    def test_privacy_flag_forces_local_regardless_of_signal(self):
        """Hard constraint: even the worst possible local signal must not
        escalate when privacy_flag is set."""
        result = router_decide(poor_local_signal(),
                                network_delay_ms=5.0,
                                predicted_cloud_latency_ms=10.0,
                                privacy_flag=True)
        assert result.decision == RouteDecision.LOCAL
        assert result.cloud_cost == float("inf")

    def test_ambiguous_aspect_ratio_increases_local_cost(self):
        """A near-square blob (ambiguous aspect ratio) should produce a
        higher local_cost than a clearly finger-shaped blob, all else
        equal — this is the sign-convention check flagged in
        confidence_signal.py's discussion."""
        confident_result = router_decide(confident_local_signal(), 10.0, 30.0, False)
        ambiguous_result = router_decide(ambiguous_local_signal(), 10.0, 30.0, False)
        assert ambiguous_result.local_cost > confident_result.local_cost

    def test_result_repr_does_not_crash(self):
        result = router_decide(confident_local_signal(), 10.0, 30.0, False)
        assert "decision=" in repr(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
