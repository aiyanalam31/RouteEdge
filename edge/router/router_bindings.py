"""
router_bindings.py

ctypes wrapper around librouter.so (built from router.c by the Makefile in
this directory). Exposes a plain-looking Python function, router_decide(),
so gesture_detect.py doesn't need to know anything about struct marshalling.

Build the library first:
    cd edge/router && make
"""

import ctypes
import os
from enum import IntEnum


class RouteDecision(IntEnum):
    LOCAL = 0
    CLOUD = 1


class _CConfidenceSignal(ctypes.Structure):
    """Must exactly match ConfidenceSignal in router.h, field order included."""
    _fields_ = [
        ("has_contour",   ctypes.c_int),
        ("aspect_margin", ctypes.c_float),
        ("mask_noise",    ctypes.c_float),
        ("flicker_rate",  ctypes.c_float),
        ("brightness",    ctypes.c_float),
    ]


class _CRouteResult(ctypes.Structure):
    """Must exactly match RouteResult in router.h."""
    _fields_ = [
        ("decision",   ctypes.c_int),
        ("local_cost", ctypes.c_float),
        ("cloud_cost", ctypes.c_float),
    ]


class RouteResult:
    """Plain Python-facing result, unwrapped from the ctypes struct."""
    __slots__ = ["decision", "local_cost", "cloud_cost"]

    def __init__(self, decision, local_cost, cloud_cost):
        self.decision   = RouteDecision(decision)
        self.local_cost = local_cost
        self.cloud_cost = cloud_cost

    def __repr__(self):
        return (f"RouteResult(decision={self.decision.name}, "
                f"local_cost={self.local_cost:.4f}, "
                f"cloud_cost={self.cloud_cost:.4f})")


_LIB_PATH = os.path.join(os.path.dirname(__file__), "librouter.so")
_lib = ctypes.CDLL(_LIB_PATH)

_lib.router_decide.argtypes = [
    _CConfidenceSignal,
    ctypes.c_float,  # network_delay_ms
    ctypes.c_float,  # predicted_cloud_latency_ms
    ctypes.c_int,    # privacy_flag
]
_lib.router_decide.restype = _CRouteResult


def router_decide(signal, network_delay_ms, predicted_cloud_latency_ms,
                   privacy_flag):
    """
    signal: a confidence_signal.ConfidenceSignal instance (Python side).
    Returns a RouteResult with .decision, .local_cost, .cloud_cost.
    """
    c_sig = _CConfidenceSignal(*signal.as_tuple())
    c_result = _lib.router_decide(
        c_sig,
        float(network_delay_ms),
        float(predicted_cloud_latency_ms),
        int(bool(privacy_flag)),
    )
    return RouteResult(c_result.decision, c_result.local_cost, c_result.cloud_cost)
