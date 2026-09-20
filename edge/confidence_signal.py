"""
confidence_signal.py

Derives a normalized ConfidenceSignal from gesture_detect.py's existing
outputs, with NO additional model inference. Pure function, called once per
frame from inside gesture_detect.py's main loop.

This is the translation layer between the CV pipeline (raw pixel counts,
contour geometry, angles) and the C router (which expects normalized floats
in [0, 1], matching the ConfidenceSignal struct defined in router/router.h).
"""

from collections import Counter

import cv2
import numpy as np


class ConfidenceSignal:
    """
    Mirrors the C struct in router/router.h:

        typedef struct {
            int   has_contour;
            float aspect_margin;
            float mask_noise;
            float flicker_rate;
            float brightness;
        } ConfidenceSignal;

    Field polarity (important — the router's cost weighting depends on this):
        has_contour = True / aspect_margin near 1.0  -> GOOD, confident local read
        mask_noise near 1.0 / flicker_rate near 1.0  -> BAD, unreliable local read
    """
    __slots__ = ["has_contour", "aspect_margin", "mask_noise",
                 "flicker_rate", "brightness"]

    def __init__(self, has_contour, aspect_margin, mask_noise,
                 flicker_rate, brightness):
        self.has_contour   = has_contour
        self.aspect_margin = aspect_margin
        self.mask_noise    = mask_noise
        self.flicker_rate  = flicker_rate
        self.brightness    = brightness

    def as_tuple(self):
        """Convenience for ctypes marshalling in router_bindings.py."""
        return (
            int(self.has_contour),
            float(self.aspect_margin),
            float(self.mask_noise),
            float(self.flicker_rate),
            float(self.brightness),
        )

    def __repr__(self):
        return (f"ConfidenceSignal(has_contour={self.has_contour}, "
                f"aspect_margin={self.aspect_margin:.3f}, "
                f"mask_noise={self.mask_noise:.3f}, "
                f"flicker_rate={self.flicker_rate:.3f}, "
                f"brightness={self.brightness:.3f})")


def _aspect_margin(dims, aspect_thresh):
    """
    How far the detected blob's aspect ratio sits from the finger-like
    threshold. 0 = right at the boundary (ambiguous shape), 1 = clearly on
    one side or the other (confident either way).

    dims comes from finger_direction()'s third return value: (w, h) of the
    minimum-area bounding rectangle around the contour, with w always the
    longer side. dims can be non-None even when direction is None (the
    aspect check can fail before the moment/tip computation runs) — that
    case is exactly when this signal is most useful.
    """
    if dims is None or dims[1] == 0:
        return 0.0  # no shape info at all -> treat as fully ambiguous
    ratio = dims[0] / dims[1]
    margin = abs(ratio - aspect_thresh) / aspect_thresh
    return float(np.clip(margin, 0.0, 1.0))


def _mask_noise(mask, max_expected_contours=20):
    """
    Counts ALL contours in the mask, before the min_area filter used
    elsewhere is applied. A noisy background (skin-colored clutter, bad
    lighting, motion blur) produces many small spurious contours even when
    the "real" hand contour is also present.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    return float(np.clip(count / max_expected_contours, 0.0, 1.0))


def _flicker_rate(history):
    """
    Fraction of recent frames whose raw reading disagrees with the current
    majority vote. High flicker = unstable local detection (the detector is
    "arguing with itself" frame to frame).
    """
    valid = [d for d in history if d is not None]
    if len(valid) < 2:
        return 0.0
    majority, _ = Counter(valid).most_common(1)[0]
    disagreements = sum(1 for d in valid if d != majority)
    return disagreements / len(valid)


def _brightness(frame):
    """
    Mean V channel of HSV, normalized to [0, 1]. Very dark or blown-out
    frames make HSV skin-thresholding unreliable regardless of what the
    contour/aspect signals say.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    return mean_v / 255.0


def build_confidence_signal(frame, mask, contour, dims, history, aspect_thresh):
    """
    Called once per frame from gesture_detect.py's main loop, using values
    it already computed (mask, contour, dims, history) plus the current
    frame for brightness. No extra model inference is performed here.
    """
    return ConfidenceSignal(
        has_contour   = contour is not None,
        aspect_margin = _aspect_margin(dims, aspect_thresh),
        mask_noise    = _mask_noise(mask),
        flicker_rate  = _flicker_rate(history),
        brightness    = _brightness(frame),
    )
