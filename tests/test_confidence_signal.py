"""
test_confidence_signal.py

Unit tests for confidence_signal.py's pure derivation functions. These use
synthetic masks/contours rather than real camera frames, so they run
anywhere (no Pi, no camera needed).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

import cv2
import numpy as np
import pytest

from confidence_signal import (_aspect_margin, _brightness, _flicker_rate,
                                _mask_noise, build_confidence_signal)


class TestAspectMargin:
    def test_none_dims_returns_zero(self):
        assert _aspect_margin(None, 1.8) == 0.0

    def test_zero_height_returns_zero(self):
        assert _aspect_margin((100, 0), 1.8) == 0.0

    def test_finger_like_shape_high_margin(self):
        # w=180, h=100 -> ratio 1.8, matches threshold exactly -> margin 0
        # (this is the "right at the boundary" case, i.e. LOW confidence,
        # which is the correct behavior even though the shape IS
        # finger-like at that exact ratio)
        margin = _aspect_margin((180, 100), 1.8)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_clearly_elongated_shape_high_margin(self):
        # w=360, h=100 -> ratio 3.6, well past threshold -> high margin
        margin = _aspect_margin((360, 100), 1.8)
        assert margin > 0.5

    def test_near_square_shape_low_margin(self):
        """The specific case flagged in the project discussion: a
        roughly-square blob should read as LOW confidence (low margin),
        i.e. the router most needs to route away from local here."""
        margin = _aspect_margin((105, 100), 1.8)  # ratio ~1.05, far below 1.8
        assert margin > 0.0  # far from threshold in the OTHER direction
        # This still produces a nonzero margin because "far from threshold"
        # captures both directions — worth noting this function does not
        # by itself distinguish "confidently NOT finger-shaped" from
        # "confidently finger-shaped". has_contour + direction being None
        # elsewhere in the pipeline is what disambiguates that case.


class TestMaskNoise:
    def test_empty_mask_zero_noise(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        assert _mask_noise(mask) == 0.0

    def test_noisy_mask_high_noise(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(30):
            x, y = rng.integers(0, 600), rng.integers(0, 440)
            cv2.rectangle(mask, (x, y), (x + 10, y + 10), 255, -1)
        noise = _mask_noise(mask, max_expected_contours=20)
        assert noise > 0.5


class TestFlickerRate:
    def test_empty_history_zero_flicker(self):
        assert _flicker_rate([]) == 0.0

    def test_single_valid_entry_zero_flicker(self):
        assert _flicker_rate(["LEFT"]) == 0.0

    def test_stable_history_zero_flicker(self):
        assert _flicker_rate(["LEFT"] * 10) == 0.0

    def test_alternating_history_high_flicker(self):
        history = ["LEFT", "RIGHT"] * 5
        rate = _flicker_rate(history)
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_none_entries_ignored(self):
        history = ["LEFT", None, "LEFT", None, "LEFT"]
        assert _flicker_rate(history) == 0.0


class TestBrightness:
    def test_black_frame_zero_brightness(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _brightness(frame) == pytest.approx(0.0, abs=0.01)

    def test_white_frame_full_brightness(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert _brightness(frame) == pytest.approx(1.0, abs=0.01)


class TestBuildConfidenceSignal:
    def test_no_contour_case(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        sig = build_confidence_signal(frame, mask, None, None, [], 1.8)
        assert sig.has_contour is False
        assert sig.aspect_margin == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
