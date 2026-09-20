"""
gesture_model.py

Loads and runs the cloud-side gesture model. This is deliberately a
different, more capable method than the Pi's classical CV pipeline — not a
shared-feature-map split of the same model (see project discussion: this is
the "independent models" pattern, since the local method isn't even a
neural network to begin with).

Default implementation uses MediaPipe Hands (pretrained, no training
required) to get 21 3D hand landmarks, then derives a direction (and
optionally a richer gesture) from landmark geometry — giving a real
robustness/capability jump over the Pi's HSV-threshold approach: lighting
and background invariance, and room to recognize more than just LEFT/RIGHT
if you extend `classify_from_landmarks`.
"""

import numpy as np

try:
    import mediapipe as mp
    _HAVE_MEDIAPIPE = True
except ImportError:
    _HAVE_MEDIAPIPE = False


class GestureModel:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.6):
        if not _HAVE_MEDIAPIPE:
            raise RuntimeError(
                "mediapipe not installed. Run scripts/setup_server.sh, "
                "or swap this class's internals for your own trained CNN."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,  # each request is an independent frame
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def infer(self, bgr_frame):
        """
        bgr_frame: numpy array, HxWx3, BGR (as decoded from the JPEG the
        Pi sends). Returns a dict: {"direction": "LEFT"|"RIGHT"|None,
        "confidence": float, "landmarks_found": bool}.
        """
        rgb_frame = bgr_frame[:, :, ::-1]  # BGR -> RGB for mediapipe
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {"direction": None, "confidence": 0.0, "landmarks_found": False}

        landmarks = results.multi_hand_landmarks[0]
        direction, confidence = self._classify_from_landmarks(landmarks)
        return {"direction": direction, "confidence": confidence,
                "landmarks_found": True}

    @staticmethod
    def _classify_from_landmarks(landmarks):
        """
        Simple direction heuristic from index-finger landmarks: compare the
        fingertip (landmark 8) x-position to the base-of-finger (landmark 5)
        x-position. Extend this to recognize more gestures (thumbs up,
        peace sign, etc.) using the full 21-point landmark set — that
        richer vocabulary is the real capability gap over the Pi's method.
        """
        pts = landmarks.landmark
        tip_x, base_x = pts[8].x, pts[5].x
        direction = "LEFT" if tip_x < base_x else "RIGHT"
        confidence = float(np.clip(abs(tip_x - base_x) * 10.0, 0.0, 1.0))
        return direction, confidence
