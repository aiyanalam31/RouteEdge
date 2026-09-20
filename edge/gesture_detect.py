"""
Index Finger Direction Detector — Raspberry Pi Version, with RouteEdge routing.

Retrofitted from the laptop version, swapping cv2.VideoCapture for Picamera2.
Extended with a per-frame routing decision: the local CV pipeline below runs
every frame regardless (it's cheap), and its own outputs are used to derive a
confidence signal. A C routing engine (router/) decides, per frame, whether to
trust the local result or escalate the frame to the cloud gesture model.

Nothing about the original detection pipeline changes — routing is added
inline in main(), at the point where all needed signals already exist.
"""

import time

import cv2
import numpy as np
from picamera2 import Picamera2

from confidence_signal import build_confidence_signal
from router.router_bindings import router_decide, RouteDecision
from network_probe import NetworkProbe
from energy_estimator import EnergyEstimator
from logger import FrameLogger
import cloud_client
import config

# ── Defaults (tweak via trackbars at runtime) ─────────────────────────────────
DEFAULTS = dict(
    h_low=0,   h_high=25,
    s_low=30,  s_high=255,
    v_low=60,  v_high=255,
    min_area=3000,
    aspect=18,      # aspect ratio × 10  (1.8 = finger-like)
    smooth=9,
)
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
# ──────────────────────────────────────────────────────────────────────────────

params = dict(DEFAULTS)


def make_trackbars(win):
    cv2.createTrackbar("H low",      win, params["h_low"],    179,   lambda v: params.update(h_low=v))
    cv2.createTrackbar("H high",     win, params["h_high"],   179,   lambda v: params.update(h_high=v))
    cv2.createTrackbar("S low",      win, params["s_low"],    255,   lambda v: params.update(s_low=v))
    cv2.createTrackbar("S high",     win, params["s_high"],   255,   lambda v: params.update(s_high=v))
    cv2.createTrackbar("V low",      win, params["v_low"],    255,   lambda v: params.update(v_low=v))
    cv2.createTrackbar("V high",     win, params["v_high"],   255,   lambda v: params.update(v_high=v))
    cv2.createTrackbar("Min area",   win, params["min_area"], 15000, lambda v: params.update(min_area=max(v, 100)))
    cv2.createTrackbar("Aspect x10", win, params["aspect"],   50,    lambda v: params.update(aspect=max(v, 5)))
    cv2.createTrackbar("Smooth",     win, params["smooth"],   30,    lambda v: params.update(smooth=max(v, 1)))


def skin_mask(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo   = np.array([params["h_low"],  params["s_low"],  params["v_low"]],  dtype=np.uint8)
    hi   = np.array([params["h_high"], params["s_high"], params["v_high"]], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    if params["h_low"] <= 10:
        lo2  = np.array([170, params["s_low"],  params["v_low"]],  dtype=np.uint8)
        hi2  = np.array([179, params["s_high"], params["v_high"]], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    return mask


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c if cv2.contourArea(c) > params["min_area"] else None


def finger_direction(contour):
    if contour is None or len(contour) < 5:
        return None, None, None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w

    aspect_thresh = params["aspect"] / 10.0
    if h == 0 or w / h < aspect_thresh:
        return None, (cx, cy), (w, h)

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, (cx, cy), (w, h)

    mcx = M["m10"] / M["m00"]
    mcy = M["m01"] / M["m00"]

    dists = np.sqrt((contour[:, 0, 0] - mcx) ** 2 +
                    (contour[:, 0, 1] - mcy) ** 2)
    tip_x = contour[np.argmax(dists), 0, 0]

    direction = "LEFT" if tip_x < mcx else "RIGHT"
    return direction, (mcx, mcy), (w, h)


class DirectionDetector:
    def __init__(self):
        self.history = []

    def update(self, frame):
        mask                = skin_mask(frame)
        contour             = largest_contour(mask)
        raw, centroid, dims = finger_direction(contour)

        self.history.append(raw)
        if len(self.history) > params["smooth"]:
            self.history.pop(0)

        valid    = [d for d in self.history if d is not None]
        smoothed = max(set(valid), key=valid.count) if valid else None

        return smoothed, raw, mask, contour, centroid, dims


def draw_overlay(frame, direction, raw, contour, centroid, dims, source=None):
    h, w = frame.shape[:2]

    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 200, 255), 2)
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
        cv2.polylines(frame, [box], True, (255, 150, 0), 2)

    if centroid is not None:
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 6, (0, 0, 255), -1)

    if dims is not None:
        ratio = dims[0] / dims[1] if dims[1] > 0 else 0
        cv2.putText(frame, f"ratio {ratio:.2f}  thresh {params['aspect']/10:.1f}",
                    (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    if direction == "LEFT":
        text, color = "<  LEFT",  (255, 80,  80)
        a_start, a_end = (3 * w // 4, h // 2), (w // 4, h // 2)
    elif direction == "RIGHT":
        text, color = "RIGHT  >", (80,  255, 80)
        a_start, a_end = (w // 4, h // 2), (3 * w // 4, h // 2)
    else:
        text, color = "---",      (160, 160, 160)
        a_start = a_end = None

    if a_start and a_end:
        cv2.arrowedLine(frame, a_start, a_end, color, 6, tipLength=0.3)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thick = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = 70
    cv2.rectangle(frame, (tx - 10, ty - th - 10),
                  (tx + tw + 10, ty + 10), (30, 30, 30), -1)
    cv2.putText(frame, text, (tx, ty), font, scale, color, thick)

    cv2.putText(frame, f"raw: {raw or '---'}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # RouteEdge addition: show which path produced this frame's result.
    # Cheap to draw, and it's the single most useful thing for the demo video —
    # lets a viewer watch the routing decision flip in real time.
    if source is not None:
        src_color = (80, 255, 80) if source == "local" else (255, 180, 60)
        cv2.putText(frame, f"source: {source.upper()}", (w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, src_color, 2)

    return frame


def main():
    print("Initialising camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}
    ))
    picam2.start()
    print("Camera ready.")

    MAIN_WIN = "Finger Direction"
    MASK_WIN = "Skin Mask + Trackbars"

    cv2.namedWindow(MAIN_WIN)
    cv2.namedWindow(MASK_WIN)
    make_trackbars(MASK_WIN)

    print("Ready — point your index finger LEFT or RIGHT.")
    print("Use the trackbars in 'Skin Mask + Trackbars' to calibrate.")
    print("Goal: your finger should appear as a clean white blob in the mask.")
    print("Press 'q' to quit, 'p' to print current settings.\n")

    cfg = config.load()

    detector = DirectionDetector()
    probe    = NetworkProbe(server_ip=cfg["server"]["ip"], port=cfg["server"]["port"])
    energy   = EnergyEstimator()
    logger   = FrameLogger(cfg["logging"]["path"])

    privacy_flag = cfg["privacy"]["default_flag"]

    t0, frames, frame_id = time.time(), 0, 0

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to read frame")
                break

            # No flip needed — Pi camera is not mirrored like a laptop webcam
            direction, raw, mask, contour, centroid, dims = detector.update(frame)

            # ── RouteEdge: derive confidence signal from what we just computed ──
            sig = build_confidence_signal(
                frame, mask, contour, dims,
                detector.history, params["aspect"] / 10.0,
            )

            network_delay_ms          = probe.current_delay_ms()
            predicted_cloud_latency_ms = probe.predicted_cloud_latency_ms()

            route_result = router_decide(
                sig,
                network_delay_ms,
                predicted_cloud_latency_ms,
                privacy_flag,
            )

            decision_start = time.time()
            if route_result.decision == RouteDecision.LOCAL:
                final_direction = direction
                source = "local"
            else:
                final_direction = cloud_client.send_frame(
                    frame, cfg["server"]["ip"], cfg["server"]["port"],
                )
                source = "cloud"
            decision_latency_ms = (time.time() - decision_start) * 1000.0

            logger.log(
                frame_id=frame_id,
                signal=sig,
                route_result=route_result,
                final_direction=final_direction,
                source=source,
                latency_ms=decision_latency_ms,
                energy_estimate=energy.sample(),
            )
            # ─────────────────────────────────────────────────────────────────

            frame = draw_overlay(frame, final_direction, raw, contour, centroid,
                                  dims, source=source)

            frames += 1
            fps = frames / (time.time() - t0)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if contour is not None:
                cv2.drawContours(mask_bgr, [contour], -1, (0, 255, 0), 2)
            cv2.imshow(MASK_WIN, mask_bgr)
            cv2.imshow(MAIN_WIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("\n── Current settings ──────────────────")
                for k, v in params.items():
                    print(f"  {k}: {v}")
                print("──────────────────────────────────────\n")
            elif key == ord('v'):
                # quick manual privacy toggle for the demo video
                privacy_flag = not privacy_flag
                print(f"[privacy_flag] now {privacy_flag}")

            frame_id += 1

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        logger.close()
        print("Done.")


if __name__ == "__main__":
    main()
