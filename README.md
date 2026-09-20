# RouteEdge
Adaptive runtime for routing LLM inference across heterogeneous edge and GPU compute based on latency, energy, privacy, and resource constraints.

Adaptive Hardware-Aware Routing for Edge and Server LLM/CV Inference| Purdue University, ECE | Chips & AI Hackathon 2026

## What this is

RouteEdge routes each gesture-detection frame between two execution targets:

- **Local (Raspberry Pi 5):** a classical CV pipeline (HSV skin-color segmentation
  + contour geometry) that decides finger direction (LEFT/RIGHT).
- **Cloud (GPU server):** a robust hand-landmark / gesture model (e.g. MediaPipe
  Hands or a small trained CNN) that handles lighting, background clutter, and a
  richer gesture vocabulary the local method can't.

A lightweight, transparent, weighted-scoring router — implemented in C for speed
and running entirely on the Pi — decides per-frame whether to trust the local
result or escalate to the cloud, based on:

- Confidence signals derived from the local detector's own outputs (no extra
  inference required): whether a contour was found, how ambiguous its aspect
  ratio is, how noisy the mask is, how much the local reading is flickering
  frame-to-frame, and scene brightness.
- Current network conditions (rolling latency/bandwidth estimate to the server).
- A hard privacy constraint (never leave the device if the camera feed is
  marked private).
- A user-selected priority (latency vs. energy).

## Repo layout

See the file tree in `docs/architecture_diagram.png` (or just browse `edge/`,
`cloud/`, `experiments/`, `dashboard/`, `analysis/`).

## Quickstart

On the Pi:
```bash
bash scripts/setup_pi.sh
python edge/gesture_detect.py
```

On the server:
```bash
bash scripts/setup_server.sh
python cloud/server.py
```

To run the full benchmark suite used in the evaluation section:
```bash
python experiments/run_baselines.py --all-scenarios
python analysis/generate_report.py
```

## Baselines compared against

- Always-edge
- Always-cloud
- Simple network-threshold rule
- RouteEdge (adaptive weighted scoring)

## Note on project structure

The original design sketch included a separate `edge/main.py` orchestrating the
camera loop. Since `gesture_detect.py` already owns a working camera loop, we
kept `main.py` as a **thin launcher** that just calls into `gesture_detect.py`,
rather than duplicating the loop. See comments in `edge/main.py`.
>>>>>>> master
