# RouteEdge
### Adaptive Hardware-Aware Routing for Edge and Server LLM Inference

## Problem and Motivation
Small language models can now execute on embedded and edge platforms, but local
execution is not always the best choice. Device memory, energy availability,
prompt length, latency requirements, network quality, and privacy constraints
change from request to request. Existing systems commonly use a fixed execution
target, wasting energy or increasing response time when conditions change.
RouteEdge investigates whether a lightweight, hardware-aware decision layer can
select between local edge inference and higher-performance server inference more
effectively than fixed policies.

## Proposed Technical Approach
RouteEdge profiles a compact gesture-recognition pipeline on two heterogeneous
targets: a Raspberry Pi 5 (CPU-based edge platform) and a GPU-capable
laptop/workstation acting as the remote server. A runtime controller, written in
C for speed and running entirely on the Pi, observes confidence signals derived
from the local detector's own outputs, current network conditions, a
user-selected latency/energy priority, and a hard privacy constraint. It
predicts the cost of each target and routes the request using a transparent
weighted scoring policy. Network delay and bandwidth are varied during testing
so the system demonstrates adaptive behavior rather than a static device
comparison.

## Prototype and Scope
- **Inference endpoints:** local classical-CV gesture detector (Pi) and a
  hand-landmark/gesture model (server), same task, different capability.
- **Profiler:** automated collection of latency, throughput, peak memory, and an
  energy proxy on both targets.
- **Routing engine:** a lightweight, C-implemented, constraint-aware scoring
  function selecting the execution target per frame.
- **Interactive demo:** a dashboard showing current constraints, the selected
  target, the resulting gesture, and measured performance.

## Evaluation and Evidence of Success
RouteEdge is compared against three baselines: always-edge, always-server, and a
simple network-threshold rule. Experiments vary lighting conditions, background
clutter, simulated bandwidth/latency, and privacy constraints. Success is
demonstrated if the adaptive policy reduces average end-to-end latency and/or
estimated energy cost while respecting hard privacy constraints and improving
robustness over the fixed baselines. Results include reproducible benchmark
tables, routing-decision traces, and plots showing when each target becomes
preferable. The final video demonstrates the same gesture being routed
differently as lighting and network conditions change.

## Connection to AI and Semiconductor Hardware
The project treats compute placement as a function of processor capability,
memory limits, communication cost, and workload characteristics — hardware-
software co-design for AI inference. It connects Algorithmic Efficiency &
Inference Acceleration, Agentic Architectures & System-Level AI, and AI Hardware
Accelerators, emphasizing heterogeneous CPU/GPU/edge systems.

## Expected Deliverable
A working proof of concept, benchmark dataset, adaptive-routing demonstration,
and concise demo video.

RouteEdge | One-Page Proposal | September 6, 2026
