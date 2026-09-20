"""
RouteEdge Router v0.5

Adaptive routing between:
    1. Raspberry Pi 5 edge inference
    2. RTX 3050 GPU server inference

Both endpoints run Qwen2.5-0.5B through Ollama.

Routing considers:
    - predicted latency
    - estimated energy proxy
    - user latency/energy priority
    - simulated network latency
    - privacy constraints
    - endpoint availability

IMPORTANT:
Energy values are ESTIMATES, not measured electrical energy.
The power constants below are configurable assumptions used
to demonstrate energy-aware routing.
"""

import time
from enum import Enum

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="RouteEdge Router",
    description="Adaptive routing for edge/cloud LLM inference",
    version="0.5.0",
)


# ============================================================
# Endpoints
# ============================================================

EDGE_URL = "http://pizero.local:11434/api/generate"
SERVER_URL = "http://localhost:11434/api/generate"

MODEL = "qwen2.5:0.5b"

REQUEST_TIMEOUT = 300
KEEP_ALIVE = "10m"

MAX_OUTPUT_TOKENS = 256


# ============================================================
# Measured Performance Model
# ============================================================

# Derived approximately from our controlled benchmark results.

EDGE_TOKENS_PER_SECOND = 18.5
SERVER_TOKENS_PER_SECOND = 193.5

# Simple fixed latency terms for the MVP model.
EDGE_BASE_LATENCY_S = 0.8
SERVER_BASE_LATENCY_S = 2.5


# ============================================================
# Energy Proxy Model
# ============================================================

# IMPORTANT:
# These are configurable modeling assumptions.
# They are NOT measured power consumption.

EDGE_POWER_PROXY_W = 8.0
SERVER_POWER_PROXY_W = 35.0


# ============================================================
# Routing Modes
# ============================================================

class RoutingMode(str, Enum):
    ROUTEEDGE = "routeedge"
    ALWAYS_EDGE = "always_edge"
    ALWAYS_SERVER = "always_server"


# ============================================================
# Request / Response Models
# ============================================================

class InferenceRequest(BaseModel):

    prompt: str

    private: bool = False

    expected_output_tokens: int = Field(
        default=128,
        ge=1,
        le=MAX_OUTPUT_TOKENS,
    )

    network_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=5000.0,
    )

    # 1.0 = latency only
    # 0.0 = energy proxy only
    # 0.5 = balanced
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    mode: RoutingMode = RoutingMode.ROUTEEDGE


class RoutingDecision(BaseModel):

    target: str
    reason: str

    predicted_edge_latency_s: float
    predicted_server_latency_s: float

    predicted_edge_energy_proxy_j: float
    predicted_server_energy_proxy_j: float

    normalized_edge_latency: float
    normalized_server_latency: float

    normalized_edge_energy: float
    normalized_server_energy: float

    edge_cost: float
    server_cost: float


class InferenceResponse(BaseModel):

    target: str
    reason: str

    response: str

    actual_latency_s: float

    predicted_edge_latency_s: float
    predicted_server_latency_s: float

    predicted_edge_energy_proxy_j: float
    predicted_server_energy_proxy_j: float

    edge_cost: float
    server_cost: float

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    generation_tokens_per_s: float | None = None


# ============================================================
# Utility Functions
# ============================================================

def ns_to_seconds(value):

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_generation_rate(
    output_tokens,
    eval_duration_ns,
):

    if not output_tokens or not eval_duration_ns:
        return None

    seconds = ns_to_seconds(
        eval_duration_ns
    )

    if seconds <= 0:
        return None

    return output_tokens / seconds


# ============================================================
# Prediction Model
# ============================================================

def predict_edge_latency(
    expected_output_tokens,
):

    generation_time = (
        expected_output_tokens
        / EDGE_TOKENS_PER_SECOND
    )

    return (
        EDGE_BASE_LATENCY_S
        + generation_time
    )


def predict_server_latency(
    expected_output_tokens,
    network_latency_ms,
):

    generation_time = (
        expected_output_tokens
        / SERVER_TOKENS_PER_SECOND
    )

    network_round_trip_s = (
        2
        * network_latency_ms
        / 1000
    )

    return (
        SERVER_BASE_LATENCY_S
        + generation_time
        + network_round_trip_s
    )


def predict_edge_energy(
    expected_output_tokens,
):

    # Energy proxy includes local edge compute time.

    compute_time = (
        EDGE_BASE_LATENCY_S
        + expected_output_tokens
        / EDGE_TOKENS_PER_SECOND
    )

    return (
        EDGE_POWER_PROXY_W
        * compute_time
    )


def predict_server_energy(
    expected_output_tokens,
):

    # This proxy estimates SERVER COMPUTE ENERGY only.
    #
    # Network energy is not modeled because we have not
    # measured it.

    compute_time = (
        SERVER_BASE_LATENCY_S
        + expected_output_tokens
        / SERVER_TOKENS_PER_SECOND
    )

    return (
        SERVER_POWER_PROXY_W
        * compute_time
    )


# ============================================================
# Normalization
# ============================================================

def normalize_pair(
    first,
    second,
):
    """
    Normalize two positive costs relative to the larger value.

    The smaller value therefore receives the smaller normalized
    cost, while the larger value becomes 1.0.
    """

    maximum = max(
        first,
        second,
    )

    if maximum <= 0:
        return 0.0, 0.0

    return (
        first / maximum,
        second / maximum,
    )


# ============================================================
# Availability
# ============================================================

def endpoint_available(
    url,
):

    health_url = url.replace(
        "/api/generate",
        "/api/tags",
    )

    try:

        response = requests.get(
            health_url,
            timeout=2,
        )

        return response.ok

    except requests.exceptions.RequestException:

        return False


# ============================================================
# Build Routing Metrics
# ============================================================

def calculate_routing_metrics(
    request,
):

    edge_latency = predict_edge_latency(
        request.expected_output_tokens
    )

    server_latency = predict_server_latency(
        request.expected_output_tokens,
        request.network_latency_ms,
    )

    edge_energy = predict_edge_energy(
        request.expected_output_tokens
    )

    server_energy = predict_server_energy(
        request.expected_output_tokens
    )

    (
        normalized_edge_latency,
        normalized_server_latency,
    ) = normalize_pair(
        edge_latency,
        server_latency,
    )

    (
        normalized_edge_energy,
        normalized_server_energy,
    ) = normalize_pair(
        edge_energy,
        server_energy,
    )

    alpha = request.priority

    edge_cost = (
        alpha
        * normalized_edge_latency
        +
        (1 - alpha)
        * normalized_edge_energy
    )

    server_cost = (
        alpha
        * normalized_server_latency
        +
        (1 - alpha)
        * normalized_server_energy
    )

    return {
        "edge_latency": edge_latency,
        "server_latency": server_latency,

        "edge_energy": edge_energy,
        "server_energy": server_energy,

        "normalized_edge_latency":
            normalized_edge_latency,

        "normalized_server_latency":
            normalized_server_latency,

        "normalized_edge_energy":
            normalized_edge_energy,

        "normalized_server_energy":
            normalized_server_energy,

        "edge_cost": edge_cost,
        "server_cost": server_cost,
    }


# ============================================================
# Build Routing Decision
# ============================================================

def make_decision(
    target,
    reason,
    metrics,
):

    return RoutingDecision(

        target=target,

        reason=reason,

        predicted_edge_latency_s=round(
            metrics["edge_latency"],
            3,
        ),

        predicted_server_latency_s=round(
            metrics["server_latency"],
            3,
        ),

        predicted_edge_energy_proxy_j=round(
            metrics["edge_energy"],
            3,
        ),

        predicted_server_energy_proxy_j=round(
            metrics["server_energy"],
            3,
        ),

        normalized_edge_latency=round(
            metrics["normalized_edge_latency"],
            4,
        ),

        normalized_server_latency=round(
            metrics["normalized_server_latency"],
            4,
        ),

        normalized_edge_energy=round(
            metrics["normalized_edge_energy"],
            4,
        ),

        normalized_server_energy=round(
            metrics["normalized_server_energy"],
            4,
        ),

        edge_cost=round(
            metrics["edge_cost"],
            4,
        ),

        server_cost=round(
            metrics["server_cost"],
            4,
        ),
    )


# ============================================================
# Routing Logic
# ============================================================

def choose_target(
    request: InferenceRequest,
) -> RoutingDecision:

    metrics = calculate_routing_metrics(
        request
    )

    # --------------------------------------------------------
    # Explicit baseline modes
    # --------------------------------------------------------

    if request.mode == RoutingMode.ALWAYS_EDGE:

        return make_decision(
            "edge",
            "Always-edge baseline selected.",
            metrics,
        )

    if request.mode == RoutingMode.ALWAYS_SERVER:

        return make_decision(
            "server",
            "Always-server baseline selected.",
            metrics,
        )

    # --------------------------------------------------------
    # Privacy hard constraint
    # --------------------------------------------------------

    if request.private:

        return make_decision(
            "edge",
            (
                "Privacy constraint requires "
                "local edge inference."
            ),
            metrics,
        )

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    edge_online = endpoint_available(
        EDGE_URL
    )

    server_online = endpoint_available(
        SERVER_URL
    )

    if not edge_online and not server_online:

        raise HTTPException(
            status_code=503,
            detail=(
                "Neither RouteEdge compute target "
                "is currently available."
            ),
        )

    if not edge_online:

        return make_decision(
            "server",
            (
                "Edge endpoint unavailable; "
                "falling back to server."
            ),
            metrics,
        )

    if not server_online:

        return make_decision(
            "edge",
            (
                "Server endpoint unavailable; "
                "falling back to edge."
            ),
            metrics,
        )

    # --------------------------------------------------------
    # Adaptive Cost Function
    # --------------------------------------------------------

    if (
        metrics["edge_cost"]
        <= metrics["server_cost"]
    ):

        target = "edge"

    else:

        target = "server"

    reason = (
        "Adaptive RouteEdge decision: "
        f"priority={request.priority:.2f}, "
        f"edge cost={metrics['edge_cost']:.3f}, "
        f"server cost={metrics['server_cost']:.3f}. "
        f"Predicted latency: "
        f"edge={metrics['edge_latency']:.2f}s, "
        f"server={metrics['server_latency']:.2f}s. "
        f"Estimated compute-energy proxy: "
        f"edge={metrics['edge_energy']:.2f}J, "
        f"server={metrics['server_energy']:.2f}J."
    )

    return make_decision(
        target,
        reason,
        metrics,
    )


# ============================================================
# Inference
# ============================================================

def run_inference(
    url,
    prompt,
    max_output_tokens,
    target_name,
):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_predict":
                max_output_tokens,
        },
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=504,
            detail=(
                f"{target_name} inference "
                "request timed out."
            ),
        )

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail=(
                f"Unable to connect to "
                f"{target_name}."
            ),
        )

    except requests.exceptions.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=(
                f"{target_name} inference "
                f"failed: {error}"
            ),
        )

    latency = (
        time.perf_counter()
        - start
    )

    data = response.json()

    output_tokens = data.get(
        "eval_count",
        0,
    )

    eval_duration = data.get(
        "eval_duration",
        0,
    )

    generation_rate = (
        calculate_generation_rate(
            output_tokens,
            eval_duration,
        )
    )

    return {
        "response": data.get(
            "response",
            "",
        ),

        "latency_s": round(
            latency,
            3,
        ),

        "prompt_tokens": data.get(
            "prompt_eval_count",
            0,
        ),

        "output_tokens":
            output_tokens,

        "generation_tokens_per_s": (
            round(
                generation_rate,
                2,
            )
            if generation_rate is not None
            else None
        ),
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():

    return {
        "service": "RouteEdge",
        "status": "online",
        "version": "0.5.0",
        "model": MODEL,

        "measured_performance": {
            "edge_tokens_per_second":
                EDGE_TOKENS_PER_SECOND,

            "server_tokens_per_second":
                SERVER_TOKENS_PER_SECOND,
        },

        "energy_proxy_assumptions": {
            "edge_power_w":
                EDGE_POWER_PROXY_W,

            "server_power_w":
                SERVER_POWER_PROXY_W,

            "note": (
                "Power values are configurable "
                "model assumptions, not measured "
                "electrical power."
            ),
        },
    }


@app.post(
    "/route",
    response_model=RoutingDecision,
)
def route(
    request: InferenceRequest,
):

    return choose_target(
        request
    )


@app.post(
    "/infer",
    response_model=InferenceResponse,
)
def infer(
    request: InferenceRequest,
):

    decision = choose_target(
        request
    )

    if decision.target == "edge":

        url = EDGE_URL
        target_name = (
            "Raspberry Pi edge endpoint"
        )

    elif decision.target == "server":

        url = SERVER_URL
        target_name = (
            "GPU server endpoint"
        )

    else:

        raise HTTPException(
            status_code=500,
            detail="Unknown RouteEdge target.",
        )

    result = run_inference(
        url=url,
        prompt=request.prompt,
        max_output_tokens=(
            request.expected_output_tokens
        ),
        target_name=target_name,
    )

    return InferenceResponse(

        target=decision.target,

        reason=decision.reason,

        response=result["response"],

        actual_latency_s=result[
            "latency_s"
        ],

        predicted_edge_latency_s=(
            decision.predicted_edge_latency_s
        ),

        predicted_server_latency_s=(
            decision.predicted_server_latency_s
        ),

        predicted_edge_energy_proxy_j=(
            decision.predicted_edge_energy_proxy_j
        ),

        predicted_server_energy_proxy_j=(
            decision.predicted_server_energy_proxy_j
        ),

        edge_cost=decision.edge_cost,

        server_cost=decision.server_cost,

        prompt_tokens=result[
            "prompt_tokens"
        ],

        output_tokens=result[
            "output_tokens"
        ],

        generation_tokens_per_s=result[
            "generation_tokens_per_s"
        ],
    )