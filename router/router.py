"""
RouteEdge Router

Provides routing and inference endpoints for RouteEdge.

Current capabilities:
    - Always-edge baseline
    - Always-server baseline
    - RouteEdge adaptive routing
    - Privacy-aware routing
    - Raspberry Pi edge inference through Ollama

The current workload-size routing rule is temporary.
It will later be replaced with measured latency/energy scoring.
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
    version="0.2.0",
)


# ============================================================
# Configuration
# ============================================================

EDGE_URL = "http://pizero.local:11434/api/generate"

# Server endpoint will be added once the server side is ready.
SERVER_URL = None

MODEL = "qwen2.5:0.5b"

REQUEST_TIMEOUT = 300

MAX_OUTPUT_TOKENS = 256

KEEP_ALIVE = "10m"


# ============================================================
# Routing modes
# ============================================================

class RoutingMode(str, Enum):
    ROUTEEDGE = "routeedge"
    ALWAYS_EDGE = "always_edge"
    ALWAYS_SERVER = "always_server"


# ============================================================
# API models
# ============================================================

class InferenceRequest(BaseModel):
    prompt: str

    # Privacy-sensitive requests must remain local.
    private: bool = False

    # 0.0 = prioritize energy
    # 1.0 = prioritize latency
    #
    # This will become part of the final weighted cost model.
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    mode: RoutingMode = RoutingMode.ROUTEEDGE


class RoutingDecision(BaseModel):
    target: str
    reason: str


class InferenceResponse(BaseModel):
    target: str
    reason: str
    response: str
    latency_s: float

    # Useful inference statistics returned by Ollama.
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    generation_tokens_per_s: float | None = None


# ============================================================
# Helper functions
# ============================================================

def ns_to_seconds(value):
    """
    Convert nanoseconds to seconds.
    """
    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_generation_rate(
    output_tokens,
    eval_duration_ns,
):
    """
    Calculate generation throughput in tokens per second.
    """
    if not output_tokens or not eval_duration_ns:
        return None

    seconds = ns_to_seconds(eval_duration_ns)

    if seconds <= 0:
        return None

    return output_tokens / seconds


# ============================================================
# Routing logic
# ============================================================

def choose_target(
    request: InferenceRequest,
) -> RoutingDecision:
    """
    Decide whether a request should run on the edge device
    or the server.

    NOTE:
    The 100-word workload threshold is temporary scaffolding.

    Later, RouteEdge will replace this with a cost model using:
        - predicted latency
        - predicted energy
        - network delay
        - user priority
        - privacy constraints
        - device availability
    """

    # --------------------------------------------------------
    # Explicit baseline: always edge
    # --------------------------------------------------------

    if request.mode == RoutingMode.ALWAYS_EDGE:
        return RoutingDecision(
            target="edge",
            reason="Always-edge baseline selected.",
        )

    # --------------------------------------------------------
    # Explicit baseline: always server
    # --------------------------------------------------------

    if request.mode == RoutingMode.ALWAYS_SERVER:
        return RoutingDecision(
            target="server",
            reason="Always-server baseline selected.",
        )

    # --------------------------------------------------------
    # Hard privacy constraint
    # --------------------------------------------------------

    if request.private:
        return RoutingDecision(
            target="edge",
            reason=(
                "Privacy constraint requires local inference."
            ),
        )

    # --------------------------------------------------------
    # Temporary workload-size heuristic
    # --------------------------------------------------------

    prompt_length = len(
        request.prompt.split()
    )

    if prompt_length <= 100:
        return RoutingDecision(
            target="edge",
            reason=(
                f"Small workload ({prompt_length} words) "
                "is suitable for edge inference."
            ),
        )

    return RoutingDecision(
        target="server",
        reason=(
            f"Large workload ({prompt_length} words) "
            "is better suited for server inference."
        ),
    )


# ============================================================
# Inference functions
# ============================================================

def run_edge_inference(
    prompt: str,
) -> dict:
    """
    Send an inference request to the Raspberry Pi Ollama
    endpoint.
    """

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            EDGE_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=(
                "Raspberry Pi inference request timed out."
            ),
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to the Raspberry Pi "
                "edge endpoint."
            ),
        )

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Edge inference request failed: {error}"
            ),
        )

    latency = time.perf_counter() - start

    data = response.json()

    output_tokens = data.get(
        "eval_count",
        0,
    )

    eval_duration = data.get(
        "eval_duration",
        0,
    )

    generation_rate = calculate_generation_rate(
        output_tokens,
        eval_duration,
    )

    return {
        "response": data.get("response", ""),
        "latency_s": round(latency, 3),
        "prompt_tokens": data.get(
            "prompt_eval_count",
            0,
        ),
        "output_tokens": output_tokens,
        "generation_tokens_per_s": (
            round(generation_rate, 2)
            if generation_rate is not None
            else None
        ),
    }


# ============================================================
# API endpoints
# ============================================================

@app.get("/")
def root():
    """
    Basic health/status endpoint.
    """

    return {
        "service": "RouteEdge",
        "status": "online",
        "version": "0.2.0",
        "model": MODEL,
        "edge_endpoint": EDGE_URL,
        "server_connected": SERVER_URL is not None,
    }


@app.post(
    "/route",
    response_model=RoutingDecision,
)
def route(
    request: InferenceRequest,
):
    """
    Return a routing decision without performing inference.
    """

    return choose_target(request)


@app.post(
    "/infer",
    response_model=InferenceResponse,
)
def infer(
    request: InferenceRequest,
):
    """
    Route the request and perform inference on the selected
    target.
    """

    decision = choose_target(request)

    # --------------------------------------------------------
    # Edge inference
    # --------------------------------------------------------

    if decision.target == "edge":

        result = run_edge_inference(
            request.prompt
        )

        return InferenceResponse(
            target="edge",
            reason=decision.reason,
            response=result["response"],
            latency_s=result["latency_s"],
            prompt_tokens=result["prompt_tokens"],
            output_tokens=result["output_tokens"],
            generation_tokens_per_s=(
                result["generation_tokens_per_s"]
            ),
        )

    # --------------------------------------------------------
    # Server inference
    # --------------------------------------------------------

    # We have not connected the GPU/server endpoint yet.
    #
    # Returning a clear error is better than pretending that
    # server inference happened.

    if SERVER_URL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RouteEdge selected the server, but the "
                "server inference endpoint is not connected yet."
            ),
        )

    # This will be implemented when the server endpoint is ready.
    raise HTTPException(
        status_code=501,
        detail="Server inference is not implemented yet.",
    )