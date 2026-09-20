"""
RouteEdge Router

Adaptive routing between:
    1. Raspberry Pi edge inference
    2. GPU server inference

Both endpoints currently run the same Qwen2.5-0.5B model
through Ollama.

Current capabilities:
    - Always-edge baseline
    - Always-server baseline
    - RouteEdge adaptive routing
    - Privacy-aware routing
    - Edge inference
    - GPU server inference
    - Latency and throughput measurements

NOTE:
The current workload-size routing heuristic is temporary.
It will later be replaced by RouteEdge's measured
latency/energy cost model.
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
    version="0.3.0",
)


# ============================================================
# Configuration
# ============================================================

# Raspberry Pi Ollama endpoint
EDGE_URL = "http://pizero.local:11434/api/generate"

# GPU-backed Ollama endpoint on the XPS
SERVER_URL = "http://localhost:11434/api/generate"

# Same model on both devices so hardware is the primary
# experimental variable.
MODEL = "qwen2.5:0.5b"

# Maximum request duration before RouteEdge gives up.
REQUEST_TIMEOUT = 300

# Prevent uncontrolled/runaway generation.
MAX_OUTPUT_TOKENS = 256

# Keep Ollama models loaded between requests.
KEEP_ALIVE = "10m"


# ============================================================
# Routing Modes
# ============================================================

class RoutingMode(str, Enum):
    ROUTEEDGE = "routeedge"
    ALWAYS_EDGE = "always_edge"
    ALWAYS_SERVER = "always_server"


# ============================================================
# API Models
# ============================================================

class InferenceRequest(BaseModel):
    """
    Request submitted to RouteEdge.
    """

    prompt: str

    # Privacy-sensitive requests must remain on the edge.
    private: bool = False

    # User preference:
    #
    # 0.0 -> prioritize energy
    # 1.0 -> prioritize latency
    #
    # This will become part of the final RouteEdge
    # weighted cost function.
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    mode: RoutingMode = RoutingMode.ROUTEEDGE


class RoutingDecision(BaseModel):
    """
    Result produced by the routing algorithm.
    """

    target: str
    reason: str


class InferenceResponse(BaseModel):
    """
    Complete response returned after inference.
    """

    target: str
    reason: str

    response: str

    latency_s: float

    prompt_tokens: int | None = None
    output_tokens: int | None = None

    generation_tokens_per_s: float | None = None


# ============================================================
# Helper Functions
# ============================================================

def ns_to_seconds(value):
    """
    Convert Ollama nanosecond timing values to seconds.
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
# Routing Logic
# ============================================================

def choose_target(
    request: InferenceRequest,
) -> RoutingDecision:
    """
    Decide whether inference should execute on the Raspberry Pi
    edge device or the GPU server.

    The current 100-word threshold is temporary scaffolding.

    It will eventually be replaced with a cost model using:

        - predicted edge latency
        - predicted server latency
        - network latency
        - estimated energy
        - user priority
        - privacy constraints
        - endpoint availability
    """

    # --------------------------------------------------------
    # Baseline: always use edge
    # --------------------------------------------------------

    if request.mode == RoutingMode.ALWAYS_EDGE:

        return RoutingDecision(
            target="edge",
            reason="Always-edge baseline selected.",
        )

    # --------------------------------------------------------
    # Baseline: always use server
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
    # Temporary workload heuristic
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
# Edge Inference
# ============================================================

def run_edge_inference(
    prompt: str,
) -> dict:
    """
    Run inference using Qwen on the Raspberry Pi.
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

        "output_tokens": output_tokens,

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
# Server Inference
# ============================================================

def run_server_inference(
    prompt: str,
) -> dict:
    """
    Run inference using Qwen on the GPU-backed XPS server.
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
            SERVER_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=504,
            detail=(
                "GPU server inference request timed out."
            ),
        )

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to the GPU server "
                "inference endpoint."
            ),
        )

    except requests.exceptions.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Server inference request failed: {error}"
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

        "output_tokens": output_tokens,

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
    """
    RouteEdge status endpoint.
    """

    return {
        "service": "RouteEdge",
        "status": "online",
        "version": "0.3.0",
        "model": MODEL,

        "edge": {
            "type": "Raspberry Pi",
            "endpoint": EDGE_URL,
        },

        "server": {
            "type": "GPU Server",
            "endpoint": SERVER_URL,
        },
    }


@app.post(
    "/route",
    response_model=RoutingDecision,
)
def route(
    request: InferenceRequest,
):
    """
    Return RouteEdge's routing decision without actually
    performing inference.
    """

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
    """
    Route a request and execute inference on the selected
    compute target.
    """

    decision = choose_target(
        request
    )

    # --------------------------------------------------------
    # Edge
    # --------------------------------------------------------

    if decision.target == "edge":

        result = run_edge_inference(
            request.prompt
        )

        return InferenceResponse(
            target="edge",

            reason=decision.reason,

            response=result[
                "response"
            ],

            latency_s=result[
                "latency_s"
            ],

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

    # --------------------------------------------------------
    # Server
    # --------------------------------------------------------

    if decision.target == "server":

        result = run_server_inference(
            request.prompt
        )

        return InferenceResponse(
            target="server",

            reason=decision.reason,

            response=result[
                "response"
            ],

            latency_s=result[
                "latency_s"
            ],

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

    # This should never occur unless the routing logic
    # returns an unsupported target.

    raise HTTPException(
        status_code=500,
        detail=(
            "RouteEdge produced an unknown routing target."
        ),
    )