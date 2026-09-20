"""
RouteEdge GPU Server Profiler

Benchmarks Qwen2.5-0.5B running through Ollama on the
RTX 3050 GPU server.

The benchmark is designed to mirror the Raspberry Pi edge
benchmark so the two compute targets can be compared.

Measures:
    - End-to-end latency
    - Ollama total duration
    - Model load duration
    - Prompt evaluation duration
    - Generation duration
    - Prompt tokens
    - Output tokens
    - Prompt throughput
    - Generation throughput
    - Success/failure
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "qwen2.5:0.5b"

REQUEST_TIMEOUT = 300

MAX_OUTPUT_TOKENS = 256

KEEP_ALIVE = "10m"

RUNS_PER_WORKLOAD = 3


# ============================================================
# Benchmark Prompts
# ============================================================

# These match the controlled workload classes used for the
# Raspberry Pi benchmark.

PROMPTS = {
    "short": (
        "In one or two sentences, explain what CPU pipelining is."
    ),

    "medium": (
        "Explain how cache memory improves CPU performance. "
        "Discuss temporal locality, spatial locality, cache hits, "
        "cache misses, and why caches reduce average memory access time. "
        "Keep the explanation concise."
    ),

    "long": (
        "Explain the major architectural components of a modern computer "
        "processor. Discuss pipelining, caches, branch prediction, "
        "instruction-level parallelism, memory hierarchy, and the role "
        "of registers. Explain how these components interact to improve "
        "processor performance while considering power and memory "
        "limitations."
    ),
}


# ============================================================
# Output Location
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "server_results.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def ns_to_seconds(value):
    """
    Convert Ollama nanoseconds to seconds.
    """

    if not value:
        return 0.0

    return value / 1_000_000_000


def calculate_rate(
    tokens,
    duration_ns,
):
    """
    Calculate tokens per second.
    """

    if not tokens or not duration_ns:
        return None

    seconds = ns_to_seconds(
        duration_ns
    )

    if seconds <= 0:
        return None

    return tokens / seconds


# ============================================================
# Benchmark
# ============================================================

def benchmark_request(
    workload,
    run_number,
    prompt,
):
    """
    Run one benchmark request against the GPU server.
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

    print(
        f"Running {workload} "
        f"{run_number}/{RUNS_PER_WORKLOAD}..."
    )

    start = time.perf_counter()

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        wall_latency = (
            time.perf_counter()
            - start
        )

        response.raise_for_status()

        data = response.json()

        prompt_tokens = data.get(
            "prompt_eval_count",
            0,
        )

        output_tokens = data.get(
            "eval_count",
            0,
        )

        prompt_eval_ns = data.get(
            "prompt_eval_duration",
            0,
        )

        eval_ns = data.get(
            "eval_duration",
            0,
        )

        prompt_rate = calculate_rate(
            prompt_tokens,
            prompt_eval_ns,
        )

        generation_rate = calculate_rate(
            output_tokens,
            eval_ns,
        )

        result = {
            "timestamp": datetime.now().isoformat(),
            "device": "xps_rtx3050",
            "model": MODEL,
            "workload": workload,
            "run": run_number,
            "success": True,
            "error": "",

            "latency_s": round(
                wall_latency,
                4,
            ),

            "total_duration_s": round(
                ns_to_seconds(
                    data.get(
                        "total_duration",
                        0,
                    )
                ),
                4,
            ),

            "load_duration_s": round(
                ns_to_seconds(
                    data.get(
                        "load_duration",
                        0,
                    )
                ),
                4,
            ),

            "prompt_eval_duration_s": round(
                ns_to_seconds(
                    prompt_eval_ns
                ),
                4,
            ),

            "eval_duration_s": round(
                ns_to_seconds(
                    eval_ns
                ),
                4,
            ),

            "prompt_tokens": prompt_tokens,

            "output_tokens": output_tokens,

            "prompt_tokens_per_s": (
                round(
                    prompt_rate,
                    2,
                )
                if prompt_rate is not None
                else ""
            ),

            "generation_tokens_per_s": (
                round(
                    generation_rate,
                    2,
                )
                if generation_rate is not None
                else ""
            ),

            "max_output_tokens": (
                MAX_OUTPUT_TOKENS
            ),

            "done_reason": data.get(
                "done_reason",
                "",
            ),
        }

        print(
            f"  latency: "
            f"{wall_latency:.2f}s | "
            f"load: "
            f"{result['load_duration_s']:.2f}s | "
            f"generation: "
            f"{generation_rate:.2f} tok/s | "
            f"tokens: {output_tokens}"
        )

        return result

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        wall_latency = (
            time.perf_counter()
            - start
        )

        print(
            f"  FAILED: timeout after "
            f"{wall_latency:.2f}s"
        )

        return failure_result(
            workload,
            run_number,
            wall_latency,
            "timeout",
        )

    # --------------------------------------------------------
    # Other request failure
    # --------------------------------------------------------

    except requests.exceptions.RequestException as error:

        wall_latency = (
            time.perf_counter()
            - start
        )

        print(
            f"  FAILED: {error}"
        )

        return failure_result(
            workload,
            run_number,
            wall_latency,
            str(error),
        )


# ============================================================
# Failure Result
# ============================================================

def failure_result(
    workload,
    run_number,
    latency,
    error,
):
    """
    Generate a CSV-compatible row for a failed request.
    """

    return {
        "timestamp": datetime.now().isoformat(),
        "device": "xps_rtx3050",
        "model": MODEL,
        "workload": workload,
        "run": run_number,
        "success": False,
        "error": error,
        "latency_s": round(
            latency,
            4,
        ),
        "total_duration_s": "",
        "load_duration_s": "",
        "prompt_eval_duration_s": "",
        "eval_duration_s": "",
        "prompt_tokens": "",
        "output_tokens": "",
        "prompt_tokens_per_s": "",
        "generation_tokens_per_s": "",
        "max_output_tokens": (
            MAX_OUTPUT_TOKENS
        ),
        "done_reason": "",
    }


# ============================================================
# Save Results
# ============================================================

def save_results(
    results,
):
    """
    Write benchmark results to server_results.csv.

    This creates a fresh benchmark file for this run.
    """

    fieldnames = [
        "timestamp",
        "device",
        "model",
        "workload",
        "run",
        "success",
        "error",
        "latency_s",
        "total_duration_s",
        "load_duration_s",
        "prompt_eval_duration_s",
        "eval_duration_s",
        "prompt_tokens",
        "output_tokens",
        "prompt_tokens_per_s",
        "generation_tokens_per_s",
        "max_output_tokens",
        "done_reason",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "RouteEdge GPU Server Profiler"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Endpoint: {OLLAMA_URL}"
    )

    print(
        f"Model:    {MODEL}"
    )

    print(
        f"Output cap: "
        f"{MAX_OUTPUT_TOKENS} tokens"
    )

    print(
        f"Timeout:    "
        f"{REQUEST_TIMEOUT}s"
    )

    print()

    results = []

    for workload, prompt in PROMPTS.items():

        for run_number in range(
            1,
            RUNS_PER_WORKLOAD + 1,
        ):

            result = benchmark_request(
                workload,
                run_number,
                prompt,
            )

            results.append(
                result
            )

    save_results(
        results
    )

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    failed = (
        len(results)
        - successful
    )

    print()

    print(
        "--------------------------------"
    )

    print(
        "Benchmark complete."
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total:      {len(results)}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()