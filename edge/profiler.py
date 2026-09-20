"""
RouteEdge Edge Profiler

Benchmarks an Ollama model running on a Raspberry Pi and records
performance information that can later be used by RouteEdge's router.

The profiler measures:
    - End-to-end latency
    - Model load time
    - Prompt evaluation time
    - Generation time
    - Prompt tokens
    - Generated tokens
    - Prompt processing rate
    - Generation rate
    - Success/failure
    - Timeout/error information

Results are written to:
    benchmarks/results/edge_results.csv
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_HOST = "http://pizero.local:11434"
ENDPOINT = f"{OLLAMA_HOST}/api/generate"

MODEL = "qwen2.5:0.5b"

# Maximum amount of time we'll allow one request to take.
REQUEST_TIMEOUT = 300

# Prevent the model from generating indefinitely.
MAX_OUTPUT_TOKENS = 256

# Keep the model loaded between benchmark requests.
KEEP_ALIVE = "10m"

# Number of times each workload is tested.
RUNS_PER_WORKLOAD = 3


# ============================================================
# Benchmark prompts
# ============================================================

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
# Output location
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_FILE = RESULTS_DIR / "edge_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def ns_to_seconds(value):
    """
    Ollama reports durations in nanoseconds.
    Convert them to seconds.
    """
    if value is None:
        return None

    return value / 1_000_000_000


def tokens_per_second(tokens, duration_ns):
    """
    Calculate token throughput using Ollama's duration values.
    """
    if not tokens or not duration_ns:
        return None

    seconds = ns_to_seconds(duration_ns)

    if seconds <= 0:
        return None

    return tokens / seconds


def benchmark_request(workload, run_number, prompt):
    """
    Send one request to the Raspberry Pi and collect performance data.
    """

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,

        # Limit generation so a runaway response does not sit there
        # generating for several minutes.
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    print(
        f"Running {workload} "
        f"{run_number}/{RUNS_PER_WORKLOAD}..."
    )

    start = time.perf_counter()

    try:
        response = requests.post(
            ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        eval_duration = data.get("eval_duration", 0)
        prompt_eval_duration = data.get("prompt_eval_duration", 0)

        generation_rate = tokens_per_second(
            eval_count,
            eval_duration
        )

        prompt_rate = tokens_per_second(
            prompt_eval_count,
            prompt_eval_duration
        )

        result = {
            "timestamp": datetime.now().isoformat(),
            "device": "raspberry_pi",
            "model": MODEL,
            "workload": workload,
            "run": run_number,
            "success": True,
            "error": "",
            "latency_s": round(elapsed, 4),
            "total_duration_s": round(
                ns_to_seconds(data.get("total_duration", 0)),
                4
            ),
            "load_duration_s": round(
                ns_to_seconds(data.get("load_duration", 0)),
                4
            ),
            "prompt_eval_duration_s": round(
                ns_to_seconds(prompt_eval_duration),
                4
            ),
            "eval_duration_s": round(
                ns_to_seconds(eval_duration),
                4
            ),
            "prompt_tokens": prompt_eval_count,
            "output_tokens": eval_count,
            "prompt_tokens_per_s": (
                round(prompt_rate, 2)
                if prompt_rate is not None
                else ""
            ),
            "generation_tokens_per_s": (
                round(generation_rate, 2)
                if generation_rate is not None
                else ""
            ),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "done_reason": data.get("done_reason", ""),
        }

        print(
            f"  latency: {elapsed:.2f}s | "
            f"generation: {generation_rate:.2f} tok/s | "
            f"tokens: {eval_count}"
        )

        return result

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start

        print(
            f"  FAILED: request timed out after "
            f"{elapsed:.2f}s"
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "device": "raspberry_pi",
            "model": MODEL,
            "workload": workload,
            "run": run_number,
            "success": False,
            "error": "timeout",
            "latency_s": round(elapsed, 4),
            "total_duration_s": "",
            "load_duration_s": "",
            "prompt_eval_duration_s": "",
            "eval_duration_s": "",
            "prompt_tokens": "",
            "output_tokens": "",
            "prompt_tokens_per_s": "",
            "generation_tokens_per_s": "",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "done_reason": "",
        }

    except requests.exceptions.RequestException as exc:
        elapsed = time.perf_counter() - start

        print(f"  FAILED: {exc}")

        return {
            "timestamp": datetime.now().isoformat(),
            "device": "raspberry_pi",
            "model": MODEL,
            "workload": workload,
            "run": run_number,
            "success": False,
            "error": str(exc),
            "latency_s": round(elapsed, 4),
            "total_duration_s": "",
            "load_duration_s": "",
            "prompt_eval_duration_s": "",
            "eval_duration_s": "",
            "prompt_tokens": "",
            "output_tokens": "",
            "prompt_tokens_per_s": "",
            "generation_tokens_per_s": "",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "done_reason": "",
        }


def save_results(results):
    """
    Append benchmark results to the CSV.

    This intentionally appends instead of overwriting so previous
    benchmark runs are preserved.
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

    file_exists = RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)


# ============================================================
# Main benchmark
# ============================================================

def main():

    print("RouteEdge Raspberry Pi Profiler")
    print("--------------------------------")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model:    {MODEL}")
    print(f"Output cap: {MAX_OUTPUT_TOKENS} tokens")
    print(f"Timeout:    {REQUEST_TIMEOUT}s")
    print()

    results = []

    for workload, prompt in PROMPTS.items():

        for run_number in range(
            1,
            RUNS_PER_WORKLOAD + 1
        ):

            result = benchmark_request(
                workload,
                run_number,
                prompt
            )

            results.append(result)

    save_results(results)

    successful = sum(
        1 for result in results
        if result["success"]
    )

    failed = len(results) - successful

    print()
    print("--------------------------------")
    print("Benchmark complete.")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(results)}")
    print()
    print("Results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()