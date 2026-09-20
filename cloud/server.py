"""
server.py

Socket listener implementing RouteEdge's length-prefixed frame protocol
(see docs/protocol_spec.md). Receives a JPEG frame from the Pi, runs the
cloud gesture model, and returns a JSON result.

Also starts load_agent.py's periodic reporting in a background thread, so
the Pi's NetworkProbe can eventually be extended to factor in real
server-side load (queue depth, concurrent connections) rather than only
round-trip latency.
"""

import json
import socket
import struct
import threading
import time

import cv2
import numpy as np
import yaml

from gesture_model import GestureModel
from load_agent import LoadAgent

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def recv_exact(conn, n):
    """Read exactly n bytes from a socket, or return None if the
    connection closes early."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_client(conn, addr, model, load_agent):
    load_agent.connection_started()
    try:
        length_prefix = recv_exact(conn, 4)
        if length_prefix is None:
            return
        (jpeg_len,) = struct.unpack(">I", length_prefix)

        jpeg_bytes = recv_exact(conn, jpeg_len)
        if jpeg_bytes is None:
            return

        start = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                              cv2.IMREAD_COLOR)
        result = model.infer(frame)
        inference_ms = (time.time() - start) * 1000.0

        response = json.dumps({
            "direction": result["direction"],
            "confidence": result["confidence"],
            "inference_ms": inference_ms,
        }) + "\n"
        conn.sendall(response.encode("utf-8"))

    except Exception as e:
        # Never let one bad frame take down the server — respond with a
        # null result so the Pi's cloud_client.py fails soft.
        try:
            conn.sendall((json.dumps({"direction": None, "error": str(e)}) + "\n").encode("utf-8"))
        except OSError:
            pass
    finally:
        load_agent.connection_finished()
        conn.close()


def main():
    cfg = load_config()
    port = cfg["server"]["port"]
    max_concurrent = cfg["server"].get("max_concurrent_connections", 8)

    print(f"Loading gesture model ({cfg['model']['path']})...")
    model = GestureModel()
    print("Model ready.")

    load_agent = LoadAgent(max_concurrent=max_concurrent)
    load_agent.start_reporting_thread()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(max_concurrent)
    print(f"RouteEdge cloud server listening on port {port}...")

    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client,
                                  args=(conn, addr, model, load_agent),
                                  daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
