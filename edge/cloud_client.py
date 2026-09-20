"""
cloud_client.py

Handles sending a frame to the cloud gesture server and parsing its
response. Implemented in plain Python (sockets) rather than through the C
router, since marshalling variable-length JPEG byte buffers through ctypes
is fiddlier than the fixed-size ConfidenceSignal struct the router itself
uses — this keeps that part easy to debug and iterate on.

(A C-level equivalent exists at router/network_client.c for anyone who
wants to push this into C as well; see the note at the top of that file.)

Wire protocol (see docs/protocol_spec.md):
    [4-byte big-endian length][JPEG bytes]   -> server
    [JSON response, newline-terminated]      <- server
"""

import json
import socket
import struct
import time

import cv2


CONNECT_TIMEOUT_S = 1.0
IO_TIMEOUT_S = 2.0


def send_frame(frame, server_ip, port, network_probe=None):
    """
    Encodes `frame` as JPEG, sends it to the cloud gesture server, and
    returns the parsed gesture result (e.g. "LEFT" / "RIGHT" / None).

    If network_probe is provided, records the observed round-trip latency
    back into it so future routing decisions reflect real recent behavior.

    Returns None on any failure (timeout, connection refused, malformed
    response) — callers should treat this the same as "local method found
    nothing," i.e. gracefully, not as a fatal error.
    """
    start = time.time()
    try:
        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        jpeg_bytes = jpeg_buf.tobytes()

        with socket.create_connection((server_ip, port), timeout=CONNECT_TIMEOUT_S) as sock:
            sock.settimeout(IO_TIMEOUT_S)

            length_prefix = struct.pack(">I", len(jpeg_bytes))
            sock.sendall(length_prefix)
            sock.sendall(jpeg_bytes)

            response_chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_chunks.append(chunk)
                if chunk.endswith(b"\n"):
                    break

            raw_response = b"".join(response_chunks).decode("utf-8").strip()
            result = json.loads(raw_response)

        elapsed_ms = (time.time() - start) * 1000.0
        if network_probe is not None:
            network_probe.record_cloud_latency(elapsed_ms)

        return result.get("direction")

    except (socket.timeout, ConnectionRefusedError, OSError, json.JSONDecodeError):
        # A bad network is exactly the condition the router should already
        # be trying to avoid routing into — but conditions can change
        # between the routing decision and this call. Fail soft.
        return None
