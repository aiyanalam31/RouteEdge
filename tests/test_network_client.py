"""
test_network_client.py

Tests edge/cloud_client.py's protocol handling against a mock local server,
so this can run without any real Pi or GPU hardware. Also documents the
expected behavior of edge/router/network_client.c against the same mock
server (see TestCNetworkClient, skipped unless librouter.so is built).
"""

import ctypes
import json
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

import numpy as np
import pytest

import cloud_client


def start_mock_server(response_dict, delay_s=0.0, close_early=False):
    """Starts a one-shot mock server that accepts a single connection,
    reads the length-prefixed JPEG payload, optionally sleeps (to simulate
    slow cloud processing), and replies with response_dict as JSON.

    Binds to port 0 (OS-assigned ephemeral port) so consecutive tests never
    race over a fixed port still in TIME_WAIT from a prior test — each test
    gets its own port and returns it for the client to connect to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)

    def _serve():
        conn, _ = sock.accept()
        try:
            length_bytes = conn.recv(4)
            (length,) = struct.unpack(">I", length_bytes)
            payload = b""
            while len(payload) < length:
                payload += conn.recv(length - len(payload))

            if close_early:
                conn.close()
                return

            time.sleep(delay_s)
            response = (json.dumps(response_dict) + "\n").encode("utf-8")
            conn.sendall(response)
        finally:
            conn.close()
            sock.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    time.sleep(0.05)  # give the server a moment to start accepting
    return port


class TestPythonCloudClient:
    def test_successful_round_trip(self):
        port = start_mock_server({"direction": "RIGHT", "confidence": 0.9})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = cloud_client.send_frame(frame, "127.0.0.1", port)
        assert result == "RIGHT"

    def test_null_direction_response(self):
        port = start_mock_server({"direction": None, "confidence": 0.0})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = cloud_client.send_frame(frame, "127.0.0.1", port)
        assert result is None

    def test_server_closes_early_fails_soft(self):
        port = start_mock_server({}, close_early=True)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = cloud_client.send_frame(frame, "127.0.0.1", port)
        assert result is None  # must not raise

    def test_connection_refused_fails_soft(self):
        # Bind and immediately close a socket to get a port guaranteed to
        # have nothing listening on it right now.
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_sock.bind(("127.0.0.1", 0))
        unused_port = probe_sock.getsockname()[1]
        probe_sock.close()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cloud_client.send_frame(frame, "127.0.0.1", unused_port)
        assert result is None

    def test_records_latency_into_network_probe(self):
        port = start_mock_server({"direction": "LEFT", "confidence": 0.8})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        class FakeProbe:
            def __init__(self):
                self.recorded = []
            def record_cloud_latency(self, ms):
                self.recorded.append(ms)

        probe = FakeProbe()
        cloud_client.send_frame(frame, "127.0.0.1", port, network_probe=probe)
        assert len(probe.recorded) == 1
        assert probe.recorded[0] > 0


class TestCNetworkClient:
    """Exercises router/network_client.c's send_frame_to_cloud() directly
    via ctypes, against the same mock server used above. Skipped if
    librouter.so hasn't been built yet (run `make` in edge/router/)."""

    LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "edge",
                             "router", "librouter.so")

    def _load_lib(self):
        if not os.path.exists(self.LIB_PATH):
            pytest.skip("librouter.so not built — run `make` in edge/router/")
        lib = ctypes.CDLL(self.LIB_PATH)
        lib.send_frame_to_cloud.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ]
        lib.send_frame_to_cloud.restype = ctypes.c_int
        return lib

    def test_c_client_successful_round_trip(self):
        lib = self._load_lib()
        port = start_mock_server({"direction": "LEFT"})

        payload = b"\xff\xd8\xff\xe0FAKEJPEGBYTES"  # doesn't need to be a real JPEG for this transport-level test
        buf = (ctypes.c_uint8 * len(payload))(*payload)
        response_buf = (ctypes.c_uint8 * 256)()

        n = lib.send_frame_to_cloud(
            b"127.0.0.1", port, buf, len(payload), response_buf, 256,
        )
        assert n > 0
        response = bytes(response_buf[:n]).decode("utf-8")
        assert "LEFT" in response

    def test_c_client_connection_refused_returns_negative_one(self):
        lib = self._load_lib()

        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_sock.bind(("127.0.0.1", 0))
        unused_port = probe_sock.getsockname()[1]
        probe_sock.close()

        payload = b"x"
        buf = (ctypes.c_uint8 * len(payload))(*payload)
        response_buf = (ctypes.c_uint8 * 256)()

        n = lib.send_frame_to_cloud(
            b"127.0.0.1", unused_port, buf, len(payload), response_buf, 256,
        )
        assert n == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
