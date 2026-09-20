# RouteEdge Wire Protocol

The Pi and cloud server communicate over a plain TCP socket using a simple
length-prefixed framing protocol. Two equivalent implementations exist:

- `edge/cloud_client.py` (Python `socket`) — the default path used by
  `gesture_detect.py`.
- `edge/router/network_client.c` (POSIX sockets via C) — an alternate,
  lower-level path, exercised by `tests/test_network_client.py`.

Both must speak the same protocol described below so either can talk to
`cloud/server.py` without changes on the server side.

## Request (Pi -> Server)

```
[4 bytes] length of JPEG payload, big-endian uint32
[N bytes] JPEG-encoded frame (BGR -> JPEG via cv2.imencode)
```

## Response (Server -> Pi)

A single JSON object, UTF-8 encoded, terminated by a newline (`\n`):

```json
{"direction": "LEFT", "confidence": 0.82, "inference_ms": 34.1}
```

- `direction`: `"LEFT"`, `"RIGHT"`, or `null` if no hand was detected.
- `confidence`: float in `[0, 1]`, from the cloud gesture model
  (see `cloud/gesture_model.py`).
- `inference_ms`: server-side processing time, useful for
  `network_probe.py`'s rolling cloud-latency estimate.

On error, the server still responds (never leaves the connection hanging):

```json
{"direction": null, "error": "description of what went wrong"}
```

## Timeouts

Both client implementations set:
- Connect timeout: ~1 second
- Read/write timeout: ~2 seconds

A client that hits either timeout should treat the result as `None`
(equivalent to "cloud found nothing") rather than raising — a bad network is
exactly the condition the router should already be trying to avoid routing
into, but conditions can change between the routing decision and the
send itself.

## Why length-prefixing instead of a higher-level protocol (HTTP, gRPC)?

For a hackathon timeline, raw sockets keep both the Pi and server sides
small and easy to reason about, and they make the byte-level cost of the
protocol trivially easy to account for in the profiler (4 bytes overhead
per request, plus whatever the JPEG encodes to). If you have time left
over, swapping this for HTTP/gRPC would not change any of the routing
logic — only `cloud_client.py`/`network_client.c` and `server.py`'s
listener would need to change.
