/*
 * network_client.h
 *
 * NOTE: This C socket client is kept as an alternate, lower-level path for
 * sending frames to the cloud server. The default RouteEdge pipeline
 * (gesture_detect.py) instead uses edge/cloud_client.py (plain Python
 * sockets) for this, since marshalling variable-length JPEG byte buffers
 * through ctypes is fiddlier than the fixed-size ConfidenceSignal struct
 * the router itself uses. This file is provided for anyone who wants to
 * push frame transmission into C as well (e.g. for a stricter latency
 * budget), and is exercised by tests/test_network_client.py against a
 * mock server.
 *
 * Wire protocol (see docs/protocol_spec.md for full details):
 *   [4-byte big-endian length][JPEG bytes]  ->  server
 *   [response bytes, small fixed buffer]    <-  server
 */

#ifndef ROUTEEDGE_NETWORK_CLIENT_H
#define ROUTEEDGE_NETWORK_CLIENT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * send_frame_to_cloud
 *
 * Opens a TCP connection to server_ip:port, sends a length-prefixed JPEG
 * payload, and reads the response into response_buf (caller-allocated,
 * response_buf_size bytes).
 *
 * Returns: number of bytes received (>= 0) on success, -1 on any failure
 * (connect failure, send failure, or timeout). Callers MUST handle -1
 * gracefully — a bad network is exactly the condition the router should
 * already be trying to avoid routing into, but network conditions can
 * change between the routing decision and the actual send.
 */
int send_frame_to_cloud(
    const char* server_ip,
    int port,
    const uint8_t* jpeg_data,
    uint32_t jpeg_len,
    uint8_t* response_buf,
    int response_buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* ROUTEEDGE_NETWORK_CLIENT_H */
