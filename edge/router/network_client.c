/*
 * network_client.c
 *
 * Plain POSIX TCP socket client. Runs as normal Linux userspace code (the
 * Pi 5 runs a full Linux OS, not bare-metal firmware) — this goes through
 * the kernel's standard socket/TCP/IP stack, same as any other network
 * program. There is no meaningful "register-level" version of this: doing
 * so would mean writing a custom NIC driver and TCP/IP stack from scratch,
 * which would not move the needle on latency (that's dominated by physical
 * network delay and cloud inference time, not kernel socket overhead).
 *
 * See network_client.h for the wire protocol and usage notes. As noted
 * there, gesture_detect.py's default path uses cloud_client.py instead;
 * this file exists for anyone who wants a C-level frame-send path and is
 * covered by tests/test_network_client.py.
 */

#include "network_client.h"

#include <arpa/inet.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define CONNECT_TIMEOUT_SEC 1
#define IO_TIMEOUT_SEC      2

int send_frame_to_cloud(
    const char* server_ip,
    int port,
    const uint8_t* jpeg_data,
    uint32_t jpeg_len,
    uint8_t* response_buf,
    int response_buf_size
) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }

    /* Timeouts matter a lot here: a live camera loop can't afford to hang
     * on a stalled connect()/recv() if the network degrades mid-demo. */
    struct timeval tv;
    tv.tv_sec  = IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port   = htons((uint16_t)port);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) != 1) {
        close(sock);
        return -1; /* bad IP string */
    }

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        return -1; /* network unreachable, refused, or timed out */
    }

    /* Length-prefixed frame: [4-byte big-endian length][JPEG bytes] */
    uint32_t len_net = htonl(jpeg_len);
    if (send(sock, &len_net, sizeof(len_net), 0) != (ssize_t)sizeof(len_net)) {
        close(sock);
        return -1;
    }

    ssize_t sent_total = 0;
    while (sent_total < (ssize_t)jpeg_len) {
        ssize_t n = send(sock, jpeg_data + sent_total,
                          jpeg_len - (uint32_t)sent_total, 0);
        if (n <= 0) {
            close(sock);
            return -1;
        }
        sent_total += n;
    }

    int received = (int)recv(sock, response_buf, (size_t)response_buf_size, 0);
    close(sock);

    return received; /* -1 on timeout/error, >=0 on success (0 = closed) */
}
