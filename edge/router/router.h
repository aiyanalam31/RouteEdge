/*
 * router.h
 *
 * Struct definitions and function declarations for RouteEdge's routing
 * engine. This runs on the Raspberry Pi 5 and decides, per frame, whether
 * to trust the local gesture-detection result or escalate to the cloud.
 *
 * Layout of ConfidenceSignal MUST match confidence_signal.py's
 * ConfidenceSignal.as_tuple() ordering exactly, since router_bindings.py
 * marshals it via ctypes.
 */

#ifndef ROUTEEDGE_ROUTER_H
#define ROUTEEDGE_ROUTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   has_contour;    /* 0 or 1 */
    float aspect_margin;  /* 0..1, higher = more confident shape read */
    float mask_noise;     /* 0..1, higher = noisier background/mask */
    float flicker_rate;   /* 0..1, higher = less stable local reading */
    float brightness;     /* 0..1, normalized mean V channel */
} ConfidenceSignal;

typedef enum {
    ROUTE_LOCAL = 0,
    ROUTE_CLOUD = 1
} RouteDecision;

typedef struct {
    RouteDecision decision;
    float local_cost;
    float cloud_cost;
} RouteResult;

/*
 * router_decide
 *
 * Computes a weighted cost for the local and cloud paths and returns
 * whichever is cheaper, subject to a hard privacy constraint (if
 * privacy_flag is nonzero, ROUTE_LOCAL is always returned regardless of
 * cost, and cloud_cost is reported as +INFINITY so the trace makes clear
 * the constraint — not the score — decided the outcome).
 */
RouteResult router_decide(
    ConfidenceSignal sig,
    float network_delay_ms,
    float predicted_cloud_latency_ms,
    int   privacy_flag
);

#ifdef __cplusplus
}
#endif

#endif /* ROUTEEDGE_ROUTER_H */
