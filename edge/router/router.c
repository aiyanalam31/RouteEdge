/*
 * router.c
 *
 * RouteEdge's transparent weighted-scoring routing policy.
 *
 * Design intent: every term here should be individually inspectable and
 * explainable in the evaluation writeup ("the router chose cloud because
 * mask_noise was high and network_delay was low") rather than hidden inside
 * a learned/opaque predictor. Weights are tuned empirically against
 * benchmark data collected under experiments/.
 */

#include "router.h"
#include <math.h>

/* ── Local-cost weights ──────────────────────────────────────────────── */
#define W_NO_CONTOUR    0.35f   /* penalize: no contour found at all */
#define W_ASPECT_MARGIN 0.20f   /* penalize: ambiguous shape */
#define W_MASK_NOISE    0.15f   /* penalize: noisy background */
#define W_FLICKER       0.15f   /* penalize: unstable frame-to-frame reading */

/* ── Cloud-cost weights ──────────────────────────────────────────────── */
#define W_NETWORK_DELAY 0.30f
#define W_CLOUD_LATENCY 0.20f

/* Normalization baselines — tune against real measurements from
 * experiments/network_conditions.py sweeps. */
#define NETWORK_DELAY_BASELINE_MS 200.0f
#define CLOUD_LATENCY_BASELINE_MS 100.0f

RouteResult router_decide(
    ConfidenceSignal sig,
    float network_delay_ms,
    float predicted_cloud_latency_ms,
    int   privacy_flag
) {
    RouteResult r;

    float local_cost =
        W_NO_CONTOUR    * (sig.has_contour ? 0.0f : 1.0f) +
        W_ASPECT_MARGIN * (1.0f - sig.aspect_margin) +
        W_MASK_NOISE    * sig.mask_noise +
        W_FLICKER       * sig.flicker_rate;

    float cloud_cost =
        W_NETWORK_DELAY * (network_delay_ms / NETWORK_DELAY_BASELINE_MS) +
        W_CLOUD_LATENCY * (predicted_cloud_latency_ms / CLOUD_LATENCY_BASELINE_MS);

    /* Hard constraint: privacy overrides the score entirely. This is a
     * gate, not one more weighted term, so it can be honestly described as
     * a guarantee rather than a preference. */
    if (privacy_flag) {
        r.decision   = ROUTE_LOCAL;
        r.local_cost = local_cost;
        r.cloud_cost = INFINITY;
        return r;
    }

    r.local_cost = local_cost;
    r.cloud_cost = cloud_cost;
    r.decision   = (local_cost < cloud_cost) ? ROUTE_LOCAL : ROUTE_CLOUD;
    return r;
}
