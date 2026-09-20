# RouteEdge Demo Video Script

Target length: ~2-3 minutes. Goal: show the SAME gesture being routed
differently as conditions change, live, with the on-screen "source:
LOCAL/CLOUD" tag visible the whole time.

## Beat 1 — Setup shot (10s)
- Wide shot: Pi 5 with camera, pointed at presenter's hand.
- Laptop screen showing the dashboard (dashboard/static/dashboard.html),
  currently idle.

## Beat 2 — Good conditions, local routing (20s)
- Good lighting, good network (default `tc netem` state, i.e. no shaping
  applied).
- Presenter points LEFT, then RIGHT a few times.
- Call out on screen: "source: LOCAL" tag stays green the whole time.
- Dashboard shows "routed local: ~100%", low latency.

## Beat 3 — Degrade lighting (30s)
- Dim the room lights (or move to backlit position).
- Same gesture repeated.
- Watch the confidence signals in the terminal/dashboard degrade
  (aspect_margin drops, flicker_rate rises, brightness drops).
- Show the "source" tag flip to orange ("CLOUD") mid-demo.
- Voiceover: "Same gesture. The router noticed the local detector was
  struggling — flickering between LEFT and RIGHT, low brightness — and
  escalated to the cloud model, which isn't fooled by lighting."

## Beat 4 — Restore lighting, degrade network instead (30s)
- Lights back on. Run `experiments/network_conditions.py --preset bad` on
  the link between Pi and server.
- Same gesture, same good lighting.
- Show the tag flip back to "LOCAL" — even though local confidence would
  otherwise be borderline, the router now avoids cloud because the network
  is expensive.
- Voiceover: "Now the opposite: good lighting, bad network. The router
  reverses its decision for the opposite reason — the same weighted
  scoring function, responding to a different constraint."

## Beat 5 — Privacy override (15s)
- Press 'v' to toggle `privacy_flag`.
- Dim the lights again (a condition that would otherwise clearly favor
  cloud).
- Show that despite everything favoring cloud, the tag stays "LOCAL" the
  entire time.
- Voiceover: "And if privacy is required, the router never even considers
  the cloud path — it's a hard constraint, not a preference."

## Beat 6 — Benchmark results (20s)
- Cut to `analysis/routing_split.png` and `analysis/latency_comparison.png`.
- Voiceover: summarize the always-edge / always-cloud / threshold-rule /
  RouteEdge comparison in one sentence each.

## Beat 7 — Close (10s)
- One slide: "RouteEdge: routing gesture recognition between Pi 5 and
  cloud based on real-time confidence, network, and privacy signals —
  transparent, hardware-aware, adaptive."
