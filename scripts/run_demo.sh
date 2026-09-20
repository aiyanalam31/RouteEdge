#!/usr/bin/env bash
# run_demo.sh
#
# One-command full-stack launch for recording the demo video. Intended to
# be run FROM THE PI, with SSH access configured to the cloud server (or
# run manually in two terminals if you'd rather watch both logs live).
#
# What this does:
#   1. Starts the cloud server (remotely, over SSH, backgrounded)
#   2. Starts the dashboard (locally, backgrounded)
#   3. Starts the gesture detector in the foreground, so you can watch
#      its OpenCV windows directly and press 'q' to quit / 'v' to toggle
#      privacy for the demo beats described in docs/demo_video_script.md
#
# Edit SERVER_HOST / SERVER_USER below to match your setup, or just run
# each piece manually in separate terminals if SSH isn't configured —
# the individual commands are echoed below either way.

set -euo pipefail

SERVER_HOST="${ROUTEEDGE_SERVER_HOST:-192.168.1.50}"
SERVER_USER="${ROUTEEDGE_SERVER_USER:-pi}"
SERVER_REPO_PATH="${ROUTEEDGE_SERVER_REPO_PATH:-~/routeedge}"

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || echo "(no local .venv found — continuing with system python)"

echo "== RouteEdge demo launch =="
echo ""
echo "1) Cloud server:"
echo "   ssh ${SERVER_USER}@${SERVER_HOST} 'cd ${SERVER_REPO_PATH} && source .venv/bin/activate && python cloud/server.py'"
read -p "   Press Enter once the cloud server is running (check its terminal for 'listening on port')... "

echo ""
echo "2) Dashboard (local, background):"
(cd dashboard && python app.py &) 
sleep 1
echo "   Dashboard running at http://localhost:5050"

echo ""
echo "3) Gesture detector (foreground) — press 'q' to quit, 'p' to print"
echo "   current CV params, 'v' to toggle the privacy flag live."
echo ""
echo "   Reminder — demo beats (see docs/demo_video_script.md):"
echo "     - Beat 2: good lighting + good network -> expect LOCAL"
echo "     - Beat 3: dim lighting + good network  -> expect flip to CLOUD"
echo "     - Beat 4: run 'python experiments/network_conditions.py"
echo "               --iface <iface> --preset bad' in another terminal,"
echo "               good lighting -> expect flip back to LOCAL"
echo "     - Beat 5: press 'v' to force privacy -> expect LOCAL regardless"
echo ""

python edge/main.py
