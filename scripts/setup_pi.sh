#!/usr/bin/env bash
# setup_pi.sh
# Run this on the Raspberry Pi 5 to prepare it for RouteEdge's edge side.
set -euo pipefail

echo "== RouteEdge: Pi 5 setup =="

echo "-- Installing system packages --"
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    python3-opencv \
    python3-picamera2 \
    build-essential \
    iputils-ping iproute2   # iproute2 gives us `tc` for network_conditions.py

echo "-- Creating virtual environment --"
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

echo "-- Installing Python dependencies --"
pip install --upgrade pip
pip install pyyaml psutil pytest

echo "-- Building the C routing engine (librouter.so) --"
cd edge/router
make clean && make
cd ../..

echo "-- Running unit tests --"
python -m pytest tests/test_confidence_signal.py tests/test_router_scoring.py -v

echo "== Setup complete. =="
echo "Edit edge/config.yaml with your server's IP address, then run:"
echo "    source .venv/bin/activate && python edge/main.py"
