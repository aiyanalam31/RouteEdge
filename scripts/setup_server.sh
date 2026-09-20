#!/usr/bin/env bash
# setup_server.sh
# Run this on the GPU laptop/workstation acting as the cloud server.
set -euo pipefail

echo "== RouteEdge: cloud server setup =="

cd "$(dirname "$0")/.."

echo "-- Creating virtual environment --"
python3 -m venv .venv
source .venv/bin/activate

echo "-- Installing Python dependencies --"
pip install --upgrade pip
pip install opencv-python-headless numpy pyyaml mediapipe flask pytest pandas matplotlib

echo "-- Starting RouteEdge cloud server --"
cd cloud
python server.py
