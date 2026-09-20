"""
dashboard/app.py

Small Flask server that tails the Pi's live JSONL log
(edge/config.yaml -> logging.path) and serves it to the browser dashboard.
Runs on the laptop (or a third viewer machine) — it's read-only and sits
outside the request's critical path, so it never blocks or slows down
routing decisions on the Pi.

For a hackathon demo, the simplest reliable setup is to have this read the
log file over a shared network mount or synced folder, or have
gesture_detect.py's logger.py POST each record to this server's
/ingest endpoint instead of writing purely to a local file. This scaffold
supports both: /ingest for push-based updates, and a background thread that
also tails a local file path if you'd rather pull.
"""

import json
import os
import threading
import time
from collections import deque

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

MAX_RECORDS = 500
_records = deque(maxlen=MAX_RECORDS)
_lock = threading.Lock()


@app.route("/ingest", methods=["POST"])
def ingest():
    """Pi-side logger.py can optionally POST records here directly instead
    of (or in addition to) writing to a local JSONL file, for a truly live
    dashboard with no file-tailing latency."""
    record = request.get_json(force=True)
    with _lock:
        _records.append(record)
    return jsonify({"status": "ok"})


@app.route("/api/records")
def api_records():
    with _lock:
        return jsonify(list(_records))


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "dashboard.html")


def tail_file(path, poll_interval_s=0.5):
    """Optional pull-based alternative to /ingest: tails a local JSONL file
    (e.g. a synced copy of the Pi's log) and appends new lines as they
    appear."""
    last_size = 0
    while True:
        if os.path.exists(path):
            with open(path, "r") as f:
                f.seek(last_size)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            with _lock:
                                _records.append(record)
                        except json.JSONDecodeError:
                            pass
                last_size = f.tell()
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    # Uncomment and point at a synced log path if using the pull-based mode:
    # threading.Thread(target=tail_file, args=("shared/session_live.jsonl",),
    #                   daemon=True).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
