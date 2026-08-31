#!/usr/bin/env python3
import json
import math
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/api/state"
start = time.monotonic()

print("Posting demo radar state to", URL)
print("Ctrl+C to stop")

while True:
    t = time.monotonic() - start
    heading = (t * 25.0) % 360.0
    payload = {
        "map": "AZ3",
        "player": {
            "x": 0.50 + math.sin(t * 0.18) * 0.13,
            "y": 0.50 + math.cos(t * 0.14) * 0.10,
            "heading": heading,
            "confidence": 0.94
        },
        "targets": [
            {
                "screenX": 0.28,
                "bearing": -36,
                "confidence": 0.89,
                "boxHeight": 0.16,
                "stableFrames": 4
            },
            {
                "screenX": 0.72,
                "bearing": 42,
                "confidence": 0.73,
                "boxHeight": 0.08,
                "stableFrames": 3
            }
        ],
        "sounds": [
            {
                "kind": "footstep",
                "bearing": -68,
                "proximity": 0.71,
                "verticalCue": 1,
                "confidence": 0.83
            },
            {
                "kind": "gunfire",
                "bearing": 57,
                "proximity": 0.34,
                "verticalCue": 0,
                "confidence": 0.77
            }
        ],
        "source": "demo_sender",
        "timestamp": time.time()
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            response.read()
    except Exception as exc:
        print("POST failed:", exc)
    time.sleep(0.20)
