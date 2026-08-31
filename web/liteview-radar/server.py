#!/usr/bin/env python3
import argparse
import json
import os
import socket
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
MAX_BODY = 128 * 1024
DISCOVERY_PORT = 8766
DISCOVERY_MAGIC = "LITEVIEW_RADAR_V1"

_state_lock = threading.Condition()
_state_version = 0
_state = {
    "map": "AZ3",
    "player": {"x": None, "y": None, "heading": 0.0, "confidence": 0.0, "mode": "unlocked"},
    "targets": [],
    "sounds": [],
    "timestamp": time.time(),
}


def snapshot():
    with _state_lock:
        return _state_version, dict(_state)


def replace_state(payload):
    global _state_version, _state
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    sanitized = {
        "map": str(payload.get("map", _state.get("map", "AZ3")))[:32],
        "player": payload.get("player") if isinstance(payload.get("player"), dict) else _state.get("player", {}),
        "targets": payload.get("targets") if isinstance(payload.get("targets"), list) else [],
        "sounds": payload.get("sounds") if isinstance(payload.get("sounds"), list) else [],
        "timestamp": float(payload.get("timestamp", time.time())),
    }
    for key in ("localization", "diagnostics", "floor", "source"):
        if key in payload:
            sanitized[key] = payload[key]
    with _state_lock:
        _state = sanitized
        _state_version += 1
        _state_lock.notify_all()
        return _state_version


def beacon_loop(stop_event, http_port):
    payload = json.dumps(
        {"service": DISCOVERY_MAGIC, "port": http_port},
        separators=(",", ":"),
    ).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while not stop_event.wait(0.85):
            try:
                sock.sendto(payload, ("255.255.255.255", DISCOVERY_PORT))
            except OSError:
                # LAN discovery is optional; HTTP still works through an explicit endpoint.
                pass
    finally:
        sock.close()


class RadarHandler(SimpleHTTPRequestHandler):
    server_version = "LiteViewRadar/0.2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            version, _ = snapshot()
            return self._json(
                {
                    "ok": True,
                    "version": version,
                    "time": time.time(),
                    "discovery": {"magic": DISCOVERY_MAGIC, "udpPort": DISCOVERY_PORT},
                }
            )
        if path == "/api/state":
            version, data = snapshot()
            data["version"] = version
            return self._json(data)
        if path == "/api/events":
            return self._sse()
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/state":
            return self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return self._json({"ok": False, "error": "invalid body size"}, HTTPStatus.BAD_REQUEST)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            version = replace_state(payload)
        except Exception as exc:
            return self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self._json({"ok": True, "version": version})

    def _sse(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        last_version = -1
        try:
            while True:
                with _state_lock:
                    if _state_version == last_version:
                        _state_lock.wait(timeout=10.0)
                    version = _state_version
                    data = dict(_state)
                if version != last_version:
                    packet = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"id: {version}\ndata: {packet}\n\n".encode("utf-8"))
                    last_version = version
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def main():
    parser = argparse.ArgumentParser(description="LiteView Web Radar local relay")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(ROOT)

    stop_event = threading.Event()
    beacon = threading.Thread(
        target=beacon_loop,
        args=(stop_event, args.port),
        name="liteview-radar-beacon",
        daemon=True,
    )
    beacon.start()

    server = ThreadingHTTPServer((args.host, args.port), RadarHandler)
    print(f"LiteView Web Radar: http://127.0.0.1:{args.port}")
    print(f"LAN viewers:        http://<this-device-ip>:{args.port}")
    print(f"POST state to:      http://<this-device-ip>:{args.port}/api/state")
    print(f"Auto discovery:     UDP broadcast {DISCOVERY_PORT} / {DISCOVERY_MAGIC}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
        beacon.join(timeout=1.0)


if __name__ == "__main__":
    main()
