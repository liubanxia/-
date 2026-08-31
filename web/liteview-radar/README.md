# LiteView Web Radar

Zero-dependency local web radar frontend and relay.

## Run

From this directory:

```bash
python server.py
```

Open:

```text
http://127.0.0.1:8765
```

A second device on the same LAN can open:

```text
http://<server-LAN-IP>:8765
```

## Verify realtime updates

In another terminal:

```bash
python demo_sender.py
```

The browser should show a continuously moving green player marker, two red visible-target markers, footstep/gunfire markers and heading updates.

You can also press `切换模拟` in the browser to test rendering without a sender.

## API

### POST /api/state

```json
{
  "map": "AZ3",
  "player": {
    "x": 0.482,
    "y": 0.617,
    "heading": 136,
    "confidence": 0.91
  },
  "targets": [
    {
      "screenX": 0.71,
      "bearing": 32,
      "confidence": 0.87,
      "boxHeight": 0.12,
      "stableFrames": 3
    }
  ],
  "sounds": [
    {
      "kind": "footstep",
      "bearing": -40,
      "proximity": 0.66,
      "verticalCue": 1,
      "confidence": 0.78
    }
  ],
  "timestamp": 0
}
```

If a target later has a calibrated map coordinate, send `mapX` and `mapY`; the renderer will place it directly instead of estimating a display position from bearing/range evidence.

### GET /api/events

Server-Sent Events stream used by the browser.

### GET /api/state

Returns the current state snapshot.

### GET /health

Relay health check.

## Map layers

`maps/az3.json` is intentionally marked `calibrated:false`.

The existing Swift `DeltaMapSeeds` coordinates are public-topology visualization hints, not true game/world coordinates. They are therefore not copied into the 1:1 calibration layer. The renderer and map layer are separated so a permitted screen-visible reference can later be calibrated without rewriting the realtime UI.

## Boundary

This module is designed for ReplayKit/screen-visible analysis, user-provided data, and other permitted data sources. It does not read game process memory, hidden entity lists, DMA, injected state, or anti-cheat-protected data.
