# Bus Occupancy Monitor — Web Interface

A real-time web dashboard for your YOLO-based bus people counting system.

## Setup

Open **http://localhost:5051** in your browser.

> **If you still get a numpy/scipy error**, try:
> ```bash
> pip uninstall numpy scipy -y && pip install numpy scipy
> ```

## How it works

1. **Upload** your bus video (MP4, MOV, AVI, MKV)
2. **Set** bus capacity (default 60)
3. **Choose a mode:**
   - **LIVE** — streams every frame with detection boxes overlaid
   - **COUNT** — uses a counting line at 55% of frame height to track entries/exits (requires supervision)
   - **BATCH** — fast mode, skips frames for speed, shows snapshots
4. **Hit Start** — watch the live video feed and stats update in real time

## Modes explained

| Mode  | Best for | How it counts |
|-------|----------|---------------|
| LIVE  | General occupancy check | Counts visible people per frame |
| COUNT | Door monitoring | Entry/exit line crossing — most accurate for in/out |
| BATCH | Long videos, quick overview | Same as LIVE but faster |

## Files

```
bus_monitor/
├── app.py            ← Flask + SocketIO backend
├── requirements.txt
├── best_new.pt       ← your trained model (place here)
├── templates/
│   └── index.html    ← web UI
└── uploads/          ← auto-created, stores uploaded videos
```

## Tips

- For COUNT mode: position camera so people cross the counting line (55% down the frame) at the door
- The interface works on mobile — open the same URL on your phone on the same WiFi
