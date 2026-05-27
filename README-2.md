# BusOccupancy AI

A real-time passenger monitoring system built with YOLOv8 and OpenCV. Detects and counts passengers boarding and exiting buses, metros, or any multi-door vehicle — with a live web dashboard, per-door video feeds, a combined occupancy timeline, and a dedicated peak-hour analytics view that aggregates all historical sessions in one place.

---

## Requirements

- Python 3.9+
- A trained YOLOv8 model file (`best_new.pt`) in the project root
- Git LFS if you store the model in the repository (see below)

---

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd BusOccupancy_AI

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate       # Mac / Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model in the project root
cp /path/to/best_new.pt .

# 5. Run
python app.py
```

Open **http://localhost:5051** in your browser.

> If `best_new.pt` is missing the app starts in **DEMO mode** — random counts are simulated so the full dashboard UI can still be tested without a model.

---

## Features

- Real-time video processing with YOLOv8 detection
- Centroid-based person tracking (no external tracker library needed)
- Entry/exit counting via a configurable virtual counting line per door
- **Single-door mode** — one camera feed with live IN/OUT counters, occupancy bar, and timeline chart
- **Multi-door mode** — add any number of doors (2–8+); each door gets its own video feed, colour-coded counting line, and independent IN/OUT counters that combine into one total
- **Starting passenger count** — set how many people are already on board before the video begins; count never drops below zero
- **Peak-hour analytics tab** — a dedicated full-screen view that reads all saved CSV logs (single-door and multi-door combined) and builds a 24-hour occupancy heatmap with summary statistics and a per-session table
- Pause / resume processing at any time
- Capacity alerts at 80% (warning) and 100% (critical)
- CSV session logging — every run is timestamped and downloadable from the dashboard
- Three processing modes: Live, Count, Batch

---

## Project Structure

```
BusOccupancy_AI/
├── app.py              ← Flask + SocketIO server (single-door & multi-door)
├── tracker.py          ← CentroidTracker, LineCrossCounter
├── drawing.py          ← OpenCV annotation helpers + 8-colour door palette
├── logger.py           ← CSV session logger + hourly log analysis
├── config.py           ← All settings in one place
├── requirements.txt
├── README.md
├── best_new.pt         ← Trained YOLOv8 model (use Git LFS if > 100 MB)
├── templates/
│   └── index.html      ← Web dashboard
├── uploads/            ← Uploaded videos at runtime (git-ignored)
│   └── .gitkeep
└── logs/               ← Session CSV logs at runtime (git-ignored)
    └── .gitkeep
```

---

## Dashboard

The left panel has three input-mode tabs that switch the entire right-hand content area.

---

### 🚪 Single Door

The standard single-camera view for vehicles monitored by one fixed camera.

1. Select the **Single** tab in the left panel.
2. Drop a video file onto the upload zone (or click to browse — MP4, MOV, AVI, MKV).
3. Drag the **Counting line** slider to position the virtual line across the door in the frame.
4. Set **Capacity** and **Starting passengers** (passengers already on board when the clip begins).
5. Choose a **Processing mode** (see table below).
6. Press **▶ START**.

The right side shows the live annotated video feed. The bottom strip shows two tabs:

- **METRICS** — count, density, capacity, entered/exited totals, and starting offset
- **OCCUPANCY CHART** — a live-updating line chart of passenger count vs time

Alert banners appear on the video feed at 80% (warning) and 100% (critical) occupancy.

---

### 🚪🚪 Multi Door

For vehicles with multiple entry/exit points — articulated buses, metro cars, trams, etc. Every door is processed in its own background thread simultaneously.

1. Select the **Multi** tab in the left panel.
2. Click a quick-set button (**2 / 3 / 4 / 6 / 8**) or press **+ Add Door** to build a custom list.
3. For each door entry, click to upload a video file and drag the **LINE** slider to the correct position.
4. Set the shared **Capacity**, **Starting passengers**, and **Processing mode**.
5. Press **▶ START**.

The right side splits into a grid of live feed cards, one per door, each colour-coded by door label. A summary strip below the grid shows combined IN/OUT counts per door. The shared occupancy chart and metrics strip reflect the combined total across all doors.

Door colour palette: cyan (A) · amber (B) · purple (C) · green (D) · orange (E) · pink (F) · teal (G) · lime (H). The palette cycles for more than 8 doors.

---

### 📊 Analytics

A dedicated full-screen view that aggregates **all** saved CSV logs — from both single-door and multi-door sessions — into a 24-hour occupancy heatmap. No extra configuration is needed; every completed session automatically contributes data.

Selecting this tab replaces the video area with four components:

**Summary cards (top row)**

| Card | What it shows |
|------|---------------|
| Peak Hour | The hour of day with the highest average occupancy across all sessions |
| Peak Occupancy | The single highest occupancy reading ever recorded, and at which hour |
| Quietest Hour | The hour with the lowest average occupancy |
| Hours w/ Data | How many of the 24 hours have at least one logged session |

**24-hour bar chart** fills the main area. Each bar is the average occupancy for that hour; a dashed red line overlays the per-hour maximum. Bars are colour-coded: green (low) → blue (moderate) → amber (busy) → red (near capacity). Hovering shows the exact values.

**Session summary table** (bottom) lists every CSV file with its recorded hour, average %, peak %, and row count — so single-door and multi-door sessions appear side by side. Newest sessions are shown first.

Use the **↻ Refresh** button in the left panel (or switch away and back) to reload after running new sessions.

---

## Processing Modes

| Mode | Description |
|------|-------------|
| LIVE | Every frame processed and streamed with detection boxes and counting line overlay |
| COUNT | Full entry/exit line-crossing tracking — most accurate for boarding counts |
| BATCH | Skips frames for speed — best for long recordings where only a snapshot count is needed |

---

## Configuration

Edit `config.py` to change defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `BUS_CAPACITY` | `60` | Maximum passenger capacity |
| `LINE_RATIO` | `0.55` | Default Door A counting line height (0.0 = top, 1.0 = bottom) |
| `DOOR_LINE_RATIOS` | `[0.55, 0.35, 0.50, 0.45]` | Fallback line positions for doors A–D in multi-door mode |
| `INITIAL_COUNT` | `0` | Passengers already on board at video start |
| `CONF_THRESHOLD` | `0.25` | YOLO detection confidence threshold |
| `TRACKER_MAX_GONE` | `2.0` | Seconds before a lost track is dropped |
| `CROSS_COOLDOWN` | `20` | Frames of cooldown after a line crossing (prevents double-counts) |
| `ALERT_WARNING` | `0.8` | Occupancy ratio that triggers a warning alert |
| `ALERT_CRITICAL` | `1.0` | Occupancy ratio that triggers a critical alert |
| `STREAM_WIDTH` | `960` | Max pixel width of frames streamed to the browser |
| `JPEG_QUALITY` | `75` | JPEG compression quality for streamed frames |

---

## CSV Log Format

Every session writes a timestamped file to `logs/` named `<videoname>_YYYYMMDD_HHMMSS.csv`. The Analytics tab reads these files automatically.

| Column | Description |
|--------|-------------|
| `wall_time` | Real clock time at the logged frame (`HH:MM:SS`) |
| `elapsed_s` | Seconds since the session started |
| `frame` | Frame index |
| `count` | Estimated passengers currently in the vehicle |
| `total_in` | Cumulative entries across all doors |
| `total_out` | Cumulative exits across all doors |
| `door_A_in` / `door_A_out` | Per-door counts — one pair of columns per door (A, B, C …) |
| `occupancy_pct` | Occupancy as a percentage of capacity |
| `density` | `LOW` / `MEDIUM` / `HIGH` / `FULL` |

The hour is parsed from the filename timestamp and used to slot each session into the correct bar in the 24-hour chart.

---

## Model

The app expects a YOLOv8 model trained to detect people, saved as `best_new.pt` in the project root. To train your own:

```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
cp runs/detect/train/weights/best.pt best_new.pt
```

---

## Git LFS (for large model files)

If `best_new.pt` exceeds 100 MB, use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes best_new.pt
git commit -m "Add model via LFS"
```

---

## .gitignore

```
__pycache__/
*.pyc
uploads/*
!uploads/.gitkeep
logs/*
!logs/.gitkeep
.env
```

Only add `*.pt` to `.gitignore` if you are **not** using Git LFS and are hosting the model elsewhere (Google Drive, S3, Hugging Face Hub, etc.).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 5051 already in use | Change `port=5051` at the bottom of `app.py` |
| `ultralytics` install fails | Run `pip install --upgrade pip` first, then retry |
| Slow on first frame | Normal — YOLOv8 JIT-compiles on first inference |
| Running on CPU only | Install the CUDA build of PyTorch that matches your driver for GPU acceleration |
| App shows DEMO mode | `best_new.pt` not found in project root — check `MODEL_PATH` in `config.py` |
| Multi-door — one door stalls | Verify each video was uploaded successfully; per-door errors appear in the browser console |
| Analytics tab is empty | Complete at least one session so a CSV is written to `logs/` |
| Analytics shows wrong hour | The hour is read from the filename timestamp (`HHMMSS`) — check your system clock is correct |
| Pause button has no effect | Ensure you are running the updated `app.py` with `pause_processing` / `resume_processing` socket handlers |
