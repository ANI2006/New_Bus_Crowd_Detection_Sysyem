# BusOccupancy AI

A real-time passenger monitoring system built with **YOLOv8** and **OpenCV**.
Detects and counts passengers boarding and exiting buses, metros, or any multi-door vehicle — with a live web dashboard, per-door video feeds, a combined occupancy timeline, and a dedicated peak-hour analytics view that aggregates all historical sessions.

> **Samsung Innovation Campus — AI Course Final Project**

---

## Requirements

- Python **3.10+** (3.9 minimum; 3.10+ recommended for `match` / `type | None` syntax used in the code)
- A trained YOLOv8 model file (`best_new.pt`) placed in the project root
- Git LFS if you store the model in the repository (see [Git LFS](#git-lfs) below)

---

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd BusOccupancy_AI

# 2. Create and activate a virtual environment (strongly recommended)
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model in the project root
cp /path/to/best_new.pt .

# 5. (Optional) Set a secure secret key
export SECRET_KEY="replace-with-a-long-random-string"

# 6. Run
python app.py
```

Open **http://localhost:5051** in your browser.

> **No model?** If `best_new.pt` is missing, the app starts in **DEMO mode** — random counts are simulated so the full dashboard UI can still be tested without a GPU or model file.

---

## Features

| Feature | Details |
|---------|---------|
| Real-time detection | YOLOv8 inference on every frame (or every Nth frame in batch mode) |
| Person tracking | Centroid-based tracker — no external library required |
| Single-door mode | One camera feed; virtual counting line tracks IN/OUT |
| Multi-door mode | 1–8+ doors processed in parallel; each door has its own feed and counter |
| Starting passenger count | Set pre-boarded passengers; count never drops below zero |
| Peak-hour analytics | Reads all saved CSV logs and plots a 24-hour occupancy heatmap |
| Capacity alerts | Warning banner at 80%, critical at 100% |
| Pause / Resume | Pause mid-video without losing counts |
| CSV session logging | Every run is timestamped and downloadable |
| Reconnect handling | Client shows a banner if the WebSocket drops and restores when it reconnects |
| Upload safety | 2 GB file size cap; only MP4/MOV/AVI/MKV/WEBM accepted |

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
│   └── index.html      ← Web dashboard (Single, Multi, Analytics tabs)
├── uploads/            ← Uploaded videos at runtime (git-ignored)
│   └── .gitkeep
└── logs/               ← Session CSV logs at runtime (git-ignored)
    └── .gitkeep
```

---

## Dashboard — How to Use

### 🚪 Single Door

1. Click the **Single** tab in the left panel.
2. Drop a video file onto the upload zone (MP4, MOV, AVI, MKV, WEBM — max 2 GB).
3. Drag the **Counting line** slider to position the virtual line across the door.
4. Set **Capacity** and **Starting passengers**.
5. Choose **COUNT** (accurate) or **BATCH** (fast) mode.
6. Press **▶ START**.

The right panel shows the live annotated feed. The bottom strip shows:
- **METRICS** — count, density, entered/exited, starting offset
- **OCCUPANCY CHART** — live line chart of passenger count vs time

---

### 🚪🚪 Multi Door

For vehicles with multiple entry/exit points (articulated buses, metro cars, trams).

1. Click the **Multi** tab.
2. Click a quick-set button (**2 / 3 / 4 / 6 / 8**) or press **+ Add Door**.
3. For each door, click to upload a video and drag the **LINE** slider to the correct position.
4. Set shared **Capacity**, **Starting passengers**, and processing mode.
5. Press **▶ START**.

The right area splits into a grid of live feed cards, one per door, colour-coded by door label. A summary strip shows combined IN/OUT counts per door.

Door colour palette: **cyan** (A) · **yellow** (B) · **purple** (C) · **green** (D) · **orange** (E) · **pink** (F) · **teal** (G) · **lime** (H). Cycles for more than 8 doors.

---

### 📊 Analytics

Click the **Analytics** tab to switch to the full-screen peak-hour view. This aggregates **all** saved CSV logs automatically.

| Card | What it shows |
|------|---------------|
| Peak Hour | Hour of day with the highest average occupancy |
| Peak Occupancy | Single highest reading ever recorded |
| Quietest Hour | Hour with the lowest average occupancy |
| Hours w/ Data | How many of the 24 hours have at least one session |

The **24-hour bar chart** colour-codes each bar: green (low) → blue (moderate) → amber (busy) → red (near capacity). A dashed red line overlays the per-hour maximum. The **session summary table** lists every CSV file with its hour, average %, peak %, and row count.

---

## Processing Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| **COUNT** | Tracks every person crossing the virtual line | Door cameras — most accurate |
| **BATCH** | Skips frames; counts detected people per sampled frame | Long recordings where approximate snapshot counts are fine |

> **LIVE mode** from the original single-door version has been removed in the multi-door rewrite. COUNT mode streams annotated frames in real time and is equivalent.

---

## Configuration

Edit `config.py` to change defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `BUS_CAPACITY` | `60` | Maximum passenger capacity |
| `LINE_RATIO` | `0.55` | Default counting line height (0.0 = top, 1.0 = bottom) |
| `DOOR_LINE_RATIOS` | `[0.55, 0.35, 0.50, 0.45]` | Fallback line positions for doors A–D when not set by the client |
| `INITIAL_COUNT` | `0` | Passengers already on board at video start |
| `CONF_THRESHOLD` | `0.25` | YOLO detection confidence threshold |
| `TRACKER_MAX_GONE` | `2.0` | Seconds before a lost track is dropped |
| `CROSS_COOLDOWN` | `20` | Frames of cooldown after a line crossing (prevents double-counts) |
| `ALERT_WARNING` | `0.8` | Occupancy ratio that triggers a warning banner |
| `ALERT_CRITICAL` | `1.0` | Occupancy ratio that triggers a critical banner |
| `STREAM_WIDTH` | `960` | Max pixel width of frames streamed to the browser |
| `JPEG_QUALITY` | `75` | JPEG compression quality for streamed frames |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SECRET_KEY` | Flask session secret — **set this in production** |

---

## CSV Log Format

Every session writes a file to `logs/` named `<videoname>_YYYYMMDD_HHMMSS.csv`.

| Column | Description |
|--------|-------------|
| `wall_time` | Real clock time at the logged frame (`HH:MM:SS`) |
| `elapsed_s` | Seconds since the session started |
| `frame` | Frame index |
| `count` | Estimated passengers currently in the vehicle |
| `total_in` | Cumulative entries across all doors |
| `total_out` | Cumulative exits across all doors |
| `door_A_in` / `door_A_out` | Per-door counts — one pair per door (A, B, C …) |
| `occupancy_pct` | Occupancy as a percentage of capacity |
| `density` | `LOW` / `MEDIUM` / `HIGH` / `FULL` |

---

## Model

The app expects a YOLOv8 model trained to detect people, saved as `best_new.pt` in the project root. To train your own:

```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
cp runs/detect/train/weights/best.pt best_new.pt
```

### GPU acceleration

Install the CUDA build of PyTorch **before** `pip install -r requirements.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct CUDA version for your GPU driver.

---

## Git LFS

If `best_new.pt` exceeds 100 MB (typical for larger YOLOv8 variants), use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes best_new.pt
git commit -m "Add model via LFS"
git push
```

Run `git lfs pull` after cloning to download the model file.

---

## Security Notes

- **`SECRET_KEY`** — the default fallback in `app.py` is safe for development but must be replaced with a long random string in any deployed environment (`export SECRET_KEY="..."`).
- **Upload limit** — files larger than 2 GB are rejected with a `413` error. Adjust `MAX_CONTENT_LENGTH` in `app.py` if your videos are larger.
- **Accepted formats** — only `mp4`, `mov`, `avi`, `mkv`, `webm` are accepted. Other extensions are rejected before saving.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 5051 already in use | Change `port=5051` at the bottom of `app.py` |
| `ultralytics` install fails | Run `pip install --upgrade pip setuptools wheel` first, then retry |
| Slow on first frame | Normal — YOLOv8 JIT-compiles on first inference |
| Running on CPU only | Install the CUDA PyTorch build (see [GPU acceleration](#gpu-acceleration)) |
| App shows DEMO mode | `best_new.pt` not found in project root — check `MODEL_PATH` in `config.py` |
| Multi-door — one door stalls | Verify each video uploaded successfully; per-door errors appear as toast messages |
| Analytics tab is empty | Complete at least one session so a CSV is written to `logs/` |
| Analytics shows wrong hour | The hour is read from the filename timestamp (`HHMMSS`) — check your system clock |
| Pause button has no effect | Ensure you are running the updated `app.py` with `pause_processing` / `resume_processing` socket handlers |
| "Disconnected" banner appears | The server restarted or the network dropped — the client reconnects automatically |
| Upload rejected | File must be MP4/MOV/AVI/MKV/WEBM and under 2 GB |
