# BusOccupancy AI

A real-time passenger monitoring system built with YOLOv8 and OpenCV. Detects and counts passengers boarding and exiting buses using a custom-trained model, with a live web dashboard, dual-door support, and historical peak-hour analytics derived from session logs.

## Requirements

- Python 3.9+
- A trained YOLOv8 model file (`best_new.pt`) placed in the project root
- Git LFS if you store the model in the repository (see below)

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd BusOccupancy_AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your trained model in the project root
cp /path/to/best_new.pt .

# 4. Run
python app.py
```

Open **http://localhost:5051** in your browser.

If `best_new.pt` is tracked with Git LFS, run `git lfs pull` after cloning to download it.

## Features

- Real-time video processing with YOLOv8 detection
- Centroid-based person tracking (no external tracker library needed)
- Entry/exit counting via a configurable virtual counting line
- **Dual-door mode** — two counting lines (Door A + Door B) tracked independently and combined into one total
- **Starting passenger count** — set how many people are already on board before the video begins; the count never goes negative
- Live occupancy chart updated as the video plays
- **Peak hour analytics** — automatically analyses all saved CSV logs to show which hours of day the bus is historically fullest
- Capacity alerts at 80% (warning) and 100% (critical)
- CSV session logging — every run is saved and downloadable from the dashboard
- Pause/resume processing
- Three processing modes: Live, Count, Batch

## Project Structure

```
BusOccupancy_AI/
├── app.py              ← Flask + SocketIO server
├── tracker.py          ← CentroidTracker, LineCrossCounter, DualLineCrossCounter
├── drawing.py          ← OpenCV annotation helpers
├── logger.py           ← CSV session logger + log analysis
├── config.py           ← All settings in one place
├── requirements.txt
├── README.md
├── best_new.pt         ← Trained YOLOv8 model (use Git LFS if >100 MB)
├── templates/
│   └── index.html      ← Web dashboard
├── uploads/            ← Uploaded videos at runtime (git-ignored, folder kept)
│   └── .gitkeep
└── logs/               ← Session CSV logs at runtime (git-ignored, folder kept)
    └── .gitkeep
```

## Processing Modes

| Mode  | Description |
|-------|-------------|
| LIVE  | Every frame processed and streamed with detection overlay |
| COUNT | Entry/exit line crossing — most accurate for boarding counts |
| BATCH | Skips frames for speed — best for long recordings |

## Dual-Door Mode

Enable the **Dual-door mode** toggle in the dashboard settings panel. A second counting line (Door B, shown in amber) appears independently of Door A (cyan). Both doors count IN/OUT separately; the passenger total combines them. Useful for buses with a front door (boarding) and a rear door (alighting).

## Starting Passenger Count

If your video starts mid-route with passengers already on board, set **Starting passengers** in the settings panel before pressing START. The live count will begin at that offset and will never drop below zero even if the video shows only exits.

## Peak Hour Analytics

Every processed session is saved as a timestamped CSV in `logs/`. Open the **PEAK HOUR ANALYTICS** tab in the dashboard to see a 24-hour bar chart of average and peak occupancy by hour of day. The more sessions you run at different times, the more accurate the predictions become. Bars are colour-coded: green (low) → blue (moderate) → amber (busy) → red (near capacity).

## Configuration

Edit `config.py` to change defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `BUS_CAPACITY` | `60` | Maximum passenger capacity |
| `LINE_RATIO` | `0.55` | Door A counting line height (0.0 = top, 1.0 = bottom) |
| `LINE_RATIO_B` | `0.35` | Door B counting line height |
| `INITIAL_COUNT` | `0` | Passengers already on board at video start |
| `DUAL_DOOR` | `False` | Enable two counting lines by default |
| `CONF_THRESHOLD` | `0.25` | YOLO detection confidence threshold |
| `TRACKER_MAX_GONE` | `2.0` | Seconds before dropping a lost track |
| `CROSS_COOLDOWN` | `20` | Frames of cooldown after a line crossing |
| `ALERT_WARNING` | `0.8` | Occupancy ratio that triggers a warning alert |
| `ALERT_CRITICAL` | `1.0` | Occupancy ratio that triggers a critical alert |
| `STREAM_WIDTH` | `960` | Max pixel width of frames streamed to browser |
| `JPEG_QUALITY` | `75` | JPEG compression quality for streamed frames |

## Model

The app expects a YOLOv8 model trained to detect people (or passengers specifically) saved as `best_new.pt` in the project root. If no model is found, the app starts in **DEMO mode** with simulated random counts so the dashboard can still be tested.

To train your own model:
```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Copy the resulting `runs/detect/train/weights/best.pt` to the project root and rename it `best_new.pt`.

## Git LFS (for large model files)

If `best_new.pt` exceeds 100 MB (typical for larger YOLOv8 variants), use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add best_new.pt
git commit -m "Add model via LFS"
```

## .gitignore

```
__pycache__/
*.pyc
*.pyo
uploads/*
!uploads/.gitkeep
logs/*
!logs/.gitkeep
.env
```

Only add `*.pt` to `.gitignore` if you are **not** using Git LFS and are hosting the model elsewhere (e.g. Google Drive, S3, Hugging Face Hub).
