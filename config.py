# ── BusOccupancy AI — Configuration ──────────────────────────────────────────


MODEL_PATH    = "best_new.pt"
UPLOAD_FOLDER = "uploads"
LOG_FOLDER    = "logs"

# Bus / vehicle settings
BUS_CAPACITY  = 60
LINE_RATIO    = 0.60   # default counting line position (0.0 = top, 1.0 = bottom)
INITIAL_COUNT = 0      # passengers already on board at video start

DOOR_LINE_RATIOS = [0.55, 0.35, 0.50, 0.45]

# Detection
CONF_THRESHOLD   = 0.25
TRACKER_MAX_GONE = 2.0   # seconds before dropping a track
CROSS_COOLDOWN   = 20    # frames cooldown after a line cross

# Multi-door performance
# INFER_WIDTH: resize frames to this width before YOLO inference.
#   Smaller = faster GPU inference per frame.

INFER_WIDTH = 640

# Density thresholds (as fractions of capacity)
DENSITY_LOW    = 0.4
DENSITY_MEDIUM = 0.7
DENSITY_HIGH   = 0.9

# Alert thresholds
ALERT_WARNING  = 0.8    # 80%  → warning alert
ALERT_CRITICAL = 1.0    # 100% → critical alert

# Streaming
STREAM_EVERY = 2        # send annotated frame to browser every N processed frames
STREAM_WIDTH = 960      # max pixel width for streamed frames
JPEG_QUALITY = 75
