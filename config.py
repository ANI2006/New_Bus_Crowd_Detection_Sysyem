# ── BusOccupancy AI — Configuration ──────────────────────────────────────────

MODEL_PATH    = "best_new.pt"
UPLOAD_FOLDER = "uploads"
LOG_FOLDER    = "logs"

# Bus settings
BUS_CAPACITY    = 60
LINE_RATIO      = 0.55   # counting line position (0.0 = top, 1.0 = bottom)
INITIAL_COUNT   = 0      # passengers already on board at video start

# Dual-door support
DUAL_DOOR       = False   # enable second counting line (door B)
LINE_RATIO_B    = 0.35    # second door line position

# Detection
CONF_THRESHOLD   = 0.25
TRACKER_MAX_GONE = 2.0   # seconds before dropping a track
CROSS_COOLDOWN   = 20    # frames cooldown after a line cross

# Density thresholds
DENSITY_LOW    = 0.4
DENSITY_MEDIUM = 0.7
DENSITY_HIGH   = 0.9

# Alert thresholds
ALERT_WARNING  = 0.8    # 80%  → warning alert
ALERT_CRITICAL = 1.0    # 100% → critical alert

# Streaming
STREAM_EVERY = 2        
STREAM_WIDTH = 960      
JPEG_QUALITY = 75
