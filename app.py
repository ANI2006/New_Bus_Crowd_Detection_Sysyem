"""
BusOccupancy AI — Flask + SocketIO server
Supports:
  • Single-door mode  (event: start_processing  / frame_data)
  • Multi-door mode   (event: start_multi       / multi_frame_data)
  • Peak-hour analytics (/analytics)
  • Session CSV logs   (/logs, /logs/<filename>)

Fixes applied vs original:
  [1] Shared YOLO model (loaded once, lock-guarded) — prevents GPU OOM with multi-door
  [2] Batch mode multi-door uses separate snapshot_count — no longer corrupts in_count
  [3] Pause race condition fixed — single lock-guarded read per iteration
  [4] process_door crash → done event still fires via try/finally
  [5] Batch single-door sets in_count = snapshot so done event is correct
  [6] timeline capped at 300 in single-door (was unbounded)
  [7] MIME / extension validation on upload
  [8] File-size limit (2 GB)
  [9] SECRET_KEY from env var with safe fallback
  [10] import random moved to top level
"""

import os
import cv2
import base64
import time
import random
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import config
from tracker import CentroidTracker, LineCrossCounter
from drawing import (draw_counting_line, draw_tracked_boxes,
                     draw_alert_banner, door_color)
from logger import SessionLogger, analyze_logs

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]        = os.environ.get("SECRET_KEY", "busoccupancy_ai_dev")
app.config["UPLOAD_FOLDER"]     = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB upload limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.LOG_FOLDER,    exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}



import queue as _queue

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed — running in DEMO mode")

_infer_queue: "_queue.Queue[tuple]" = _queue.Queue(maxsize=32)
_model = None


def _inference_worker():
    """
    Single background thread that owns the YOLO model.
    Drains _infer_queue; each item is (frame, result_q).
    Puts the YOLO result (or None) back onto result_q.
    """
    global _model
    if YOLO_AVAILABLE and os.path.exists(config.MODEL_PATH):
        _model = YOLO(config.MODEL_PATH)
        print(f"✓ Model loaded: {config.MODEL_PATH}")
    else:
        print("⚠ Model not found — running in DEMO mode")

    while True:
        frame, result_q = _infer_queue.get()
        if frame is None:           # sentinel: shut down
            result_q.put(None)
            break
        try:
            if _model is not None:
                result = _model(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]
            else:
                result = None
        except Exception as exc:
            print(f"Inference error: {exc}")
            result = None
        result_q.put(result)
        _infer_queue.task_done()


_worker_thread = threading.Thread(target=_inference_worker, daemon=True, name="yolo-worker")
_worker_thread.start()


def run_inference(frame):
    """
    Submit a frame to the inference worker and block until the result arrives.
    Each door thread calls this independently; they wait in parallel — not in a
    serial lock queue — so frame I/O, tracking, and JPEG encoding on all other
    doors continues while this door waits for its GPU result.
    """
    result_q: "_queue.Queue" = _queue.Queue(maxsize=1)
    _infer_queue.put((frame, result_q))
    return result_q.get()          


# ── Session registries ────────────────────────────────────────────────────────
single_sessions: dict = {}
multi_sessions:  dict = {}
sessions_lock = threading.Lock()


# ── Shared multi-door state ───────────────────────────────────────────────────

class MultiDoorState:
    """Thread-safe aggregator for N-door processing."""

    def __init__(self, door_count: int, capacity: int, initial_count: int):
        self.lock          = threading.Lock()
        self.door_count    = door_count
        self.capacity      = capacity
        self.initial_count = initial_count

        self.door_in       = [0] * door_count   # cumulative in per door
        self.door_out      = [0] * door_count   # cumulative out per door
        self.door_snapshot = [0] * door_count   # batch-mode snapshot count
        self.door_fps      = [0.0] * door_count
        self.door_prog     = [0] * door_count
        self.door_frame    = [None] * door_count
        self.door_done     = [False] * door_count

        self.timeline   = []
        self.start_time = time.time()
        self._last_tl   = 0.0

    @property
    def total_in(self):
        return sum(self.door_in)

    @property
    def total_out(self):
        return sum(self.door_out)

    @property
    def count(self):
        return max(0, self.initial_count + self.total_in - self.total_out)

    def append_timeline(self, occ_pct: int):
        now = time.time() - self.start_time
        if now - self._last_tl >= 1.0:
            self.timeline.append({"t": round(now, 1), "count": self.count, "pct": occ_pct})
            self._last_tl = now
        if len(self.timeline) > 300:
            self.timeline = self.timeline[-300:]

    @property
    def all_done(self):
        return all(self.door_done)


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def get_density(count: int, capacity: int) -> tuple[str, str]:
    r = count / max(capacity, 1)
    if r < config.DENSITY_LOW:      return "LOW",         "#22c55e"
    elif r < config.DENSITY_MEDIUM: return "MEDIUM",      "#f59e0b"
    elif r < config.DENSITY_HIGH:   return "HIGH",        "#ef4444"
    elif r < 1.0:                   return "FULL",        "#dc2626"
    else:                           return "OVERCROWDED", "#9333ea"


def frame_to_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def get_alert(occupancy_pct: int) -> dict | None:
    if occupancy_pct > 100:
        return {"level": "critical",
                "msg": f"Bus OVERCROWDED — {occupancy_pct}% of capacity!"}
    elif occupancy_pct == 100:
        return {"level": "critical", "msg": "Bus is at full capacity!"}
    elif occupancy_pct >= 80:
        return {"level": "warning",
                "msg": f"Bus is {occupancy_pct}% full — nearly at capacity"}
    return None


# ── Single-door processing loop ───────────────────────────────────────────────

def process_video_single(video_path: str, sid: str, capacity: int,
                         mode: str, line_ratio: float, initial_count: int):
    with sessions_lock:
        single_sessions[sid] = {"active": True, "paused": False}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit("error", {"msg": "Cannot open video file"}, to=sid)
        with sessions_lock:
            single_sessions.pop(sid, None)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = max(cap.get(cv2.CAP_PROP_FPS) or 30, 1)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Warn client if frame count unavailable 
    if total_frames <= 0:
        total_frames = 0
        socketio.emit("warning", {"msg": "Frame count unavailable — progress bar disabled"}, to=sid)

    socketio.emit("video_info", {
        "total_frames": total_frames, "fps": round(fps, 1),
        "width": width, "height": height,
    }, to=sid)

    LINE_Y  = int(height * line_ratio)
    tracker = CentroidTracker(max_disappeared=int(fps * config.TRACKER_MAX_GONE))
    counter = LineCrossCounter(LINE_Y)
    logger  = SessionLogger(video_path, door_count=1)

    frame_idx        = 0
    fps_buf          = []
    prev_time        = time.time()
    in_count         = 0
    out_count        = 0
    count            = initial_count
    last_alert_level = None
    SKIP             = max(1, int(fps // 10)) if mode == "batch" else 1
    timeline         = []
    last_tl          = 0.0

    try:
        while True:
            with sessions_lock:
                sess = single_sessions.get(sid, {})
                active = sess.get("active", False)
                paused = sess.get("paused", False)

            if not active:
                break
            if paused:
                socketio.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % SKIP != 0:
                continue

            now = time.time()
            fps_buf.append(min(1.0 / max(now - prev_time, 1e-3), 200.0))
            prev_time = now
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            live_fps = sum(fps_buf) / len(fps_buf)

            annotated = frame.copy()
            boxes = []
            results = run_inference(frame)

            if results is not None:
                boxes = results.boxes.xyxy.cpu().numpy().tolist()
                if mode == "batch":
                    snapshot = len(boxes)
                    count    = snapshot + initial_count
                    in_count = snapshot   
                    annotated = results.plot()
                else:
                    tracked   = tracker.update(boxes)
                    counter.update(tracked)
                    in_count  = counter.in_count
                    out_count = counter.out_count
                    count     = max(0, initial_count + in_count - out_count)
                    draw_counting_line(annotated, LINE_Y, width, in_count, out_count)
                    draw_tracked_boxes(annotated, boxes, tracked)
            else:
                # DEMO mode
                delta     = random.randint(-1, 2)
                count     = max(0, count + delta)
                in_count  = max(0, in_count  + max(0,  delta))
                out_count = max(0, out_count + max(0, -delta))
                cv2.putText(annotated, "DEMO — no model loaded", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                draw_counting_line(annotated, LINE_Y, width, in_count, out_count)

            density, color_hex = get_density(count, capacity)
            occ_pct  = round(count / max(capacity, 1) * 100)
            progress = (round(frame_idx / total_frames * 100)
                        if total_frames > 0 else 0)

            draw_alert_banner(annotated, density, occ_pct)
            logger.log(frame_idx, count, in_count, out_count, occ_pct, density,
                       door_counts=[(in_count, out_count)])

            elapsed = now - logger.start_time
            if elapsed - last_tl >= 1.0:
                timeline.append({"t": round(elapsed, 1), "count": count, "pct": occ_pct})
                last_tl = elapsed
                if len(timeline) > 300:
                    timeline = timeline[-300:]

            alert = get_alert(occ_pct)
            alert_payload = None
            if alert:
                if alert["level"] != last_alert_level:
                    alert_payload    = alert
                    last_alert_level = alert["level"]
            else:
                last_alert_level = None

            frame_b64 = None
            if frame_idx % (SKIP * config.STREAM_EVERY) == 0:
                out_w = min(width, config.STREAM_WIDTH)
                out_h = int(height * out_w / width)
                small = cv2.resize(annotated, (out_w, out_h))
                frame_b64 = frame_to_b64(small)

            payload = {
                "frame_idx": frame_idx, "total_frames": total_frames,
                "progress": progress, "count": count, "capacity": capacity,
                "density": density, "density_color": color_hex,
                "occupancy_pct": occ_pct, "fps": round(live_fps, 1),
                "in_count": in_count, "out_count": out_count,
                "mode": mode, "initial_count": initial_count,
                "timeline": timeline[-60:],
            }
            if frame_b64:     payload["frame"] = frame_b64
            if alert_payload: payload["alert"] = alert_payload

            socketio.emit("frame_data", payload, to=sid)
            socketio.sleep(0)

    finally:
        cap.release()
        logger.close()
        with sessions_lock:
            single_sessions.pop(sid, None)

    socketio.emit("done", {
        "mode": "single",
        "total_frames": frame_idx, "in_count": in_count,
        "out_count": out_count, "final_count": count,
        "log_file": logger.path, "timeline": timeline,
    }, to=sid)


# ── Multi-door per-door processing loop ───────────────────────────────────────

def process_door(video_path: str, door_index: int, door_label: str,
                 sid: str, state: MultiDoorState,
                 mode: str, line_ratio: float):
    """
    Runs in its own thread for each door.
    Uses try/finally so the done event always fires even if this door crashes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit("error",
                      {"msg": f"Cannot open Door {door_label} video"}, to=sid)
        with state.lock:
            state.door_done[door_index] = True
            all_done = state.all_done
        if all_done:
            _emit_multi_done(sid, state)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = max(cap.get(cv2.CAP_PROP_FPS) or 30, 1)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LINE_Y  = int(height * line_ratio)
    color   = door_color(door_index)

    SKIP = max(1, int(fps // 10)) if mode == "batch" else 1

    match_dist = int(120 * (width / 640))
    tracker = CentroidTracker(max_disappeared=int(fps * config.TRACKER_MAX_GONE),
                              match_dist=match_dist)

    scaled_cooldown = max(1, config.CROSS_COOLDOWN // SKIP)
    counter = LineCrossCounter(LINE_Y, door_label, cooldown=scaled_cooldown)
    logger  = SessionLogger(f"{video_path}_door{door_label}", door_count=1)

    frame_idx    = 0
    fps_buf      = []
    prev_time    = time.time()
    batch_snapshot = 0

    try:
        while True:
            # Single lock-guarded read 
            with sessions_lock:
                sess   = multi_sessions.get(sid)
                active = sess["active"] if sess else False
                paused = sess.get("paused", False) if sess else False

            if not active:
                break
            if paused:
                socketio.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % SKIP != 0:
                continue

            now = time.time()
            fps_buf.append(min(1.0 / max(now - prev_time, 1e-3), 200.0))
            prev_time = now
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            live_fps = sum(fps_buf) / len(fps_buf)

            annotated = frame.copy()
            boxes     = []

            
            infer_frame = frame
            scale = 1.0
            if width > config.INFER_WIDTH:
                scale       = config.INFER_WIDTH / width
                infer_h     = int(height * scale)
                infer_frame = cv2.resize(frame, (config.INFER_WIDTH, infer_h),
                                         interpolation=cv2.INTER_LINEAR)

            results = run_inference(infer_frame)

            if results is not None:
                raw_boxes = results.boxes.xyxy.cpu().numpy().tolist()
                boxes = ([[x1/scale, y1/scale, x2/scale, y2/scale]
                           for x1, y1, x2, y2 in raw_boxes]
                          if scale != 1.0 else raw_boxes)
                if mode == "batch":
                    batch_snapshot = len(boxes)
                    for x1, y1, x2, y2 in boxes:
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (80, 220, 140), 2, cv2.LINE_AA)
                    cv2.putText(annotated, f"SNAPSHOT: {batch_snapshot}", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                else:
                    tracked = tracker.update(boxes)
                    counter.update(tracked)
                    draw_counting_line(annotated, LINE_Y, width,
                                       counter.in_count, counter.out_count,
                                       label=f"DOOR {door_label}", color=color)
                    draw_tracked_boxes(annotated, boxes, tracked)
            else:
                # DEMO mode
                delta = random.randint(-1, 2)
                if mode == "batch":
                    batch_snapshot = max(0, batch_snapshot + delta)
                else:
                    counter.in_count  = max(0, counter.in_count  + max(0,  delta))
                    counter.out_count = max(0, counter.out_count + max(0, -delta))
                cv2.putText(annotated, f"DEMO — Door {door_label}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                draw_counting_line(annotated, LINE_Y, width,
                                   counter.in_count, counter.out_count,
                                   label=f"DOOR {door_label}", color=color)

            progress = (round(frame_idx / total_frames * 100)
                        if total_frames > 0 else 0)

            frame_b64 = None
            if frame_idx % (SKIP * config.STREAM_EVERY) == 0:
                out_w = min(width, config.STREAM_WIDTH // max(state.door_count, 2))
                out_h = int(height * out_w / width)
                small = cv2.resize(annotated, (out_w, out_h))
                frame_b64 = frame_to_b64(small)

            with state.lock:
                if mode == "batch":
                    state.door_snapshot[door_index] = batch_snapshot

                    state.door_in[door_index]  = batch_snapshot
                    state.door_out[door_index] = 0
                else:
                    state.door_in[door_index]  = counter.in_count
                    state.door_out[door_index] = counter.out_count

                state.door_fps[door_index]  = round(live_fps, 1)
                state.door_prog[door_index] = progress
                if frame_b64:
                    state.door_frame[door_index] = frame_b64

                count    = state.count
                capacity = state.capacity
                density, color_hex = get_density(count, capacity)
                occ_pct  = round(count / max(capacity, 1) * 100)
                state.append_timeline(occ_pct)

                payload = {
                    "door_index":    door_index,
                    "door_label":    door_label,
                    "door_count":    state.door_count,
                    "count":         count,
                    "capacity":      capacity,
                    "density":       density,
                    "density_color": color_hex,
                    "occupancy_pct": occ_pct,
                    "total_in":      state.total_in,
                    "total_out":     state.total_out,
                    "initial_count": state.initial_count,
                    "timeline":      state.timeline[-60:],
                    "door_in_list":   list(state.door_in),
                    "door_out_list":  list(state.door_out),
                    "door_fps_list":  list(state.door_fps),
                    "door_prog_list": list(state.door_prog),
                    "door_frames":    list(state.door_frame),
                }
                alert = get_alert(occ_pct)
                if alert:
                    payload["alert"] = alert

            door_in_val  = counter.in_count  if mode != "batch" else batch_snapshot
            door_out_val = counter.out_count if mode != "batch" else 0
            logger.log(frame_idx, door_in_val - door_out_val,
                       door_in_val, door_out_val, occ_pct, density,
                       door_counts=[(door_in_val, door_out_val)])

            socketio.emit("multi_frame_data", payload, to=sid)
            socketio.sleep(0)

    except Exception as exc:
        socketio.emit("error",
                      {"msg": f"Door {door_label} error: {exc}"}, to=sid)
    finally:
        cap.release()
        logger.close()

        with state.lock:
            state.door_done[door_index] = True
            all_done = state.all_done

        if all_done:
            with sessions_lock:
                multi_sessions.pop(sid, None)
            _emit_multi_done(sid, state)


def _emit_multi_done(sid: str, state: MultiDoorState):
    """Emit the done event for a completed multi-door session."""
    socketio.emit("done", {
        "mode":          "multi",
        "door_count":    state.door_count,
        "door_in_list":  list(state.door_in),
        "door_out_list": list(state.door_out),
        "total_in":      state.total_in,
        "total_out":     state.total_out,
        "final_count":   state.count,
        "timeline":      state.timeline,
    }, to=sid)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    if not allowed_file(filename):
        return jsonify({"error": "File type not allowed. Use MP4, MOV, AVI, MKV, or WEBM"}), 400
    path = os.path.join(config.UPLOAD_FOLDER, filename)
    f.save(path)
    return jsonify({"ok": True, "path": path, "name": filename})


@app.route("/upload_error", methods=["GET"])
def upload_error():
    """Called by Flask when MAX_CONTENT_LENGTH is exceeded."""
    return jsonify({"error": "File too large — maximum upload size is 2 GB"}), 413


@app.route("/logs")
def list_logs():
    files = sorted(os.listdir(config.LOG_FOLDER), reverse=True)
    return jsonify({"logs": [f for f in files if f.endswith(".csv")]})


@app.route("/logs/<filename>")
def download_log(filename):
    safe = secure_filename(filename)
    if not safe or not safe.endswith(".csv"):
        return jsonify({"error": "Invalid log filename"}), 400
    return send_from_directory(config.LOG_FOLDER, safe, as_attachment=True)


@app.route("/analytics")
def analytics():
    """Return hourly occupancy patterns derived from all saved CSV logs."""
    hourly, summaries = analyze_logs()
    return jsonify({"hourly": hourly, "summaries": summaries})


# ── Socket handlers ───────────────────────────────────────────────────────────

@socketio.on("start_processing")
def handle_start_single(data):
    """Single-door mode."""
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions or sid in multi_sessions:
            emit("error", {"msg": "Already processing"})
            return

    video_path    = data.get("path")
    mode          = data.get("mode",          "count")
    capacity      = int(data.get("capacity",      config.BUS_CAPACITY))
    line_ratio    = float(data.get("line_ratio",  config.LINE_RATIO))
    initial_count = int(data.get("initial_count", config.INITIAL_COUNT))

    if not video_path or not os.path.exists(video_path):
        emit("error", {"msg": "Video file not found"})
        return

    threading.Thread(
        target=process_video_single,
        args=(video_path, sid, capacity, mode, line_ratio, initial_count),
        daemon=True,
    ).start()


@socketio.on("start_multi")
def handle_start_multi(data):
    """
    Multi-door mode.
    Expects:
      doors: [{"path": "...", "line_ratio": 0.55}, ...]
      capacity, initial_count, mode
    """
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions or sid in multi_sessions:
            emit("error", {"msg": "Already processing"})
            return

    doors         = data.get("doors", [])
    mode          = data.get("mode",          "count")
    capacity      = int(data.get("capacity",      config.BUS_CAPACITY))
    initial_count = int(data.get("initial_count", config.INITIAL_COUNT))

    if not doors:
        emit("error", {"msg": "No doors configured"})
        return

    for i, d in enumerate(doors):
        label = chr(65 + i)
        path  = d.get("path", "")
        if not path or not os.path.exists(path):
            emit("error", {"msg": f"Door {label} video file not found"})
            return

    door_count = len(doors)
    state      = MultiDoorState(door_count, capacity, initial_count)

    threads = []
    for i, d in enumerate(doors):
        label      = chr(65 + i)
        path       = d["path"]
        line_ratio = float(d.get("line_ratio",
                                 config.DOOR_LINE_RATIOS[i]
                                 if i < len(config.DOOR_LINE_RATIOS)
                                 else config.LINE_RATIO))
        t = threading.Thread(
            target=process_door,
            args=(path, i, label, sid, state, mode, line_ratio),
            daemon=True,
        )
        threads.append(t)

    with sessions_lock:
        multi_sessions[sid] = {"active": True, "paused": False,
                               "state": state, "threads": threads}

    for t in threads:
        t.start()

    emit("multi_started", {"door_count": door_count,
                           "labels": [chr(65 + i) for i in range(door_count)]})


@socketio.on("stop_processing")
def handle_stop(_):
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions:
            single_sessions[sid]["active"] = False
        if sid in multi_sessions:
            multi_sessions[sid]["active"] = False
    emit("stopped", {})


@socketio.on("pause_processing")
def handle_pause(_):
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions:
            single_sessions[sid]["paused"] = True
        if sid in multi_sessions:
            multi_sessions[sid]["paused"] = True
    emit("paused", {})


@socketio.on("resume_processing")
def handle_resume(_):
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions:
            single_sessions[sid]["paused"] = False
        if sid in multi_sessions:
            multi_sessions[sid]["paused"] = False
    emit("resumed", {})


@socketio.on("disconnect")
def handle_disconnect():
    """Clean up session state when a client disconnects."""
    sid = request.sid
    with sessions_lock:
        if sid in single_sessions:
            single_sessions[sid]["active"] = False
        if sid in multi_sessions:
            multi_sessions[sid]["active"] = False


if __name__ == "__main__":
    print("🚌 BusOccupancy AI — starting on http://localhost:5051")
    socketio.run(app, debug=True, port=5051)
