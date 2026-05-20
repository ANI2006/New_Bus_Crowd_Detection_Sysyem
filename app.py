import os, cv2, base64, time, threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

import config
from tracker import CentroidTracker, LineCrossCounter
from drawing import draw_counting_line, draw_tracked_boxes, draw_alert_banner, C_CYAN, C_YELLOW
from logger  import SessionLogger, analyze_logs

app = Flask(__name__)
app.config["SECRET_KEY"]    = "busoccupancy_ai_2026"
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.LOG_FOLDER,    exist_ok=True)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed — running in DEMO mode")


# Shared dual-door state  

class DualDoorState:
    def __init__(self, capacity, initial_count):
        self.lock          = threading.Lock()
        self.capacity      = capacity
        self.initial_count = initial_count

        # per-door counts
        self.door_a_in  = self.door_a_out  = 0
        self.door_b_in  = self.door_b_out  = 0

        # per-door processing status
        self.door_a_done = self.door_b_done = False
        self.door_a_frame = self.door_b_frame = None
        self.door_a_fps   = self.door_b_fps   = 0.0
        self.door_a_progress = self.door_b_progress = 0

        self.timeline   = []
        self.start_time = time.time()
        self._last_tl   = 0

    @property
    def in_count(self):
        return self.door_a_in + self.door_b_in

    @property
    def out_count(self):
        return self.door_a_out + self.door_b_out

    @property
    def count(self):
        return max(0, self.initial_count + self.in_count - self.out_count)

    def append_timeline(self, occ_pct):
        now = time.time() - self.start_time
        if now - self._last_tl >= 1.0:
            self.timeline.append({"t": round(now, 1), "count": self.count, "pct": occ_pct})
            self._last_tl = now
        if len(self.timeline) > 300:
            self.timeline = self.timeline[-300:]


sessions = {}
sessions_lock = threading.Lock()

single_sessions = {}


Helpers 

def get_density(count, capacity):
    r = count / max(capacity, 1)
    if r < config.DENSITY_LOW:      return "LOW",    "#22c55e"
    elif r < config.DENSITY_MEDIUM: return "MEDIUM", "#f59e0b"
    elif r < config.DENSITY_HIGH:   return "HIGH",   "#ef4444"
    else:                           return "FULL",   "#dc2626"


def frame_to_b64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def get_alert(occupancy_pct):
    if occupancy_pct >= 100:
        return {"level": "critical", "msg": "Bus is at full capacity!"}
    elif occupancy_pct >= 80:
        return {"level": "warning",  "msg": f"Bus is {occupancy_pct}% full — nearly at capacity"}
    return None


# Single-video processing loop 

def process_video_single(video_path, sid, capacity, mode, line_ratio,
                          initial_count):
    with sessions_lock:
        single_sessions[sid] = {"active": True}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit("error", {"msg": "Cannot open video file"}, to=sid)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    socketio.emit("video_info", {
        "total_frames": total_frames, "fps": round(fps, 1),
        "width": width, "height": height,
    }, to=sid)

    LINE_Y  = int(height * line_ratio)
    model   = YOLO(config.MODEL_PATH) if (YOLO_AVAILABLE and os.path.exists(config.MODEL_PATH)) else None
    tracker = CentroidTracker(max_disappeared=int(fps * config.TRACKER_MAX_GONE))
    counter = LineCrossCounter(LINE_Y)
    logger  = SessionLogger(video_path)

    frame_idx = 0
    fps_buf   = []
    prev_time = time.time()
    in_count  = out_count = 0
    count     = initial_count
    last_alert_level = None
    SKIP         = max(1, int(fps // 10)) if mode == "batch" else 1
    timeline     = []
    last_tl_time = 0

    while single_sessions.get(sid, {}).get("active", False):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP != 0:
            continue

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        if len(fps_buf) > 30: fps_buf.pop(0)
        live_fps = sum(fps_buf) / len(fps_buf)

        annotated = frame.copy()
        boxes = []

        if model:
            results = model(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy().tolist()
            if mode == "batch":
                count     = len(boxes) + initial_count
                annotated = results.plot()
            else:
                tracked  = tracker.update(boxes)
                counter.update(tracked)
                in_count  = counter.in_count
                out_count = counter.out_count
                count     = max(0, initial_count + in_count - out_count)
                draw_counting_line(annotated, LINE_Y, width, in_count, out_count)
                draw_tracked_boxes(annotated, boxes, tracked)
        else:
            import random
            delta     = random.randint(-1, 2)
            count     = max(0, count + delta)
            in_count  = max(0, in_count + max(0, delta))
            out_count = max(0, out_count + max(0, -delta))
            cv2.putText(annotated, "DEMO — no model loaded", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            draw_counting_line(annotated, LINE_Y, width, in_count, out_count)

        density, color_hex = get_density(count, capacity)
        occ_pct  = round(min(count / max(capacity, 1), 1.0) * 100)
        progress = round(frame_idx / max(total_frames, 1) * 100)

        draw_alert_banner(annotated, density, occ_pct)
        logger.log(frame_idx, count, in_count, out_count, occ_pct, density)

        elapsed = now - logger.start_time
        if elapsed - last_tl_time >= 1.0:
            timeline.append({"t": round(elapsed, 1), "count": count, "pct": occ_pct})
            last_tl_time = elapsed

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
            "mode": mode, "dual_door": False,
            "initial_count": initial_count, "timeline": timeline[-60:],
        }
        if frame_b64:    payload["frame"]  = frame_b64
        if alert_payload: payload["alert"] = alert_payload

        socketio.emit("frame_data", payload, to=sid)
        socketio.sleep(0)

    cap.release()
    logger.close()
    with sessions_lock:
        single_sessions.pop(sid, None)

    socketio.emit("done", {
        "total_frames": frame_idx, "in_count": in_count,
        "out_count": out_count, "final_count": count,
        "log_file": logger.path, "timeline": timeline,
    }, to=sid)


# Dual-video per-door processing loop 

def process_door(video_path, door_label, sid, state: DualDoorState,
                 mode, line_ratio):
    """Runs in its own thread. Updates shared DualDoorState."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        socketio.emit("error", {"msg": f"Cannot open Door {door_label} video"}, to=sid)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LINE_Y  = int(height * line_ratio)
    color   = C_CYAN if door_label == "A" else C_YELLOW
    model   = YOLO(config.MODEL_PATH) if (YOLO_AVAILABLE and os.path.exists(config.MODEL_PATH)) else None
    tracker = CentroidTracker(max_disappeared=int(fps * config.TRACKER_MAX_GONE))
    counter = LineCrossCounter(LINE_Y, door_label)
    logger  = SessionLogger(f"{video_path}_door{door_label}")

    frame_idx = 0
    fps_buf   = []
    prev_time = time.time()
    SKIP      = max(1, int(fps // 10)) if mode == "batch" else 1

    while True:
        with sessions_lock:
            sess = sessions.get(sid)
            if sess is None or not sess["active"]:
                break

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP != 0:
            continue

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        if len(fps_buf) > 30: fps_buf.pop(0)
        live_fps = sum(fps_buf) / len(fps_buf)

        annotated = frame.copy()
        boxes = []

        if model:
            results = model(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy().tolist()
            tracked = tracker.update(boxes)
            counter.update(tracked)
            draw_counting_line(annotated, LINE_Y, width,
                               counter.in_count, counter.out_count,
                               label=f"DOOR {door_label}", color=color)
            draw_tracked_boxes(annotated, boxes, tracked)
        else:
            import random
            delta = random.randint(-1, 2)
            counter.in_count  = max(0, counter.in_count  + max(0,  delta))
            counter.out_count = max(0, counter.out_count + max(0, -delta))
            cv2.putText(annotated, f"DEMO Door {door_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            draw_counting_line(annotated, LINE_Y, width,
                               counter.in_count, counter.out_count,
                               label=f"DOOR {door_label}", color=color)

        frame_b64 = None
        if frame_idx % (SKIP * config.STREAM_EVERY) == 0:
            out_w = min(width, config.STREAM_WIDTH // 2)   
            out_h = int(height * out_w / width)
            small = cv2.resize(annotated, (out_w, out_h))
            frame_b64 = frame_to_b64(small)

        progress = round(frame_idx / max(total_frames, 1) * 100)

        with state.lock:
            if door_label == "A":
                state.door_a_in   = counter.in_count
                state.door_a_out  = counter.out_count
                state.door_a_fps  = round(live_fps, 1)
                state.door_a_progress = progress
                if frame_b64: state.door_a_frame = frame_b64
            else:
                state.door_b_in   = counter.in_count
                state.door_b_out  = counter.out_count
                state.door_b_fps  = round(live_fps, 1)
                state.door_b_progress = progress
                if frame_b64: state.door_b_frame = frame_b64

            count    = state.count
            capacity = state.capacity
            density, color_hex = get_density(count, capacity)
            occ_pct  = round(min(count / max(capacity, 1), 1.0) * 100)
            state.append_timeline(occ_pct)

            payload = {
                "dual_door":     True,
                "door_label":    door_label,
                "progress_a":    state.door_a_progress,
                "progress_b":    state.door_b_progress,
                "count":         count,
                "capacity":      capacity,
                "density":       density,
                "density_color": color_hex,
                "occupancy_pct": occ_pct,
                "fps_a":         state.door_a_fps,
                "fps_b":         state.door_b_fps,
                "door_a_in":     state.door_a_in,
                "door_a_out":    state.door_a_out,
                "door_b_in":     state.door_b_in,
                "door_b_out":    state.door_b_out,
                "in_count":      state.in_count,
                "out_count":     state.out_count,
                "initial_count": state.initial_count,
                "timeline":      state.timeline[-60:],
                "frame_a":       state.door_a_frame,
                "frame_b":       state.door_b_frame,
            }
            alert = get_alert(occ_pct)
            if alert: payload["alert"] = alert

        logger.log(frame_idx, count, counter.in_count, counter.out_count,
                   occ_pct, density)
        socketio.emit("dual_frame_data", payload, to=sid)
        socketio.sleep(0)

    cap.release()
    logger.close()

    with state.lock:
        if door_label == "A":
            state.door_a_done = True
        else:
            state.door_b_done = True
        both_done = state.door_a_done and state.door_b_done

    if both_done:
        with sessions_lock:
            sessions.pop(sid, None)
        socketio.emit("done", {
            "dual_door":   True,
            "door_a_in":   state.door_a_in,  "door_a_out": state.door_a_out,
            "door_b_in":   state.door_b_in,  "door_b_out": state.door_b_out,
            "in_count":    state.in_count,   "out_count":  state.out_count,
            "final_count": state.count,
            "timeline":    state.timeline,
        }, to=sid)



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
    path = os.path.join(config.UPLOAD_FOLDER, f.filename)
    f.save(path)
    return jsonify({"ok": True, "path": path, "name": f.filename})


@app.route("/logs")
def list_logs():
    files = sorted(os.listdir(config.LOG_FOLDER), reverse=True)
    csv_files = [f for f in files if f.endswith(".csv")]
    return jsonify({"logs": csv_files})


@app.route("/logs/<filename>")
def download_log(filename):
    return send_from_directory(config.LOG_FOLDER, filename, as_attachment=True)


@app.route("/analytics")
def analytics():
    hourly, summaries = analyze_logs()
    return jsonify({"hourly": hourly, "summaries": summaries})



@socketio.on("start_processing")
def handle_start(data):
    sid = request.sid

    with sessions_lock:
        if sid in sessions or sid in single_sessions:
            emit("error", {"msg": "Already processing"})
            return

    video_path    = data.get("path")
    mode          = data.get("mode", "live")
    capacity      = int(data.get("capacity",      config.BUS_CAPACITY))
    line_ratio    = float(data.get("line_ratio",  config.LINE_RATIO))
    initial_count = int(data.get("initial_count", config.INITIAL_COUNT))

    if not video_path or not os.path.exists(video_path):
        emit("error", {"msg": "Video file not found"})
        return

    t = threading.Thread(
        target=process_video_single,
        args=(video_path, sid, capacity, mode, line_ratio, initial_count),
        daemon=True
    )
    t.start()


@socketio.on("start_dual_processing")
def handle_start_dual(data):
    sid = request.sid

    with sessions_lock:
        if sid in sessions or sid in single_sessions:
            emit("error", {"msg": "Already processing"})
            return

    path_a        = data.get("path_a")
    path_b        = data.get("path_b")
    mode          = data.get("mode",          "live")
    capacity      = int(data.get("capacity",      config.BUS_CAPACITY))
    line_ratio_a  = float(data.get("line_ratio_a", config.LINE_RATIO))
    line_ratio_b  = float(data.get("line_ratio_b", config.LINE_RATIO_B))
    initial_count = int(data.get("initial_count",  config.INITIAL_COUNT))

    for label, path in [("A", path_a), ("B", path_b)]:
        if not path or not os.path.exists(path):
            emit("error", {"msg": f"Door {label} video file not found"})
            return

    state = DualDoorState(capacity, initial_count)

    t_a = threading.Thread(
        target=process_door,
        args=(path_a, "A", sid, state, mode, line_ratio_a),
        daemon=True
    )
    t_b = threading.Thread(
        target=process_door,
        args=(path_b, "B", sid, state, mode, line_ratio_b),
        daemon=True
    )

    with sessions_lock:
        sessions[sid] = {"active": True, "state": state, "threads": [t_a, t_b]}

    t_a.start()
    t_b.start()


@socketio.on("stop_processing")
def handle_stop(_):
    sid = request.sid
    with sessions_lock:
        if sid in sessions:
            sessions[sid]["active"] = False
        if sid in single_sessions:
            single_sessions[sid]["active"] = False
    emit("stopped", {})


if __name__ == "__main__":
    print("🚌 BusOccupancy AI — starting on http://localhost:5051")
    socketio.run(app, debug=True, port=5051)
