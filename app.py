import os, cv2, base64, time, threading, math
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "busmonitor2024"
app.config["UPLOAD_FOLDER"] = "uploads"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

os.makedirs("uploads", exist_ok=True)

BUS_CAPACITY = 60
processing_active = False
processing_thread = None
LINE_RATIO = 0.55   

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed — running in DEMO mode")

MODEL_PATH = "best_new.pt"



class CentroidTracker:
    """Assigns persistent IDs to detections by nearest-centroid matching."""
    def __init__(self, max_disappeared=30):
        self.next_id         = 0
        self.objects         = {}   # id -> (cx, cy)
        self.disappeared     = {}   # id -> frames missing
        self.max_disappeared = max_disappeared

    def _register(self, cx, cy):
        self.objects[self.next_id]    = (cx, cy)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, boxes):
        """boxes: list of [x1,y1,x2,y2]. Returns dict {id: (cx,cy)}."""
        if not boxes:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for x1,y1,x2,y2 in boxes]

        if not self.objects:
            for c in centroids:
                self._register(*c)
        else:
            oids       = list(self.objects.keys())
            ocentroids = list(self.objects.values())
            used_rows, used_cols = set(), set()
            pairs = sorted(
                [(math.hypot(oc[0]-nc[0], oc[1]-nc[1]), r, c)
                 for r, oc in enumerate(ocentroids)
                 for c, nc in enumerate(centroids)]
            )
            for d, r, c in pairs:
                if r in used_rows or c in used_cols:
                    continue
                if d > 120:
                    break
                oid = oids[r]
                self.objects[oid]    = centroids[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for r, oid in enumerate(oids):
                if r not in used_rows:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self._deregister(oid)

            for c in range(len(centroids)):
                if c not in used_cols:
                    self._register(*centroids[c])

        return dict(self.objects)


class LineCrossCounter:
    """Counts upward/downward crossings of a horizontal line."""
    def __init__(self, line_y):
        self.line_y    = line_y
        self.prev_cy   = {}   
        self.in_count  = 0
        self.out_count = 0
        self.cooldown  = {}   

    def update(self, tracked):
        for oid, (cx, cy) in tracked.items():
            if oid not in self.cooldown:
                self.cooldown[oid] = 0
            if self.cooldown[oid] > 0:
                self.cooldown[oid] -= 1

            if oid in self.prev_cy and self.cooldown[oid] == 0:
                prev = self.prev_cy[oid]
                if prev < self.line_y <= cy:        
                    self.in_count += 1
                    self.cooldown[oid] = 20
                elif prev > self.line_y >= cy:
                    self.out_count += 1
                    self.cooldown[oid] = 20

            self.prev_cy[oid] = cy

        for oid in list(self.prev_cy):
            if oid not in tracked:
                del self.prev_cy[oid]
                self.cooldown.pop(oid, None)



def get_density(count, capacity):
    r = count / max(capacity, 1)
    if r < 0.4:   return "LOW",    "#22c55e"
    elif r < 0.7: return "MEDIUM", "#f59e0b"
    elif r < 0.9: return "HIGH",   "#ef4444"
    else:         return "FULL",   "#dc2626"


def frame_to_b64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


def draw_counting_line(frame, line_y, width, in_count, out_count):
    cv2.line(frame, (0, line_y), (width, line_y), (20, 80, 80), 12, cv2.LINE_AA)
    x = 0
    while x < width:
        cv2.line(frame, (x, line_y), (min(x+22, width), line_y),
                 (60, 210, 255), 2, cv2.LINE_AA)
        x += 32
        
    cv2.rectangle(frame, (0, line_y - 30), (145, line_y + 26), (10, 10, 10), -1)
    cv2.putText(frame, f"  IN : {in_count}",  (4, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, f"  OUT: {out_count}", (4, line_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 90, 230), 1, cv2.LINE_AA)
    lbl = " COUNTING LINE "
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    px = width - tw - 20
    cv2.rectangle(frame, (px - 4, line_y - th - 10), (px + tw + 4, line_y + 4), (10, 10, 10), -1)
    cv2.putText(frame, lbl, (px, line_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 210, 255), 1, cv2.LINE_AA)


def draw_tracked_boxes(frame, boxes, tracked):
    color = (80, 220, 140)
    for (x1, y1, x2, y2) in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        best_id, best_d = None, 9999
        for oid, (ox, oy) in tracked.items():
            d = math.hypot(cx - ox, cy - oy)
            if d < best_d:
                best_d, best_id = d, oid

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (60, 210, 255), -1, cv2.LINE_AA)
        if best_id is not None and best_d < 120:
            lbl = f"#{best_id}"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - lh - 8),
                          (int(x1) + lw + 6, int(y1)), (10, 10, 10), -1)
            cv2.putText(frame, lbl, (int(x1) + 3, int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)



def process_video_live(video_path, sid, capacity, mode, line_ratio):
    global processing_active

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
        "width": width, "height": height
    }, to=sid)

    LINE_Y  = int(height * line_ratio)
    model   = YOLO(MODEL_PATH) if (YOLO_AVAILABLE and os.path.exists(MODEL_PATH)) else None
    tracker = CentroidTracker(max_disappeared=int(fps * 2))
    counter = LineCrossCounter(LINE_Y)

    frame_idx = 0
    fps_buf   = []
    prev_time = time.time()
    count = in_count = out_count = 0
    SKIP         = max(1, int(fps // 10)) if mode == "batch" else 1
    STREAM_EVERY = 2

    while processing_active:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP != 0:
            continue

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        live_fps = sum(fps_buf) / len(fps_buf)

        annotated = frame.copy()
        boxes     = []

        if model:
            results = model(frame, conf=0.25, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy().tolist()

            if mode == "batch":
                count     = len(boxes)
                annotated = results.plot()
            else:
                tracked   = tracker.update(boxes)
                counter.update(tracked)
                in_count  = counter.in_count
                out_count = counter.out_count
                count     = max(0, in_count - out_count)
                draw_tracked_boxes(annotated, boxes, tracked)
                draw_counting_line(annotated, LINE_Y, width, in_count, out_count)
        else:
            import random
            count = random.randint(10, 45)
            cv2.putText(annotated, "DEMO — no model loaded", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            draw_counting_line(annotated, LINE_Y, width, in_count, out_count)

        density, color_hex = get_density(count, capacity)
        occ_pct  = round(min(count / max(capacity, 1), 1.0) * 100)
        progress = round(frame_idx / max(total_frames, 1) * 100)

        frame_b64 = None
        if frame_idx % (SKIP * STREAM_EVERY) == 0:
            out_w = min(width, 960)
            out_h = int(height * out_w / width)
            small = cv2.resize(annotated, (out_w, out_h))
            frame_b64 = frame_to_b64(small)

        payload = {
            "frame_idx": frame_idx, "total_frames": total_frames,
            "progress": progress,   "count": count,
            "capacity": capacity,   "density": density,
            "density_color": color_hex, "occupancy_pct": occ_pct,
            "fps": round(live_fps, 1),
            "in_count": in_count,   "out_count": out_count,
            "mode": mode,
        }
        if frame_b64:
            payload["frame"] = frame_b64

        socketio.emit("frame_data", payload, to=sid)
        socketio.sleep(0)

    cap.release()
    processing_active = False
    socketio.emit("done", {
        "total_frames": frame_idx, "in_count": in_count,
        "out_count": out_count,    "final_count": count,
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
    path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(path)
    return jsonify({"ok": True, "path": path, "name": f.filename})


@socketio.on("start_processing")
def handle_start(data):
    global processing_active, processing_thread, BUS_CAPACITY, LINE_RATIO
    if processing_active:
        emit("error", {"msg": "Already processing"})
        return
    video_path = data.get("path")
    mode       = data.get("mode", "live")
    capacity   = data.get("capacity", BUS_CAPACITY)
    line_ratio = float(data.get("line_ratio", LINE_RATIO))
    if not video_path or not os.path.exists(video_path):
        emit("error", {"msg": "Video file not found"})
        return
    processing_active = True
    sid = request.sid
    processing_thread = threading.Thread(
        target=process_video_live,
        args=(video_path, sid, capacity, mode, line_ratio),
        daemon=True
    )
    processing_thread.start()


@socketio.on("stop_processing")
def handle_stop(_):
    global processing_active
    processing_active = False
    emit("stopped", {})


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5051)
