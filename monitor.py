from ultralytics import YOLO
import cv2
from collections import deque
import time

# ── CONFIG ──────────────────────────────────────────────
MODEL_PATH   = "best_new.pt"       # your trained model
SOURCE       = 0               # 0 = webcam | "test_video.mp4" = video file
BUS_CAPACITY = 60              # change to your bus's max capacity
CONF         = 0.3             # detection confidence threshold
SMOOTH_FRAMES = 10             # how many frames to average count over
# ────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

count_buffer = deque(maxlen=SMOOTH_FRAMES)
fps_buffer   = deque(maxlen=30)
prev_time    = time.time()

def get_density(count, capacity):
    ratio = count / capacity
    if ratio < 0.4:
        return "LOW",    (34, 197, 94)    # green
    elif ratio < 0.7:
        return "MEDIUM", (251, 146, 60)   # orange
    elif ratio < 0.9:
        return "HIGH",   (239, 68, 68)    # red
    else:
        return "FULL",   (127, 29, 29)    # dark red

def draw_overlay(frame, count, capacity, fps):
    h, w = frame.shape[:2]

    density_label, density_color = get_density(count, capacity)
    pct = int((count / capacity) * 100)

    # ── top-left info box ──
    cv2.rectangle(frame, (0, 0), (320, 110), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 0), (320, 110), (50, 50, 50), 1)

    # people count
    cv2.putText(frame, f"People: {count}", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # density badge
    cv2.rectangle(frame, (12, 48), (160, 78), density_color, -1)
    cv2.putText(frame, density_label, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # capacity %
    cv2.putText(frame, f"{pct}% of {capacity} seats", (12, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── capacity bar (bottom of screen) ──
    bar_x, bar_y, bar_w, bar_h = 10, h - 25, w - 20, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    fill_w = int(bar_w * min(count / capacity, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  density_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), 1)

    # ── top-right: FPS + timestamp ──
    fps_text = f"FPS: {fps:.1f}"
    ts_text  = time.strftime("%H:%M:%S")
    cv2.putText(frame, fps_text, (w - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    cv2.putText(frame, ts_text, (w - 110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    return frame

print("✅ Bus Monitor started — press Q to quit, S to screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  End of video or stream lost")
        break

    # ── FPS calculation ──
    now = time.time()
    fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    fps = sum(fps_buffer) / len(fps_buffer)

    # ── Run detection ──
    results = model(frame, conf=CONF, verbose=False)

    # ── Smooth the count ──
    raw_count = len(results[0].boxes)
    count_buffer.append(raw_count)
    stable_count = round(sum(count_buffer) / len(count_buffer))

    # ── Draw YOLO boxes ──
    annotated = results[0].plot(
        labels=True,
        conf=True,
        line_width=2
    )

    # ── Draw our custom overlay ──
    annotated = draw_overlay(annotated, stable_count, BUS_CAPACITY, fps)

    # ── Show window ──
    cv2.imshow("Bus Occupancy Monitor", annotated)

    # ── Key controls ──
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"📸 Screenshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()