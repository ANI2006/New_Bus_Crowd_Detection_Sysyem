import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "bus_enter.mov"   # ← change to your video
MODEL_PATH   = "best_new.pt"
LINE_RATIO   = 0.55               # counting line: 55% down the frame
BUS_CAPACITY = 60

# ── Palette (BGR) ──────────────────────────────────────────────────────────────
C_BG     = (15,  17,  20)
C_BORDER = (50,  55,  65)
C_WHITE  = (235, 235, 235)
C_DIM    = (120, 120, 130)
C_GREEN  = (80,  220, 140)    # entered
C_RED    = (80,  90,  230)    # exited
C_YELLOW = (40,  210, 255)    # in bus
C_LINE   = (60,  180, 255)    # counting line
C_LGLOW  = (30,  90,  130)    # line glow

FONT   = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ────────────────────────────────────────────────────────────

def alpha_rect(img, x, y, w, h, color, alpha=0.60):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def border_rect(img, x, y, w, h, color, t=2):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, t, cv2.LINE_AA)


def stat_card(img, x, y, w, h, label, value, color):
    alpha_rect(img, x, y, w, h, C_BG, alpha=0.78)
    border_rect(img, x, y, w, h, C_BORDER)
    # top accent bar
    cv2.rectangle(img, (x + 10, y + 1), (x + w - 10, y + 4), color, -1)
    # label
    cv2.putText(img, label, (x + 14, y + 26),
                FONT_S, 0.48, C_DIM, 1, cv2.LINE_AA)
    # value — centred
    val = str(value)
    (tw, _), _ = cv2.getTextSize(val, FONT, 1.5, 2)
    cv2.putText(img, val, (x + (w - tw) // 2, y + h - 14),
                FONT, 1.5, color, 2, cv2.LINE_AA)


def capacity_bar(img, x, y, w, in_bus, capacity):
    ratio = min(in_bus / max(capacity, 1), 1.0)
    bar_h = 7
    cv2.rectangle(img, (x, y), (x + w, y + bar_h), C_BORDER, -1)
    bar_color = C_GREEN if ratio < 0.5 else (C_YELLOW if ratio < 0.8 else C_RED)
    if int(w * ratio) > 0:
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + bar_h), bar_color, -1)
    cv2.putText(img, f"{ratio:.0%}", (x + w + 8, y + bar_h),
                FONT_S, 0.42, C_DIM, 1, cv2.LINE_AA)


def counting_line(img, line_y, w):
    # glow
    cv2.line(img, (0, line_y), (w, line_y), C_LGLOW, 8, cv2.LINE_AA)
    # dashed line
    x, dash, gap = 0, 20, 10
    while x < w:
        cv2.line(img, (x, line_y), (min(x + dash, w), line_y), C_LINE, 2, cv2.LINE_AA)
        x += dash + gap
    # label
    lbl = " COUNTING LINE "
    (tw, th), _ = cv2.getTextSize(lbl, FONT_S, 0.44, 1)
    px = w - tw - 20
    alpha_rect(img, px - 4, line_y - th - 8, tw + 8, th + 10, C_BG, alpha=0.85)
    cv2.putText(img, lbl, (px, line_y - 5), FONT_S, 0.44, C_LINE, 1, cv2.LINE_AA)


def hud(img, in_count, out_count, in_bus, frame_idx, fps_val, capacity):
    h, w = img.shape[:2]

    # ── Right-side stat cards ─────────────────────────────────────────────────
    cw, ch, gap = 205, 82, 8
    cx = w - cw - 14
    cy = 14
    stat_card(img, cx, cy,              cw, ch, "ENTERED", in_count,  C_GREEN)
    stat_card(img, cx, cy + ch + gap,   cw, ch, "EXITED",  out_count, C_RED)
    stat_card(img, cx, cy + (ch+gap)*2, cw, ch, "IN BUS",  in_bus,    C_YELLOW)

    # ── Capacity bar card ─────────────────────────────────────────────────────
    by = cy + (ch + gap) * 3 + 4
    alpha_rect(img, cx, by, cw, 44, C_BG, alpha=0.78)
    border_rect(img, cx, by, cw, 44, C_BORDER)
    cv2.putText(img, "CAPACITY", (cx + 14, by + 18),
                FONT_S, 0.44, C_DIM, 1, cv2.LINE_AA)
    capacity_bar(img, cx + 14, by + 28, cw - 55, in_bus, capacity)

    # ── Top-left title bar ────────────────────────────────────────────────────
    alpha_rect(img, 10, 10, 270, 56, C_BG, alpha=0.80)
    border_rect(img, 10, 10, 270, 56, C_BORDER)
    cv2.rectangle(img, (10, 10), (14, 66), C_YELLOW, -1)   # left accent stripe
    cv2.putText(img, "BUS OCCUPANCY MONITOR", (22, 34),
                FONT_S, 0.54, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {frame_idx:05d}   {fps_val:.0f} fps", (22, 56),
                FONT_S, 0.38, C_DIM, 1, cv2.LINE_AA)

    # ── Bottom density badge ──────────────────────────────────────────────────
    ratio = in_bus / max(capacity, 1)
    if ratio < 0.5:
        density, dc = "LOW DENSITY",  C_GREEN
    elif ratio < 0.8:
        density, dc = "MED DENSITY",  C_YELLOW
    else:
        density, dc = "HIGH DENSITY", C_RED

    (dw, dh), _ = cv2.getTextSize(density, FONT_S, 0.52, 1)
    bx, by2 = 10, h - 14
    alpha_rect(img, bx, by2 - dh - 10, dw + 30, dh + 16, C_BG, alpha=0.80)
    cv2.rectangle(img, (bx, by2 - dh - 10), (bx + 4, by2 + 6), dc, -1)
    cv2.putText(img, density, (bx + 12, by2), FONT_S, 0.52, dc, 1, cv2.LINE_AA)


# ── Setup ──────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30

print(f"Video: {width}x{height} @ {fps:.1f}fps")
print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

LINE_Y     = int(height * LINE_RATIO)
LINE_START = sv.Point(width, LINE_Y)
LINE_END   = sv.Point(0,     LINE_Y)

tracker = sv.ByteTrack(
    track_activation_threshold=0.20,
    lost_track_buffer=int(fps * 3),
    minimum_matching_threshold=0.70,
    frame_rate=int(fps),
)

line_zone       = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator   = sv.BoxAnnotator(
    thickness=2,
    color=sv.ColorPalette.from_hex(["#50DC8C"]),
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.42,
    text_thickness=1,
    text_padding=4,
    color=sv.ColorPalette.from_hex(["#1a1a1a"]),
    text_color=sv.Color.from_hex("#50DC8C"),
)

frame_idx = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results    = model(frame, conf=0.20, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    crossed_in, crossed_out = line_zone.trigger(detections)
    if crossed_in.any():
        print(f"[{frame_idx:05d}] ✅ ENTERED — total IN:  {line_zone.in_count}")
    if crossed_out.any():
        print(f"[{frame_idx:05d}] 🚪 EXITED  — total OUT: {line_zone.out_count}")

    labels    = [f"#{tid}" for tid in detections.tracker_id]
    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

    in_bus = max(0, line_zone.in_count - line_zone.out_count)

    counting_line(annotated, LINE_Y, width)
    hud(annotated, line_zone.in_count, line_zone.out_count,
        in_bus, frame_idx, fps, BUS_CAPACITY)

    cv2.imshow("Bus Occupancy Monitor", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'─' * 38}")
print(f"  Total entered   : {line_zone.in_count}")
print(f"  Total exited    : {line_zone.out_count}")
print(f"  Currently inside: {max(0, line_zone.in_count - line_zone.out_count)}")
print(f"{'─' * 38}")