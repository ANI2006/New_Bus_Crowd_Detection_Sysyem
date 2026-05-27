import cv2, math

C_GREEN  = (80,  220, 140)
C_RED    = (80,  90,  230)
C_CYAN   = (60,  210, 255)
C_YELLOW = (40,  210, 255)
C_BLACK  = (10,  10,  10)
C_WHITE  = (235, 235, 235)
C_AMBER  = (40,  180, 255)

# Colour palette for up to 8 doors — cycles beyond that
_DOOR_PALETTE = [
    (60,  210, 255),   # cyan   — Door A
    (40,  210, 255),   # amber  — Door B
    (140, 100, 255),   # purple — Door C
    (80,  220, 140),   # green  — Door D
    (255, 160,  60),   # orange — Door E
    (255,  80, 160),   # pink   — Door F
    (100, 220, 220),   # teal   — Door G
    (200, 200,  80),   # lime   — Door H
]


def door_color(index: int) -> tuple:
    """Return a BGR colour for the door at zero-based index."""
    return _DOOR_PALETTE[index % len(_DOOR_PALETTE)]


def draw_counting_line(frame, line_y, width, in_count, out_count,
                       label="COUNTING LINE", color=None):
    color = color or C_CYAN
    # Glow
    cv2.line(frame, (0, line_y), (width, line_y), (20, 80, 80), 12, cv2.LINE_AA)
    # Dashed line
    x = 0
    while x < width:
        cv2.line(frame, (x, line_y), (min(x + 22, width), line_y), color, 2, cv2.LINE_AA)
        x += 32
    # IN / OUT labels (left side)
    cv2.rectangle(frame, (0, line_y - 32), (160, line_y + 28), C_BLACK, -1)
    cv2.putText(frame, f"  IN : {in_count}",  (4, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"  OUT: {out_count}", (4, line_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED,   1, cv2.LINE_AA)
    # Door label (right side)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    px = width - tw - 20
    cv2.rectangle(frame, (px - 4, line_y - th - 10), (px + tw + 4, line_y + 4), C_BLACK, -1)
    cv2.putText(frame, f" {label} ", (px, line_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def draw_tracked_boxes(frame, boxes, tracked):
    for (x1, y1, x2, y2) in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        best_id, best_d = None, 9999
        for oid, (ox, oy) in tracked.items():
            d = math.hypot(cx - ox, cy - oy)
            if d < best_d:
                best_d, best_id = d, oid

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), C_GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, C_CYAN, -1, cv2.LINE_AA)
        if best_id is not None and best_d < 120:
            lbl = f"#{best_id}"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - lh - 8),
                          (int(x1) + lw + 6, int(y1)), C_BLACK, -1)
            cv2.putText(frame, lbl, (int(x1) + 3, int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREEN, 1, cv2.LINE_AA)


def draw_alert_banner(frame, density, occupancy_pct):
    """Draw a red/amber alert banner when the vehicle is nearly full or full."""
    h, w = frame.shape[:2]
    if occupancy_pct >= 100:
        color = (0, 0, 200)
        msg   = "!! FULL — CAPACITY REACHED !!"
    elif occupancy_pct >= 80:
        color = (0, 120, 255)
        msg   = "WARNING — NEARLY FULL"
    else:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, msg, ((w - tw) // 2, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2, cv2.LINE_AA)
