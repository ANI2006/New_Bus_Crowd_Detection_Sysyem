import csv, os, time
from datetime import datetime
from config import LOG_FOLDER

os.makedirs(LOG_FOLDER, exist_ok=True)


class SessionLogger:
    """
    Logs per-frame stats to a CSV file for later export/analysis.
    Supports single-door and multi-door sessions.
    Filename: <videoname>_YYYYMMDD_HHMMSS.csv
    """

    def __init__(self, video_name: str, door_count: int = 1):
        ts        = time.strftime("%Y%m%d_%H%M%S")
        safe_name = os.path.splitext(os.path.basename(str(video_name)))[0]
        self.path       = os.path.join(LOG_FOLDER, f"{safe_name}_{ts}.csv")
        self.door_count = door_count
        self._file      = open(self.path, "w", newline="")
        self._writer    = csv.writer(self._file)

        # Base columns + per-door columns
        door_cols = []
        for i in range(door_count):
            lbl = chr(65 + i)
            door_cols += [f"door_{lbl}_in", f"door_{lbl}_out"]

        self._writer.writerow([
            "wall_time", "elapsed_s", "frame", "count",
            "total_in", "total_out",
            *door_cols,
            "occupancy_pct", "density",
        ])
        self.start_time   = time.time()
        self.session_hour = datetime.now().hour

    def log(self, frame_idx: int, count: int,
            total_in: int, total_out: int,
            occupancy_pct: int, density: str,
            door_counts: list | None = None):
        """
        door_counts: list of (in, out) tuples, one per door.
                     If None, logs zeros for all door columns.
        """
        elapsed   = round(time.time() - self.start_time, 2)
        wall_time = datetime.now().strftime("%H:%M:%S")

        door_vals = []
        if door_counts:
            for d_in, d_out in door_counts:
                door_vals += [d_in, d_out]
        else:
            door_vals = [0, 0] * self.door_count

        self._writer.writerow([
            wall_time, elapsed, frame_idx, count,
            total_in, total_out,
            *door_vals,
            occupancy_pct, density,
        ])

    def close(self):
        self._file.close()

    def summary(self, total_frames: int) -> dict:
        return {
            "log_file":     os.path.basename(self.path),
            "total_frames": total_frames,
            "duration_s":   round(time.time() - self.start_time, 1),
        }


def analyze_logs() -> tuple[dict, list]:
    """
    Read all CSV logs and compute hourly occupancy patterns.
    Filename format: <videoname>_YYYYMMDD_HHMMSS.csv
    The hour is parsed from the last HHMMSS segment.

    Returns:
        hourly: dict  {0..23: {avg_pct, max_pct, sample_count}}
        summaries: list of per-log dicts
    """
    hourly = {h: {"total_pct": 0.0, "max_pct": 0.0, "samples": 0}
              for h in range(24)}
    log_summaries = []

    if not os.path.isdir(LOG_FOLDER):
        return _build_result(hourly), log_summaries

    for fname in sorted(os.listdir(LOG_FOLDER)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(LOG_FOLDER, fname)
        try:
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            if not rows:
                continue

            # Extract hour: last underscore-separated token is HHMMSS.csv
            hour = None
            try:
                hhmmss = fname.rsplit("_", 1)[-1].replace(".csv", "")
                if len(hhmmss) == 6 and hhmmss.isdigit():
                    hour = int(hhmmss[:2])
            except Exception:
                pass

            pcts = []
            for row in rows:
                try:
                    pct = float(row.get("occupancy_pct", 0))
                    pcts.append(pct)
                    if hour is not None:
                        hourly[hour]["total_pct"] += pct
                        hourly[hour]["max_pct"] = max(hourly[hour]["max_pct"], pct)
                        hourly[hour]["samples"] += 1
                except Exception:
                    pass

            if pcts:
                log_summaries.append({
                    "file":    fname,
                    "avg_pct": round(sum(pcts) / len(pcts), 1),
                    "max_pct": round(max(pcts), 1),
                    "rows":    len(pcts),
                    "hour":    hour,
                })
        except Exception:
            continue

    return _build_result(hourly), log_summaries


def _build_result(hourly: dict) -> dict:
    result = {}
    for h, data in hourly.items():
        if data["samples"] > 0:
            result[h] = {
                "avg_pct":      round(data["total_pct"] / data["samples"], 1),
                "max_pct":      round(data["max_pct"], 1),
                "sample_count": data["samples"],
            }
        else:
            result[h] = {"avg_pct": None, "max_pct": None, "sample_count": 0}
    return result
