import csv, os, time
from datetime import datetime
from config import LOG_FOLDER

os.makedirs(LOG_FOLDER, exist_ok=True)


class SessionLogger:

    def __init__(self, video_name: str):
        ts        = time.strftime("%Y%m%d_%H%M%S")
        safe_name = os.path.splitext(os.path.basename(video_name))[0]
        self.path = os.path.join(LOG_FOLDER, f"{safe_name}_{ts}.csv")
        self._file   = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "wall_time", "elapsed_s", "frame", "count",
            "in_count", "out_count",
            "door_a_in", "door_a_out",
            "door_b_in", "door_b_out",
            "occupancy_pct", "density"
        ])
        self.start_time   = time.time()
        self.session_hour = datetime.now().hour   # recorded at open time, used by analytics

    def log(self, frame_idx, count, in_count, out_count, occupancy_pct, density,
            door_a_in=0, door_a_out=0, door_b_in=0, door_b_out=0):
        elapsed   = round(time.time() - self.start_time, 2)
        wall_time = datetime.now().strftime("%H:%M:%S")
        self._writer.writerow([
            wall_time, elapsed, frame_idx, count,
            in_count, out_count,
            door_a_in, door_a_out,
            door_b_in, door_b_out,
            occupancy_pct, density
        ])

    def close(self):
        self._file.close()

    def summary(self, total_frames):
        return {
            "log_file":     os.path.basename(self.path),
            "total_frames": total_frames,
            "duration_s":   round(time.time() - self.start_time, 1),
        }


def analyze_logs():
    """
    Read all CSV logs and compute hourly occupancy patterns.
    Filename format: <videoname>_YYYYMMDD_HHMMSS.csv
    The hour is in the LAST underscore-segment (HHMMSS), first 2 chars.
    """
    hourly = {h: {"total_pct": 0, "max_pct": 0, "samples": 0} for h in range(24)}
    log_summaries = []

    if not os.path.isdir(LOG_FOLDER):
        return hourly, log_summaries

    for fname in sorted(os.listdir(LOG_FOLDER)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(LOG_FOLDER, fname)
        try:
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue


            hour = None
            try:
                hhmmss = fname.rsplit("_", 1)[-1].replace(".csv", "")  
                if len(hhmmss) == 6 and hhmmss.isdigit():
                    hour = int(hhmmss[:2])   # 23
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

    return result, log_summaries
