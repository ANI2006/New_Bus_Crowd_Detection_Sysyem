import math
from config import CROSS_COOLDOWN


class CentroidTracker:
    """Assigns persistent IDs to detections by nearest-centroid matching."""

    def __init__(self, max_disappeared=30):
        self.next_id         = 0
        self.objects         = {}
        self.disappeared     = {}
        self.max_disappeared = max_disappeared

    def _register(self, cx, cy):
        self.objects[self.next_id]     = (cx, cy)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, boxes):
        if not boxes:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for x1, y1, x2, y2 in boxes]

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
                self.objects[oid]     = centroids[c]
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

    def reset(self):
        self.next_id     = 0
        self.objects     = {}
        self.disappeared = {}


class LineCrossCounter:

    def __init__(self, line_y, label="A"):
        self.line_y    = line_y
        self.label     = label
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
                    self.cooldown[oid] = CROSS_COOLDOWN
                elif prev > self.line_y >= cy:     
                    self.out_count += 1
                    self.cooldown[oid] = CROSS_COOLDOWN

            self.prev_cy[oid] = cy

        for oid in list(self.prev_cy):
            if oid not in tracked:
                del self.prev_cy[oid]
                self.cooldown.pop(oid, None)

    def reset(self):
        self.prev_cy   = {}
        self.in_count  = 0
        self.out_count = 0
        self.cooldown  = {}


class DualLineCrossCounter:

    def __init__(self, line_y_a, line_y_b):
        self.door_a = LineCrossCounter(line_y_a, "A")
        self.door_b = LineCrossCounter(line_y_b, "B")

    def update(self, tracked):
        self.door_a.update(tracked)
        self.door_b.update(tracked)

    @property
    def in_count(self):
        return self.door_a.in_count + self.door_b.in_count

    @property
    def out_count(self):
        return self.door_a.out_count + self.door_b.out_count

    @property
    def door_a_in(self):  return self.door_a.in_count
    @property
    def door_a_out(self): return self.door_a.out_count
    @property
    def door_b_in(self):  return self.door_b.in_count
    @property
    def door_b_out(self): return self.door_b.out_count

    def reset(self):
        self.door_a.reset()
        self.door_b.reset()
