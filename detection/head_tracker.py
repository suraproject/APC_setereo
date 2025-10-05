"""Simple centroid tracker + crossing logic. Not a full SORT implementation but lightweight and robust."""
import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> frames disappeared
        self.history = {}  # object_id -> list of centroids
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.history[oid] = [centroid]
        self.next_object_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.history[oid]

    def update(self, rects):
        # rects: list of bbox tuples (x1,y1,x2,y2)
        if len(rects) == 0:
            # mark disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.history

        input_centroids = np.zeros((len(rects), 2), dtype='int')
        for (i, (x1,y1,x2,y2)) in enumerate(rects):
            cX = int((x1+x2)/2.0); cY = int((y1+y2)/2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = distance.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set(); used_cols = set()
            for (r, c) in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = object_ids[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.history[oid].append(tuple(input_centroids[c]))
                self.disappeared[oid] = 0
                used_rows.add(r); used_cols.add(c)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            # mark disappeared
            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            # register new
            for c in unused_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects, self.history

# helper: check crossing with center band

def check_crossing(history, band_top, band_bottom):
    # history: list of (x,y)
    if len(history) < 2:
        return None
    y_prev = history[-2][1]
    y_cur = history[-1][1]
    if y_prev < band_top and y_cur >= band_bottom:
        return 'in'
    elif y_prev > band_bottom and y_cur <= band_top:
        return 'out'
    return None
