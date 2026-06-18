from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """
    Simple centroid-based person tracker.
    Assigns each detected person a stable ID.
    """

    def __init__(self, max_disappeared=25, max_distance=90):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": bbox
        }

        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        input_centroids = []

        for (x, y, w, h) in boxes:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(tuple(centroid), boxes[i])

            return self.objects

        object_ids = list(self.objects.keys())

        object_centroids = np.array(
            [self.objects[object_id]["centroid"] for object_id in object_ids]
        )

        distances = self.compute_distances(object_centroids, input_centroids)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]

            self.objects[object_id] = {
                "centroid": tuple(input_centroids[col]),
                "bbox": boxes[col]
            }

            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        if distances.shape[0] >= distances.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        else:
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), boxes[col])

        return self.objects

    def compute_distances(self, a, b):
        distances = np.zeros((len(a), len(b)))

        for i in range(len(a)):
            for j in range(len(b)):
                distances[i, j] = np.linalg.norm(a[i] - b[j])

        return distances