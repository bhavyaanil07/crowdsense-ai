import time
import numpy as np


class CrowdFeatureExtractor:
    """
    Extracts numerical crowd features from tracked people.
    These features will later be used for anomaly detection.
    """

    def __init__(self):
        self.previous_centroids = {}
        self.previous_time = time.time()
        self.entry_times = {}

    def extract(self, tracked_objects):
        current_time = time.time()
        delta_time = current_time - self.previous_time

        if delta_time <= 0:
            delta_time = 1e-6

        person_count = len(tracked_objects)

        centroids = []
        velocities = []
        aspect_ratios = []
        dwell_times = []

        current_ids = set(tracked_objects.keys())

        for object_id, data in tracked_objects.items():
            centroid = data["centroid"]
            bbox = data["bbox"]

            x, y, w, h = bbox

            centroids.append(centroid)

            if h != 0:
                aspect_ratios.append(w / h)

            if object_id not in self.entry_times:
                self.entry_times[object_id] = current_time

            dwell_times.append(current_time - self.entry_times[object_id])

            if object_id in self.previous_centroids:
                previous_centroid = self.previous_centroids[object_id]

                distance = np.linalg.norm(
                    np.array(centroid) - np.array(previous_centroid)
                )

                velocity = distance / delta_time
                velocities.append(velocity)

        # Remove old IDs that disappeared
        old_ids = set(self.entry_times.keys()) - current_ids

        for old_id in old_ids:
            del self.entry_times[old_id]

        if len(velocities) > 0:
            avg_velocity = float(np.mean(velocities))
            velocity_variance = float(np.var(velocities))
        else:
            avg_velocity = 0.0
            velocity_variance = 0.0

        if len(centroids) > 1:
            centroids_array = np.array(centroids)
            group_center = np.mean(centroids_array, axis=0)

            distances_from_center = np.linalg.norm(
                centroids_array - group_center,
                axis=1
            )

            group_dispersion = float(np.mean(distances_from_center))
        else:
            group_dispersion = 0.0

        if len(aspect_ratios) > 0:
            aspect_ratio_mean = float(np.mean(aspect_ratios))
        else:
            aspect_ratio_mean = 0.0

        if len(dwell_times) > 0:
            dwell_time_mean = float(np.mean(dwell_times))
        else:
            dwell_time_mean = 0.0

        self.previous_centroids = {
            object_id: data["centroid"]
            for object_id, data in tracked_objects.items()
        }

        self.previous_time = current_time

        features = {
            "person_count": person_count,
            "avg_velocity": avg_velocity,
            "velocity_variance": velocity_variance,
            "group_dispersion": group_dispersion,
            "aspect_ratio_mean": aspect_ratio_mean,
            "dwell_time_mean": dwell_time_mean
        }

        return features