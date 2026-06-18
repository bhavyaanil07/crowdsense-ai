import csv
import os
from datetime import datetime


class SessionLogger:
    """
    Saves per-frame crowd features into a CSV log file.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.log_dir, f"session_{timestamp}.csv")

        self.file = open(self.file_path, mode="w", newline="")
        self.writer = None

    def log(self, frame_number, features):
        """
        Writes one row of feature data to CSV.
        """

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_number": frame_number,
            **features
        }

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
            self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()