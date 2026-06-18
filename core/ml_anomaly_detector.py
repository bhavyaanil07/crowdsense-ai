import os
import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "person_count",
    "avg_velocity",
    "velocity_variance",
    "group_dispersion",
    "aspect_ratio_mean",
    "dwell_time_mean"
]


class MLAnomalyDetector:
    """
    Uses trained Isolation Forest model to detect abnormal crowd behavior.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"ML model loaded from: {self.model_path}")
        else:
            print(f"Warning: ML model not found at {self.model_path}")
            print("Live system will continue using rule-based alerts only.")

    def predict(self, features):
        """
        Returns ML anomaly result.

        Isolation Forest output:
        1  = normal
        -1 = anomaly
        """

        if self.model is None:
            return {
                "enabled": False,
                "ml_status": "MODEL_NOT_FOUND",
                "ml_prediction": 0,
                "anomaly_score": 0.0
            }

        row = {
            column: features[column]
            for column in FEATURE_COLUMNS
        }

        X = pd.DataFrame([row])

        prediction = int(self.model.predict(X)[0])
        score = float(self.model.decision_function(X)[0])

        if prediction == -1:
            status = "ANOMALY"
        else:
            status = "NORMAL"

        return {
            "enabled": True,
            "ml_status": status,
            "ml_prediction": prediction,
            "anomaly_score": score
        }