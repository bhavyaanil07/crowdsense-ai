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


class AnomalyClassifier:
    """
    Uses a trained Random Forest classifier to predict anomaly type.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Random Forest classifier loaded from: {self.model_path}")
        else:
            print(f"Warning: Random Forest classifier not found at {self.model_path}")
            print("Anomaly type classification will be disabled.")

    def predict(self, features):
        if self.model is None:
            return {
                "enabled": False,
                "predicted_type": "CLASSIFIER_NOT_FOUND",
                "confidence": 0.0
            }

        row = {
            column: features[column]
            for column in FEATURE_COLUMNS
        }

        X = pd.DataFrame([row])

        prediction = self.model.predict(X)[0]

        confidence = 0.0

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(max(probabilities))

        return {
            "enabled": True,
            "predicted_type": str(prediction),
            "confidence": confidence
        }