class AlertEngine:
    """
    Rule-based anomaly alert engine.

    This is the first alert layer.
    Later, ML models will replace or support these rules.
    """

    def __init__(self, config):
        self.config = config["alerts"]

    def evaluate(self, features):
        person_count = features["person_count"]
        avg_velocity = features["avg_velocity"]
        velocity_variance = features["velocity_variance"]
        group_dispersion = features["group_dispersion"]

        alert = {
            "level": "NORMAL",
            "type": "Normal Crowd Activity",
            "message": "No anomaly detected"
        }

        # Overcrowding
        if person_count >= self.config["person_count_high"]:
            alert = {
                "level": "HIGH",
                "type": "Overcrowding",
                "message": "High number of people detected"
            }

        elif person_count >= self.config["person_count_medium"]:
            alert = {
                "level": "MEDIUM",
                "type": "Crowd Build-up",
                "message": "Moderate crowd density detected"
            }

        # Sudden movement / panic-like motion
        if avg_velocity >= self.config["velocity_high"]:
            alert = {
                "level": "HIGH",
                "type": "Rapid Crowd Movement",
                "message": "Very high average movement detected"
            }

        elif avg_velocity >= self.config["velocity_medium"] and alert["level"] == "NORMAL":
            alert = {
                "level": "MEDIUM",
                "type": "Unusual Movement",
                "message": "Crowd movement is higher than normal"
            }

        # Chaotic motion
        if velocity_variance > 5000:
            alert = {
                "level": "HIGH",
                "type": "Chaotic Movement",
                "message": "Large variation in movement speed detected"
            }

        # Very tight clustering
        if person_count >= 3 and group_dispersion <= self.config["dispersion_low"]:
            alert = {
                "level": "MEDIUM",
                "type": "Tight Crowd Cluster",
                "message": "People are unusually close together"
            }

        # Very scattered sudden movement
        if person_count >= 3 and group_dispersion >= self.config["dispersion_high"]:
            if avg_velocity >= self.config["velocity_medium"]:
                alert = {
                    "level": "HIGH",
                    "type": "Dispersed Rapid Movement",
                    "message": "People are moving rapidly in different directions"
                }

        # Critical condition
        if (
            person_count >= self.config["person_count_high"]
            and avg_velocity >= self.config["velocity_high"]
        ):
            alert = {
                "level": "CRITICAL",
                "type": "Possible Panic Situation",
                "message": "High crowd count with rapid movement detected"
            }

        return alert