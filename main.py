import cv2
import yaml
import argparse
import time

from core.detector import PersonDetector
from core.tracker import CentroidTracker
from core.feature_extractor import CrowdFeatureExtractor
from core.session_logger import SessionLogger
from core.ml_anomaly_detector import MLAnomalyDetector
from core.anomaly_classifier import AnomalyClassifier
from alerts.alert_engine import AlertEngine


def load_config(path="config/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def resize_frame(frame, width):
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))


def parse_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def draw_feature_panel(frame, features, ml_result, classifier_result):
    x = 20
    y = 110
    line_height = 28

    cv2.putText(
        frame,
        "Crowd Features",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += line_height

    for key, value in features.items():
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        else:
            text = f"{key}: {value}"

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y += line_height

    y += 10

    cv2.putText(
        frame,
        "ML Anomaly Detection",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"IF Status: {ml_result['ml_status']}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"IF Score: {ml_result['anomaly_score']:.4f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"RF Type: {classifier_result['predicted_type']}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"RF Confidence: {classifier_result['confidence']:.2f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def combine_alerts(rule_alert, ml_result, classifier_result):
    final_alert = {
        "level": rule_alert["level"],
        "type": rule_alert["type"],
        "message": rule_alert["message"]
    }

    ml_flagged = ml_result["enabled"] and ml_result["ml_status"] == "ANOMALY"
    rf_enabled = classifier_result["enabled"]
    predicted_type = classifier_result["predicted_type"]
    confidence = classifier_result["confidence"]

    if ml_flagged:
        if final_alert["level"] == "NORMAL":
            final_alert = {
                "level": "MEDIUM",
                "type": "ML Anomaly",
                "message": "Isolation Forest detected abnormal crowd behavior"
            }
        else:
            final_alert["message"] += " | Isolation Forest also flagged anomaly"

    if rf_enabled and predicted_type != "normal":
        if confidence >= 0.70:
            final_alert = {
                "level": "HIGH",
                "type": predicted_type,
                "message": f"Random Forest classified anomaly as {predicted_type}"
            }

        elif confidence >= 0.45 and final_alert["level"] in ["NORMAL", "LOW"]:
            final_alert = {
                "level": "MEDIUM",
                "type": predicted_type,
                "message": f"Possible {predicted_type} detected by Random Forest"
            }

    if (
        ml_flagged
        and rf_enabled
        and predicted_type != "normal"
        and confidence >= 0.70
    ):
        final_alert = {
            "level": "CRITICAL",
            "type": predicted_type,
            "message": f"Both IF and RF detected high-confidence {predicted_type}"
        }

    return final_alert


def draw_alert_panel(frame, final_alert):
    level = final_alert["level"]
    alert_type = final_alert["type"]
    message = final_alert["message"]

    if level == "NORMAL":
        color = (0, 255, 0)
    elif level == "LOW":
        color = (255, 255, 0)
    elif level == "MEDIUM":
        color = (0, 255, 255)
    elif level == "HIGH":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, height - 90), (width, height), color, -1)

    cv2.putText(
        frame,
        f"ALERT LEVEL: {level}",
        (20, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"{alert_type} - {message}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=None,
        help="Video source: 0 for webcam or path to video file"
    )

    args = parser.parse_args()

    config = load_config()

    source = args.source if args.source is not None else config["video"]["source"]
    source = parse_source(str(source))

    detector = PersonDetector(config)

    tracker = CentroidTracker(
        max_disappeared=config["tracker"]["max_disappeared"],
        max_distance=config["tracker"]["max_distance"]
    )

    feature_extractor = CrowdFeatureExtractor()
    session_logger = SessionLogger()
    alert_engine = AlertEngine(config)

    ml_detector = MLAnomalyDetector(
        config["models"]["isolation_forest_path"]
    )

    anomaly_classifier = AnomalyClassifier(
        config["models"]["random_forest_path"]
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    previous_time = time.time()
    frame_number = 0

    print(f"Logging session data to: {session_logger.file_path}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame = resize_frame(frame, config["video"]["resize_width"])

        boxes = detector.detect(frame)
        tracked_objects = tracker.update(boxes)

        features = feature_extractor.extract(tracked_objects)

        rule_alert = alert_engine.evaluate(features)
        ml_result = ml_detector.predict(features)
        classifier_result = anomaly_classifier.predict(features)

        final_alert = combine_alerts(
            rule_alert,
            ml_result,
            classifier_result
        )

        for object_id, data in tracked_objects.items():
            x, y, w, h = data["bbox"]
            cx, cy = data["centroid"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID {object_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            frame,
            f"People Count: {len(tracked_objects)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        draw_feature_panel(
            frame,
            features,
            ml_result,
            classifier_result
        )

        draw_alert_panel(frame, final_alert)

        log_row = {
            **features,
            "rule_alert_level": rule_alert["level"],
            "rule_alert_type": rule_alert["type"],
            "if_status": ml_result["ml_status"],
            "if_prediction": ml_result["ml_prediction"],
            "if_anomaly_score": ml_result["anomaly_score"],
            "rf_predicted_type": classifier_result["predicted_type"],
            "rf_confidence": classifier_result["confidence"],
            "final_alert_level": final_alert["level"],
            "final_alert_type": final_alert["type"],
            "final_alert_message": final_alert["message"]
        }

        session_logger.log(frame_number, log_row)

        cv2.imshow("CrowdSense AI - Two Layer ML System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    session_logger.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Session ended. Log file saved successfully.")


if __name__ == "__main__":
    main()