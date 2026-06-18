import cv2
import yaml
import argparse
import time

from core.detector import PersonDetector
from core.tracker import CentroidTracker
from core.feature_extractor import CrowdFeatureExtractor
from core.session_logger import SessionLogger


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


def draw_feature_panel(frame, features):
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

        session_logger.log(frame_number, features)

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

        draw_feature_panel(frame, features)

        cv2.imshow("CrowdSense AI - CSV Logging", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    session_logger.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Session ended. Log file saved successfully.")


if __name__ == "__main__":
    main()