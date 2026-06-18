import cv2
import yaml
import csv
import argparse
import os
import time
from datetime import datetime

from core.detector import PersonDetector
from core.tracker import CentroidTracker
from core.feature_extractor import CrowdFeatureExtractor


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


def save_row(output_path, row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_exists = os.path.exists(output_path)

    with open(output_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def draw_collection_panel(frame, label, frame_number, output_path):
    cv2.putText(
        frame,
        f"Collecting Label: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames Saved: {frame_number}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Output: {output_path}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to stop",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0 for webcam or path to video file"
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Label for this data collection session, example: normal"
    )

    parser.add_argument(
        "--output",
        default="data/features.csv",
        help="Output CSV file path"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Optional duration in seconds. Use 0 for manual stop."
    )

    args = parser.parse_args()

    config = load_config()
    source = parse_source(args.source)

    detector = PersonDetector(config)

    tracker = CentroidTracker(
        max_disappeared=config["tracker"]["max_disappeared"],
        max_distance=config["tracker"]["max_distance"]
    )

    feature_extractor = CrowdFeatureExtractor()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Feature collection started.")
    print(f"Label: {args.label}")
    print(f"Saving to: {args.output}")

    start_time = time.time()
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame = resize_frame(frame, config["video"]["resize_width"])

        boxes = detector.detect(frame)
        tracked_objects = tracker.update(boxes)
        features = feature_extractor.extract(tracked_objects)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_number": frame_number,
            **features,
            "label": args.label
        }

        save_row(args.output, row)

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

        draw_collection_panel(frame, args.label, frame_number, args.output)

        cv2.imshow("CrowdSense AI - Feature Collector", frame)

        if args.duration > 0:
            elapsed_time = time.time() - start_time

            if elapsed_time >= args.duration:
                break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Feature collection completed.")
    print(f"Saved rows: {frame_number}")
    print(f"Dataset path: {args.output}")


if __name__ == "__main__":
    main()