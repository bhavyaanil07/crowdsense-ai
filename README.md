# CrowdSense AI

**CrowdSense AI** is a real-time crowd behavior analysis and anomaly detection system built using **OpenCV** and **Machine Learning**.

The system detects people from video/webcam input, tracks them across frames, extracts crowd-level features, logs session data, and detects possible crowd anomalies using both rule-based logic and ML models.

---

## Project Overview

CrowdSense AI is designed for real-world environments such as:

* Malls
* Campuses
* Metro stations
* Events
* Public spaces
* Surveillance monitoring systems

The goal is to analyze crowd behavior in real time and detect unusual situations like overcrowding, rapid movement, chaotic movement, or tight crowd clustering.

---

## Key Features

* Real-time person detection using OpenCV HOG + SVM
* Centroid-based multi-person tracking
* Crowd feature extraction
* CSV session logging
* Rule-based alert engine
* Isolation Forest anomaly detection
* Random Forest anomaly type classification
* Live OpenCV dashboard overlay
* Model training pipeline
* Evaluation script with accuracy, classification report, confusion matrix, and feature importance

---

## Tech Stack

| Layer                | Technology   |
| -------------------- | ------------ |
| Programming Language | Python       |
| Computer Vision      | OpenCV       |
| Numerical Computing  | NumPy        |
| Data Processing      | Pandas       |
| Machine Learning     | Scikit-learn |
| Model Saving         | Joblib       |
| Config Management    | PyYAML       |
| Version Control      | Git + GitHub |

---

## Project Architecture

```text
Video/Webcam Input
        |
        v
Person Detection
        |
        v
Centroid Tracking
        |
        v
Crowd Feature Extraction
        |
        v
CSV Logging
        |
        v
Rule-Based Alerts + ML Anomaly Detection
        |
        v
Live Alert Display
```

---

## Project Structure

```text
crowdsense-ai/
│
├── alerts/
│   ├── __init__.py
│   └── alert_engine.py
│
├── config/
│   └── config.yaml
│
├── core/
│   ├── __init__.py
│   ├── anomaly_classifier.py
│   ├── detector.py
│   ├── feature_extractor.py
│   ├── ml_anomaly_detector.py
│   ├── session_logger.py
│   └── tracker.py
│
├── data/
│   └── features.csv
│
├── logs/
│   └── session logs
│
├── models/
│   ├── isolation_forest.pkl
│   └── random_forest_classifier.pkl
│
├── training/
│   ├── __init__.py
│   ├── collect_features.py
│   ├── evaluate.py
│   ├── train_anomaly.py
│   └── train_classifier.py
│
├── requirements.txt
├── main.py
└── README.md
```

---

## ML Features Extracted

The system extracts numerical crowd features from every frame.

| Feature             | Description                                      |
| ------------------- | ------------------------------------------------ |
| `person_count`      | Number of tracked people                         |
| `avg_velocity`      | Average movement speed of tracked people         |
| `velocity_variance` | Variation in crowd movement speed                |
| `group_dispersion`  | How spread out the crowd is                      |
| `aspect_ratio_mean` | Average width-to-height ratio of detected people |
| `dwell_time_mean`   | Average time people remain visible               |

These features are used for rule-based alerts and ML-based anomaly detection.

---

## Machine Learning Pipeline

CrowdSense AI uses a two-layer ML approach.

### 1. Isolation Forest

Used for unsupervised anomaly detection.

* Trained mainly on normal crowd behavior
* Detects unusual crowd patterns
* Outputs `NORMAL` or `ANOMALY`

### 2. Random Forest Classifier

Used for supervised anomaly type classification.

Possible labels:

```text
normal
overcrowding
rapid_movement
chaotic_movement
tight_cluster
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd crowdsense-ai
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Live System

For webcam:

```bash
python main.py --source 0
```

For a video file:

```bash
python main.py --source data/sample_crowd.mp4
```

Press `Q` to stop the program.

---

## Collect Training Data

Collect normal behavior:

```bash
python -m training.collect_features --source 0 --label normal --duration 15
```

Collect rapid movement behavior:

```bash
python -m training.collect_features --source 0 --label rapid_movement --duration 15
```

The collected features are saved to:

```text
data/features.csv
```

---

## Train Isolation Forest Model

```bash
python -m training.train_anomaly --data data/features.csv
```

This saves the model to:

```text
models/isolation_forest.pkl
```

---

## Train Random Forest Classifier

```bash
python -m training.train_classifier --data data/features.csv
```

This saves the classifier to:

```text
models/random_forest_classifier.pkl
```

---

## Evaluate Model

```bash
python -m training.evaluate --data data/features.csv
```

Evaluation output includes:

* Accuracy
* Classification report
* Confusion matrix
* Feature importance

---

## Example Output

The live OpenCV window displays:

* People count
* FPS
* Crowd features
* Isolation Forest status
* Isolation Forest anomaly score
* Random Forest predicted type
* Random Forest confidence
* Final alert level

Alert levels include:

```text
NORMAL
LOW
MEDIUM
HIGH
CRITICAL
```

---

## Current Status

Completed:

* Person detection
* Object tracking
* Feature extraction
* CSV logging
* Rule-based alert system
* Training data collector
* Isolation Forest model training
* Random Forest classifier training
* Live two-layer ML alert system
* Model evaluation script

---

## Future Improvements

* Add YOLOv8-nano as an optional detector
* Add ROI zone-based crowd density analysis
* Add Flask dashboard
* Add Chart.js live graphs
* Add sound alerts
* Add Docker support
* Add RTSP camera support
* Add LSTM-based temporal anomaly detection
* Improve dataset quality using public crowd datasets

---

## Author

**Bhavya Anil**

B.Tech Artificial Intelligence and Data Science
Interested in Machine Learning, Computer Vision, and real-world AI systems.

---

## Disclaimer

This project is intended for educational and research purposes. It should not be used for real surveillance or safety-critical deployment without proper testing, privacy review, and ethical approval.

