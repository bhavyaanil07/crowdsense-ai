import os
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


FEATURE_COLUMNS = [
    "person_count",
    "avg_velocity",
    "velocity_variance",
    "group_dispersion",
    "aspect_ratio_mean",
    "dwell_time_mean"
]


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for supervised training.")

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    return df


def train_random_forest(df):
    label_counts = df["label"].value_counts()

    print("\nLabel Distribution")
    print("------------------")
    print(label_counts)

    if df["label"].nunique() < 2:
        raise ValueError(
            "Random Forest needs at least 2 different labels. "
            "Collect more data using labels like normal, rapid_movement, overcrowding."
        )

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    test_size = 0.2

    if len(df) < 30:
        print("\nWarning: Dataset is small. Training without a test split.")
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nEvaluation")
    print("----------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(y_test, predictions))

    return model


def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/features.csv",
        help="Path to feature CSV dataset"
    )

    parser.add_argument(
        "--output",
        default="models/random_forest_classifier.pkl",
        help="Path to save trained Random Forest classifier"
    )

    args = parser.parse_args()

    df = load_dataset(args.data)

    model = train_random_forest(df)

    save_model(model, args.output)

    print(f"\nRandom Forest classifier saved to: {args.output}")


if __name__ == "__main__":
    main()