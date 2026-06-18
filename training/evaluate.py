import os
import argparse
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


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
        raise ValueError("Dataset must contain a 'label' column.")

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    return df


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def get_test_split(df):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    if len(df) < 30 or y.nunique() < 2:
        print("Dataset is small. Evaluating on the full dataset.")
        return X, y

    try:
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_test, y_test

    except ValueError:
        print("Stratified split failed. Evaluating on the full dataset.")
        return X, y


def print_feature_importance(model):
    try:
        rf_model = model.named_steps["random_forest"]
        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": importances
            }
        ).sort_values(by="importance", ascending=False)

        print("\nFeature Importance")
        print("------------------")

        for _, row in importance_df.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    except Exception:
        print("\nFeature importance not available for this model.")


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Samples evaluated: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(y_test, predictions))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/features.csv",
        help="Path to feature CSV dataset"
    )

    parser.add_argument(
        "--model",
        default="models/random_forest_classifier.pkl",
        help="Path to trained Random Forest classifier"
    )

    args = parser.parse_args()

    df = load_dataset(args.data)
    model = load_model(args.model)

    X_test, y_test = get_test_split(df)

    evaluate_model(model, X_test, y_test)
    print_feature_importance(model)


if __name__ == "__main__":
    main()