import os
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    return df


def train_isolation_forest(df):
    """
    Trains Isolation Forest only on normal samples if a label column exists.
    Otherwise trains on the full dataset.
    """

    if "label" in df.columns:
        normal_df = df[df["label"].astype(str).str.lower() == "normal"]

        if len(normal_df) == 0:
            print("Warning: No rows with label='normal'. Training on full dataset.")
            train_df = df
        else:
            train_df = normal_df
    else:
        train_df = df

    X_train = train_df[FEATURE_COLUMNS]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=150,
                    contamination=0.10,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train)

    return model, train_df


def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def show_training_summary(model, df):
    X = df[FEATURE_COLUMNS]

    predictions = model.predict(X)
    scores = model.decision_function(X)

    normal_count = int((predictions == 1).sum())
    anomaly_count = int((predictions == -1).sum())

    print("\nTraining Summary")
    print("----------------")
    print(f"Rows used for training: {len(df)}")
    print(f"Normal predictions: {normal_count}")
    print(f"Anomaly predictions: {anomaly_count}")
    print(f"Average normality score: {scores.mean():.4f}")
    print(f"Minimum score: {scores.min():.4f}")
    print(f"Maximum score: {scores.max():.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/features.csv",
        help="Path to feature CSV dataset"
    )

    parser.add_argument(
        "--output",
        default="models/isolation_forest.pkl",
        help="Path to save trained Isolation Forest model"
    )

    args = parser.parse_args()

    df = load_dataset(args.data)

    if len(df) < 20:
        print("Warning: Dataset has fewer than 20 rows. Collect more data for better results.")

    model, train_df = train_isolation_forest(df)

    save_model(model, args.output)

    show_training_summary(model, train_df)

    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()