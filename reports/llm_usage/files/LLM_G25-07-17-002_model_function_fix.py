import argparse
import json
import sys
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _build_classifier(classifier: str, seed: int) -> Any:
    """
    Instantiate a scikit-learn classifier by name.

    Parameters
    ----------
    classifier : str
        One of "random_forest", "logistic_regression", "gradient_boosting".
    seed : int
        Random seed passed as random_state where supported.

    Raises
    ------
    ValueError
        If the classifier name is not recognized.
    """
    registry = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
    }

    if classifier not in registry:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            f"Available options: {list(registry.keys())}"
        )

    return registry[classifier]()


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
) -> None:
    """
    Train a classifier on the reduced training dataset and save it, along with
    training statistics.

    Parameters
    ----------
    classifier : str
        Name of the classifier to train ("random_forest", "logistic_regression",
        or "gradient_boosting").
    reducedTrain_csv : str
        Path to the reduced training dataset (selected features + target).
    target_column : str
        Name of the target column.
    model_path : str
        Path where the trained model will be saved (joblib format).
    metrics_json : str
        Path where training statistics will be saved (JSON format).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Raises
    ------
    ValueError
        If target_column is not found in the dataset, or classifier is unknown.
    """
    t0 = time.perf_counter()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.perf_counter() - t0

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()

    n_samples = len(df)
    n_features = len(feature_columns)
    target_1_percentage = float((y == 1).mean() * 100)

    model = _build_classifier(classifier, seed)

    t1 = time.perf_counter()
    model.fit(X, y)
    training_time = time.perf_counter() - t1

    joblib.dump({"model": model, "classifier_name": classifier}, model_path)

    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": round(target_1_percentage, 4),
        "dataset_input_time": round(dataset_input_time, 4),
        "training_time": round(training_time, 4),
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> None:
    """
    Apply a trained classifier to the reduced test dataset, saving per-record
    predictions and overall classification statistics.

    Parameters
    ----------
    reduced_Test_csv : str
        Path to the reduced test dataset (selected features + target).
    target_column : str
        Name of the target column.
    model_path : str
        Path to the saved trained model (joblib format).
    predictions_csv : str
        Path where per-record predictions will be saved (CSV format).
    classif_stats_json : str
        Path where classification statistics will be saved (JSON format).

    Raises
    ------
    ValueError
        If target_column is not found in the dataset.
    """
    df = pd.read_csv(reduced_Test_csv)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].to_numpy()
    y_true = df[target_column].to_numpy()

    saved_obj = joblib.load(model_path)
    model = saved_obj["model"]
    classifier_name = saved_obj["classifier_name"]

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    # --- Save per-record predictions ---
    predictions_df = pd.DataFrame(
        {
            "row_n": np.arange(len(df)),
            "target": y_true,
            "prediction": y_pred,
            "score": y_score,
        }
    )
    predictions_df.to_csv(predictions_csv, index=False)

    # --- Compute classification statistics ---
    n_samples = len(df)
    target_1_count = int((y_true == 1).sum())
    target_1_percentage = float((target_1_count / n_samples) * 100) if n_samples else 0.0

    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        # roc_auc_score fails if y_true contains only one class
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])    

    stats: Dict[str, Any] = {
        "classifier": classifier_name,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 4),
        "accuracy": accuracy,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
    }

    with open(classif_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


def main() -> None:
    """Command-line interface for train and predict, via subcommands."""
    parser = argparse.ArgumentParser(
        description="Train a classifier or run predictions for the QUBO "
        "feature selection project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train a classifier.")
    train_parser.add_argument("--classifier", required=True)
    train_parser.add_argument("--in-reduced", required=True, dest="in_reduced")
    train_parser.add_argument("--target", required=True)
    train_parser.add_argument("--out-model", required=True, dest="out_model")
    train_parser.add_argument("--out-metrics", required=True, dest="out_metrics")
    train_parser.add_argument("--seed", type=int, default=42)

    # --- predict subcommand ---
    predict_parser = subparsers.add_parser(
        "predict", help="Run predictions with a trained classifier."
    )
    predict_parser.add_argument(
        "--input-testset", required=True, dest="input_testset"
    )
    predict_parser.add_argument("--target", required=True)
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument(
        "--out-predictions", required=True, dest="out_predictions"
    )
    predict_parser.add_argument("--out-stats", required=True, dest="out_stats")

    args = parser.parse_args()

    try:
        if args.command == "train":
            train(
                classifier=args.classifier,
                reducedTrain_csv=args.in_reduced,
                target_column=args.target,
                model_path=args.out_model,
                metrics_json=args.out_metrics,
                seed=args.seed,
            )
            print(f"Model saved to: {args.out_model}")
            print(f"Training metrics saved to: {args.out_metrics}")

        elif args.command == "predict":
            predict(
                reduced_Test_csv=args.input_testset,
                target_column=args.target,
                model_path=args.model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats,
            )
            print(f"Predictions saved to: {args.out_predictions}")
            print(f"Classification statistics saved to: {args.out_stats}")

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()