"""
preprocessing.py

Preprocessing module for the QUBO-based feature selection project.

Provides the mandatory `fit_normalize` function, which:
1. Reads a numeric CSV dataset with a target column.
2. Drops feature columns with too few valid (non-zero, non-missing) values.
3. Normalizes the remaining features using z-score standardization.
4. Saves the normalized dataset and a JSON report of the operation.

Can also be run from the command line, see `main()` below.
"""

import argparse
import json
import sys
import time
from typing import List

import numpy as np
import pandas as pd


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05,
) -> None:
    """
    Read a dataset, drop low-validity feature columns, normalize the remaining
    features with z-score standardization, and save results.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file. First row must contain column headers.
    target_column : str
        Name of the binary target column (0/1). Not assumed to have a fixed name.
    normalized_csv : str
        Path where the normalized dataset (features + unchanged target) will be
        written.
    outInitalRes_json : str
        Path where the JSON report of this preprocessing step will be written.
    minPercValid : float, optional
        Minimum fraction (0-1) of valid (non-NaN, non-zero) values a feature
        column must have to be kept. Default is 0.05 (5%).

    Raises
    ------
    ValueError
        If target_column is not found in the dataset.
    """
    # --- 1. Read dataset, measuring input time ---
    t0 = time.perf_counter()
    df = pd.read_csv(input_csv)
    dataset_input_time = time.perf_counter() - t0

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    t1 = time.perf_counter()

    dataset_size = len(df)
    feature_columns: List[str] = [c for c in df.columns if c != target_column]
    n_input_features = len(feature_columns)

    # --- 2. Identify columns to drop based on validity threshold ---
    dropped_feature_names: List[str] = []
    kept_feature_columns: List[str] = []

    for col in feature_columns:
        col_values = df[col]
        valid_mask = col_values.notna() & (col_values != 0)
        valid_perc = valid_mask.mean() if dataset_size > 0 else 0.0

        if valid_perc < minPercValid:
            dropped_feature_names.append(col)
        else:
            kept_feature_columns.append(col)

    n_kept_features = len(kept_feature_columns)

    # --- 3. Normalize remaining features with z-score standardization ---
    features_df = df[kept_feature_columns].astype(float)

    means = features_df.mean(axis=0)
    stds = features_df.std(axis=0, ddof=0)

    # Avoid division by zero for constant columns: leave them at 0 after
    # centering, rather than producing NaN/inf.
    safe_stds = stds.replace(0, np.nan)
    normalized_features = (features_df - means) / safe_stds
    normalized_features = normalized_features.fillna(0.0)

    # --- 4. Reassemble target column (normalized to int 0/1, unchanged values) ---
    target_series = df[target_column]
    target_series = target_series.round().astype(int)

    normalized_df = normalized_features.copy()
    normalized_df[target_column] = target_series

    # Preserve original column order (kept features in original order, target
    # in its original relative position is not strictly required by the spec;
    # we place kept features first, then target, matching the CSV structure
    # described in the assignment).
    normalized_df = normalized_df[kept_feature_columns + [target_column]]

    dataset_processing_time = time.perf_counter() - t1

    # --- Save normalized dataset ---
    normalized_df.to_csv(normalized_csv, index=False)

    # --- Save JSON report ---
    result = {
        "n_input_features": n_input_features,
        "n_kept_features": n_kept_features,
        "dataset_size": dataset_size,
        "dataset_input_time": round(dataset_input_time, 4),
        "dataset_processing_time": round(dataset_processing_time, 4),
        "dropped_feature_names": dropped_feature_names,
    }

    with open(outInitalRes_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


def main() -> None:
    """Command-line interface for fit_normalize."""
    parser = argparse.ArgumentParser(
        description="Preprocess a numeric CSV dataset: drop low-validity "
        "columns and apply z-score normalization."
    )
    parser.add_argument("--input", required=True, help="Input dataset CSV path.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--out-data", required=True, help="Output path for normalized dataset CSV."
    )
    parser.add_argument(
        "--out-json", required=True, help="Output path for the statistics JSON file."
    )
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        help="Minimum fraction of valid (non-zero, non-missing) values required "
        "to keep a column (default: 0.05).",
    )

    args = parser.parse_args()

    try:
        fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Normalized dataset written to: {args.out_data}")
    print(f"Preprocessing report written to: {args.out_json}")


if __name__ == "__main__":
    main()