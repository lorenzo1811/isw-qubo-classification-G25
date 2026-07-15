"""
inspect_and_sample.py

Utility script to:
1. Inspect a CSV dataset (target column, missing/zero value statistics, class balance).
2. Generate a small stratified sample dataset for automated testing.

Usage:
    python inspect_and_sample.py \
        --input data/input_dataset.csv \
        --target target \
        --out-sample data/sample_test_dataset.csv \
        --sample-size 300 \
        --min-minority-perc 0.10 \
        --seed 42
"""

import argparse
import sys

import pandas as pd


def inspect_dataset(df: pd.DataFrame, target_column: str) -> None:
    """Print basic statistics about the dataset and per-column validity."""
    n_rows, n_cols = df.shape
    print(f"Dataset shape: {n_rows} rows, {n_cols} columns")

    if target_column not in df.columns:
        print(f"WARNING: target column '{target_column}' not found in dataset.")
        return

    target_counts = df[target_column].value_counts(dropna=False)
    print("\nTarget class distribution:")
    print(target_counts)
    print("\nTarget class percentage:")
    print((target_counts / n_rows * 100).round(2))

    print("\nPer-column validity (missing % / zero % / combined invalid %):")
    feature_cols = [c for c in df.columns if c != target_column]
    stats = []
    for col in feature_cols:
        missing_perc = df[col].isna().mean() * 100
        zero_perc = (df[col] == 0).mean() * 100
        invalid_perc = ((df[col].isna()) | (df[col] == 0)).mean() * 100
        stats.append((col, missing_perc, zero_perc, invalid_perc))

    stats_df = pd.DataFrame(
        stats, columns=["column", "missing_perc", "zero_perc", "invalid_perc"]
    ).sort_values("invalid_perc", ascending=False)

    with pd.option_context("display.max_rows", None):
        print(stats_df.round(2).to_string(index=False))

    n_high_invalid = (stats_df["invalid_perc"] >= 90).sum()
    print(
        f"\nColumns with >= 90% invalid (missing or zero) values: {n_high_invalid} "
        f"out of {len(feature_cols)}"
    )


def create_stratified_sample(
    df: pd.DataFrame,
    target_column: str,
    sample_size: int,
    min_minority_perc: float,
    seed: int,
) -> pd.DataFrame:
    """
    Create a small stratified sample of the dataset, ensuring the minority class
    is represented with at least `min_minority_perc` of the total sample size.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    n_minority = max(int(sample_size * min_minority_perc), 1)
    n_majority = sample_size - n_minority

    minority_df = df[df[target_column] == minority_class]
    majority_df = df[df[target_column] == majority_class]

    if len(minority_df) < n_minority:
        print(
            f"WARNING: only {len(minority_df)} minority samples available, "
            f"fewer than requested {n_minority}. Using all available."
        )
        n_minority = len(minority_df)
        n_majority = sample_size - n_minority

    minority_sample = minority_df.sample(n=n_minority, random_state=seed)
    majority_sample = majority_df.sample(
        n=min(n_majority, len(majority_df)), random_state=seed
    )

    sample_df = pd.concat([minority_sample, majority_sample]).sample(
        frac=1, random_state=seed
    )  # shuffle rows
    return sample_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a dataset and create a small stratified sample for testing."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument(
        "--out-sample",
        required=True,
        help="Path to write the sample CSV dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of rows in the sample dataset (default: 300).",
    )
    parser.add_argument(
        "--min-minority-perc",
        type=float,
        default=0.10,
        help="Minimum fraction of minority class samples in the sample (default: 0.10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    inspect_dataset(df, args.target)

    sample_df = create_stratified_sample(
        df,
        target_column=args.target,
        sample_size=args.sample_size,
        min_minority_perc=args.min_minority_perc,
        seed=args.seed,
    )

    sample_df.to_csv(args.out_sample, index=False)
    print(f"\nSample dataset written to: {args.out_sample}")
    print(f"Sample shape: {sample_df.shape}")
    print("Sample target distribution:")
    print(sample_df[args.target].value_counts())


if __name__ == "__main__":
    main()