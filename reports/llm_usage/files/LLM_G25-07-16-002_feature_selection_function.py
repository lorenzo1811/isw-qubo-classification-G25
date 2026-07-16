"""
feature_selection.py

Feature selection module based on a QUBO (Quadratic Unconstrained Binary
Optimization) formulation.

Provides the mandatory `select_features` function, which:
1. Builds a QUBO cost function balancing feature-target relevance (Spearman
   correlation) against feature-feature redundancy.
2. Searches over the alpha weighting parameter to select approximately
   `percSelected * n_features` features (+/- allowance).
3. Solves each QUBO instance using simulated annealing (dimod + neal).
4. Splits the dataset into training/test sets using only the selected features.

Can also be run from the command line, see `main()` below.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import dimod
import neal


def _build_qubo_matrix(
    rho_vj: np.ndarray, rho_jk: np.ndarray, alpha: float
) -> Dict[Tuple[int, int], float]:
    """
    Build the QUBO matrix (as an upper-triangular dict, dimod convention) for
    a given alpha.

    f(x) = -alpha * sum_j(x_j * rho_vj[j])
           + (1 - alpha) * sum_{j,k, k!=j} (x_j * x_k * rho_jk[j,k])

    Diagonal terms:    Q[j,j] = -alpha * rho_vj[j]
    Off-diagonal (j<k): Q[j,k] = 2 * (1 - alpha) * rho_jk[j,k]
    """
    n = len(rho_vj)
    Q: Dict[Tuple[int, int], float] = {}

    for j in range(n):
        Q[(j, j)] = -alpha * rho_vj[j]

    for j in range(n):
        for k in range(j + 1, n):
            coeff = 2.0 * (1.0 - alpha) * rho_jk[j, k]
            if coeff != 0.0:
                Q[(j, k)] = coeff

    return Q


def _solve_qubo(
    Q: Dict[Tuple[int, int], float], n: int, seed: int
) -> Tuple[np.ndarray, float]:
    """
    Solve a QUBO instance using simulated annealing (neal.SimulatedAnnealingSampler).

    Returns
    -------
    x_star : np.ndarray of shape (n,), dtype int
        Binary vector of the best solution found.
    energy : float
        The QUBO objective value (f(x)) at x_star.
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=20, seed=seed)
    best = sampleset.first

    x_star = np.zeros(n, dtype=int)
    for idx, val in best.sample.items():
        x_star[idx] = int(val)

    return x_star, float(best.energy)


def _compute_correlations(
    features_df: pd.DataFrame, target_series: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute absolute Spearman correlations:
    - rho_vj: correlation between each feature and the target (shape: n,)
    - rho_jk: correlation between each pair of features (shape: n, n)
    """
    n = features_df.shape[1]

    # Feature-feature Spearman correlation matrix (vectorized via pandas)
    corr_matrix = features_df.corr(method="spearman").to_numpy()
    rho_jk = np.abs(np.nan_to_num(corr_matrix, nan=0.0))
    np.fill_diagonal(rho_jk, 0.0)  # exclude self-correlation (k != j)

    # Feature-target Spearman correlation
    rho_vj = np.zeros(n)
    for i, col in enumerate(features_df.columns):
        corr = features_df[col].corr(target_series, method="spearman")
        rho_vj[i] = abs(corr) if not np.isnan(corr) else 0.0

    return rho_vj, rho_jk


def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100,
) -> None:
    """
    Select a subset of features via QUBO optimization, then split the reduced
    dataset into training and test sets.

    See module docstring for details. Raises ValueError if target_column is
    missing from the dataset.
    """
    df = pd.read_csv(normalized_csv)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    feature_columns: List[str] = [c for c in df.columns if c != target_column]
    n_features = len(feature_columns)
    features_df = df[feature_columns]
    target_series = df[target_column]

    target_k = round(percSelected * n_features)
    target_k = max(0, min(n_features, target_k))

    # --- Precompute correlations once (independent of alpha) ---
    t0 = time.perf_counter()
    rho_vj, rho_jk = _compute_correlations(features_df, target_series)
    q_matrix_creation_time = time.perf_counter() - t0

    # --- Alpha search (bounded binary search) ---
    attempts: List[Tuple[float, float, int, float]] = []  # alpha, time, n_sel, cost
    opt_times: List[float] = []

    best_result = None  # (alpha, x_star, n_selected, cost)
    best_diff = None

    lo, hi = 0.0, 1.0
    rng = np.random.default_rng(seed)

    for attempt_idx in range(alpha_computations):
        alpha = (lo + hi) / 2.0

        Q = _build_qubo_matrix(rho_vj, rho_jk, alpha)

        t_opt0 = time.perf_counter()
        x_star, energy = _solve_qubo(Q, n_features, seed=seed + attempt_idx)
        opt_time = time.perf_counter() - t_opt0
        opt_times.append(opt_time)

        n_selected = int(x_star.sum())
        attempts.append((alpha, opt_time, n_selected, energy))

        diff = abs(n_selected - target_k)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_result = (alpha, x_star, n_selected, energy)

        if diff <= allowance:
            break  # good enough, stop searching

        # Monotonic assumption: more alpha -> more features selected
        if n_selected < target_k:
            lo = alpha
        else:
            hi = alpha

    alpha_final, x_star, n_selected, cost_final = best_result

    # Sort attempts by increasing alpha for the output CSV
    attempts_sorted = sorted(attempts, key=lambda t: t[0])

    # --- Save optimizations CSV ---
    ottim_df = pd.DataFrame(
        attempts_sorted,
        columns=["alpha", "optimization_time", "n_selected_features", "cost_value"],
    )
    ottim_df.to_csv(output_ottim_csv, index=False)

    # --- Selected feature names ---
    selected_feature_names = [
        feature_columns[i] for i in range(n_features) if x_star[i] == 1
    ]

    # --- Split into train/test using selected features (hard cut at sample M) ---
    dataset_size = len(df)
    n_test = round(percTest * dataset_size)
    n_train = dataset_size - n_test

    reduced_columns = selected_feature_names + [target_column]
    reduced_df = df[reduced_columns]

    train_df = reduced_df.iloc[:n_train]
    test_df = reduced_df.iloc[n_train:]

    train_df.to_csv(reducedTrain_csv, index=False)
    test_df.to_csv(reducedTest_csv, index=False)

    # --- Save JSON report ---
    mean_opt_time = float(np.mean(opt_times)) if opt_times else 0.0
    std_opt_time = float(np.std(opt_times)) if opt_times else 0.0

    result = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": n_selected,
        "alpha": round(float(alpha_final), 6),
        "selected_vector": x_star.tolist(),
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(attempts),
        "percTest": percTest,
        "training_dataset_size": len(train_df),
        "test_dataset_size": len(test_df),
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean_optimization_time": round(mean_opt_time, 4),
        "std_dev_optimization_time": round(std_opt_time, 4),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


def main() -> None:
    """Command-line interface for select_features."""
    parser = argparse.ArgumentParser(
        description="Select features via QUBO optimization and split into "
        "train/test sets."
    )
    parser.add_argument("--in-normalized", required=True, dest="in_normalized")
    parser.add_argument("--out-train", required=True, dest="out_train")
    parser.add_argument("--out-test", required=True, dest="out_test")
    parser.add_argument(
        "--out-optimizations", required=True, dest="out_optimizations"
    )
    parser.add_argument("--out-json", required=True, dest="out_json")
    parser.add_argument("--target", required=True, dest="target")
    parser.add_argument(
        "--perc-selected", type=float, default=0.20, dest="perc_selected"
    )
    parser.add_argument("--allowance", type=int, default=1, dest="allowance")
    parser.add_argument("--perc-test", type=float, default=0.30, dest="perc_test")
    parser.add_argument("--seed", type=int, default=42, dest="seed")
    parser.add_argument(
        "--alpha-computations", type=int, default=100, dest="alpha_computations"
    )

    args = parser.parse_args()

    try:
        select_features(
            normalized_csv=args.in_normalized,
            reducedTrain_csv=args.out_train,
            reducedTest_csv=args.out_test,
            output_ottim_csv=args.out_optimizations,
            output_json=args.out_json,
            target_column=args.target,
            percTest=args.perc_test,
            percSelected=args.perc_selected,
            allowance=args.allowance,
            seed=args.seed,
            alpha_computations=args.alpha_computations,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Reduced training set written to: {args.out_train}")
    print(f"Reduced test set written to: {args.out_test}")
    print(f"Optimization attempts written to: {args.out_optimizations}")
    print(f"Feature selection report written to: {args.out_json}")


if __name__ == "__main__":
    main()