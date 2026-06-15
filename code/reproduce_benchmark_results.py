# -*- coding: utf-8 -*-
"""
M15 Benchmark — Reproduce Raw Metric Results
==============================================
Runs the full adversarial benchmark: 10 scenarios × N simulations.
Computes M15 raw metrics and classical metrics for each scenario.

NOTE: This script outputs RAW METRIC VALUES only. The ordinal
classification logic (PASS/WARN/ALERT/FAIL) with calibrated thresholds
is not included in this public release. See the paper (Section 4) for
a high-level description of the classification protocol.

Usage:
    python reproduce_benchmark_results.py [--simulations 100]

Reference:
  Abella, R. & Picón, J. (2026). M15 Paper, Section 4.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg as la
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans

# Import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m15_metrics import M15GeometricAudit
from synthetic_data_generator import build_adversarial_scenarios


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL PROJECTION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trustworthiness(X_high, X_low, k=5):
    n = X_high.shape[0]
    D_high = squareform(pdist(X_high, 'euclidean'))
    D_low = squareform(pdist(X_low, 'euclidean'))
    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)

    sum_val = 0.0
    for i in range(n):
        low_neighs = np.argsort(D_low[i])[1:k+1]
        high_neighs = np.argsort(D_high[i])[1:k+1]
        u_i = [j for j in low_neighs if j not in high_neighs]
        for j in u_i:
            sum_val += (ranks_high[i, j] - k)

    norm_factor = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - norm_factor * sum_val


def compute_continuity(X_high, X_low, k=5):
    n = X_high.shape[0]
    D_high = squareform(pdist(X_high, 'euclidean'))
    D_low = squareform(pdist(X_low, 'euclidean'))
    ranks_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    sum_val = 0.0
    for i in range(n):
        low_neighs = np.argsort(D_low[i])[1:k+1]
        high_neighs = np.argsort(D_high[i])[1:k+1]
        v_i = [j for j in high_neighs if j not in low_neighs]
        for j in v_i:
            sum_val += (ranks_low[i, j] - k)

    norm_factor = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - norm_factor * sum_val


def compute_stress(X_high, X_low):
    D_high = squareform(pdist(X_high, 'euclidean'))
    D_low = squareform(pdist(X_low, 'euclidean'))
    D_high_norm = D_high / D_high.max() if D_high.max() > 0 else D_high
    D_low_norm = D_low / D_low.max() if D_low.max() > 0 else D_low
    num = np.sum((D_high_norm - D_low_norm) ** 2)
    den = np.sum(D_high_norm ** 2)
    return np.sqrt(num / den) if den > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# M15 RAW METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mantel_r(X_high, X_low):
    """Fast Mantel correlation (no permutation test)."""
    D_high = squareform(pdist(X_high, 'euclidean'))
    D_low = squareform(pdist(X_low, 'euclidean'))
    tri_idx = np.triu_indices(X_high.shape[0], k=1)
    r, _ = stats.pearsonr(D_high[tri_idx], D_low[tri_idx])
    return float(r)


def build_normalized_laplacian_knn(X, k=5):
    """Build symmetric normalized Laplacian from a KNN graph."""
    n = X.shape[0]
    D = squareform(pdist(X, 'euclidean'))
    D_norm = D / D.max() if D.max() > 0 else D
    W = np.zeros((n, n))
    for i in range(n):
        sorted_idx = np.argsort(D_norm[i])
        for m in range(1, min(k + 1, n)):
            j = sorted_idx[m]
            W[i, j] = 1.0 - D_norm[i, j]
            W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    deg_inv_sqrt = np.zeros(n)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    L = np.eye(n) - np.diag(deg_inv_sqrt) @ W @ np.diag(deg_inv_sqrt)
    return L


def compute_heat_trace_divergence(X_high, X_low, k_neigh=5):
    """Spectral divergence via logarithmic heat traces on homogeneous KNN graphs."""
    n = X_high.shape[0]
    L_8D = build_normalized_laplacian_knn(X_high, k_neigh)
    L_2D = build_normalized_laplacian_knn(X_low, k_neigh)
    evals_8D = la.eigvalsh(L_8D)
    evals_2D = la.eigvalsh(L_2D)
    t_times = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    trace_8D = [np.sum(np.exp(-t * evals_8D)) / n for t in t_times]
    trace_2D = [np.sum(np.exp(-t * evals_2D)) / n for t in t_times]
    return float(np.sqrt(np.mean((np.array(trace_8D) - np.array(trace_2D)) ** 2)))


def compute_ph0_profile(X_high, X_low):
    """PH0 profile: normalized Wasserstein-1, MST ratio, and variance ratio."""
    D_high = squareform(pdist(X_high, 'euclidean'))
    D_low = squareform(pdist(X_low, 'euclidean'))
    D_high_norm = D_high / D_high.max() if D_high.max() > 0 else D_high
    D_low_norm = D_low / D_low.max() if D_low.max() > 0 else D_low

    def get_mst_deaths(D):
        mst = minimum_spanning_tree(D)
        deaths = mst.toarray()[mst.toarray() > 0]
        return np.sort(deaths)

    death_8D = get_mst_deaths(D_high_norm)
    death_2D = get_mst_deaths(D_low_norm)
    min_len = min(len(death_8D), len(death_2D))

    if min_len == 0:
        return {"w1_norm": 1.0, "mst_ratio": 0.0, "var_ratio": 5.0}

    d8 = death_8D[:min_len]
    d2 = death_2D[:min_len]

    mean_8D = np.mean(d8) if np.mean(d8) > 0 else 1.0
    mean_2D = np.mean(d2) if np.mean(d2) > 0 else 1.0

    d8_norm = d8 / mean_8D
    d2_norm = d2 / mean_2D

    w1_norm = np.sum(np.abs(d8_norm - d2_norm)) / min_len
    mst_ratio = np.sum(d2) / np.sum(d8) if np.sum(d8) > 0 else 0.0

    var_8D = np.var(d8_norm)
    var_2D = np.var(d2_norm)
    var_ratio = abs(np.log(var_2D / var_8D)) if var_8D > 0 and var_2D > 0 else 0.0

    return {
        "w1_norm": float(w1_norm),
        "mst_ratio": float(mst_ratio),
        "var_ratio": float(var_ratio)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK RUNNER (RAW METRICS ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_simulations=100):
    print(f"Starting M15 Benchmark with {n_simulations} simulations per scenario...")
    print("NOTE: This script outputs raw metric values. The ordinal classification")
    print("      (PASS/WARN/ALERT/FAIL) is described in the paper, Section 4.\n")
    start = time.time()

    engine = M15GeometricAudit()
    results = []

    scenarios = [
        "A_CLEAN", "B1_LOW_NOISE", "B2_MEDIUM_NOISE", "B3_HIGH_NOISE",
        "C_RANDOM", "D_BRIDGE_COLLAPSE", "E_FALSE_OUTLIER",
        "F_REAL_FISSURE", "G_CLUSTER_SWAP", "H_SPECTRAL_REWIRE"
    ]

    for seed in range(n_simulations):
        if (seed + 1) % 10 == 0:
            print(f"  Simulation {seed + 1}/{n_simulations}...")

        X_8D, layouts, cluster_ids, names = build_adversarial_scenarios(n_nodes=100, seed=seed)

        # Fissure Index for base scenarios
        fissure_base = engine.calculate_fissure_index(
            X_high=X_8D, cluster_ids=cluster_ids, names=names, permutations=100
        )
        z_base = max([n["z_score"] for n in fissure_base["nodes"]] + [0.0]) if fissure_base["status"] == "PASS" else 0.0

        # Fissure for F_REAL_FISSURE
        _, X_8D_F = layouts["F_REAL_FISSURE"]
        fissure_F = engine.calculate_fissure_index(
            X_high=X_8D_F, cluster_ids=cluster_ids, names=names, permutations=100
        )
        z_F = max([n["z_score"] for n in fissure_F["nodes"]] + [0.0]) if fissure_F["status"] == "PASS" else 0.0

        # Fissure for G_CLUSTER_SWAP (K-Means on 2D)
        X_2D_G = layouts["G_CLUSTER_SWAP"]
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_ids_G = kmeans.fit_predict(X_2D_G)
        fissure_G = engine.calculate_fissure_index(
            X_high=X_8D, cluster_ids=cluster_ids_G, names=names, permutations=100
        )
        z_G = max([n["z_score"] for n in fissure_G["nodes"]] + [0.0]) if fissure_G["status"] == "PASS" else 0.0

        for sc in scenarios:
            X_high = X_8D

            if sc == "F_REAL_FISSURE":
                X_low, X_high = layouts[sc]
                max_pos_z = z_F
            elif sc == "G_CLUSTER_SWAP":
                X_low = layouts[sc]
                max_pos_z = z_G
            elif sc == "H_SPECTRAL_REWIRE":
                X_low, G_rew = layouts[sc]
                max_pos_z = z_base
                L_H = nx.normalized_laplacian_matrix(G_rew).toarray()
                evals_H, evecs_H = la.eigh(L_H)
                X_high = evecs_H[:, 1:9]
                X_high = X_high * (np.std(X_8D) / np.std(X_high))
            else:
                X_low = layouts[sc]
                max_pos_z = z_base

            # Classical metrics
            trust = compute_trustworthiness(X_high, X_low, k=5)
            cont = compute_continuity(X_high, X_low, k=5)
            stress = compute_stress(X_high, X_low)

            # M15 raw metrics
            r_M = compute_mantel_r(X_high, X_low)
            hks_div = compute_heat_trace_divergence(X_high, X_low, k_neigh=5) if sc != "H_SPECTRAL_REWIRE" else _compute_htd_rewire(X_low, G_rew)
            ph0 = compute_ph0_profile(X_high, X_low)

            results.append({
                "simulation": seed,
                "scenario": sc,
                "r_M": r_M,
                "hks_div": hks_div,
                "w1_norm": ph0["w1_norm"],
                "mst_ratio": ph0["mst_ratio"],
                "var_ratio": ph0["var_ratio"],
                "max_abs_z": max_pos_z,
                "trustworthiness": trust,
                "continuity": cont,
                "stress": stress,
            })

    df = pd.DataFrame(results)

    # Save CSV
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "benchmark_reproduced.csv")
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - start
    print(f"\nBenchmark completed in {elapsed:.1f}s")
    print(f"Results saved to: {csv_path}")

    # Print summary (raw metrics only, no classification)
    print("\n" + "=" * 80)
    print("RAW METRIC SUMMARY (per scenario, averaged over simulations)")
    print("=" * 80)

    print(f"\n{'Scenario':<25} {'r_M':<10} {'D_HT':<10} {'W1_norm':<10} {'MST_ratio':<10} {'max_Z':<10}")
    print("-" * 75)
    for sc in scenarios:
        df_sc = df[df["scenario"] == sc]
        print(f"{sc:<25} {df_sc['r_M'].mean():.4f}    {df_sc['hks_div'].mean():.4f}    "
              f"{df_sc['w1_norm'].mean():.4f}    {df_sc['mst_ratio'].mean():.4f}    "
              f"{df_sc['max_abs_z'].mean():.2f}")

    print(f"\nTotal rows: {len(df)} ({n_simulations} simulations × {len(scenarios)} scenarios)")
    print("\nTo reproduce the classification results from the paper, apply the ordinal")
    print("classifier described in Section 4 to these raw metrics.")


def _compute_htd_rewire(X_low, G_rew):
    """Heat trace divergence for spectral rewire scenario."""
    n = X_low.shape[0]
    W_8D = nx.to_numpy_array(G_rew)
    deg_8D = W_8D.sum(axis=1)
    deg_8D_inv = np.zeros(n)
    deg_8D_inv[deg_8D > 0] = 1.0 / np.sqrt(deg_8D[deg_8D > 0])
    L_8D = np.eye(n) - np.diag(deg_8D_inv) @ W_8D @ np.diag(deg_8D_inv)
    L_2D = build_normalized_laplacian_knn(X_low, k=5)
    evals_8D = la.eigvalsh(L_8D)
    evals_2D = la.eigvalsh(L_2D)
    t_times = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    trace_8D = [np.sum(np.exp(-t * evals_8D)) / n for t in t_times]
    trace_2D = [np.sum(np.exp(-t * evals_2D)) / n for t in t_times]
    return float(np.sqrt(np.mean((np.array(trace_8D) - np.array(trace_2D)) ** 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce M15 Benchmark Raw Metrics")
    parser.add_argument("--simulations", type=int, default=100, help="Simulations per scenario (default: 100)")
    args = parser.parse_args()
    run_benchmark(n_simulations=args.simulations)
