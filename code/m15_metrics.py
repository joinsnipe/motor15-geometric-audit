# -*- coding: utf-8 -*-
"""
M15 Geometric Audit — Core Metrics Engine
==========================================
Post-hoc multiscale geometric fidelity audit for 2D projections of latent spaces.

Three complementary layers:
  1. Mantel Test (Global Metric Fidelity)
  2. PH0 Profile over MST (Hierarchical Topological Stability)
  3. Fissure Index (Local Semantic-Structural Dissonance)

Reference:
  Abella, R. & Picón, J. (2026). M15: A Multiscale Post-Hoc Protocol for
  Auditing the Geometric Fidelity of Latent Projections.
  DOI: 10.5281/zenodo.20700231
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

logger = logging.getLogger("M15.Metrics")


class M15GeometricAudit:
    """
    Multiscale geometric fidelity audit for 2D projections of latent spaces.

    Provides three complementary layers of analysis:
      - Global metric fidelity (Mantel Test)
      - Hierarchical topological stability (PH0 Profile / MST)
      - Local semantic-structural dissonance (Fissure Index)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PROJECTION ISOMORPHISM VERIFICATION (Mantel + Wasserstein + HKS)
    # ═══════════════════════════════════════════════════════════════════════════

    def verify_projection_isomorphism(
        self,
        X_high: Optional[np.ndarray],
        X_low: Optional[np.ndarray],
        permutations: int = 1000,
        sigma: float = 0.15,
        t_node: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluate whether a 2D projection preserves the geometric properties
        of the original high-dimensional latent space.

        Parameters
        ----------
        X_high : np.ndarray of shape (N, D)
            Original high-dimensional latent representations.
        X_low : np.ndarray of shape (N, 2)
            2D projection coordinates.
        permutations : int
            Number of Monte Carlo permutations for the Mantel test.
        sigma : float
            Bandwidth for the Gaussian kernel Laplacian (2D).
        t_node : float
            Diffusion time for node-level HKS comparison.

        Returns
        -------
        dict
            Contains Mantel r/p, Wasserstein-1 (0D persistence), and HKS correlations.
        """
        if X_high is None or X_low is None:
            return {"status": "SKIP", "reason": "Input arrays X_high or X_low are None."}

        X_high = np.asarray(X_high, dtype=np.float64)
        X_low = np.asarray(X_low, dtype=np.float64)

        if X_high.shape[0] != X_low.shape[0]:
            return {"status": "SKIP", "reason": f"Dimension mismatch: high={X_high.shape[0]} vs low={X_low.shape[0]}."}

        if X_high.shape[0] < 5:
            return {"status": "SKIP", "reason": f"Insufficient nodes (n={X_high.shape[0]} < 5)."}

        n = X_high.shape[0]
        results = {"status": "PASS", "n_nodes": n}

        try:
            # ── Distance Matrices ──
            D_high = squareform(pdist(X_high, 'euclidean'))
            D_high_norm = D_high / D_high.max() if D_high.max() > 0 else D_high
            D_low = squareform(pdist(X_low, 'euclidean'))
            D_low_norm = D_low / D_low.max() if D_low.max() > 0 else D_low

            # ── A. Mantel Test (Global Metric Isomorphism) ──
            tri_idx = np.triu_indices(n, k=1)
            v_high = D_high_norm[tri_idx]
            v_low = D_low_norm[tri_idx]

            r_observed, _ = stats.pearsonr(v_high, v_low)
            better_count = 0
            idx = np.arange(n)

            rng = np.random.default_rng(self.random_state)
            for _ in range(permutations):
                rng.shuffle(idx)
                D_low_shuffled = D_low_norm[idx, :][:, idx]
                r_sh, _ = stats.pearsonr(v_high, D_low_shuffled[tri_idx])
                if abs(r_sh) >= abs(r_observed):
                    better_count += 1

            p_val = better_count / permutations
            results["mantel"] = {
                "r_M": float(r_observed),
                "p_value": float(p_val),
                "permutations": permutations,
            }

            # ── B. MST Death Times L1 Distance (PH0 Approximation) ──
            def _compute_mst_death_times(D_matrix):
                mst = minimum_spanning_tree(D_matrix)
                death_times = mst.toarray()[mst.toarray() > 0]
                return np.sort(death_times)

            death_high = _compute_mst_death_times(D_high_norm)
            death_low = _compute_mst_death_times(D_low_norm)
            min_len = min(len(death_high), len(death_low))

            if min_len > 0:
                mst_l1_dist = np.sum(np.abs(death_high[:min_len] - death_low[:min_len])) / min_len
            else:
                mst_l1_dist = float('nan')

            results["mst_death_l1"] = {
                "l1_distance": float(mst_l1_dist),
                "n_edges": int(min_len)
            }

            # ── C. Heat Kernel Signatures (Spectral Consistency) ──
            def _compute_laplacian_knn(D, k=5):
                W = np.zeros((n, n))
                for i in range(n):
                    sorted_idx = np.argsort(D[i])
                    for m in range(1, min(k + 1, n)):
                        j = sorted_idx[m]
                        W[i, j] = 1.0 - D[i, j]
                        W[j, i] = W[i, j]
                deg = W.sum(axis=1)
                deg_inv_sqrt = np.zeros(n)
                deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
                D_inv_sqrt = np.diag(deg_inv_sqrt)
                return np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

            def _compute_laplacian_gaussian(D_norm, sig):
                W = np.exp(- (D_norm ** 2) / (2 * sig ** 2))
                np.fill_diagonal(W, 0)
                deg = W.sum(axis=1)
                deg_inv_sqrt = np.zeros(n)
                deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
                D_inv_sqrt = np.diag(deg_inv_sqrt)
                return np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

            L_high = _compute_laplacian_knn(D_high_norm, k=5)
            L_low = _compute_laplacian_gaussian(D_low_norm, sigma)

            evals_high, evecs_high = la.eigh(L_high)
            evals_low, evecs_low = la.eigh(L_low)

            times = np.linspace(0.1, 10.0, 100)
            hks_high_trace = np.array([np.sum(np.exp(-t * evals_high)) for t in times])
            hks_low_trace = np.array([np.sum(np.exp(-t * evals_low)) for t in times])
            global_hks_corr, _ = stats.pearsonr(hks_high_trace, hks_low_trace)

            hks_high_nodes = np.sum(np.exp(-t_node * evals_high) * (evecs_high ** 2), axis=1)
            hks_low_nodes = np.sum(np.exp(-t_node * evals_low) * (evecs_low ** 2), axis=1)
            node_hks_corr, node_hks_p = stats.pearsonr(hks_high_nodes, hks_low_nodes)

            results["hks"] = {
                "global_correlation": float(global_hks_corr),
                "node_correlation_t": float(node_hks_corr),
                "node_p_value": float(node_hks_p),
                "t": t_node,
                "sigma": sigma,
            }

        except Exception as e:
            logger.error("Error computing projection isomorphism: %s", str(e))
            return {"status": "ERROR", "error": str(e)}

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. LOCAL DISSONANCE DIAGNOSTIC (Fissure Index via Karrer-Newman Shuffle)
    # ═══════════════════════════════════════════════════════════════════════════

    def calculate_fissure_index(
        self,
        X_high: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        names: Optional[List[str]],
        permutations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Detect misalignments between a node's structural position (community)
        and its semantic profile in the latent space.

        Parameters
        ----------
        X_high : np.ndarray of shape (N, D)
            High-dimensional latent representations.
        cluster_ids : np.ndarray of shape (N,)
            Community/cluster assignment for each node.
        names : list of str
            Node identifiers.
        permutations : int
            Number of permutations for the null model.

        Returns
        -------
        dict
            Contains node-level Z-scores, p-values, and dissonance details.
        """
        if X_high is None or cluster_ids is None or names is None:
            return {"status": "SKIP", "reason": "Missing required inputs."}

        X_high = np.asarray(X_high, dtype=np.float64)
        cluster_ids = np.asarray(cluster_ids)
        n = X_high.shape[0]

        if n != len(cluster_ids) or n != len(names):
            return {"status": "SKIP", "reason": f"Size mismatch: X={n}, clusters={len(cluster_ids)}, names={len(names)}."}

        if n < 10:
            return {"status": "SKIP", "reason": f"Insufficient nodes (n={n} < 10)."}

        try:
            unique_clusters = np.unique(cluster_ids)
            if len(unique_clusters) < 2:
                return {"status": "SKIP", "reason": "All nodes belong to a single cluster."}

            # ── 1. Covariance and Mahalanobis ──
            cov_matrix = np.cov(X_high, rowvar=False)
            d_dim = X_high.shape[1] if len(X_high.shape) > 1 else 1
            cov_matrix_reg = cov_matrix + 1e-5 * np.eye(d_dim)
            inv_cov = la.inv(cov_matrix_reg)

            def _mahalanobis_dist(u, v):
                diff = u - v
                return np.sqrt(diff.T @ inv_cov @ diff)

            def _compute_centroids(X, c_ids):
                cents = {}
                for cid in unique_clusters:
                    mask = (c_ids == cid)
                    cents[cid] = X[mask].mean(axis=0) if mask.any() else X.mean(axis=0)
                return cents

            centroids = _compute_centroids(X_high, cluster_ids)

            # ── 2. Fissure computation function ──
            def _compute_fissures(X_map, cents):
                deltas = np.zeros(n)
                details = []

                for i in range(n):
                    x_i = X_map[i]
                    cid_str = cluster_ids[i]
                    C_str = cents[cid_str]

                    d_str = _mahalanobis_dist(x_i, C_str)

                    min_sem_d = float('inf')
                    cid_sem = cid_str
                    for cid in unique_clusters:
                        if cid == cid_str:
                            continue
                        d_alt = _mahalanobis_dist(x_i, cents[cid])
                        if d_alt < min_sem_d:
                            min_sem_d = d_alt
                            cid_sem = cid

                    delta_mah = d_str - min_sem_d
                    deltas[i] = delta_mah

                    details.append({
                        "name": names[i],
                        "structural_cluster": int(cid_str),
                        "semantic_cluster_match": int(cid_sem),
                        "delta_mahalanobis": float(delta_mah),
                    })

                return deltas, details

            observed_deltas, node_details = _compute_fissures(X_high, centroids)

            # ── 3. Null Model (Community-constrained semantic feature shuffle) ──
            all_null_deltas = np.zeros((permutations, n))
            shuffled_idx = np.arange(n)
            rng = np.random.default_rng(self.random_state)

            for p in range(permutations):
                rng.shuffle(shuffled_idx)
                X_shuffled = X_high[shuffled_idx]
                cents_shuffled = _compute_centroids(X_shuffled, cluster_ids)
                null_deltas, _ = _compute_fissures(X_shuffled, cents_shuffled)
                all_null_deltas[p] = null_deltas

            # ── 4. Z-score consolidation ──
            for i in range(n):
                node_nulls = all_null_deltas[:, i]
                null_mean = node_nulls.mean()
                null_std = node_nulls.std()
                if null_std == 0:
                    null_std = 1.0

                z_score = (observed_deltas[i] - null_mean) / null_std
                p_one_tailed = 1.0 - stats.norm.cdf(z_score)
                p_two_tailed = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

                node_details[i]["null_mean"] = float(null_mean)
                node_details[i]["null_std"] = float(null_std)
                node_details[i]["z_score"] = float(z_score)
                node_details[i]["p_value_one_tailed"] = float(p_one_tailed)
                node_details[i]["p_value_two_tailed"] = float(p_two_tailed)

            node_details.sort(key=lambda x: abs(x["z_score"]), reverse=True)

            return {
                "status": "PASS",
                "nodes": node_details,
            }

        except Exception as e:
            logger.error("Error calculating fissure index: %s", str(e))
            return {"status": "ERROR", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST WITH SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=== SMOKE TEST M15 ===")

    np.random.seed(42)
    n_samples = 50

    X_A = np.random.normal(loc=35.0, scale=8.0, size=(25, 8))
    X_B = np.random.normal(loc=75.0, scale=10.0, size=(25, 8))
    X_A[:, 5] = np.random.normal(loc=30.0, scale=5.0, size=25)
    X_B[:, 5] = np.random.normal(loc=85.0, scale=5.0, size=25)

    X_high = np.vstack([X_A, X_B])
    cluster_ids = np.array([0] * 25 + [1] * 25)
    names = [f"Actor_{i:02d}" for i in range(50)]

    X_low = X_high[:, [5, 7]] / 10.0 + np.random.normal(0, 0.5, size=(50, 2))

    engine = M15GeometricAudit()

    iso_res = engine.verify_projection_isomorphism(X_high, X_low, permutations=100)
    print("\n[+] Isomorphism Verification:")
    print(f"  Status: {iso_res['status']}")
    if iso_res['status'] == "PASS":
        print(f"  Mantel r_M: {iso_res['mantel']['r_M']:.4f} (p={iso_res['mantel']['p_value']:.4f})")
        print(f"  MST Death L1: {iso_res['mst_death_l1']['l1_distance']:.4f}")
        print(f"  HKS Global Corr: {iso_res['hks']['global_correlation']:.4f}")

    fiss_res = engine.calculate_fissure_index(X_high, cluster_ids, names, permutations=100)
    print("\n[+] Fissure Index:")
    print(f"  Status: {fiss_res['status']}")
    if fiss_res['status'] == "PASS":
        print("  Top 3 Anomalies:")
        for f in fiss_res['nodes'][:3]:
            print(f"    - {f['name']}: Z={f['z_score']:.3f}")

    print("\n=== SMOKE TEST COMPLETED ===")
