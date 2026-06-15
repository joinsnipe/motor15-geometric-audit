# M15 Ordinal Classification Pipeline — High-Level Pseudocode

## Overview

The M15 ordinal classifier assigns one of four labels to a 2D projection:

| Label | Meaning | Action Required |
|-------|---------|-----------------|
| **PASS** | Projection preserves geometric fidelity | Safe for interpretation |
| **WARN** | Minor deviations detected | Review with caution |
| **ALERT** | Significant distortions or local dissonance | Do not trust without investigation |
| **FAIL** | Projection is geometrically unfaithful | Discard or re-project |

---

## Layer 1: Global Metric Fidelity (Mantel Test)

```
INPUT:  X_high ∈ ℝ^{N×D}   (latent representations)
        X_low  ∈ ℝ^{N×2}   (2D projection)

1. D_high ← pairwise_euclidean_distance(X_high)
2. D_low  ← pairwise_euclidean_distance(X_low)
3. Normalize both matrices by their respective maxima
4. v_high ← upper_triangle(D_high_norm)
5. v_low  ← upper_triangle(D_low_norm)
6. r_M ← pearson_correlation(v_high, v_low)

OUTPUT: r_M ∈ [-1, 1]   (Mantel correlation coefficient)
```

**Interpretation**: Higher r_M indicates better global metric preservation. Values near 1.0 mean the projection faithfully preserves pairwise distances.

---

## Layer 2: Hierarchical Topological Stability (PH0 Profile)

```
INPUT:  D_high_norm, D_low_norm  (normalized distance matrices)

1. MST_high ← minimum_spanning_tree(D_high_norm)
2. MST_low  ← minimum_spanning_tree(D_low_norm)
3. deaths_high ← sorted_edge_weights(MST_high)
4. deaths_low  ← sorted_edge_weights(MST_low)
5. L ← min(len(deaths_high), len(deaths_low))

# Normalize by mean to remove scale bias
6. d8 ← deaths_high[:L] / mean(deaths_high[:L])
7. d2 ← deaths_low[:L]  / mean(deaths_low[:L])

# Three sub-metrics
8. w1_norm   ← sum(|d8 - d2|) / L                          # Wasserstein-1 normalized
9. mst_ratio ← sum(deaths_low[:L]) / sum(deaths_high[:L])  # Scale ratio
10. var_ratio ← |log(var(d2) / var(d8))|                    # Variance divergence

OUTPUT: ph0 = {w1_norm, mst_ratio, var_ratio}
```

**Interpretation**: These metrics capture whether the hierarchical skeleton (MST) is preserved. Large deviations in w1_norm or extreme mst_ratio values indicate topological distortion.

---

## Layer 3: Spectral Consistency (Heat Trace Divergence)

```
INPUT:  X_high, X_low, k (number of nearest neighbors)

1. L_high ← normalized_laplacian_knn(X_high, k)
2. L_low  ← normalized_laplacian_knn(X_low, k)
3. λ_high ← eigenvalues(L_high)
4. λ_low  ← eigenvalues(L_low)

5. t_times ← [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

6. FOR each t in t_times:
       trace_high[t] ← (1/N) * Σ exp(-t * λ_high_i)
       trace_low[t]  ← (1/N) * Σ exp(-t * λ_low_i)

7. D_HT ← sqrt(mean((trace_high - trace_low)²))

OUTPUT: D_HT ≥ 0   (Heat Trace Divergence)
```

**Interpretation**: D_HT measures whether the diffusion dynamics (information flow through the graph) are preserved in the projection. Higher values indicate spectral inconsistency.

---

## Layer 4: Local Semantic-Structural Dissonance (Fissure Index)

```
INPUT:  X_high ∈ ℝ^{N×D}, cluster_ids ∈ ℤ^N, K permutations

1. Compute regularized covariance and its inverse
2. Compute centroid for each cluster

3. FOR each node i:
       d_structural ← mahalanobis(node_i, centroid_of_its_cluster)
       d_semantic   ← min mahalanobis(node_i, centroid_of_other_clusters)
       δ_observed[i] ← d_structural - d_semantic

4. FOR p = 1 to K:
       Shuffle semantic profiles across nodes (preserving structure)
       Recompute centroids and deltas → δ_null[p]

5. FOR each node i:
       z[i] ← (δ_observed[i] - mean(δ_null[:, i])) / std(δ_null[:, i])

6. max_z ← max(z)

OUTPUT: max_z   (maximum positive Z-score across all nodes)
```

**Interpretation**: A high Z-score for a node indicates that its semantic profile (latent features) is inconsistent with the community it was assigned to structurally. Standard statistical thresholds (Z > 2 for moderate, Z > 3 for strong) apply.

---

## Ordinal Classification Logic

The four metrics (r_M, D_HT, ph0, max_z) are combined into an ordinal classification using a rule-based decision tree:

```
FUNCTION classify_m15(r_M, D_HT, ph0, max_z):

    # Step 1: Global classification
    #   Evaluate r_M, ph0.w1_norm, ph0.mst_ratio, and D_HT
    #   against calibrated thresholds for each severity level.
    #   Assign the most severe matching label.
    label ← evaluate_global_metrics(r_M, D_HT, ph0)
    #   Returns one of: PASS, WARN, ALERT, FAIL

    # Step 2: Local dissonance override
    #   If max_z exceeds standard statistical thresholds,
    #   escalate the label (never downgrade).
    label ← apply_local_override(label, max_z)

    RETURN label
```

> **Note**: The specific calibrated thresholds for each metric and severity level are part of the production implementation and are not included in this public release. See the paper (Section 4) for a qualitative description of the classification boundaries. The general principle is:
>
> - **FAIL**: At least one metric shows extreme deviation from fidelity
> - **ALERT**: Multiple metrics show moderate-to-significant deviation
> - **WARN**: Minor deviations detectable but within acceptable margins
> - **PASS**: All metrics within fidelity bounds
>
> The local dissonance layer (Fissure Index) acts as an **override** — it can only escalate the label, never downgrade it. This ensures that global fidelity alone cannot mask local anomalies.
