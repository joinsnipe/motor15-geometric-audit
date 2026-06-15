# M15 Ordinal Classification Pipeline — Pseudocode

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
3. D_high_norm ← D_high / max(D_high)
4. D_low_norm  ← D_low  / max(D_low)
5. v_high ← upper_triangle(D_high_norm)
6. v_low  ← upper_triangle(D_low_norm)
7. r_M ← pearson_correlation(v_high, v_low)

OUTPUT: r_M ∈ [-1, 1]   (Mantel correlation coefficient)
```

### Interpretation
- `r_M > 0.90`: Strong global metric preservation
- `r_M ∈ [0.75, 0.90]`: Moderate preservation
- `r_M < 0.50`: Severe distortion

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

### Interpretation
- `w1_norm > 1.0` → Severe skeleton deformation
- `mst_ratio < 0.08 or > 0.35` → Scale collapse/explosion
- `var_ratio > 2.0` → Heterogeneous distortion

---

## Layer 3: Spectral Consistency (Heat Trace Divergence)

```
INPUT:  X_high, X_low, k=5

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

### Interpretation
- `D_HT < 0.014` → Consistent diffusion dynamics
- `D_HT > 0.015` → Diffusion flow altered by projection

---

## Layer 4: Local Semantic-Structural Dissonance (Fissure Index)

```
INPUT:  X_high ∈ ℝ^{N×D}, cluster_ids ∈ ℤ^N, K permutations

1. Σ_reg ← covariance(X_high) + εI    (regularized)
2. Σ_inv ← inverse(Σ_reg)

3. FOR each cluster c:
       centroid[c] ← mean(X_high[cluster_ids == c])

4. FOR each node i:
       d_structural ← mahalanobis(X_high[i], centroid[cluster_ids[i]], Σ_inv)
       d_semantic   ← min over c ≠ cluster_ids[i]: mahalanobis(X_high[i], centroid[c], Σ_inv)
       δ_observed[i] ← d_structural - d_semantic

5. FOR p = 1 to K:
       X_shuffled ← permute_rows(X_high)
       centroids_shuffled ← recompute_centroids(X_shuffled, cluster_ids)
       δ_null[p] ← compute_deltas(X_shuffled, centroids_shuffled)

6. FOR each node i:
       z[i] ← (δ_observed[i] - mean(δ_null[:, i])) / std(δ_null[:, i])

7. max_z ← max(z)

OUTPUT: max_z   (maximum positive Z-score across all nodes)
```

### Interpretation
- `max_z < 2.0` → No local dissonance detected
- `max_z ∈ [2.0, 3.0]` → Moderate local anomaly (WARN override)
- `max_z > 3.0` → Strong local dissonance (ALERT override)

---

## Ordinal Classification Rules

```
FUNCTION classify_m15(r_M, D_HT, ph0, max_z):

    # Step 1: Global classification from metrics
    IF r_M < 0.50 OR
       ph0.mst_ratio > 0.35 OR ph0.mst_ratio < 0.08 OR
       ph0.w1_norm > 1.0 OR ph0.w1_norm < 0.20:
        label ← FAIL

    ELSE IF r_M < 0.75 OR
            ph0.mst_ratio > 0.26 OR ph0.mst_ratio < 0.13 OR
            ph0.w1_norm > 0.65 OR ph0.w1_norm < 0.28 OR
            D_HT > 0.015:
        label ← ALERT

    ELSE IF r_M < 0.90 OR
            ph0.mst_ratio > 0.22 OR ph0.mst_ratio < 0.15 OR
            ph0.w1_norm > 0.55 OR ph0.w1_norm < 0.35 OR
            D_HT > 0.014:
        label ← WARN

    ELSE:
        label ← PASS

    # Step 2: Local dissonance override
    IF max_z > 3.0 AND label ∈ {PASS, WARN}:
        label ← ALERT

    ELSE IF max_z > 2.0 AND label == PASS:
        label ← WARN

    RETURN label
```

---

## Threshold Summary Table

| Metric | PASS | WARN | ALERT | FAIL |
|--------|------|------|-------|------|
| **r_M** (Mantel) | ≥ 0.90 | [0.75, 0.90) | [0.50, 0.75) | < 0.50 |
| **w1_norm** (Wasserstein) | [0.35, 0.55] | (0.28, 0.35) ∪ (0.55, 0.65) | — | < 0.20 or > 1.0 |
| **mst_ratio** | [0.15, 0.22] | (0.13, 0.15) ∪ (0.22, 0.26) | — | < 0.08 or > 0.35 |
| **D_HT** (Heat Trace) | < 0.014 | [0.014, 0.015] | > 0.015 | — |
| **max_z** (Fissure) | < 2.0 | [2.0, 3.0) | ≥ 3.0 | — |

> **Note**: The local dissonance layer acts as an **override** — it can only escalate the label, never downgrade it. This ensures that global fidelity alone cannot mask local anomalies.
