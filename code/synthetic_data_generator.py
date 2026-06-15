# -*- coding: utf-8 -*-
"""
M15 Benchmark — Synthetic Adversarial Scenario Generator
=========================================================
Generates 10 adversarial scenarios (A-H) for benchmarking the M15 protocol.

Each scenario introduces a specific class of distortion to test the
sensitivity of different audit layers:
  A_CLEAN: Faithful spring-layout projection (control)
  B1_LOW_NOISE: Minor metric deformation
  B2_MEDIUM_NOISE: Material metric deformation
  B3_HIGH_NOISE: Severe metric deformation
  C_RANDOM: Random 2D layout (chaos control)
  D_BRIDGE_COLLAPSE: Topological bridge collapse (clusters merge visually)
  E_FALSE_OUTLIER: Single node displaced far from its community
  F_REAL_FISSURE: Semantic profile swapped for one node
  G_CLUSTER_SWAP: Two communities swap 2D positions
  H_SPECTRAL_REWIRE: Inter-community edges rewired

Reference:
  Abella, R. & Picón, J. (2026). M15 Paper, Section 4.
"""

import numpy as np
import networkx as nx
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform


def build_adversarial_scenarios(n_nodes=100, seed=42):
    """
    Generate a base 8D latent space and 10 adversarial 2D projections.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (default: 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_8D : np.ndarray of shape (n_nodes, 8)
        Original 8D latent representations.
    layouts : dict
        Maps scenario name -> 2D coordinates (or tuple for special cases).
    cluster_ids : np.ndarray of shape (n_nodes,)
        Cluster assignment for each node.
    names : list of str
        Node identifiers.
    """
    rng = np.random.default_rng(seed)
    n_half = n_nodes // 2

    # Two clusters in 8D with strong separation in dimension 0
    C1_8D = rng.normal(loc=10.0, scale=1.5, size=(n_half, 8))
    C1_8D[:, 0] = rng.normal(loc=5.0, scale=0.5, size=n_half)

    C2_8D = rng.normal(loc=10.0, scale=1.5, size=(n_nodes - n_half, 8))
    C2_8D[:, 0] = rng.normal(loc=25.0, scale=0.5, size=(n_nodes - n_half))

    X_8D = np.vstack([C1_8D, C2_8D])
    cluster_ids = np.array([0] * n_half + [1] * (n_nodes - n_half))

    # KNN graph in 8D (k=5)
    D = squareform(pdist(X_8D, 'euclidean'))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        sorted_idx = np.argsort(D[i])[1:6]
        for j in sorted_idx:
            G.add_edge(i, j, weight=1.0 - D[i, j] / D.max())

    names = [f"Actor_{i:03d}" for i in range(n_nodes)]
    layouts = {}

    # 1. Scenario A: Clean Projection
    pos_clean = nx.spring_layout(G, iterations=180, seed=seed)
    X_2D_A = np.array([pos_clean[i] for i in range(n_nodes)])
    layouts["A_CLEAN"] = X_2D_A

    # 2. Scenarios B: Progressive Metric Noise
    layouts["B1_LOW_NOISE"] = X_2D_A + rng.normal(loc=0.0, scale=0.10, size=X_2D_A.shape)
    layouts["B2_MEDIUM_NOISE"] = X_2D_A + rng.normal(loc=0.0, scale=0.35, size=X_2D_A.shape)
    layouts["B3_HIGH_NOISE"] = X_2D_A + rng.normal(loc=0.0, scale=0.75, size=X_2D_A.shape)

    # 3. Scenario C: Random Projection
    layouts["C_RANDOM"] = rng.uniform(low=-1.0, high=1.0, size=X_2D_A.shape)

    # 4. Scenario D: Bridge Collapse
    X_2D_D = X_2D_A.copy()
    c1_center = X_2D_D[cluster_ids == 0].mean(axis=0)
    c2_center = X_2D_D[cluster_ids == 1].mean(axis=0)
    shift_vec = c1_center - c2_center
    X_2D_D[cluster_ids == 1] += shift_vec + rng.normal(loc=0.0, scale=0.05, size=X_2D_D[cluster_ids == 1].shape)
    layouts["D_BRIDGE_COLLAPSE"] = X_2D_D

    # 5. Scenario E: False Outlier
    X_2D_E = X_2D_A.copy()
    target_node = n_half // 2
    X_2D_E[target_node] = np.array([5.0, 5.0])
    layouts["E_FALSE_OUTLIER"] = X_2D_E

    # 6. Scenario F: Real Fissure (semantic swap)
    X_8D_F = X_8D.copy()
    X_8D_F[0] = C2_8D[0]
    layouts["F_REAL_FISSURE"] = (X_2D_A, X_8D_F)

    # 7. Scenario G: Cluster Swap
    X_2D_G = X_2D_A.copy()
    c1_idx = (cluster_ids == 0)
    c2_idx = (cluster_ids == 1)
    c1_mean = X_2D_G[c1_idx].mean(axis=0)
    c2_mean = X_2D_G[c2_idx].mean(axis=0)
    X_2D_G[c1_idx] += (c2_mean - c1_mean)
    X_2D_G[c2_idx] += (c1_mean - c2_mean)
    # 15% of nodes stay trapped in original position (convergence failure)
    trap_idx = rng.choice(n_nodes, size=int(n_nodes * 0.15), replace=False)
    X_2D_G[trap_idx] = X_2D_A[trap_idx]
    layouts["G_CLUSTER_SWAP"] = X_2D_G

    # 8. Scenario H: Spectral Rewire
    G_rewired = G.copy()
    c1_nodes = [i for i in range(n_nodes) if cluster_ids[i] == 0]
    c2_nodes = [i for i in range(n_nodes) if cluster_ids[i] == 1]
    for _ in range(15):
        n1 = rng.choice(c1_nodes)
        n2 = rng.choice(c2_nodes)
        n1_neighbors = list(G_rewired.neighbors(n1))
        if n1_neighbors:
            G_rewired.remove_edge(n1, rng.choice(n1_neighbors))
        n2_neighbors = list(G_rewired.neighbors(n2))
        if n2_neighbors:
            G_rewired.remove_edge(n2, rng.choice(n2_neighbors))
        G_rewired.add_edge(n1, n2, weight=0.1)

    layouts["H_SPECTRAL_REWIRE"] = (X_2D_A, G_rewired)

    return X_8D, layouts, cluster_ids, names
