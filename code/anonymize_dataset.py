# -*- coding: utf-8 -*-
"""
M15 — Dataset Anonymization Utility
=====================================
Removes personally identifiable information (PII) from the empirical
case study dataset while preserving all numeric columns needed for
reproducing M15 metrics (Mantel, PH0, Fissure Index).

Usage:
    python anonymize_dataset.py --input path/to/nominal.csv --output path/to/anonymized.csv

Columns removed: name, role, Partido (contain real actor identities)
Columns preserved: all numeric metrics (cluster, network, dimensional scores)
"""

import os
import argparse
import pandas as pd


# Columns that contain personally identifiable information
PII_COLUMNS = ["name", "role", "Partido"]

# Columns to preserve (all numeric metrics)
METRIC_COLUMNS = [
    "clusterId", "clusterCohesion", "degree", "betweenness", "pagerank",
    "bridgeRatio_conductance", "bridgeScore", "stability", "certainty",
    "raw_CRED", "raw_CAP", "raw_LEAD", "raw_PROX",
    "raw_INTG", "raw_POL", "raw_COH", "raw_SAL",
    "norm_CRED", "norm_CAP", "norm_LEAD", "norm_PROX",
    "norm_INTG", "norm_POL", "norm_COH", "norm_SAL",
]


def anonymize_dataset(input_path: str, output_path: str) -> None:
    """
    Read a nominal CSV, strip PII columns, add anonymous IDs, and save.

    Parameters
    ----------
    input_path : str
        Path to the original CSV with nominal data.
    output_path : str
        Path where the anonymized CSV will be saved.
    """
    df = pd.read_csv(input_path)
    print(f"[+] Loaded {len(df)} rows from {input_path}")
    print(f"    Columns: {list(df.columns)}")

    # Verify expected columns exist
    available_metrics = [c for c in METRIC_COLUMNS if c in df.columns]
    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing:
        print(f"[!] Warning: Missing expected columns: {missing}")

    # Build anonymized dataframe
    df_anon = pd.DataFrame()
    df_anon["actor_id"] = [f"actor_{i:03d}" for i in range(len(df))]

    for col in available_metrics:
        df_anon[col] = df[col].values

    # Verify no PII leakage
    for col in PII_COLUMNS:
        if col in df_anon.columns:
            raise RuntimeError(f"PII column '{col}' found in anonymized output!")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_anon.to_csv(output_path, index=False)
    print(f"[+] Anonymized dataset saved to {output_path}")
    print(f"    Shape: {df_anon.shape}")
    print(f"    Columns: {list(df_anon.columns)}")

    # Final PII scan
    text_content = df_anon.to_string()
    pii_keywords = ["Abascal", "Sánchez", "Sanchez", "Díaz", "Diaz", "Feijóo",
                     "Feijoo", "Iglesias", "Errejón", "Errejon", "VOX", "PSOE",
                     "Sumar", "SUMAR", "Podemos"]
    for kw in pii_keywords:
        if kw.lower() in text_content.lower():
            raise RuntimeError(f"PII keyword '{kw}' found in anonymized output!")

    print("[+] PII scan passed — no identifiable information detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize M15 empirical dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to nominal CSV")
    parser.add_argument("--output", type=str, default=None, help="Path for anonymized CSV")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset_anonymized.csv"
        )

    anonymize_dataset(args.input, args.output)
