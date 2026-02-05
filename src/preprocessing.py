# src/preprocessing.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np

from features import extract_features


LABEL_MAP = {
    "legitimate": 0,
    "phishing": 1,
}


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def basic_explore(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())

    if "status" in df.columns:
        print("\nClass counts:")
        print(df["status"].value_counts(dropna=False))
        print("\nClass %:")
        print((df["status"].value_counts(normalize=True, dropna=False) * 100).round(2))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    assert "url" in df.columns, "El dataset debe tener columna 'url'"

    feats = df["url"].apply(extract_features)
    feat_df = pd.DataFrame(list(feats))

    out = pd.concat([df.copy(), feat_df], axis=1)
    return out

def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    assert "status" in df.columns, "El dataset debe tener columna 'status'"
    df = df.copy()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].map(LABEL_MAP)

    if df["status"].isna().any():
        unknown = df[df["status"].isna()]
        raise ValueError(f"Hay etiquetas desconocidas en 'status'. Ejemplos:\n{unknown.head()}")

    df["status"] = df["status"].astype(int)
    return df

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def select_features(df: pd.DataFrame, variance_threshold: float = 0.0, corr_threshold: float = 0.98) -> tuple[pd.DataFrame, list[str]]:
    # Devuelve df con features + status y lista final de features
    df = df.copy()

    # Features numéricas = todo menos url/status
    feature_cols = [c for c in df.columns if c not in ("url", "status")]

    X = df[feature_cols].copy()

    # varianza
    variances = X.var(numeric_only=True)
    keep_var = variances[variances > variance_threshold].index.tolist()
    X = X[keep_var]

    # correlacion alta
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] >= corr_threshold)]
    X = X.drop(columns=drop_cols, errors="ignore")

    final_features = X.columns.tolist()

    out = pd.concat([df[["url", "status"]], X], axis=1)
    return out, final_features

def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
