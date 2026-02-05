# src/split_data.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from preprocessing import (
    load_raw, basic_explore, add_features, preprocess_labels,
    drop_leakage_columns, select_features, ensure_dirs
)

RANDOM_SEED = 42

RAW_PATH = "data/dataset_pishing.csv"
OUT_DIR = "data/processed"


def split_train_val_test(df: pd.DataFrame, train=0.55, val=0.15, test=0.30, seed=RANDOM_SEED):
    assert abs((train + val + test) - 1.0) < 1e-9, "train+val+test debe sumar 1"
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


def main():
    print("== Cargando dataset ==")
    df = load_raw(RAW_PATH)
    basic_explore(df)

    print("\n== Features ==")
    df = add_features(df)
    num_features = df.shape[1] - 2
    print(f"Features creadas: {num_features}")

    print("\n== Preprocesando labels ==")
    df = preprocess_labels(df)
    print("Etiquetas convertidas a formato binario (0 = legitimate, 1 = phishing)")

    print("\n== Limpieza mínima ==")
    df = drop_leakage_columns(df)
    print("Columnas de leakage verificadas / no aplicables en este dataset")

    print("\n== Selección simple de features ==")
    df, feats = select_features(df, variance_threshold=0.0, corr_threshold=0.98)
    print(f"Features finales: {len(feats)}")
    print(feats)

    print("\n== Split train / val / test ==")
    print("train: 55%")
    print("val:   15%")
    print("test:  30%")
    train_df, val_df, test_df = split_train_val_test(df, 0.55, 0.15, 0.30)
    print(f"train size: {train_df.shape[0]}")
    print(f"val size:   {val_df.shape[0]}")
    print(f"test size:  {test_df.shape[0]}")

    ensure_dirs(OUT_DIR)
    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path = os.path.join(OUT_DIR, "val.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n== Export listo ==")
    print(train_path, train_df.shape)
    print(val_path, val_df.shape)
    print(test_path, test_df.shape)


if __name__ == "__main__":
    main()
