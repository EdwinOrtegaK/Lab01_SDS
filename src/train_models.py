# src/train_models.py
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

DATA_DIR = "data/processed"
OUT_DIR = "reports/model_outputs"
RANDOM_SEED = 42

def to_jsonable(obj):
    """Convierte objetos con numpy/pandas a tipos serializables por JSON."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    return pd.read_csv(path)


def get_Xy(df: pd.DataFrame):
    # Excluir url + status
    feature_cols = [c for c in df.columns if c not in ("url", "status")]
    X = df[feature_cols].values
    y = df["status"].values.astype(int)
    return X, y, feature_cols


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_confusion(cm: np.ndarray, title: str, outpath: str):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["legit(0)", "phish(1)"])
    plt.yticks([0, 1], ["legit(0)", "phish(1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, outpath: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def evaluate(model, X, y):
    # Probabilidad de phishing (clase 1) para ROC/AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        # fallback: decision_function
        y_score = model.decision_function(X)

    y_pred = (y_score >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_score)

    return {
        "confusion_matrix": cm.tolist(),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
        "y_score": y_score,  # para ROC plot
        "y_pred": y_pred,
    }


def alarms_for_base_rate(metrics: dict, base_rate_phish: float, total_emails: int):
    """
    Usa TPR (= recall) y FPR para estimar alarmas en un escenario real.
    - base_rate_phish: proporción real de phishing (ej: 0.15)
    - total_emails: ej: 50000
    """
    cm = np.array(metrics["confusion_matrix"])
    # cm = [[TN, FP],
    #       [FN, TP]]
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # tasas en el set evaluado
    tpr = TP / (TP + FN) if (TP + FN) else 0.0  # recall
    fpr = FP / (FP + TN) if (FP + TN) else 0.0

    phish = int(round(total_emails * base_rate_phish))
    legit = total_emails - phish

    exp_TP = int(round(phish * tpr))
    exp_FN = phish - exp_TP
    exp_FP = int(round(legit * fpr))
    exp_TN = legit - exp_FP

    positives = exp_TP + exp_FP  # “alarmas”
    negatives = exp_TN + exp_FN  # no alarmas

    return {
        "base_rate_phish": base_rate_phish,
        "total_emails": total_emails,
        "expected_TP": exp_TP,
        "expected_FP": exp_FP,
        "expected_FN": exp_FN,
        "expected_TN": exp_TN,
        "alarms_positive": positives,
        "alarms_negative": negatives,
        "tpr_recall_used": float(tpr),
        "fpr_used": float(fpr),
    }


def main():
    ensure_dir(OUT_DIR)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train, feature_cols = get_Xy(train_df)
    X_val, y_val, _ = get_Xy(val_df)
    X_test, y_test, _ = get_Xy(test_df)

    # Modelo 1: Logistic Regression
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )

    # Modelo 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight=None,
    )

    models = {
        "logreg": lr,
        "random_forest": rf,
    }

    summary_lines = []
    all_results = {}

    for name, model in models.items():
        print(f"\n== Training: {name} ==")
        model.fit(X_train, y_train)

        # Evaluación val y test
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        all_results[name] = {"val": val_metrics, "test": test_metrics}

        # Plots
        plot_confusion(np.array(val_metrics["confusion_matrix"]),
                       f"{name} - Confusion Matrix (VAL)",
                       os.path.join(OUT_DIR, f"{name}_cm_val.png"))
        plot_confusion(np.array(test_metrics["confusion_matrix"]),
                       f"{name} - Confusion Matrix (TEST)",
                       os.path.join(OUT_DIR, f"{name}_cm_test.png"))

        plot_roc(y_val, val_metrics["y_score"],
                 f"{name} - ROC (VAL)",
                 os.path.join(OUT_DIR, f"{name}_roc_val.png"))
        plot_roc(y_test, test_metrics["y_score"],
                 f"{name} - ROC (TEST)",
                 os.path.join(OUT_DIR, f"{name}_roc_test.png"))

        # Resumen
        summary_lines.append(f"\nMODEL: {name}")
        summary_lines.append(f"VAL  precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} auc={val_metrics['auc']:.4f}")
        summary_lines.append(f"TEST precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} auc={test_metrics['auc']:.4f}")
        summary_lines.append(f"VAL  CM={val_metrics['confusion_matrix']}")
        summary_lines.append(f"TEST CM={test_metrics['confusion_matrix']}")

        # Escenario de empresa (BR=15%, 50,000 emails) usando métricas del TEST
        alarms = alarms_for_base_rate(test_metrics, base_rate_phish=0.15, total_emails=50000)
        all_results[name]["base_rate_scenario"] = alarms
        summary_lines.append(f"BR scenario (15%, 50k) alarms+={alarms['alarms_positive']} alarms-={alarms['alarms_negative']}, FP={alarms['expected_FP']}, TP={alarms['expected_TP']}")

    # Guardar outputs
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        compact = {}
        for mname, rr in all_results.items():
            compact[mname] = {
                "val": {k: v for k, v in rr["val"].items() if k != "y_score"},
                "test": {k: v for k, v in rr["test"].items() if k != "y_score"},
                "base_rate_scenario": rr["base_rate_scenario"],
            }
        json.dump(to_jsonable(compact), f, indent=2)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\nSaved outputs in:", OUT_DIR)
    print(" - summary.txt")
    print(" - results.json")
    print(" - *_cm_*.png and *_roc_*.png")


if __name__ == "__main__":
    main()
