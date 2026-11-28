"""
Shared set-up for every model notebook (04 onwards).
Everyone imports from here, so all models use the same data, features and scores.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

TARGET = "is_canceled"

# features agreed by the team (arrival year / week / day are NOT used: they are time stamps)
NUMERIC = ["lead_time", "total_stay", "total_guests", "adr",
           "historical_cancellation_rate", "has_booking_history",
           "total_of_special_requests", "booking_changes",
           "days_in_waiting_list", "is_repeated_guest"]

CATEGORICAL = ["hotel", "deposit_type", "market_segment", "distribution_channel",
               "customer_type", "meal", "reserved_room_type",
               "arrival_date_month", "booking_month", "country_grouped"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_splits():
    """Load train and validation. The test set is NOT loaded here: it is used once, at the end."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    # country_grouped: the list of common countries is learned from TRAIN only (Lab 4)
    counts = train["country"].value_counts()
    keep = counts[counts >= 500].index
    for d in (train, val):
        d["country_grouped"] = d["country"].where(d["country"].isin(keep), "Other")
    return train, val


def make_prep(numeric, categorical, scaler=None):
    """Lab 5 preparation: scale numeric columns, one-hot encode categorical ones."""
    scaler = StandardScaler() if scaler is None else scaler
    return ColumnTransformer([
        ("num", scaler, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def evaluate(name, y_true, predicted, score=None):
    """Lab 5 confusion-matrix scores. `score` = predicted probability of cancelling (for ROC-AUC)."""
    y_true, predicted = np.asarray(y_true), np.asarray(predicted)
    TP = ((predicted == 1) & (y_true == 1)).sum()
    FP = ((predicted == 1) & (y_true == 0)).sum()
    FN = ((predicted == 0) & (y_true == 1)).sum()
    TN = ((predicted == 0) & (y_true == 0)).sum()
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "model": name,
        "accuracy": round(float((TP + TN) / len(y_true)), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 3) if score is not None else np.nan,
    }


def save_results(rows, file_name):
    """
    Save this notebook's validation scores to results/<file_name>.csv.
    One file per notebook, so teammates never overwrite each other (no git conflicts).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"{file_name}.csv", index=False)
    return table


def load_all_results():
    """Combine every notebook's results into one comparison table."""
    files = sorted(RESULTS_DIR.glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)