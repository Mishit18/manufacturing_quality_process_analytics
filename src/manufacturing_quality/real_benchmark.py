from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


DATASET_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
FEATURES = [
    "x_minimum", "x_maximum", "y_minimum", "y_maximum", "pixels_areas",
    "x_perimeter", "y_perimeter", "sum_luminosity", "min_luminosity",
    "max_luminosity", "conveyor_length", "steel_a300", "steel_a400",
    "plate_thickness", "edges_index", "empty_index", "square_index",
    "outside_x_index", "edges_x_index", "edges_y_index", "outside_global_index",
    "log_areas", "log_x_index", "log_y_index", "orientation_index",
    "luminosity_index", "sigmoid_areas",
]
FAULTS = [
    "pastry", "z_scratch", "k_scratch", "stains", "dirtiness", "bumps", "other_faults",
]


def fetch_steel_faults(timeout: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    response = requests.get(DATASET_URL, timeout=timeout)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        frame = pd.read_csv(
            archive.open("Faults.NNA"), sep=r"\s+", header=None, names=FEATURES + FAULTS
        )
    return frame[FEATURES], frame[FAULTS].idxmax(axis=1).rename("fault_type")


def run_steel_faults_benchmark(seed: int = 42) -> tuple[dict[str, float], pd.DataFrame]:
    features, target = fetch_steel_faults()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed, stratify=target
    )
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "records": float(len(features)),
        "classes": float(target.nunique()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
    importance = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return metrics, importance
