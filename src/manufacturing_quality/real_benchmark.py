from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
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

MODELED_FAILURE_WEIGHTS = {
    "pastry": 1.2,
    "z_scratch": 1.5,
    "k_scratch": 1.5,
    "stains": 1.1,
    "dirtiness": 1.0,
    "bumps": 1.3,
    "other_faults": 1.0,
}


def fetch_steel_faults(timeout: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    response = requests.get(DATASET_URL, timeout=timeout)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        frame = pd.read_csv(
            archive.open("Faults.NNA"), sep=r"\s+", header=None, names=FEATURES + FAULTS
        )
    return frame[FEATURES], frame[FAULTS].idxmax(axis=1).rename("fault_type")


def _fit_steel_faults(seed: int = 42):
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
    return features, target, x_test, y_test, model, predictions


def build_failure_priority(class_metrics: pd.DataFrame) -> pd.DataFrame:
    priority = class_metrics.copy()
    priority["modeled_failure_weight"] = priority["fault_type"].map(MODELED_FAILURE_WEIGHTS)
    priority["missed_cases"] = (priority["support"] * (1 - priority["recall"])).round().astype(int)
    priority["failure_priority_score"] = (
        priority["missed_cases"] * priority["modeled_failure_weight"]
    ).round(2)
    return priority.sort_values("failure_priority_score", ascending=False).reset_index(drop=True)


def run_steel_faults_evidence(
    seed: int = 42, shap_sample_size: int = 300
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, target, x_test, y_test, model, predictions = _fit_steel_faults(seed)
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

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame(
        [
            {
                "fault_type": fault,
                "precision": report[fault]["precision"],
                "recall": report[fault]["recall"],
                "f1_score": report[fault]["f1-score"],
                "support": int(report[fault]["support"]),
            }
            for fault in FAULTS
        ]
    )
    failure_priority = build_failure_priority(class_metrics)

    sample = x_test.sample(min(shap_sample_size, len(x_test)), random_state=seed)
    shap_values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    if shap_values.ndim == 3:
        if shap_values.shape[1] == len(FEATURES):
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"feature": FEATURES, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    return metrics, importance, class_metrics, failure_priority, shap_importance


def run_steel_faults_benchmark(seed: int = 42) -> tuple[dict[str, float], pd.DataFrame]:
    metrics, importance, _, _, _ = run_steel_faults_evidence(seed)
    return metrics, importance
