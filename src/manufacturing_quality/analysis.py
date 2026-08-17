from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def compute_oee(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["line", "machine"], as_index=False).agg(
        planned_runtime_min=("planned_runtime_min", "sum"),
        actual_runtime_min=("actual_runtime_min", "sum"),
        planned_units=("planned_units", "sum"),
        good_units=("good_units", "sum"),
        defect_units=("defect_units", "sum"),
        target_cycle_time_sec=("target_cycle_time_sec", "mean"),
        actual_cycle_time_sec=("actual_cycle_time_sec", "mean"),
        unplanned_downtime_min=("unplanned_downtime_min", "sum"),
        scrap_cost_inr=("scrap_cost_inr", "sum"),
    )
    grouped["availability"] = grouped["actual_runtime_min"] / grouped["planned_runtime_min"]
    grouped["performance"] = (grouped["target_cycle_time_sec"] / grouped["actual_cycle_time_sec"]).clip(0, 1.15)
    grouped["quality"] = grouped["good_units"] / grouped["planned_units"]
    grouped["oee"] = grouped["availability"] * grouped["performance"] * grouped["quality"]
    grouped["defect_rate"] = grouped["defect_units"] / grouped["planned_units"]
    return grouped.sort_values("oee")


def detect_spc_breaches(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.assign(day=((df["batch_id"] - 1) // 240) + 1).groupby(["line", "day"], as_index=False).agg(
        defect_rate=("defect_rate", "mean"),
        downtime=("unplanned_downtime_min", "sum"),
        batches=("batch_id", "count"),
    )
    stats = daily.groupby("line").agg(mean_defect=("defect_rate", "mean"), std_defect=("defect_rate", "std")).reset_index()
    daily = daily.merge(stats, on="line")
    daily["ucl"] = daily["mean_defect"] + 3 * daily["std_defect"]
    daily["lcl"] = (daily["mean_defect"] - 3 * daily["std_defect"]).clip(lower=0)
    daily["spc_breach"] = (daily["defect_rate"] > daily["ucl"]) | (daily["defect_rate"] < daily["lcl"])
    return daily


def fit_defect_model(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict[str, float]]:
    modeling = df.copy()
    modeling["high_defect_batch"] = (modeling["defect_rate"] >= modeling["defect_rate"].quantile(0.80)).astype(int)
    features = [
        "operator_tenure_months",
        "target_cycle_time_sec",
        "actual_cycle_time_sec",
        "ambient_temp_c",
        "humidity_pct",
        "pressure_bar",
        "vibration_mm_s",
        "tool_age_hours",
        "unplanned_downtime_min",
    ]
    X = modeling[features]
    y = modeling["high_defect_batch"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=40, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=seed, n_jobs=-1)
    importances = pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    return importances, metrics


def identify_bottlenecks(oee: pd.DataFrame) -> pd.DataFrame:
    bottlenecks = oee.copy()
    bottlenecks["lost_units_proxy"] = (
        bottlenecks["planned_units"] * (1 - bottlenecks["availability"]).clip(0)
        + bottlenecks["defect_units"]
    )
    bottlenecks["bottleneck_score"] = (
        0.45 * (1 - bottlenecks["oee"])
        + 0.30 * bottlenecks["defect_rate"]
        + 0.25 * (bottlenecks["unplanned_downtime_min"] / bottlenecks["unplanned_downtime_min"].max())
    )
    return bottlenecks.sort_values("bottleneck_score", ascending=False)


def recommend_actions(bottlenecks: pd.DataFrame, spc: pd.DataFrame, importances: pd.DataFrame) -> pd.DataFrame:
    top = bottlenecks.head(6)[["line", "machine", "oee", "defect_rate", "unplanned_downtime_min", "scrap_cost_inr"]].copy()
    breach_counts = spc.groupby("line")["spc_breach"].sum().rename("spc_breach_days").reset_index()
    top = top.merge(breach_counts, on="line", how="left")
    leading = ", ".join(importances.head(3)["feature"].tolist())
    top["recommended_action"] = np.where(
        top["spc_breach_days"] > 0,
        "Prioritize SPC audit, tool-age checks, and shift-level root-cause review",
        "Move to preventive maintenance and operator-standard-work review",
    )
    top["primary_defect_drivers"] = leading
    return top

