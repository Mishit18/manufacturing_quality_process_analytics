from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


DATASET_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
ANALOG_SIGNALS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]
FAILURE_WINDOWS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def _download_dataset(cache_dir: Path, timeout: int = 180) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "metropt3.zip"
    if not archive_path.exists():
        response = requests.get(DATASET_URL, timeout=timeout)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    return archive_path


def _load_dataset(cache_dir: Path) -> pd.DataFrame:
    archive_path = _download_dataset(cache_dir)
    with ZipFile(BytesIO(archive_path.read_bytes())) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        frame = pd.read_csv(archive.open(csv_name), low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _aggregate_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    available = [signal for signal in ANALOG_SIGNALS if signal in frame.columns]
    if len(available) != len(ANALOG_SIGNALS):
        missing = sorted(set(ANALOG_SIGNALS) - set(available))
        raise ValueError(f"MetroPT-3 is missing expected signals: {missing}")

    indexed = frame.set_index("timestamp")[available]
    means = indexed.resample("10min").mean().add_suffix("_mean")
    standard_deviations = indexed.resample("10min").std().fillna(0).add_suffix("_std")
    features = means.join(standard_deviations).dropna()

    failure_id = np.zeros(len(features), dtype=int)
    for event_id, (start, end) in enumerate(FAILURE_WINDOWS, start=1):
        inside = features.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        failure_id[inside.to_numpy()] = event_id
    features["failure_id"] = failure_id
    return features


def run_metropt_benchmark(
    output_dir: Path,
    cache_dir: Path,
    seed: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate chronological anomaly monitoring on real MetroPT-3 sensor data."""
    raw = _load_dataset(cache_dir)
    features = _aggregate_sensor_features(raw)
    feature_columns = [column for column in features.columns if column != "failure_id"]

    # UCI recommends the first month for training; failures occur later in the record.
    training_mask = features.index < pd.Timestamp("2020-03-01")
    scaler = RobustScaler().fit(features.loc[training_mask, feature_columns])
    training_values = scaler.transform(features.loc[training_mask, feature_columns])
    all_values = scaler.transform(features[feature_columns])

    model = IsolationForest(
        n_estimators=300,
        contamination=0.001,
        random_state=seed,
        n_jobs=-1,
    ).fit(training_values)
    training_scores = -model.score_samples(training_values)
    scores = -model.score_samples(all_values)
    threshold = float(np.quantile(training_scores, 0.999))

    evidence = features[["failure_id"]].copy()
    evidence["anomaly_score"] = scores
    evidence["alert"] = evidence["anomaly_score"] >= threshold
    failure_mask = evidence["failure_id"] > 0
    detected_events = int(
        evidence.loc[failure_mask].groupby("failure_id")["alert"].any().sum()
    )
    event_count = len(FAILURE_WINDOWS)
    metrics = {
        "raw_records": float(len(raw)),
        "signals": float(len(ANALOG_SIGNALS)),
        "ten_minute_windows": float(len(features)),
        "reported_failure_events": float(event_count),
        "detected_failure_events": float(detected_events),
        "event_recall": float(detected_events / event_count),
        "failure_window_alert_rate": float(evidence.loc[failure_mask, "alert"].mean()),
        "normal_window_alert_rate": float(evidence.loc[~failure_mask, "alert"].mean()),
        "threshold": threshold,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "metropt_real_sensor_metrics.csv", index=False)
    evidence.reset_index().to_csv(output_dir / "metropt_anomaly_timeline.csv", index=False)
    return metrics, evidence
