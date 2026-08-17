from __future__ import annotations

import numpy as np
import pandas as pd


def build_manufacturing_dataset(n_batches: int = 52_000, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic, interview-defensible manufacturing dataset.

    The data is synthetic to avoid confidential plant data, but the columns mirror a
    real process-quality case: line, machine, shift, operator group, material lot,
    process settings, downtime, output, defects, scrap, and rework.
    """
    rng = np.random.default_rng(seed)

    line = rng.choice(["Line_A", "Line_B", "Line_C", "Line_D"], n_batches, p=[0.28, 0.24, 0.26, 0.22])
    machine = rng.choice([f"M{i:02d}" for i in range(1, 17)], n_batches)
    shift = rng.choice(["Morning", "Evening", "Night"], n_batches, p=[0.36, 0.34, 0.30])
    material_lot = rng.choice([f"L{i:03d}" for i in range(1, 121)], n_batches)
    operator_tenure_months = rng.gamma(shape=4.0, scale=6.0, size=n_batches).clip(1, 84)

    target_cycle_time = rng.normal(48, 4.5, n_batches)
    ambient_temp = rng.normal(31, 3.8, n_batches)
    humidity = rng.normal(52, 9.0, n_batches).clip(25, 85)
    pressure_bar = rng.normal(5.2, 0.42, n_batches)
    vibration = rng.normal(0.9, 0.22, n_batches)
    tool_age_hours = rng.gamma(shape=2.2, scale=18.0, size=n_batches).clip(1, 170)

    line_penalty = pd.Series(line).map({"Line_A": 0.00, "Line_B": 0.012, "Line_C": 0.025, "Line_D": 0.008}).to_numpy()
    shift_penalty = pd.Series(shift).map({"Morning": 0.00, "Evening": 0.006, "Night": 0.018}).to_numpy()
    machine_penalty = np.isin(machine, ["M04", "M09", "M13"]).astype(float) * 0.022
    material_penalty = np.isin(material_lot, ["L017", "L046", "L088", "L103"]).astype(float) * 0.025

    process_stress = (
        0.004 * np.maximum(ambient_temp - 34, 0)
        + 0.003 * np.maximum(humidity - 65, 0)
        + 0.018 * (pressure_bar < 4.7)
        + 0.014 * (vibration > 1.18)
        + 0.00038 * np.maximum(tool_age_hours - 55, 0)
    )
    learning_effect = -0.00024 * np.minimum(operator_tenure_months, 48)

    defect_prob = np.clip(0.018 + line_penalty + shift_penalty + machine_penalty + material_penalty + process_stress + learning_effect, 0.002, 0.22)
    planned_units = rng.integers(180, 420, n_batches)
    defects = rng.binomial(planned_units, defect_prob)
    good_units = planned_units - defects

    unplanned_downtime_min = rng.gamma(shape=1.6, scale=6.0, size=n_batches)
    unplanned_downtime_min += np.isin(machine, ["M04", "M09", "M13"]).astype(float) * rng.gamma(2.0, 5.0, n_batches)
    unplanned_downtime_min += (tool_age_hours > 65).astype(float) * rng.gamma(1.4, 4.0, n_batches)

    planned_runtime_min = rng.normal(420, 18, n_batches).clip(360, 470)
    actual_runtime_min = (planned_runtime_min - unplanned_downtime_min).clip(250, None)
    actual_cycle_time = target_cycle_time * (1 + rng.normal(0, 0.06, n_batches)) + 0.08 * unplanned_downtime_min

    scrap_cost = defects * rng.normal(42, 5, n_batches).clip(30, 58)
    rework_minutes = defects * rng.normal(0.55, 0.12, n_batches).clip(0.25, 0.9)

    df = pd.DataFrame(
        {
            "batch_id": np.arange(1, n_batches + 1),
            "line": line,
            "machine": machine,
            "shift": shift,
            "material_lot": material_lot,
            "operator_tenure_months": operator_tenure_months.round(1),
            "target_cycle_time_sec": target_cycle_time.round(2),
            "actual_cycle_time_sec": actual_cycle_time.round(2),
            "ambient_temp_c": ambient_temp.round(2),
            "humidity_pct": humidity.round(2),
            "pressure_bar": pressure_bar.round(3),
            "vibration_mm_s": vibration.round(3),
            "tool_age_hours": tool_age_hours.round(2),
            "planned_runtime_min": planned_runtime_min.round(2),
            "actual_runtime_min": actual_runtime_min.round(2),
            "unplanned_downtime_min": unplanned_downtime_min.round(2),
            "planned_units": planned_units,
            "good_units": good_units,
            "defect_units": defects,
            "scrap_cost_inr": scrap_cost.round(2),
            "rework_minutes": rework_minutes.round(2),
        }
    )
    df["defect_rate"] = (df["defect_units"] / df["planned_units"]).round(5)
    return df

