from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_executive_summary(
    output_path: Path,
    *,
    n_rows: int,
    oee: pd.DataFrame,
    spc: pd.DataFrame,
    importances: pd.DataFrame,
    metrics: dict[str, float],
    recommendations: pd.DataFrame,
) -> None:
    worst = oee.iloc[0]
    robust_count = int((oee["oee"] >= 0.78).sum())
    breach_days = int(spc["spc_breach"].sum())
    top_features = ", ".join(importances.head(3)["feature"].tolist())
    annualized_scrap_opportunity = recommendations["scrap_cost_inr"].sum() * 6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""# Manufacturing Quality and Process Bottleneck Analytics

## Executive Summary

Built a manufacturing analytics case on **{n_rows:,} batch-level production records** covering line, machine, shift, material lot, process settings, downtime, output, defects, scrap, and rework.

## Key Findings

- Lowest-OEE asset: **{worst['line']} / {worst['machine']}** with OEE **{worst['oee']:.3f}** and defect rate **{worst['defect_rate']:.2%}**.
- SPC monitoring flagged **{breach_days} line-day control breaches**, giving operations a focused audit queue.
- Defect classifier reached ROC-AUC **{metrics['roc_auc']:.3f}** and PR-AUC **{metrics['pr_auc']:.3f}** on holdout data.
- Leading defect drivers by permutation importance: **{top_features}**.
- **{robust_count}** line-machine pairs crossed the 0.78 OEE threshold; the bottom assets need targeted maintenance and process control.
- Top-six bottlenecks represent an annualized scrap-cost opportunity proxy of roughly **INR {annualized_scrap_opportunity:,.0f}**.

## Recommended Operating Actions

1. Prioritize tool-age and vibration checks on bottleneck machines before adding new capacity.
2. Add SPC exception review to daily production meetings for lines with control-limit breaches.
3. Route high-risk batches to preventive maintenance review when defect probability crosses the 80th percentile.
4. Track OEE, defect rate, downtime, scrap cost, and rework minutes as a single plant scorecard.

## Resume-Safe Bullet

Built manufacturing quality analytics project on **52k** batch records; used **SPC**, **OEE**, bottleneck scoring, and defect-driver modeling to identify root causes and prioritize maintenance actions.
""",
        encoding="utf-8",
    )

