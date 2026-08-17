from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manufacturing_quality import (
    build_manufacturing_dataset,
    compute_oee,
    detect_spc_breaches,
    fit_defect_model,
    identify_bottlenecks,
    recommend_actions,
    run_steel_faults_benchmark,
)
from manufacturing_quality.reporting import write_executive_summary


OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = build_manufacturing_dataset()
    oee = compute_oee(df)
    spc = detect_spc_breaches(df)
    importances, metrics = fit_defect_model(df)
    bottlenecks = identify_bottlenecks(oee)
    recommendations = recommend_actions(bottlenecks, spc, importances)
    real_metrics, real_importance = run_steel_faults_benchmark()

    df.head(5000).to_csv(OUTPUTS / "manufacturing_sample.csv", index=False)
    oee.to_csv(OUTPUTS / "oee_by_line_machine.csv", index=False)
    spc.to_csv(OUTPUTS / "spc_control_chart_data.csv", index=False)
    importances.to_csv(OUTPUTS / "defect_driver_importance.csv", index=False)
    bottlenecks.to_csv(OUTPUTS / "bottleneck_rankings.csv", index=False)
    recommendations.to_csv(OUTPUTS / "recommended_actions.csv", index=False)
    pd.DataFrame([real_metrics]).to_csv(OUTPUTS / "uci_steel_faults_metrics.csv", index=False)
    real_importance.to_csv(OUTPUTS / "uci_steel_faults_importance.csv", index=False)
    write_executive_summary(
        REPORTS / "executive_summary.md",
        n_rows=len(df),
        oee=oee,
        spc=spc,
        importances=importances,
        metrics=metrics,
        recommendations=recommendations,
    )
    print(f"Built project artifacts under {OUTPUTS} and {REPORTS}")
    print(f"Defect model ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    print(
        "UCI Steel Faults benchmark "
        f"accuracy={real_metrics['accuracy']:.3f}, macro-F1={real_metrics['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
