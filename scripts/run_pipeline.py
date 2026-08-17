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
    run_metropt_benchmark,
    run_steel_faults_evidence,
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
    real_metrics, real_importance, class_metrics, failure_priority, shap_importance = run_steel_faults_evidence()
    metro_metrics, _ = run_metropt_benchmark(
        output_dir=OUTPUTS,
        cache_dir=ROOT / ".cache",
    )

    df.head(5000).to_csv(OUTPUTS / "manufacturing_sample.csv", index=False)
    oee.to_csv(OUTPUTS / "oee_by_line_machine.csv", index=False)
    spc.to_csv(OUTPUTS / "spc_control_chart_data.csv", index=False)
    importances.to_csv(OUTPUTS / "defect_driver_importance.csv", index=False)
    bottlenecks.to_csv(OUTPUTS / "bottleneck_rankings.csv", index=False)
    recommendations.to_csv(OUTPUTS / "recommended_actions.csv", index=False)
    pd.DataFrame([real_metrics]).to_csv(OUTPUTS / "uci_steel_faults_metrics.csv", index=False)
    real_importance.to_csv(OUTPUTS / "uci_steel_faults_importance.csv", index=False)
    class_metrics.to_csv(OUTPUTS / "uci_steel_faults_class_metrics.csv", index=False)
    failure_priority.to_csv(OUTPUTS / "uci_steel_faults_failure_priority.csv", index=False)
    shap_importance.to_csv(OUTPUTS / "uci_steel_faults_shap_importance.csv", index=False)
    write_executive_summary(
        REPORTS / "executive_summary.md",
        n_rows=len(df),
        oee=oee,
        spc=spc,
        importances=importances,
        metrics=metrics,
        recommendations=recommendations,
    )
    top_fault = failure_priority.iloc[0]
    top_shap = ", ".join(shap_importance.head(5)["feature"].tolist())
    (REPORTS / "real_data_validation.md").write_text(
        f"# Real Industrial Data Validation\n\n"
        f"## MetroPT-3 Predictive-Maintenance Benchmark\n\n"
        f"Analyzed **{int(metro_metrics['raw_records']):,} real operational sensor readings** "
        f"from a metro-train air compressor and evaluated chronological anomaly monitoring "
        f"against {int(metro_metrics['reported_failure_events'])} company-reported air-leak events.\n\n"
        f"- Detected failure events: **{int(metro_metrics['detected_failure_events'])}/"
        f"{int(metro_metrics['reported_failure_events'])}**\n"
        f"- Failure-window alert rate: **{metro_metrics['failure_window_alert_rate']:.2%}**\n"
        f"- Normal-window alert rate: **{metro_metrics['normal_window_alert_rate']:.2%}**\n\n"
        f"## UCI Steel Faults Classification\n\n"
        f"The primary external benchmark uses **{int(real_metrics['records']):,} real public records**, "
        f"27 measured features, and {int(real_metrics['classes'])} fault classes.\n\n"
        f"## Holdout Results\n\n"
        f"- Accuracy: **{real_metrics['accuracy']:.3f}**\n"
        f"- Balanced accuracy: **{real_metrics['balanced_accuracy']:.3f}**\n"
        f"- Macro-F1: **{real_metrics['macro_f1']:.3f}**\n"
        f"- Leading SHAP features: **{top_shap}**\n"
        f"- Highest modeled review priority: **{top_fault['fault_type']}**, based on missed cases and an explicit severity assumption.\n\n"
        "## Claim Boundary\n\n"
        "The failure-priority score is a modeled review aid, not observed plant cost. "
        "SPC and OEE outputs come from the separately labeled simulated operations scenario because the UCI data has no downtime or production-count fields.\n",
        encoding="utf-8",
    )
    print(f"Built project artifacts under {OUTPUTS} and {REPORTS}")
    print(f"Defect model ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    print(
        "UCI Steel Faults benchmark "
        f"accuracy={real_metrics['accuracy']:.3f}, macro-F1={real_metrics['macro_f1']:.3f}"
    )
    print(
        "MetroPT-3 benchmark "
        f"events={int(metro_metrics['detected_failure_events'])}/"
        f"{int(metro_metrics['reported_failure_events'])}, "
        f"normal alert rate={metro_metrics['normal_window_alert_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
