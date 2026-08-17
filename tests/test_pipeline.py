from manufacturing_quality import (
    build_manufacturing_dataset,
    compute_oee,
    detect_spc_breaches,
    fit_defect_model,
    identify_bottlenecks,
    recommend_actions,
    run_steel_faults_benchmark,
)


def test_dataset_has_expected_shape_and_columns():
    df = build_manufacturing_dataset(n_batches=2_000)
    assert len(df) == 2_000
    for col in ["line", "machine", "planned_units", "good_units", "defect_units", "defect_rate"]:
        assert col in df.columns
    assert df["defect_rate"].between(0, 1).all()


def test_oee_and_bottleneck_outputs_are_valid():
    df = build_manufacturing_dataset(n_batches=4_000)
    oee = compute_oee(df)
    bottlenecks = identify_bottlenecks(oee)
    assert {"availability", "performance", "quality", "oee", "bottleneck_score"}.issubset(bottlenecks.columns)
    assert bottlenecks["oee"].between(0, 1.2).all()
    assert bottlenecks["bottleneck_score"].is_monotonic_decreasing


def test_model_and_recommendations_are_defensible():
    df = build_manufacturing_dataset(n_batches=6_000)
    oee = compute_oee(df)
    spc = detect_spc_breaches(df)
    importances, metrics = fit_defect_model(df)
    recs = recommend_actions(identify_bottlenecks(oee), spc, importances)
    assert metrics["roc_auc"] > 0.70
    assert metrics["pr_auc"] > 0.30
    assert len(importances) >= 5
    assert len(recs) == 6


def test_real_uci_benchmark_is_reproducible():
    metrics, importance = run_steel_faults_benchmark()
    assert metrics["records"] == 1_941
    assert metrics["classes"] == 7
    assert metrics["accuracy"] > 0.70
    assert metrics["macro_f1"] > 0.60
    assert len(importance) == 27
