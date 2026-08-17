# External Review Packet

## Claims to verify

- Real benchmark: 1,941 UCI steel records, 27 features, and seven fault classes.
- Held-out accuracy 0.792 and macro-F1 0.806 under a stratified 75/25 split.
- SHAP feature ranking and class-level error analysis are generated from the fitted real-data model.
- SPC/OEE and scrap-cost outputs are a separately labeled simulated operations scenario.

## Reproduce

```bash
python scripts/run_pipeline.py
python -m pytest -q
```

## Evidence

- `outputs/uci_steel_faults_metrics.csv`
- `outputs/uci_steel_faults_class_metrics.csv`
- `outputs/uci_steel_faults_shap_importance.csv`
- `outputs/uci_steel_faults_failure_priority.csv`
- `reports/real_data_validation.md`

## Reviewer checklist

- Confirm split, stratification, metrics, and absence of target columns in features.
- Review whether SHAP rankings are interpreted as associations rather than causal drivers.
- Confirm modeled severity weights are visibly separated from observed data.
- Review whether SPC/OEE recommendations are operationally plausible.
- Record reviewer name, role, date, and scope reviewed.
