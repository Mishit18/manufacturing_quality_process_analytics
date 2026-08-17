# Real UCI Steel Faults Validation

The primary external benchmark uses **1,941 real public records**, 27 measured features, and 7 fault classes.

## Holdout Results

- Accuracy: **0.792**
- Balanced accuracy: **0.791**
- Macro-F1: **0.806**
- Leading SHAP features: **conveyor_length, orientation_index, steel_a400, pixels_areas, square_index**
- Highest modeled review priority: **bumps**, based on missed cases and an explicit severity assumption.

## Claim Boundary

The failure-priority score is a modeled review aid, not observed plant cost. SPC and OEE outputs come from the separately labeled simulated operations scenario because the UCI data has no downtime or production-count fields.
