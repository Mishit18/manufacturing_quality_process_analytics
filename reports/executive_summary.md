# Manufacturing Quality and Process Bottleneck Analytics

## Executive Summary

Built a manufacturing analytics case on **52,000 batch-level production records** covering line, machine, shift, material lot, process settings, downtime, output, defects, scrap, and rework.

## Key Findings

- Lowest-OEE asset: **Line_C / M13** with OEE **0.846** and defect rate **7.61%**.
- SPC monitoring flagged **1 line-day control breaches**, giving operations a focused audit queue.
- Defect classifier reached ROC-AUC **0.760** and PR-AUC **0.492** on holdout data.
- Leading defect drivers by permutation importance: **tool_age_hours, pressure_bar, humidity_pct**.
- **64** line-machine pairs crossed the 0.78 OEE threshold; the bottom assets need targeted maintenance and process control.
- Top-six bottlenecks represent an annualized scrap-cost opportunity proxy of roughly **INR 25,100,899**.

## Recommended Operating Actions

1. Prioritize tool-age and vibration checks on bottleneck machines before adding new capacity.
2. Add SPC exception review to daily production meetings for lines with control-limit breaches.
3. Route high-risk batches to preventive maintenance review when defect probability crosses the 80th percentile.
4. Track OEE, defect rate, downtime, scrap cost, and rework minutes as a single plant scorecard.

## Resume-Safe Bullet

Built manufacturing quality analytics project on **52k** batch records; used **SPC**, **OEE**, bottleneck scoring, and defect-driver modeling to identify root causes and prioritize maintenance actions.
