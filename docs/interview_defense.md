# Interview Defense: Manufacturing Quality and Process Bottleneck Analytics

## 30-Second Pitch

I built a manufacturing operations analytics case on 52,000 batch records to show how I would move from raw plant data to an operating recommendation. The pipeline computes SPC control breaches, OEE decomposition, line-machine bottleneck rankings, defect-driver importance, and dashboard-ready action tables for maintenance and root-cause review.

## What The Project Proves

- I can work outside e-commerce and customer analytics, in a manufacturing/process-improvement setting.
- I understand the difference between model accuracy and operational usefulness.
- I can connect analytics outputs to plant actions: audits, maintenance prioritization, and daily KPI review.
- I can explain SPC, OEE, defect rate, downtime, scrap cost, and bottleneck prioritization in business language.

## Key Metrics

| Metric | Value |
|---|---:|
| Batch records | 52,000 |
| Defect-risk ROC-AUC | 0.760 |
| Defect-risk PR-AUC | 0.492 |
| Lowest-OEE asset | Line_C / M13 |
| Lowest-OEE value | 0.846 |
| SPC line-day breaches | 1 |

## Methods To Defend

- **SPC**: I used line-day defect rates and 3-sigma control limits to flag statistically unusual quality behavior.
- **OEE**: I decomposed equipment effectiveness into availability, performance, and quality instead of using only defect rate.
- **Bottleneck score**: I combined OEE loss, defect rate, and downtime into a prioritization score for line-machine assets.
- **Defect-risk model**: I used a Random Forest for nonlinear defect-driver discovery, then used permutation importance to keep the explanation interview-friendly.

## Honest Caveats

- The dataset is synthetic for reproducibility and privacy; I should not call it live factory data.
- The model is a decision-support tool, not an automated production-control system.
- Before deployment, I would validate against real MES/ERP data, add shift/operator controls, and confirm recommended actions with process engineers.

## Likely Interview Questions

**Why not only use machine learning?**  
Manufacturing quality needs statistical process control and explainability. A model can rank risk, but SPC and OEE make the result easier for operators and managers to trust.

**Why use OEE?**  
OEE separates availability, performance, and quality losses, so the recommendation can distinguish downtime issues from cycle-time issues and defect-quality issues.

**Why use synthetic data?**  
Plant data is often confidential. I used synthetic data to make the project reproducible while preserving the workflow and fields a real plant analytics project would use.

**What would you improve next?**  
I would connect the pipeline to a live dashboard, add root-cause annotations from maintenance logs, and run intervention tracking to measure whether recommended actions reduce defect rates.
