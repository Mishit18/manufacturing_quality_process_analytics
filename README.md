# Manufacturing Quality and Process Bottleneck Analytics

Manufacturing analytics case study for operations, strategy, and data analytics roles. The project creates a reproducible plant-quality dataset and analyzes **SPC control breaches**, **OEE**, bottleneck assets, defect drivers, scrap-cost exposure, and recommended maintenance actions.

The dataset is synthetic to avoid confidential plant data, but the fields mirror a practical production-quality workflow: line, machine, shift, material lot, process settings, downtime, output, defects, scrap, and rework.

## What This Demonstrates

- Manufacturing process analytics beyond e-commerce/customer data
- SPC control charts and exception monitoring
- OEE decomposition across availability, performance, and quality
- Bottleneck scoring across defect rate, downtime, and OEE loss
- Root-cause analytics using interpretable defect-driver modeling
- Dashboard-ready CSV outputs and executive recommendation memo

## Project Outputs

| Artifact | Purpose |
|---|---|
| `outputs/oee_by_line_machine.csv` | OEE, defect rate, downtime, and scrap by line/machine |
| `outputs/spc_control_chart_data.csv` | Line-day defect rates, UCL/LCL, and breach flags |
| `outputs/defect_driver_importance.csv` | Permutation importance for process defect drivers |
| `outputs/bottleneck_rankings.csv` | Ranked bottlenecks using OEE, downtime, and defect losses |
| `outputs/recommended_actions.csv` | Maintenance and root-cause recommendations |
| `reports/executive_summary.md` | Interview-ready management summary |

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
pytest
```

## Resume Bullets

- Built manufacturing quality analytics project on **52k** batch records; used **SPC**, **OEE**, bottleneck scoring, and defect-driver modeling to identify root causes and prioritize maintenance actions.
- Ranked line-machine bottlenecks using downtime, defect rate, scrap cost, and OEE loss; generated dashboard-ready exception tables and management recommendations.
- Trained interpretable defect-risk model and used permutation importance to link tool age, vibration, cycle time, and process conditions to quality failures.

