# Model Monitoring & Responsible-Use Operations Plan

## 1. Post-Deployment Monitoring Strategy

To protect the business logic from degradation post-deployment, the scoring service will track metrics across the following vectors:

### A. Data Drift Metrics
* **Feature Distribution Tracking:** Run weekly Population Stability Index (PSI) audits comparing real-time inference data profiles with original parameters in `rfm_modeling_snapshot.csv`. 
* **Categorical Feature Shift:** Track categorical feature frequencies (e.g., changes in baseline `acquisition_channel` or `loyalty_tier` ratios). Alerts trigger if any single category drifts by over 15%.

### B. Prediction Distribution Monitoring
* **Threshold Violations:** Calculate rolling daily averages of the churn assignment rates. If the overall percentage of predicted churners moves out of historical normal bounds (e.g., shifts unexpectedly from ~20% up to 45%), a structural evaluation protocol triggers.

### C. Operational System Context
* **API Error Logs:** Automated capture of HTTP 400 bad payload inputs and raw engine execution tracking via continuous monitoring.
* **Latency Flags:** Track response times; warning thresholds trigger if serialization and feature encoding re-indexing takes more than 200ms per record block.

### D. Model Retraining Triggers
* Retraining pipelines are executed automatically under any of the following conditions:
  1. A measured feature drift marker shows a structural PSI score $\ge 0.25$.
  2. Sustained precision or recall decay drops by 5% or more over a rolling 30-day monitoring block evaluated against finalized true labels.

---

## 2. Responsible-Use Framework

The API output should be consumed strategically by the CRM retention team according to these strict operational guardrails:

* ** Intended Use:** Target high-risk cohorts with tailored product recommendations, loyalty enrollment incentives, or priority customer service ticket resolution tracks.
* ** Prohibited Use:** Do NOT use high risk scores to automatically cancel customer accounts, deny support availability, or aggressively spampugn accounts with constant high-discount messaging that could dilute brand value.