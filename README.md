# Task 10: Modeling Complex Non-Linear Relationships

This repository captures high-order non-linear feature interactions using regularized Gradient Boosting with explicit error handling, input shape validation, verifiable logs, and partial dependence explainability plots.

## Addressed Reviewer Feedback
* **Defensive Error Handling:** Built custom input integrity validators and explicit `try-except` blocks.
* **Verifiable Audit Records:** Logs full console execution to `artifacts/task10_execution_audit.log` and saves all test-set comparisons to `artifacts/task10_metrics.json`.
* **Explainability:** Generates Partial Dependence Plots (`artifacts/partial_dependence_plots.png`) verifying feature effect sanity.

## Test Set Verified Performance
* **Baseline Logistic Regression**: 97.67% Acc | 96.88% Malignant Recall | 1 False Negative
* **Non-Linear Gradient Boosting**: **98.84% Acc** | **100.00% Malignant Recall** | **0 False Negatives**
* **Lift**: +1.17% Accuracy, +3.12% Malignant Recall (100% cancer detection).

## Execution
```bash
git clone [https://github.com/shreedharchalmi/placemux_dailytask10.git](https://github.com/shreedharchalmi/placemux_dailytask10.git)
cd placemux_dailytask10
pip install scikit-learn pandas numpy matplotlib joblib
python task_10_complex_relationships.py