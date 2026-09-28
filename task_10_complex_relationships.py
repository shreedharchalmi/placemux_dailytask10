import os
import sys
import json
import logging
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------
# Logging Configuration (Verifiable Audit Trail)
# ---------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("artifacts/task10_execution_audit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Task10")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# Defensive Input Validation & Error Handling
# ---------------------------------------------------------
def validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Validates dataframe schema, shapes, missingness, and classes."""
    try:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame and y must be a Series/array.")
        
        if X.empty or len(y) == 0:
            raise ValueError("Input data cannot be empty.")
            
        if len(X) != len(y):
            raise ValueError(f"Shape mismatch: X has {len(X)} rows, y has {len(y)} labels.")
            
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Task requires binary classification. Found classes: {unique_classes}")
            
        # Check variance
        zero_var = [col for col in X.columns if X[col].nunique() <= 1]
        if zero_var:
            logger.warning(f"Columns with zero variance detected: {zero_var}")
            
        logger.info(f"Input verification passed: {X.shape[0]} samples, {X.shape[1]} features.")
    except Exception as e:
        logger.error(f"Input validation failure: {str(e)}")
        raise

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    logger.info("Starting Task 10: Complex Non-Linear Relationships Pipeline")
    
    try:
        # 1. Ingestion
        raw = load_breast_cancer(as_frame=True)
        X = raw.data.copy()
        y = raw.target.copy()
        
        # Add engineered non-linear interaction features
        X['severe_nuclear_distortion'] = X['worst concavity'] * X['worst compactness']
        X['radial_variance_spread'] = X['worst radius'] - X['mean radius']
        
        validate_inputs(X, y)
        
        # 2. Stratified Data Splitting
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=SEED
        )
        logger.info(f"Splits generated -> Dev: {len(X_dev)}, Test: {len(X_test)}")
        
        # 3. Model Definition: Linear vs. Regularized Non-Linear
        linear_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1, penalty="l2", solver="liblinear", class_weight="balanced", random_state=SEED
            ))
        ])
        
        nonlinear_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.5,
            early_stopping=True,
            scoring="loss",
            random_state=SEED
        )
        
        # 4. Out-of-Fold Cross-Validation Comparison
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scoring = {"acc": "accuracy", "roc": "roc_auc"}
        
        cv_linear = cross_validate(linear_pipeline, X_dev, y_dev, cv=cv, scoring=scoring)
        cv_nonlinear = cross_validate(nonlinear_model, X_dev, y_dev, cv=cv, scoring=scoring)
        
        logger.info(f"5-Fold CV Accuracy -> Linear: {cv_linear['test_acc'].mean():.4f} | Non-Linear: {cv_nonlinear['test_acc'].mean():.4f}")
        logger.info(f"5-Fold CV ROC-AUC  -> Linear: {cv_linear['test_roc'].mean():.4f} | Non-Linear: {cv_nonlinear['test_roc'].mean():.4f}")
        
        # 5. Fit & Test-Set Evaluation
        linear_pipeline.fit(X_dev, y_dev)
        nonlinear_model.fit(X_dev, y_dev)
        
        # Test predictions
        y_test_pred_lin = linear_pipeline.predict(X_test)
        y_test_prob_lin = linear_pipeline.predict_proba(X_test)[:, 1]
        
        y_test_pred_non = nonlinear_model.predict(X_test)
        y_test_prob_non = nonlinear_model.predict_proba(X_test)[:, 1]
        
        cm_lin = confusion_matrix(y_test, y_test_pred_lin)
        cm_non = confusion_matrix(y_test, y_test_pred_non)
        
        results = {
            "linear_baseline": {
                "accuracy": float(accuracy_score(y_test, y_test_pred_lin)),
                "recall_malignant": float(recall_score(y_test, y_test_pred_lin, pos_label=0)),
                "precision_malignant": float(precision_score(y_test, y_test_pred_lin, pos_label=0)),
                "f1_weighted": float(f1_score(y_test, y_test_pred_lin, average="weighted")),
                "roc_auc": float(roc_auc_score(y_test, y_test_prob_lin)),
                "false_negatives": int(cm_lin[0, 1]),
                "false_positives": int(cm_lin[1, 0])
            },
            "nonlinear_gradient_boosting": {
                "accuracy": float(accuracy_score(y_test, y_test_pred_non)),
                "recall_malignant": float(recall_score(y_test, y_test_pred_non, pos_label=0)),
                "precision_malignant": float(precision_score(y_test, y_test_pred_non, pos_label=0)),
                "f1_weighted": float(f1_score(y_test, y_test_pred_non, average="weighted")),
                "roc_auc": float(roc_auc_score(y_test, y_test_prob_non)),
                "false_negatives": int(cm_non[0, 1]),
                "false_positives": int(cm_non[1, 0])
            },
            "performance_lift": {
                "accuracy_delta": float(accuracy_score(y_test, y_test_pred_non) - accuracy_score(y_test, y_test_pred_lin)),
                "recall_delta": float(recall_score(y_test, y_test_pred_non, pos_label=0) - recall_score(y_test, y_test_pred_lin, pos_label=0)),
                "false_negative_reduction": int(cm_lin[0, 1] - cm_non[0, 1])
            }
        }
        
        logger.info(f"Held-Out Test Results -> Non-Linear Accuracy: {results['nonlinear_gradient_boosting']['accuracy']:.4f}")
        logger.info(f"Malignant Recall (Class 0) -> Non-Linear: {results['nonlinear_gradient_boosting']['recall_malignant']:.4f} (FN: {results['nonlinear_gradient_boosting']['false_negatives']})")
        
        # 6. Explainability: Partial Dependence Plots (PDP)
        pdp_features = ['worst concave points', 'severe_nuclear_distortion', 'worst radius']
        fig, ax = plt.subplots(figsize=(12, 4))
        PartialDependenceDisplay.from_estimator(
            nonlinear_model, X_dev, pdp_features, ax=ax, line_kw={"color": "darkred", "lw": 2}
        )
        plt.suptitle("Task 10: Partial Dependence Plots for Complex Non-Linear Signals", fontsize=13)
        plt.tight_layout()
        pdp_path = "artifacts/partial_dependence_plots.png"
        plt.savefig(pdp_path, dpi=200)
        plt.close()
        logger.info(f"Saved explainability plots to {pdp_path}")
        
        # 7. Persist Artifacts
        model_path = "artifacts/nonlinear_gradient_boost_model.joblib"
        metrics_path = "artifacts/task10_metrics.json"
        
        joblib.dump(nonlinear_model, model_path)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Trained non-linear model saved to {model_path}")
        logger.info(f"Evaluation metrics JSON saved to {metrics_path}")
        logger.info("Task 10 execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()