"""Train the attrition model (SYNTHETIC labels) and write a model card.

Baseline logistic regression vs XGBoost, stratified 80/20 split, metrics: ROC-AUC, PR-AUC, Brier, top-decile lift.
Global explanations: mean |SHAP| grouped by source feature. Tracked in MLflow (./mlruns), artefacts in
ml/registry/attrition/. Run: python -m ml.attrition.train
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import duckdb
import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import ROOT, db_path
from ml.features.attrition import FEATURE_LABELS, LABEL, base_feature, design_matrix, load_frame

log = logging.getLogger(__name__)
REGISTRY = ROOT / "ml" / "registry" / "attrition"
SEED = 42
XGB_PARAMS = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
              "colsample_bytree": 0.9, "min_child_weight": 5, "eval_metric": "logloss", "random_state": SEED}


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "base_rate": round(float(y.mean()), 4),
            "top_decile_lift": round(float(y[top].mean() / y.mean()), 2)}


def grouped_importance(model: xgb.XGBClassifier, X: pd.DataFrame) -> list[dict[str, float | str]]:
    values = shap.TreeExplainer(model).shap_values(X)
    per_col = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    grouped = per_col.groupby([base_feature(c) for c in X.columns]).sum().sort_values(ascending=False)
    return [{"feature": f, "label": FEATURE_LABELS.get(f, f), "mean_abs_shap": round(float(v), 4)}
            for f, v in grouped.items()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with duckdb.connect(str(db_path()), read_only=True) as con:
        df = load_frame(con)
    X, y = design_matrix(df), df[LABEL].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    model = xgb.XGBClassifier(**XGB_PARAMS).fit(X_tr, y_tr)
    metrics = {"logistic_regression": evaluate(y_te, baseline.predict_proba(X_te)[:, 1]),
               "xgboost": evaluate(y_te, model.predict_proba(X_te)[:, 1])}
    importance = grouped_importance(model, X_te.sample(min(5000, len(X_te)), random_state=SEED))

    REGISTRY.mkdir(parents=True, exist_ok=True)
    model.save_model(REGISTRY / "model.json")
    card = {
        "name": "hrum-attrition-xgb", "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "warning": "Trained on SYNTHETIC termination labels: demonstrates the method; drivers mirror the "
                   "generator rules in docs/data_dictionary.md and are not findings about SF employees.",
        "target": "left during FY2026 (synthetic)", "rows": len(df), "test_rows": len(y_te),
        "features": list(X.columns), "excluded_on_purpose": ["gender"],
        "params": XGB_PARAMS, "metrics_test": metrics, "global_importance": importance,
    }
    (REGISTRY / "model_card.json").write_text(json.dumps(card, indent=2))

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("hrum-attrition") is None:
        mlflow.create_experiment("hrum-attrition", artifact_location=(ROOT / "mlruns" / "artifacts").as_uri())
    mlflow.set_experiment("hrum-attrition")
    with mlflow.start_run(run_name="xgb-vs-logreg"):
        mlflow.log_params(XGB_PARAMS)
        for name, m in metrics.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
        mlflow.log_artifact(str(REGISTRY / "model_card.json"))
    log.info("test metrics: %s", json.dumps(metrics))
    log.info("top drivers: %s", [i["feature"] for i in importance[:5]])


if __name__ == "__main__":
    main()
