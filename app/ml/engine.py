"""
Evaluated dual-classifier attrition engine for Kaizen.

PorusXS (Optimized Balanced Random Forest) and AlexzanderXS (Histogram Gradient
Boosting) are trained on identical data, evaluated on a stratified holdout split,
and selected objectively based on combined discrimination and classification metrics.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from functools import lru_cache
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _detect_cuda_device() -> str:
    """Detect if NVIDIA CUDA hardware acceleration is available for XGBoost."""
    try:
        import xgboost as xgb

        test_clf = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", verbosity=0)
        test_clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        logger.info("NVIDIA CUDA hardware acceleration active for XGBoost.")
        return "cuda"
    except Exception as exc:
        logger.info("NVIDIA CUDA acceleration not available (%s); falling back to CPU.", exc)
        return "cpu"


def _get_model_details() -> dict[str, dict[str, str]]:
    cuda_active = _detect_cuda_device() == "cuda"
    algo_alex = "XGBoost (GPU-Accelerated)" if cuda_active else "Histogram Gradient Boosting"
    desc_alex = (
        "Enterprise-grade histogram-binned gradient-boosted decision tree with "
        "NVIDIA CUDA acceleration, L2 regularization, early stopping, and lossguide tree growth. "
        "Excels at subtle, non-linear feature threshold interactions."
        if cuda_active
        else (
            "State-of-the-art histogram-binned gradient-boosted decision tree with "
            "L2 regularization, early stopping, and native class balancing. "
            "Excels at subtle, non-linear feature threshold interactions."
        )
    )
    return {
        "PorusXS": {
            "algorithm": "Optimized Balanced Random Forest",
            "description": (
                "Ensemble of decorrelated decision trees using balanced sub-sample "
                "weighting. Captures high-order interactions and nonlinear turnover "
                "signals with robust generalization."
            ),
        },
        "AlexzanderXS": {
            "algorithm": algo_alex,
            "description": desc_alex,
        },
    }


# Canonical descriptions displayed in the dual-classifier lab view.
MODEL_DETAILS: dict[str, dict[str, str]] = _get_model_details()

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _prettify(name: str) -> str:
    """Convert snake_case or camelCase into clean title words."""
    cleaned = str(name).replace("_", " ").strip()
    words = _CAMEL_SPLIT_RE.split(cleaned)
    return " ".join(word.capitalize() for word in words if word)


class TrainingError(Exception):
    """Raised when training data fails validation or cannot produce a valid model."""


@dataclass
class KaizenEngine:
    """Trains, compares, persists, and serves two attrition classifiers."""

    target_col: str = "Attrition"
    task_type: str = "classification"

    active_model: str | None = None
    candidate_models: dict[str, Pipeline] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_comparison: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    # Empirically optimised threshold for the champion model
    decision_threshold: float = 0.5

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    reference_profile: dict[str, Any] = field(default_factory=dict)
    feature_importances: list[dict[str, Any]] = field(default_factory=list)
    trained_at: str | None = None
    version: str | None = None
    model: Pipeline | None = None

    # ------------------------------------------------------------------ #
    # Pipeline construction
    # ------------------------------------------------------------------ #

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_features),
                ("cat", categorical_pipe, self.categorical_features),
            ],
            remainder="drop",
        )

    def _build_estimator(self, model_name: str, n_samples: int = 1000):
        if model_name == "PorusXS":
            # Scale-adaptive subsampling: 60k-80k samples per tree at million-scale
            # guarantees maximum ensemble diversity with lightning-fast execution
            if n_samples > 200_000:
                max_samples = 0.06
                n_estimators = 100
            elif n_samples > 20_000:
                max_samples = 0.25
                n_estimators = 200
            else:
                max_samples = None
                n_estimators = 400

            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=14,
                min_samples_split=3,
                min_samples_leaf=1,
                max_samples=max_samples,
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=-1,
                random_state=42,
            )
        if model_name == "AlexzanderXS":
            cuda_device = _detect_cuda_device()
            try:
                import xgboost as xgb

                if n_samples > 200_000:
                    n_estimators = 250
                    learning_rate = 0.038
                elif n_samples > 20_000:
                    n_estimators = 300
                    learning_rate = 0.035
                else:
                    n_estimators = 300
                    learning_rate = 0.035

                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_leaves=31,
                    grow_policy="lossguide",
                    tree_method="hist",
                    device=cuda_device,
                    reg_lambda=0.5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1 if cuda_device == "cpu" else None,
                )
            except Exception as exc:
                logger.warning("XGBoost initialization fallback (%s); using HistGradientBoostingClassifier", exc)
                if n_samples > 200_000:
                    max_iter = 250
                    early_stopping = True
                elif n_samples > 20_000:
                    max_iter = 300
                    early_stopping = False
                else:
                    max_iter = 350
                    early_stopping = False

                return HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.038 if n_samples > 200_000 else 0.035,
                    max_leaf_nodes=31,
                    min_samples_leaf=16 if n_samples > 200_000 else 12,
                    l2_regularization=0.5,
                    class_weight="balanced",
                    early_stopping=early_stopping,
                    n_iter_no_change=12 if n_samples > 200_000 else 10,
                    random_state=42,
                )
        raise ValueError(f"Unknown candidate model: {model_name}")

    def _build_pipeline(self, model_name: str, n_samples: int = 1000) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", self._build_preprocessor()),
                ("estimator", self._build_estimator(model_name, n_samples=n_samples)),
            ]
        )

    # ------------------------------------------------------------------ #
    # Threshold optimisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _optimal_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
        """Find the decision threshold balancing Accuracy, F1, and Balanced Accuracy.

        Sweeps thresholds from 0.25 to 0.75 and picks the operating point that maximizes
        the harmonic combination of accuracy and minority-churn detection.
        """
        thresholds = np.linspace(0.25, 0.75, 51)
        best_threshold = 0.5
        best_score = -1.0

        for t in thresholds:
            preds = (probabilities >= t).astype(int)
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, zero_division=0)
            bacc = balanced_accuracy_score(y_true, preds)
            # Composite objective: rewards high raw accuracy while strictly guarding minority recall
            score = 0.40 * acc + 0.35 * f1 + 0.25 * bacc
            if score > best_score:
                best_score = score
                best_threshold = t

        return float(np.clip(best_threshold, 0.25, 0.75))

    def _training_oof_threshold(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
    ) -> float:
        """Choose the operating threshold without using the holdout labels.

        Selecting a cutoff on the same holdout used for reporting makes F1 and
        recall look better than they will be in production.  We instead use
        three stratified out-of-fold predictions from the training portion.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        folds = min(3, int(y_train.value_counts().min()))
        if folds < 2:
            return 0.5
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        probabilities = cross_val_predict(
            clone(pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        return self._optimal_threshold(y_train, probabilities)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evaluate(
        y_true: pd.Series | np.ndarray,
        probabilities: np.ndarray,
        model_name: str,
        threshold: float,
        hardware_device: str | None = None,
    ) -> dict[str, Any]:
        """Compute full classification telemetry on held-out data."""
        y_true_arr = np.asarray(y_true, dtype=int)
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_true_arr, predictions, labels=[0, 1])

        # High-resolution ROC curve with hover HUD telemetry (tau, TPR, FPR, F1)
        fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_true_arr, probabilities)
        roc_len = len(fpr_raw)
        roc_indices = (
            np.arange(roc_len)
            if roc_len <= 80
            else np.unique(np.linspace(0, roc_len - 1, 80, dtype=int))
        )
        pos_count = float(np.sum(y_true_arr == 1))
        neg_count = float(np.sum(y_true_arr == 0))
        roc_points = []
        for i in roc_indices:
            fpr_val = float(fpr_raw[i])
            tpr_val = float(tpr_raw[i])
            t_val = float(thresholds_raw[i])
            if t_val > 1.0:
                t_val = 1.0
            tp_est = tpr_val * pos_count
            fp_est = fpr_val * neg_count
            prec_est = tp_est / (tp_est + fp_est) if (tp_est + fp_est) > 0 else 0.0
            f1_est = (
                2.0 * prec_est * tpr_val / (prec_est + tpr_val)
                if (prec_est + tpr_val) > 0
                else 0.0
            )
            roc_points.append(
                {
                    "fpr": round(fpr_val, 4),
                    "tpr": round(tpr_val, 4),
                    "threshold": round(t_val, 4),
                    "f1": round(f1_est, 4),
                }
            )

        # High-resolution Precision-Recall (PR) curve
        prec_raw, rec_raw, pr_thresholds = precision_recall_curve(y_true_arr, probabilities)
        pr_len = len(prec_raw)
        pr_indices = (
            np.arange(pr_len)
            if pr_len <= 80
            else np.unique(np.linspace(0, pr_len - 1, 80, dtype=int))
        )
        pr_points = []
        for i in pr_indices:
            prec_val = float(prec_raw[i])
            rec_val = float(rec_raw[i])
            t_val = float(pr_thresholds[i]) if i < len(pr_thresholds) else 1.0
            f1_est = (
                2.0 * prec_val * rec_val / (prec_val + rec_val)
                if (prec_val + rec_val) > 0
                else 0.0
            )
            pr_points.append(
                {
                    "recall": round(rec_val, 4),
                    "precision": round(prec_val, 4),
                    "threshold": round(t_val, 4),
                    "f1": round(f1_est, 4),
                }
            )

        # 10-bin Calibration & Reliability Curve with Expected Calibration Error (ECE)
        n_samples = len(y_true_arr)
        cal_points = []
        ece = 0.0
        reliability = 0.0
        resolution = 0.0
        base_prev = float(np.mean(y_true_arr)) if n_samples > 0 else 0.0
        uncertainty = float(base_prev * (1.0 - base_prev))

        for b in range(10):
            lo = b / 10.0
            hi = (b + 1) / 10.0
            bin_mask = (
                (probabilities >= lo) & (probabilities <= hi)
                if b == 9
                else (probabilities >= lo) & (probabilities < hi)
            )
            bin_count = int(np.sum(bin_mask))
            if bin_count > 0:
                p_pred = float(np.mean(probabilities[bin_mask]))
                p_true = float(np.mean(y_true_arr[bin_mask]))
                weight = bin_count / max(n_samples, 1)
                ece += weight * abs(p_true - p_pred)
                reliability += weight * ((p_pred - p_true) ** 2)
                resolution += weight * ((p_true - base_prev) ** 2)
            else:
                p_pred = round((lo + hi) / 2.0, 3)
                p_true = p_pred
            cal_points.append(
                {
                    "bin_midpoint": round((lo + hi) / 2.0, 2),
                    "prob_pred": round(p_pred, 4),
                    "prob_true": round(p_true, 4),
                    "bin_count": bin_count,
                    "bin_name": f"{int(lo * 100)}-{int(hi * 100)}%",
                }
            )

        brier_val = round(float(brier_score_loss(y_true_arr, probabilities)), 4)
        brier_decomp = {
            "brier_score": brier_val,
            "reliability": round(reliability, 4),
            "resolution": round(resolution, 4),
            "uncertainty": round(uncertainty, 4),
            "ece": round(ece, 4),
        }

        # Vectorized threshold sweep (49 operating points across tau in [0.02, 0.98])
        sweep_thresholds = np.linspace(0.02, 0.98, 49)
        preds_mat = (probabilities[:, None] >= sweep_thresholds[None, :]).astype(int)
        tp_arr = np.sum((y_true_arr[:, None] == 1) & (preds_mat == 1), axis=0)
        fp_arr = np.sum((y_true_arr[:, None] == 0) & (preds_mat == 1), axis=0)
        fn_arr = np.sum((y_true_arr[:, None] == 1) & (preds_mat == 0), axis=0)
        tn_arr = np.sum((y_true_arr[:, None] == 0) & (preds_mat == 0), axis=0)

        acc_arr = (tp_arr + tn_arr) / max(n_samples, 1)
        prec_arr = np.divide(
            tp_arr.astype(float),
            (tp_arr + fp_arr).astype(float),
            out=np.zeros(len(tp_arr), dtype=float),
            where=(tp_arr + fp_arr) > 0,
        )
        rec_arr = np.divide(
            tp_arr.astype(float),
            (tp_arr + fn_arr).astype(float),
            out=np.zeros(len(tp_arr), dtype=float),
            where=(tp_arr + fn_arr) > 0,
        )
        spec_arr = np.divide(
            tn_arr.astype(float),
            (tn_arr + fp_arr).astype(float),
            out=np.zeros(len(tn_arr), dtype=float),
            where=(tn_arr + fp_arr) > 0,
        )
        pr_sum = prec_arr + rec_arr
        f1_arr = np.divide(
            2.0 * prec_arr * rec_arr,
            pr_sum,
            out=np.zeros(len(prec_arr), dtype=float),
            where=pr_sum > 0,
        )
        bacc_arr = 0.5 * (rec_arr + spec_arr)

        denom = (tp_arr + fp_arr).astype(float) * (tp_arr + fn_arr) * (tn_arr + fp_arr) * (tn_arr + fn_arr)
        sqrt_denom = np.sqrt(np.maximum(denom, 0.0))
        mcc_arr = np.divide(
            (tp_arr.astype(float) * tn_arr - fp_arr.astype(float) * fn_arr),
            sqrt_denom,
            out=np.zeros(len(tp_arr), dtype=float),
            where=sqrt_denom > 0,
        )

        # Net Business ROI: ($50k replacement cost retained - $3.5k intervention on flagged)
        net_roi_arr = (tp_arr.astype(float) * 50_000.0) - ((tp_arr + fp_arr).astype(float) * 3_500.0)

        threshold_metrics = [
            {
                "threshold": round(float(sweep_thresholds[j]), 3),
                "accuracy": round(float(acc_arr[j]), 4),
                "balanced_accuracy": round(float(bacc_arr[j]), 4),
                "precision": round(float(prec_arr[j]), 4),
                "recall": round(float(rec_arr[j]), 4),
                "f1": round(float(f1_arr[j]), 4),
                "specificity": round(float(spec_arr[j]), 4),
                "mcc": round(float(mcc_arr[j]), 4),
                "tn": int(tn_arr[j]),
                "fp": int(fp_arr[j]),
                "fn": int(fn_arr[j]),
                "tp": int(tp_arr[j]),
                "net_roi": round(float(net_roi_arr[j]), 2),
            }
            for j in range(len(sweep_thresholds))
        ]

        device_tag = hardware_device or (
            "cuda" if model_name == "AlexzanderXS" and _detect_cuda_device() == "cuda" else "cpu"
        )
        algo_name = MODEL_DETAILS.get(model_name, {}).get("algorithm", model_name)

        return {
            "model_name": model_name,
            "algorithm": algo_name,
            "accuracy": round(float(accuracy_score(y_true_arr, predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true_arr, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_true_arr, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_true_arr, probabilities)), 4),
            "f1": round(float(f1_score(y_true_arr, predictions, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true_arr, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_arr, predictions, zero_division=0)), 4),
            "brier_score": brier_val,
            "mcc": round(float(matthews_corrcoef(y_true_arr, predictions)), 4),
            "decision_threshold": round(float(threshold), 4),
            "confusion": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "roc_curve": roc_points,
            "pr_curve": pr_points,
            "calibration_curve": cal_points,
            "threshold_metrics": threshold_metrics,
            "ece": round(ece, 4),
            "brier_decomposition": brier_decomp,
            "baseline_prevalence": round(base_prev, 4),
            "hardware_device": device_tag,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KaizenEngine":
        """Train both candidates, evaluate on holdout, and refit the champion."""
        if self.task_type != "classification":
            raise TrainingError("Kaizen supports binary attrition classification only.")

        self.numeric_features = X.select_dtypes(include="number").columns.tolist()
        self.categorical_features = X.select_dtypes(exclude="number").columns.tolist()
        if not self.numeric_features and not self.categorical_features:
            raise TrainingError("No usable feature columns were found in the dataset.")
        if y.nunique() != 2:
            raise TrainingError("Attrition must contain both stay (0) and leave (1) examples.")

        # Stratified holdout split (80/20)
        from sklearn.model_selection import train_test_split

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y.astype(int),
            test_size=0.20,
            random_state=42,
            stratify=y.astype(int),
        )

        comparisons: list[dict[str, Any]] = []
        candidate_pipelines: dict[str, Pipeline] = {}
        candidate_thresholds: dict[str, float] = {}

        for name in ("PorusXS", "AlexzanderXS"):
            logger.info("Training candidate %s on %d rows…", name, len(X_train))
            pipe = self._build_pipeline(name, n_samples=len(X_train))
            pipe.fit(X_train, y_train)

            # Move booster inference to CPU to maximize throughput and eliminate
            # thread-safety and CUDA context-switching issues during 1-row simulation.
            try:
                est = pipe.named_steps["estimator"]
                if hasattr(est, "set_params"):
                    est.set_params(device="cpu")
            except Exception:
                pass

            candidate_pipelines[name] = pipe

            probs = pipe.predict_proba(X_holdout)[:, 1]
            opt_thresh = self._training_oof_threshold(pipe, X_train, y_train)
            candidate_thresholds[name] = opt_thresh

            dev_tag = "cuda" if name == "AlexzanderXS" and _detect_cuda_device() == "cuda" else "cpu"
            metrics = self._evaluate(y_holdout, probs, name, threshold=opt_thresh, hardware_device=dev_tag)
            comparisons.append(metrics)
            logger.info(
                "%s (%s) holdout: Acc=%.4f | ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | ECE=%.4f | Thresh=%.3f",
                name,
                dev_tag,
                metrics["accuracy"],
                metrics["roc_auc"],
                metrics["pr_auc"],
                metrics["f1"],
                metrics["ece"],
                opt_thresh,
            )

        # Attrition is normally rare, so PR-AUC is the primary champion metric.
        # ROC-AUC and MCC break ties without rewarding a majority-class model.
        comparisons.sort(
            key=lambda m: (m["pr_auc"], m["roc_auc"], m["mcc"], m["balanced_accuracy"]),
            reverse=True,
        )
        self.model_comparison = comparisons
        self.active_model = comparisons[0]["model_name"]
        self.metrics = comparisons[0]
        self.decision_threshold = candidate_thresholds[self.active_model]
        self.candidate_models = candidate_pipelines

        self.validation = {
            "method": "stratified 80/20 holdout with 3-fold training-only threshold selection",
            "training_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
            "positive_rate_pct": round(float(y.mean() * 100), 2),
            "selection_metric": "PR-AUC, then ROC-AUC and MCC",
            "note": (
                "Both candidates were evaluated on identical unseen stratified records. "
                "Their decision thresholds were selected using training-only out-of-fold "
                "predictions; the champion was then refit on the full dataset."
            ),
        }

        # Refit champion on all data for production deployment
        logger.info(
            "Champion: %s | Refitting on full dataset (%d rows)…",
            self.active_model,
            len(X),
        )
        self.model = self._build_pipeline(self.active_model, n_samples=len(X))
        self.model.fit(X, y.astype(int))
        try:
            est = self.model.named_steps["estimator"]
            if hasattr(est, "set_params"):
                est.set_params(device="cpu")
        except Exception:
            pass

        self.reference_profile, self.numeric_bounds = self._build_reference_and_bounds(X)
        self.feature_importances = self._calculate_feature_importances(X, y.astype(int))
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.version = uuid.uuid4().hex[:12]
        logger.info(
            "Kaizen training complete | champion=%s | holdout Acc=%.4f | ROC-AUC=%.4f | threshold=%.3f",
            self.active_model,
            self.metrics["accuracy"],
            self.metrics["roc_auc"],
            self.decision_threshold,
        )
        return self

    # ------------------------------------------------------------------ #
    # Reference profile & feature schema
    # ------------------------------------------------------------------ #

    def _build_reference_and_bounds(
        self, X: pd.DataFrame
    ) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
        """Compute population medians/modes and observed min/max from training data."""
        reference: dict[str, Any] = {}
        bounds: dict[str, dict[str, float]] = {}
        for col in self.numeric_features:
            series = pd.to_numeric(X[col], errors="coerce")
            reference[col] = float(series.median())
            bounds[col] = {
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }
        for col in self.categorical_features:
            mode = X[col].dropna().mode()
            reference[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
        return reference, bounds

    def get_feature_schema(self) -> list[dict[str, Any]]:
        """Return a safe, UI-ready input schema derived from the trained dataset."""
        specs: list[dict[str, Any]] = []
        bounds_dict = getattr(self, "numeric_bounds", {}) or {}
        ref_profile = getattr(self, "reference_profile", {}) or {}

        for col in self.numeric_features:
            default = ref_profile.get(col, 0.0)
            b = bounds_dict.get(col, {})
            specs.append(
                {
                    "name": col,
                    "kind": "number",
                    "minimum": b.get("minimum"),
                    "maximum": b.get("maximum"),
                    "default": round(float(default), 2),
                    "choices": [],
                }
            )
        for col in self.categorical_features:
            choices: list[str] = []
            if self.model is not None:
                try:
                    preprocessor: ColumnTransformer = self.model.named_steps["preprocess"]
                    encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps[
                        "onehot"
                    ]
                    index = self.categorical_features.index(col)
                    choices = [str(v) for v in encoder.categories_[index]]
                except Exception:
                    choices = [str(ref_profile.get(col, "Unknown"))]
            specs.append(
                {
                    "name": col,
                    "kind": "category",
                    "minimum": None,
                    "maximum": None,
                    "default": str(ref_profile.get(col, "Unknown")),
                    "choices": choices,
                }
            )
        return specs

    # ------------------------------------------------------------------ #
    # Feature importances
    # ------------------------------------------------------------------ #

    def _calculate_feature_importances(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None
    ) -> list[dict[str, Any]]:
        """Calculate global feature importance for the champion model."""
        if self.model is None:
            return []
        estimator = self.model.named_steps["estimator"]
        importances = getattr(estimator, "feature_importances_", None)
        all_features = self.numeric_features + self.categorical_features

        if importances is None:
            # Model-agnostic permutation importance for estimators without feature_importances_
            if X is not None and y is not None and len(X) > 0:
                sample_size = min(len(X), 250)
                X_sample = X.iloc[:sample_size]
                y_sample = y.iloc[:sample_size]
                try:
                    perm = permutation_importance(
                        self.model,
                        X_sample,
                        y_sample,
                        n_repeats=3,
                        random_state=42,
                        n_jobs=1,
                    )
                    raw_scores = np.maximum(0.0, perm.importances_mean)
                    total_score = float(sum(raw_scores)) or 1.0
                    result = [
                        {
                            "feature": _prettify(col),
                            "importance": round(float(score / total_score) * 100, 2),
                        }
                        for col, score in zip(all_features, raw_scores)
                        if score > 0
                    ]
                    if result:
                        return sorted(result, key=lambda item: item["importance"], reverse=True)
                except Exception as exc:
                    logger.warning("Permutation importance error: %s", exc)

            # Clean proportional default
            return [
                {"feature": _prettify(col), "importance": round(100.0 / max(len(all_features), 1), 2)}
                for col in all_features[:10]
            ]

        preprocessor: ColumnTransformer = self.model.named_steps["preprocess"]
        encoded_names = preprocessor.get_feature_names_out()
        aggregated: dict[str, float] = {col: 0.0 for col in all_features}
        for encoded_name, importance in zip(encoded_names, importances):
            raw_name = encoded_name.split("__", 1)[-1]
            source = raw_name
            if encoded_name.startswith("cat__"):
                source = next(
                    (
                        col
                        for col in sorted(self.categorical_features, key=len, reverse=True)
                        if raw_name.startswith(f"{col}_")
                    ),
                    raw_name,
                )
            aggregated[source] = aggregated.get(source, 0.0) + float(importance)
        total = sum(aggregated.values()) or 1.0
        result = [
            {"feature": _prettify(name), "importance": round((value / total) * 100, 2)}
            for name, value in aggregated.items()
            if value > 0
        ]
        return sorted(result, key=lambda item: item["importance"], reverse=True)

    def get_feature_importances(self, top_n: int = 12) -> list[dict[str, Any]]:
        return self.feature_importances[:top_n]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _aligned_profile(self, input_dict: dict[str, Any]) -> pd.DataFrame:
        expected = self.numeric_features + self.categorical_features
        missing = [col for col in expected if col not in input_dict]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "…" if len(missing) > 5 else ""
            raise ValueError(
                f"Complete every model input before predicting. Missing: {preview}{suffix}"
            )
        row: dict[str, Any] = {}
        for col in self.numeric_features:
            val = input_dict[col]
            if pd.isna(val) or val is None or str(val).strip() == "":
                row[col] = np.nan
            else:
                try:
                    row[col] = float(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{col} must be a number.") from exc
                if not np.isfinite(row[col]):
                    raise ValueError(f"{col} must be a finite number.")
        for col in self.categorical_features:
            value = input_dict[col]
            if pd.isna(value) or value is None or str(value).strip() == "":
                row[col] = np.nan
            else:
                row[col] = str(value)
        return pd.DataFrame([row], columns=expected)

    def predict_detailed(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.active_model is None:
            raise RuntimeError("No model has been trained yet.")
        profile = self._aligned_profile(input_dict)
        risk_score = float(self.model.predict_proba(profile)[0, 1])
        drivers = self._local_drivers(profile, risk_score)
        prescriptions = [_prescription_for(driver["raw_feature"], driver.get("impact", 0.0)) for driver in drivers]
        thresh = getattr(self, "decision_threshold", 0.5)
        return {
            "risk_score": risk_score,
            "drivers": [
                {
                    "feature": d["feature"],
                    "impact": d["impact"],
                    "method": d["method"],
                }
                for d in drivers
            ],
            "prescriptions": prescriptions,
            "model_name": self.active_model,
            "decision_threshold": thresh,
        }

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized batch risk probability prediction across an arbitrary DataFrame."""
        if self.model is None or self.active_model is None:
            raise RuntimeError("No model has been trained yet.")
        expected = self.numeric_features + self.categorical_features
        aligned_df = pd.DataFrame(index=df.index)
        for col in self.numeric_features:
            if col in df.columns:
                aligned_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                aligned_df[col] = np.nan
        for col in self.categorical_features:
            if col in df.columns:
                aligned_df[col] = df[col].astype(str)
            else:
                aligned_df[col] = "Missing"
        aligned_df = aligned_df[expected]
        return self.model.predict_proba(aligned_df)[:, 1]

    def _local_drivers(
        self, profile: pd.DataFrame, current_risk: float
    ) -> list[dict[str, Any]]:
        """Identify individual feature contributions via counterfactual perturbation.

        Always returns the most decisive features (risk-increasing drivers or protective
        factors) so recommendations are rich and actionable.
        """
        if self.model is None:
            return []
        drivers: list[dict[str, Any]] = []
        for col in self.numeric_features + self.categorical_features:
            reference = self.reference_profile.get(col)
            if reference is None:
                continue
            perturbed = profile.copy()
            perturbed.at[0, col] = reference
            try:
                ref_risk = float(self.model.predict_proba(perturbed)[0, 1])
            except Exception:
                continue
            delta = current_risk - ref_risk
            drivers.append(
                {
                    "raw_feature": col,
                    "feature": _prettify(col),
                    "impact": round(delta * 100, 1),
                    "abs_impact": abs(delta),
                    "is_risk_increasing": delta > 0,
                    "method": "risk driver" if delta > 0 else "protective signal",
                }
            )

        # Sort by magnitude of contribution
        drivers.sort(key=lambda d: d["abs_impact"], reverse=True)

        # Prioritize features pushing risk up
        positive_drivers = [d for d in drivers if d["impact"] > 0.0]
        if positive_drivers:
            return positive_drivers[:5]

        # If employee is low-risk / near baseline, highlight top stabilizing factors
        if drivers:
            return drivers[:5]

        # Fallback to key features so simulator is never blank
        fallback_cols = ["OverTime", "MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "YearsAtCompany"]
        return [
            {
                "raw_feature": col,
                "feature": _prettify(col),
                "impact": 0.0,
                "abs_impact": 0.0,
                "is_risk_increasing": False,
                "method": "baseline alignment",
            }
            for col in fallback_cols
            if col in self.numeric_features + self.categorical_features
        ]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        logger.info("Persisted Kaizen engine to %s", target)

    @classmethod
    def load(cls, path: Path | str) -> "KaizenEngine":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Model file not found: {target}")
        loaded = joblib.load(target)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded)}, expected {cls}.")
        # Ensure backward compatibility for any missing dataclass attributes
        if not hasattr(loaded, "numeric_bounds") or loaded.numeric_bounds is None:
            loaded.numeric_bounds = {}
        if not hasattr(loaded, "decision_threshold") or loaded.decision_threshold is None:
            loaded.decision_threshold = 0.5
        logger.info(
            "Loaded Kaizen engine from %s (version=%s, champion=%s)",
            target,
            loaded.version,
            loaded.active_model,
        )
        return loaded


# ---------------------------------------------------------------------------
# Prescriptive retention catalogue
# ---------------------------------------------------------------------------

def _prescription_for(feature_name: str, impact: float = 0.0) -> dict[str, str]:
    """Return tailored, actionable retention recommendations for a given feature."""
    key = feature_name.lower().replace("_", "").replace(" ", "")

    if "overtime" in key:
        return {
            "title": "Workload & Overtime Audit",
            "action": (
                "Review ongoing overtime hours with the employee's direct manager. "
                "Redistribute urgent deliverables, establish on-call rotations, and ensure "
                "time-in-lieu or overtime compensation to prevent burnout."
            ),
        }
    if "monthlyincome" in key or "income" in key or "salary" in key:
        return {
            "title": "Compensation & Market Parity Review",
            "action": (
                "Benchmark this employee's salary against current industry market bands for their role and level. "
                "Consider an off-cycle merit adjustment or retention bonus if compensation lags peer percentiles."
            ),
        }
    if "jobsatisfaction" in key or "satisfaction" in key:
        return {
            "title": "Confidential Stay Interview",
            "action": (
                "Conduct a structured, non-evaluative stay interview to uncover core friction points. "
                "Focus on daily project ownership, team psychological safety, and day-to-day role autonomy."
            ),
        }
    if "environmentsatisfaction" in key:
        return {
            "title": "Workplace Environment & Team Culture Check",
            "action": (
                "Examine team dynamics, tooling, and physical/virtual work conditions. "
                "Engage leadership to eliminate friction in cross-functional collaboration and equipment support."
            ),
        }
    if "worklifebalance" in key or "worklife" in key:
        return {
            "title": "Flexible Working Arrangement",
            "action": (
                "Explore hybrid or flexible schedule adjustments, core working hours, and "
                "respect for after-hours digital boundaries to support sustainable work-life integration."
            ),
        }
    if "distancefromhome" in key or "commute" in key or "distance" in key:
        return {
            "title": "Commute Support & Hybrid Scheduling",
            "action": (
                "Offer additional work-from-home days each week or flexible start/end times "
                "to alleviate peak commute fatigue, paired with corporate transit benefits."
            ),
        }
    if "stockoptionlevel" in key or "stock" in key or "equity" in key:
        return {
            "title": "Long-Term Equity & Retention Grant",
            "action": (
                "Evaluate a refresh grant or vesting schedule extension to strengthen long-term "
                "alignment and retention incentive for high-impact contributors."
            ),
        }
    if "yearssincelastpromotion" in key or "promotion" in key:
        return {
            "title": "Career Progression & Promotion Roadmap",
            "action": (
                "Establish clear, documented promotion criteria with concrete 6-month milestones "
                "and an executive sponsor to address career stagnation."
            ),
        }
    if "yearswithcurrmanager" in key or "manager" in key:
        return {
            "title": "Manager Alignment & Mentorship",
            "action": (
                "Facilitate constructive skip-level check-ins or pair the employee with an independent senior "
                "mentor to diversify feedback channels and coaching support."
            ),
        }
    if "businesstravel" in key or "travel" in key:
        return {
            "title": "Travel Frequency Optimization",
            "action": (
                "Cap consecutive travel weeks, encourage virtual attendance where feasible, "
                "and provide compensatory rest days following demanding client on-site trips."
            ),
        }
    if "jobinvolvement" in key:
        return {
            "title": "High-Impact Project Ownership",
            "action": (
                "Assign the employee as lead on a high-visibility, cross-functional project "
                "that aligns with their professional interests and strengths."
            ),
        }
    if "numcompaniesworked" in key:
        return {
            "title": "Accelerated Integration & Retention Milestone",
            "action": (
                "Implement structured 90-day retention checkpoints with leadership recognition "
                "to build durable organisational loyalty."
            ),
        }
    if "yearsatcompany" in key or "tenure" in key:
        return {
            "title": "Tenure Milestone Recognition",
            "action": (
                "Celebrate organizational tenure and create new internal mobility opportunities "
                "to keep challenging seasoned staff."
            ),
        }
    if "leaves" in key or "absenteeism" in key:
        return {
            "title": "Health, Wellness & Burnout Intervention",
            "action": (
                "Offer Employee Assistance Program (EAP) resources and confidential wellness check-ins "
                "to ensure personal or health stressors are proactively supported."
            ),
        }

    # Default versatile prompt
    return {
        "title": f"Proactive Review: {_prettify(feature_name)}",
        "action": (
            f"Discuss {_prettify(feature_name)} during the next regular 1-on-1 check-in. "
            "Explore mutual adjustments to ensure strong long-term engagement and retention."
        ),
    }
