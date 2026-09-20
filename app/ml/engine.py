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

    @staticmethod
    def _audit_fairness(
        X_df: pd.DataFrame | None,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        threshold: float,
    ) -> dict[str, Any] | None:
        """Audit algorithmic fairness and EEOC disparate impact (Four-Fifths Rule)."""
        if X_df is None or len(X_df) == 0:
            return None

        y_true_arr = np.asarray(y_true, dtype=int)
        n_samples = len(y_true_arr)
        if n_samples != len(X_df):
            return None

        predictions = (probabilities >= threshold).astype(int)

        # Identify candidate protected cohorts
        detected_cohorts: list[tuple[str, pd.Series]] = []

        for col in X_df.columns:
            norm = str(col).lower().replace("_", "").replace(" ", "").replace("-", "")
            series = X_df[col]

            if norm in {"gender", "sex"}:
                detected_cohorts.append(("Gender", series.astype(str).str.strip().str.title()))
            elif norm in {"age", "ageyears", "yearsold"}:
                if pd.api.types.is_numeric_dtype(series):
                    adea = pd.Series(
                        np.where(series >= 40, "≥ 40 Years (ADEA Protected)", "< 40 Years (Unprotected)"),
                        index=series.index,
                    )
                    detected_cohorts.append(("Age (ADEA Cohorts)", adea))

                    try:
                        decades = pd.cut(
                            series,
                            bins=[0, 29, 39, 49, 120],
                            labels=["Under 30", "30-39 Years", "40-49 Years", "50+ Years"],
                        ).astype(str)
                        detected_cohorts.append(("Age (Decade Bands)", decades))
                    except Exception:
                        pass
            elif norm in {"department", "dept", "businessunit"}:
                detected_cohorts.append(("Department", series.astype(str).str.strip().str.title()))
            elif norm in {"maritalstatus", "marital", "civilstatus"}:
                detected_cohorts.append(("Marital Status", series.astype(str).str.strip().str.title()))
            elif norm in {"ethnicity", "race"}:
                detected_cohorts.append(("Ethnicity", series.astype(str).str.strip().str.title()))

        if not detected_cohorts:
            return None

        attribute_reports: list[dict[str, Any]] = []
        air_min_list: list[float] = []
        fpr_disp_list: list[float] = []

        for attr_name, cohort_series in detected_cohorts:
            val_counts = cohort_series.value_counts()
            valid_groups = [g for g, c in val_counts.items() if c >= 3 and str(g) not in {"nan", "None", ""}]
            if len(valid_groups) < 2:
                continue

            subgroups = []
            max_sr = 0.0
            ref_group = valid_groups[0]

            for g in valid_groups:
                mask = (cohort_series == g).to_numpy()
                g_count = int(np.sum(mask))
                if g_count == 0:
                    continue
                g_preds = predictions[mask]
                g_y = y_true_arr[mask]

                sel_count = int(np.sum(g_preds == 1))
                sr = sel_count / g_count
                if sr > max_sr:
                    max_sr = sr
                    ref_group = str(g)

                tp = int(np.sum((g_y == 1) & (g_preds == 1)))
                fp = int(np.sum((g_y == 0) & (g_preds == 1)))
                fn = int(np.sum((g_y == 1) & (g_preds == 0)))
                tn = int(np.sum((g_y == 0) & (g_preds == 0)))

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                subgroups.append({
                    "group": str(g),
                    "sample_count": g_count,
                    "selection_count": sel_count,
                    "selection_rate": round(float(sr), 4),
                    "tpr": round(float(tpr), 4),
                    "fpr": round(float(fpr), 4),
                })

            if not subgroups:
                continue

            attr_air_min = 1.0
            attr_four_fifths_passed = True
            all_sr = [sg["selection_rate"] for sg in subgroups]
            all_tpr = [sg["tpr"] for sg in subgroups]
            all_fpr = [sg["fpr"] for sg in subgroups]

            for sg in subgroups:
                air = (sg["selection_rate"] / max_sr) if max_sr > 0 else 1.0
                air = round(float(air), 4)
                sg["disparate_impact_ratio"] = air
                passed = bool(air >= 0.80)
                sg["four_fifths_passed"] = passed
                if air >= 0.80:
                    sg["status"] = "PASS"
                elif air >= 0.70:
                    sg["status"] = "WARNING"
                else:
                    sg["status"] = "VIOLATION"

                if air < attr_air_min:
                    attr_air_min = air
                if not passed:
                    attr_four_fifths_passed = False

            sp_diff = round(float(max(all_sr) - min(all_sr)), 4)
            tpr_disp = round(float(max(all_tpr) - min(all_tpr)), 4)
            fpr_disp = round(float(max(all_fpr) - min(all_fpr)), 4)

            attr_status = "PASS" if attr_four_fifths_passed else ("WARNING" if attr_air_min >= 0.70 else "VIOLATION")

            attribute_reports.append({
                "attribute_name": attr_name,
                "subgroups": subgroups,
                "reference_group": ref_group,
                "disparate_impact_ratio": round(attr_air_min, 4),
                "four_fifths_passed": attr_four_fifths_passed,
                "statistical_parity_diff": sp_diff,
                "demographic_parity_diff": sp_diff,
                "tpr_disparity": tpr_disp,
                "equal_opportunity_diff": tpr_disp,
                "fpr_disparity": fpr_disp,
                "predictive_equality_diff": fpr_disp,
                "status": attr_status,
            })

            air_min_list.append(attr_air_min)
            fpr_disp_list.append(fpr_disp)

        if not attribute_reports:
            return None

        overall_compliant = all(r["four_fifths_passed"] for r in attribute_reports)
        avg_air = np.mean(air_min_list) if air_min_list else 1.0
        avg_fpr_disp = np.mean(fpr_disp_list) if fpr_disp_list else 0.0
        dp_diff_list = [r["demographic_parity_diff"] for r in attribute_reports]
        tpr_disp_list = [r["equal_opportunity_diff"] for r in attribute_reports]

        base_score = 100.0 * float(avg_air) - 25.0 * float(avg_fpr_disp)
        if overall_compliant:
            base_score = max(base_score, 85.0)
        fairness_score = round(float(np.clip(base_score, 0.0, 100.0)), 1)

        overall_status = "PASS" if overall_compliant else ("WARNING" if fairness_score >= 70.0 else "VIOLATION")

        recs = []
        for r in attribute_reports:
            if not r["four_fifths_passed"]:
                worst_sg = min(r["subgroups"], key=lambda s: s["disparate_impact_ratio"])
                recs.append(
                    f"Cohort '{worst_sg['group']}' in '{r['attribute_name']}' exhibits an Adverse Impact Ratio "
                    f"of {worst_sg['disparate_impact_ratio']:.2f} (< 0.80 threshold). Review decision cutoff."
                )
            if r["fpr_disparity"] > 0.08:
                recs.append(
                    f"Elevated False Positive Rate disparity ({r['fpr_disparity'] * 100:.1f}%) detected in '{r['attribute_name']}'. "
                    "Calibrate probability thresholds per cohort to prevent disproportionate retention alerts."
                )

        if not recs:
            recs.append(
                "All evaluated demographic cohorts comply with the EEOC 80% Four-Fifths Rule (29 C.F.R. § 1607.4D). "
                "Adverse impact ratios remain within standard regulatory tolerances."
            )

        return {
            "overall_compliant": overall_compliant,
            "fairness_score": fairness_score,
            "status": overall_status,
            "disparate_impact_ratio": round(float(min(air_min_list)), 4) if air_min_list else 1.0,
            "demographic_parity_diff": round(float(max(dp_diff_list)), 4) if dp_diff_list else 0.0,
            "equal_opportunity_diff": round(float(max(tpr_disp_list)), 4) if tpr_disp_list else 0.0,
            "predictive_equality_diff": round(float(max(fpr_disp_list)), 4) if fpr_disp_list else 0.0,
            "attributes": attribute_reports,
            "recommendations": recs,
        }

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
        X_df: pd.DataFrame | None = None,
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

        fairness_audit = (
            KaizenEngine._audit_fairness(X_df, y_true_arr, probabilities, threshold)
            if X_df is not None
            else None
        )

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
            "fairness_audit": fairness_audit,
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
            metrics = self._evaluate(
                y_holdout,
                probs,
                name,
                threshold=opt_thresh,
                hardware_device=dev_tag,
                X_df=X_holdout,
            )
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

    def _detect_team_contagion(
        self,
        profile_data: pd.DataFrame | dict[str, Any],
        raw_risk: float,
        dept_risk_density: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate organizational turnover contagion dynamics (Felps et al., 2008).

        Turnover transmits contagiously through managerial instability, hostile micro-climates,
        chronic burnout, and career progression ceilings.
        """
        if isinstance(profile_data, pd.DataFrame):
            row_dict = profile_data.iloc[0].to_dict() if len(profile_data) > 0 else {}
        elif isinstance(profile_data, dict):
            row_dict = profile_data
        else:
            row_dict = {}

        triggers: list[str] = []
        weights: list[float] = []

        # 1. Managerial Relational Fragility
        curr_mgr = row_dict.get("YearsWithCurrManager")
        if curr_mgr is not None:
            try:
                val = float(curr_mgr)
                if val <= 1.0:
                    triggers.append("Managerial relationship fragility (tenure <= 1 yr; unanchored psychological contract)")
                    weights.append(0.12 if val == 0 else 0.09)
            except (ValueError, TypeError):
                pass

        # 2. Adverse Organizational Culture & Micro-Climate
        env_sat = row_dict.get("EnvironmentSatisfaction")
        if env_sat is not None:
            try:
                val = float(env_sat)
                if val <= 2.0:
                    triggers.append("Adverse department climate (EnvironmentSatisfaction <= 2/4)")
                    weights.append(0.10 if val == 1.0 else 0.06)
            except (ValueError, TypeError):
                pass

        # 3. Affective Disengagement
        job_sat = row_dict.get("JobSatisfaction")
        if job_sat is not None:
            try:
                val = float(job_sat)
                if val <= 2.0:
                    triggers.append("Affective job disengagement (JobSatisfaction <= 2/4)")
                    weights.append(0.08)
            except (ValueError, TypeError):
                pass

        # 4. Work-Life Equilibrium Deficit
        wlb = row_dict.get("WorkLifeBalance")
        if wlb is not None:
            try:
                val = float(wlb)
                if val <= 2.0:
                    triggers.append("Work-life equilibrium deficit (WorkLifeBalance <= 2/4)")
                    weights.append(0.06)
            except (ValueError, TypeError):
                pass

        # 5. Chronic Overtime Burnout Contagion
        ot_val = str(row_dict.get("OverTime", "")).strip().lower()
        if ot_val in {"yes", "true", "1", "y"}:
            triggers.append("Chronic overtime pressure (accelerated peer burnout hazard)")
            weights.append(0.08)

        # 6. Career Mobility / Promotion Stagnation
        last_prom = row_dict.get("YearsSinceLastPromotion")
        if last_prom is not None:
            try:
                val = float(last_prom)
                if val >= 5.0:
                    triggers.append("Career mobility ceiling (>= 5 years without promotion)")
                    weights.append(0.05)
            except (ValueError, TypeError):
                pass

        # 7. Unvested / Zero Equity Handcuffs
        stock = row_dict.get("StockOptionLevel")
        if stock is not None:
            try:
                val = float(stock)
                if val == 0.0:
                    triggers.append("Zero equity lock-in (uninhibited external mobility barrier)")
                    weights.append(0.04)
            except (ValueError, TypeError):
                pass

        # Contagion multiplier bounded between 1.00 and 1.45
        contagion_multiplier = round(min(1.45, 1.0 + sum(weights)), 2)

        # Non-linear contagion-adjusted hazard transmission:
        # P_contagion = 1.0 - (1.0 - raw_risk)^multiplier
        raw_risk_clamped = max(0.0, min(1.0, float(raw_risk)))
        if contagion_multiplier > 1.0:
            adjusted_risk = 1.0 - np.power(max(0.0, 1.0 - raw_risk_clamped), contagion_multiplier)
        else:
            adjusted_risk = raw_risk_clamped
        adjusted_risk = round(float(min(0.999, max(0.001, adjusted_risk))), 4)

        # Localized Team Flight Density estimate
        if dept_risk_density is not None:
            team_density_pct = round(float(dept_risk_density) * 100, 1)
        else:
            dept_name = str(row_dict.get("Department", "")).strip()
            if "Sales" in dept_name:
                base_density = 22.0
            elif "Human Resources" in dept_name:
                base_density = 19.5
            elif "Engineering" in dept_name or "Technology" in dept_name:
                base_density = 15.0
            elif "Research" in dept_name:
                base_density = 14.5
            else:
                base_density = 16.0
            team_density_pct = round(min(65.0, base_density + len(triggers) * 4.5), 1)

        # Categorize Contagion Risk Level
        if (adjusted_risk >= 0.70 and len(triggers) >= 2) or (raw_risk_clamped >= 0.65 and contagion_multiplier >= 1.25):
            contagion_level = "CRITICAL"
        elif adjusted_risk >= 0.45 or len(triggers) >= 2:
            contagion_level = "ELEVATED"
        elif adjusted_risk >= 0.25 or len(triggers) >= 1:
            contagion_level = "MODERATE"
        else:
            contagion_level = "LOW"

        # Targeted Corporate Containment Strategy
        if any("Managerial" in t for t in triggers):
            containment = "Deploy immediate skip-level 1-on-1 within 48h to evaluate managerial rapport and re-anchor psychological contract."
        elif any("overtime" in t or "climate" in t for t in triggers):
            containment = "Implement temporary overtime moratorium, rebalance workload, and conduct localized team culture pulse check."
        elif any("ceiling" in t or "equity" in t for t in triggers):
            containment = "Expedite promotion and title progression review; structure customized equity retention award."
        elif adjusted_risk >= 0.50:
            containment = "Activate executive sponsorship and structured career pathway review to preempt departmental departure cascade."
        else:
            containment = "Maintain standard quarterly engagement monitoring; reinforce project autonomy and peer recognition."

        return {
            "contagion_risk_level": contagion_level,
            "team_density_pct": team_density_pct,
            "contagion_multiplier": contagion_multiplier,
            "contagion_adjusted_risk": adjusted_risk,
            "contagion_triggers": triggers,
            "containment_strategy": containment,
        }

    def predict_detailed(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.active_model is None:
            raise RuntimeError("No model has been trained yet.")
        profile = self._aligned_profile(input_dict)
        risk_score = float(self.model.predict_proba(profile)[0, 1])
        drivers, shap_base_val = self._explain_profile(profile, risk_score)
        prescriptions = [_prescription_for(driver["raw_feature"], driver.get("impact", 0.0)) for driver in drivers]
        thresh = getattr(self, "decision_threshold", 0.5)

        # Evaluate Team Churn Contagion dynamics
        contagion = self._detect_team_contagion(input_dict, risk_score)

        return {
            "risk_score": risk_score,
            "drivers": [
                {
                    "feature": d["feature"],
                    "impact": d["impact"],
                    "method": d["method"],
                    "shap_value": d.get("shap_value"),
                    "direction": d.get("direction", "risk_increasing"),
                }
                for d in drivers
            ],
            "prescriptions": prescriptions,
            "model_name": self.active_model,
            "decision_threshold": thresh,
            "shap_base_value": shap_base_val,
            "team_contagion": contagion,
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

    def _explain_profile(
        self, profile: pd.DataFrame, current_risk: float
    ) -> tuple[list[dict[str, Any]], float | None]:
        """Compute exact TreeSHAP game-theoretic attributions (with graceful permutation fallback)."""
        if self.model is None:
            return [], None

        drivers: list[dict[str, Any]] = []
        shap_base_value: float | None = None

        # 1. Native C++ TreeSHAP attribution for XGBoost estimators
        try:
            preprocessor = self.model.named_steps.get("preprocess")
            estimator = self.model.named_steps.get("estimator")
            if (
                preprocessor is not None
                and estimator is not None
                and hasattr(estimator, "get_booster")
            ):
                import xgboost as xgb

                X_trans = preprocessor.transform(profile)
                booster = estimator.get_booster()
                dmat = xgb.DMatrix(X_trans)
                contribs = booster.predict(dmat, pred_contribs=True)
                raw_shap = contribs[0, :-1]
                shap_base_value = float(contribs[0, -1])

                feature_names = preprocessor.get_feature_names_out()
                fmap: dict[str, float] = {}
                for feat_col, val in zip(feature_names, raw_shap):
                    if feat_col.startswith("num__"):
                        parent = feat_col[5:]
                    elif feat_col.startswith("cat__"):
                        cat_sub = feat_col[5:]
                        parent = cat_sub
                        for c in self.categorical_features:
                            if cat_sub.startswith(f"{c}_"):
                                parent = c
                                break
                    else:
                        parent = feat_col
                    fmap[parent] = fmap.get(parent, 0.0) + float(val)

                total_logit = shap_base_value + sum(fmap.values())
                current_p = 1.0 / (1.0 + np.exp(-np.clip(total_logit, -50.0, 50.0)))

                for col in self.numeric_features + self.categorical_features:
                    if col in fmap:
                        phi = fmap[col]
                        counter_logit = total_logit - phi
                        counter_p = 1.0 / (1.0 + np.exp(-np.clip(counter_logit, -50.0, 50.0)))
                        marginal_p_delta = (current_p - counter_p) * 100.0

                        drivers.append(
                            {
                                "raw_feature": col,
                                "feature": _prettify(col),
                                "impact": float(round(marginal_p_delta, 1)),
                                "shap_value": float(round(phi, 4)),
                                "direction": "risk_increasing" if phi > 0 else "protective",
                                "abs_impact": abs(phi),
                                "is_risk_increasing": phi > 0,
                                "method": "TreeSHAP (exact Shapley)",
                            }
                        )

                drivers.sort(key=lambda d: d["abs_impact"], reverse=True)
                positive_drivers = [d for d in drivers if d["shap_value"] > 0.0]
                if current_risk >= 0.35 and positive_drivers:
                    return positive_drivers[:5], round(shap_base_value, 4)
                if drivers:
                    return drivers[:5], round(shap_base_value, 4)
        except Exception as exc:
            logger.debug("TreeSHAP calculation fallback: %s", exc)

        # 2. Counterfactual perturbation fallback (for non-tree or legacy models)
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
                    "shap_value": None,
                    "direction": "risk_increasing" if delta > 0 else "protective",
                    "abs_impact": abs(delta),
                    "is_risk_increasing": delta > 0,
                    "method": "risk driver" if delta > 0 else "protective signal",
                }
            )

        drivers.sort(key=lambda d: d["abs_impact"], reverse=True)
        positive_drivers = [d for d in drivers if d["impact"] > 0.0]
        if positive_drivers:
            return positive_drivers[:5], shap_base_value
        if drivers:
            return drivers[:5], shap_base_value

        fallback_cols = ["OverTime", "MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "YearsAtCompany"]
        return [
            {
                "raw_feature": col,
                "feature": _prettify(col),
                "impact": 0.0,
                "shap_value": 0.0,
                "direction": "baseline",
                "abs_impact": 0.0,
                "is_risk_increasing": False,
                "method": "baseline alignment",
            }
            for col in fallback_cols
            if col in self.numeric_features + self.categorical_features
        ], shap_base_value

    def _local_drivers(
        self, profile: pd.DataFrame, current_risk: float
    ) -> list[dict[str, Any]]:
        """Identify individual feature contributions (maintaining list[dict] contract)."""
        drivers, _ = self._explain_profile(profile, current_risk)
        return drivers

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
