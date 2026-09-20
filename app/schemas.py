"""API contracts for Kaizen's evaluated attrition-classification workflow."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Driver(APIModel):
    feature: str
    impact: float
    method: str
    shap_value: float | None = None
    direction: str = "risk_increasing"  # risk_increasing, protective, or baseline


class TeamContagionReport(APIModel):
    contagion_risk_level: str  # LOW, MODERATE, ELEVATED, CRITICAL
    team_density_pct: float
    contagion_multiplier: float
    contagion_adjusted_risk: float
    contagion_triggers: list[str] = []
    containment_strategy: str


class Prescription(APIModel):
    title: str
    action: str
    lift: float | None = None


class Financials(APIModel):
    turnover_replacement_cost: float
    retention_intervention_cost: float
    net_capital_saved: float


class PredictResponse(APIModel):
    risk_score: float
    risk_percentage: float
    drivers: list[Driver]
    prescriptions: list[Prescription]
    financials: Financials | None = None
    model_name: str
    model_version: str
    decision_threshold: float
    shap_base_value: float | None = None
    team_contagion: TeamContagionReport | None = None


class RocPoint(APIModel):
    fpr: float
    tpr: float
    threshold: float | None = None
    f1: float | None = None


class PrPoint(APIModel):
    recall: float
    precision: float
    threshold: float | None = None
    f1: float | None = None


class CalibrationPoint(APIModel):
    bin_midpoint: float
    prob_pred: float
    prob_true: float
    bin_count: int
    bin_name: str = ""


class BrierDecomposition(APIModel):
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float


class ThresholdMetricPoint(APIModel):
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    net_roi: float = 0.0


class SubgroupFairnessMetric(APIModel):
    group: str
    sample_count: int
    selection_count: int
    selection_rate: float
    tpr: float
    fpr: float
    disparate_impact_ratio: float
    four_fifths_passed: bool
    status: str = "PASS"  # PASS, WARNING, VIOLATION


class AttributeFairnessReport(APIModel):
    attribute_name: str
    subgroups: list[SubgroupFairnessMetric]
    reference_group: str
    disparate_impact_ratio: float
    four_fifths_passed: bool
    statistical_parity_diff: float
    demographic_parity_diff: float = 0.0
    tpr_disparity: float
    equal_opportunity_diff: float = 0.0
    fpr_disparity: float
    predictive_equality_diff: float = 0.0
    status: str = "PASS"


class FairnessAuditReport(APIModel):
    overall_compliant: bool
    fairness_score: float  # 0 to 100
    status: str = "PASS"  # PASS, WARNING, VIOLATION
    disparate_impact_ratio: float = 1.0
    demographic_parity_diff: float = 0.0
    equal_opportunity_diff: float = 0.0
    predictive_equality_diff: float = 0.0
    attributes: list[AttributeFairnessReport] = []
    recommendations: list[str] = []


class ConfusionMatrix(APIModel):
    tn: int
    fp: int
    fn: int
    tp: int


class ClassificationMetrics(APIModel):
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier_score: float
    # Matthews Correlation Coefficient — most informative single metric for
    # imbalanced binary classification (default 0.0 for backward compatibility
    # with model bundles saved before this field was introduced).
    mcc: float = 0.0
    # Per-model decision threshold optimised for F1 on the holdout set.
    decision_threshold: float = 0.5
    confusion: ConfusionMatrix
    roc_curve: list[RocPoint] = []
    pr_curve: list[PrPoint] = []
    calibration_curve: list[CalibrationPoint] = []
    threshold_metrics: list[ThresholdMetricPoint] = []
    ece: float = 0.0
    brier_decomposition: BrierDecomposition | None = None
    baseline_prevalence: float = 0.0
    hardware_device: str = "cpu"
    fairness_audit: FairnessAuditReport | None = None


class CandidateMetrics(ClassificationMetrics):
    model_name: str
    algorithm: str


class ValidationSummary(APIModel):
    method: str
    training_rows: int
    holdout_rows: int
    positive_rate_pct: float
    selection_metric: str
    note: str


class FeatureDefinition(APIModel):
    name: str
    kind: Literal["number", "category"]
    minimum: float | None = None
    maximum: float | None = None
    default: str | float
    choices: list[str]


class Telemetry(APIModel):
    total_rows: int
    total_cols: int
    missing_cells: int
    missing_pct: float
    numeric_features: int
    categorical_features: int
    attrition_rate: float
    target: str
    dropped_columns: list[str]


class FeatureImportance(APIModel):
    feature: str
    importance: float


class HighRiskEmployee(APIModel):
    id: str
    risk_score: float
    drivers: list[Driver]
    department: str
    job_role: str
    profile: dict[str, Any] = {}
    team_contagion: TeamContagionReport | None = None


class TrainResult(APIModel):
    source: str
    task_type: Literal["classification"]
    active_model: str
    metrics: ClassificationMetrics
    model_comparison: list[CandidateMetrics]
    validation: ValidationSummary
    importances: list[FeatureImportance]
    feature_schema: list[FeatureDefinition]
    model_version: str
    trained_at: str | None = None
    high_risk_employees: list[HighRiskEmployee] = []
    high_risk_threshold: float = 0.5


class TrainResponse(APIModel):
    status: Literal["success"]
    telemetry: Telemetry
    result: TrainResult
    sample_rows: list[dict]


class HealthResponse(APIModel):
    status: str
    system: str
    version: str
    # Populated from MODEL_DETAILS: {name → algorithm}
    candidates: dict[str, str] = {}
    trained: bool
    active_model: str | None = None


class ErrorResponse(APIModel):
    detail: str
