"""Blind Test of Kaizen Pre-Trained Engine Against Real-Life External Benchmark Cohort.

1. Takes external 1,470-employee cohort.
2. Completely strips the 'Attrition' column to create a blind dataset.
3. Feeds blind dataset into KaizenEngine (pre-trained stacking ensemble / champion).
4. Matches blind predictions against ground-truth Attrition.
5. Computes exact Confusion Matrix, ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, and Brier Score.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, brier_score_loss

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ml.engine import KaizenEngine

MODEL_PATH = Path("storage/models/kaizen_attrition_bundle.joblib")
RAW_DATA_PATH = Path("data/redsmoothy.csv")
BLIND_PATH = Path("data/real_life_blind_1470.csv")
TRUTH_PATH = Path("data/real_life_ground_truth_1470.csv")

def main():
    print("=" * 80)
    print("      KAIZEN BLIND EVALUATION: INDEPENDENT REAL-LIFE BENCHMARK COHORT")
    print("=" * 80)

    # 1. Load raw dataset
    df_raw = pd.read_csv(RAW_DATA_PATH)
    total_records = len(df_raw)
    print(f"[*] Loaded raw external dataset: {total_records:,} employee records")

    # 2. Extract ground truth and create blind file (NO ATTRITION)
    if "EmployeeNumber" in df_raw.columns:
        id_col = "EmployeeNumber"
    elif "EmployeeID" in df_raw.columns:
        id_col = "EmployeeID"
    else:
        df_raw["EmployeeID"] = [f"EMP-{i+1:05d}" for i in range(total_records)]
        id_col = "EmployeeID"

    # Separate ground truth
    df_truth = df_raw[[id_col, "Attrition"]].copy()
    df_truth.to_csv(TRUTH_PATH, index=False)
    print(f"[*] Saved Ground Truth dataset ({TRUTH_PATH.name}): {len(df_truth):,} rows with 'Attrition'")

    # Create BLIND dataset (WITHOUT Attrition)
    df_blind = df_raw.drop(columns=["Attrition"]).copy()
    df_blind.to_csv(BLIND_PATH, index=False)
    print(f"[*] Saved Blind dataset ({BLIND_PATH.name}): {len(df_blind):,} rows (ATTRITION COMPLETELY REMOVED)")
    print(f"    Blind columns: {list(df_blind.columns[:6])} ... (+{len(df_blind.columns)-6} more)")

    # 3. Load pre-trained Kaizen Engine
    print(f"\n[*] Loading pre-trained Kaizen model bundle: {MODEL_PATH}")
    engine = KaizenEngine.load(MODEL_PATH)
    print(f"    Active Champion Model : {engine.active_model}")
    print(f"    Decision Cutoff Thresh: {engine.decision_threshold:.4f}")

    # 4. Predict on Blind Data
    print(f"[*] Executing blind inference across all {total_records:,} employees...")
    probs = engine.predict_batch(df_blind, blend_ensemble=True)
    preds = (probs >= engine.decision_threshold).astype(int)

    # 5. Extract Truth Binary
    truth_binary = (df_truth["Attrition"].astype(str).str.lower().isin(["yes", "true", "1"])).astype(int).values
    actual_positives = int(np.sum(truth_binary))
    actual_negatives = total_records - actual_positives

    # 6. Confusion Matrix & Mathematical Metrics
    tn, fp, fn, tp = confusion_matrix(truth_binary, preds).ravel()
    accuracy = (tp + tn) / total_records
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    roc_auc = float(roc_auc_score(truth_binary, probs))
    pr_auc = float(average_precision_score(truth_binary, probs))
    brier = float(brier_score_loss(truth_binary, probs))

    print("\n" + "=" * 80)
    print("                MATHEMATICAL EVALUATION RESULTS ON BLIND DATA")
    print("=" * 80)
    print(f" Total Cohort Evaluated        : {total_records:,} employees")
    print(f" Real Attrition (Ground Truth) : {actual_positives:,} ({actual_positives/total_records*100:.2f}%)")
    print(f" Model Predicted Departures    : {tp + fp:,} ({(tp + fp)/total_records*100:.2f}%)")
    print(f" Decision Threshold Used       : {engine.decision_threshold:.4f}")
    print("-" * 80)
    print(" CONFUSION MATRIX:")
    print(f"   True Positives  (TP) : {tp:5d}  [Accurately flagged leavers]")
    print(f"   False Positives (FP) : {fp:5d}  [False alarms]")
    print(f"   True Negatives  (TN) : {tn:5d}  [Accurately confirmed retained]")
    print(f"   False Negatives (FN) : {fn:5d}  [Unpredicted departures]")
    print("-" * 80)
    print(" PERFORMANCE METRICS:")
    print(f"   ROC-AUC Score              : {roc_auc:.4f}  (Ability to rank risks)")
    print(f"   PR-AUC Score               : {pr_auc:.4f}")
    print(f"   Overall Accuracy           : {accuracy*100:.2f}%")
    print(f"   Recall / Sensitivity       : {recall*100:.2f}%  (Percentage of actual leavers caught)")
    print(f"   Precision                  : {precision*100:.2f}%  (Reliability of risk flags)")
    print(f"   F1-Score                   : {f1:.4f}")
    print(f"   Brier Calibration Score    : {brier:.4f}")
    print("=" * 80)

    # 7. Show Sample Blind Matches
    df_results = df_blind[[id_col, "Department", "JobRole", "MonthlyIncome"]].copy()
    df_results["Actual_Attrition"] = df_truth["Attrition"]
    df_results["Predicted_Risk_Pct"] = np.round(probs * 100, 1)
    df_results["Prediction"] = np.where(preds == 1, "Flight Risk (YES)", "Retained (NO)")
    df_results["Match"] = np.where((preds == 1) == (truth_binary == 1), "CORRECT MATCH", "MISMATCH")

    print("\n[+] SAMPLE PREDICTIONS (BLIND INPUT vs. ACTUAL OUTCOME):")
    sample_show = pd.concat([
        df_results[df_results["Match"] == "CORRECT MATCH"].head(6),
        df_results[df_results["Match"] == "MISMATCH"].head(4)
    ])
    for _, row in sample_show.iterrows():
        print(f"   {row[id_col]:<10} | {row['JobRole']:<26} | Risk: {row['Predicted_Risk_Pct']:>5.1f}% | Pred: {row['Prediction']:<18} | Actual: {row['Actual_Attrition']:<3} | [{row['Match']}]")

    print("=" * 80)

if __name__ == "__main__":
    main()
