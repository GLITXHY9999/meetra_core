"""Evaluation and Ground-Truth Matcher for Kaizen Model Laboratory.

Compares model predictions against the real ground-truth labels in:
data/test_workforce_ground_truth.csv

Usage:
  # Option A: Evaluate directly via the live Kaizen server API or active engine:
  python match_predictions.py

  # Option B: Evaluate a CSV exported from the Kaizen Web UI:
  python match_predictions.py --predictions scored_workforce_predictions.csv
  python match_predictions.py -p "C:\\Users\\Manmeet\\Downloads\\scored_workforce_predictions.csv"
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_BLIND_PATH = DATA_DIR / "test_workforce_blind.csv"
DEFAULT_TRUTH_PATH = DATA_DIR / "test_workforce_ground_truth.csv"


def score_via_api(blind_path: Path, api_url: str = "http://127.0.0.1:8000/api/batch-predict/export") -> pd.DataFrame:
    """Send blind dataset to live Kaizen server and return scored DataFrame."""
    import urllib.request
    import io

    print(f"[*] Querying live Kaizen inference server ({api_url})...")
    with open(blind_path, "rb") as f:
        file_bytes = f.read()

    boundary = "----KaizenFormBoundary" + os.urandom(16).hex()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{blind_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    return pd.read_csv(io.BytesIO(content))


def score_via_engine(blind_path: Path) -> pd.DataFrame:
    """Fallback: score using local Kaizen engine bundle directly."""
    import joblib
    model_path = ROOT_DIR / "storage" / "models" / "kaizen_attrition_bundle.joblib"
    print(f"[*] Scoring using local model bundle ({model_path.name})...")
    engine = joblib.load(model_path)
    df = pd.read_csv(blind_path)
    probs = engine.predict_batch(df)
    thresh = float(engine.decision_threshold)

    scored_df = df.copy()
    scored_df["Predicted_Attrition_Risk_Pct"] = np.round(probs * 100, 2)
    scored_df["Risk_Category"] = np.where(probs >= 0.70, "Critical", np.where(probs >= 0.40, "Elevated", "Low"))
    scored_df["Retention_Action_Required"] = np.where(probs >= thresh, "YES", "NO")
    scored_df["_prob"] = probs
    return scored_df


def match_and_evaluate(scored_df: pd.DataFrame, truth_df: pd.DataFrame):
    """Align predictions with ground truth and display rigorous scientific evaluation."""
    # Align on EmployeeID if present, else on index
    if "EmployeeID" in scored_df.columns and "EmployeeID" in truth_df.columns:
        merged = pd.merge(truth_df, scored_df[["EmployeeID", "Predicted_Attrition_Risk_Pct", "Risk_Category", "Retention_Action_Required"]], on="EmployeeID", how="inner")
    else:
        merged = truth_df.copy()
        merged["Predicted_Attrition_Risk_Pct"] = scored_df["Predicted_Attrition_Risk_Pct"].values
        merged["Risk_Category"] = scored_df["Risk_Category"].values
        merged["Retention_Action_Required"] = scored_df["Retention_Action_Required"].values

    # True binary outcome (1 for Attrition=Yes, 0 for Attrition=No)
    y_true = (merged["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int).values
    
    # Predicted probabilities (0.0 to 1.0)
    y_prob = (merged["Predicted_Attrition_Risk_Pct"].values / 100.0)
    
    # Predicted binary decision (1 for Retention_Action_Required=YES, 0 for NO)
    y_pred = (merged["Retention_Action_Required"].astype(str).str.strip().str.upper() == "YES").astype(int).values

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 70)
    print("      KAIZEN MODEL LABORATORY — GROUND TRUTH BENCHMARK MATCH")
    print("=" * 70)
    print(f" Total Workforce Evaluated : {len(merged):,} employees")
    print(f" Real Attrition (Ground Truth): {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f" Model Predicted Attrition : {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    print("-" * 70)
    print(" CONFUSION MATRIX:")
    print(f"   True Positives  (TP) : {tp:4d}  [Correctly identified leavers]")
    print(f"   False Positives (FP) : {fp:4d}  [Retained flagged as leavers - false alarms]")
    print(f"   True Negatives  (TN) : {tn:4d}  [Correctly identified retained staff]")
    print(f"   False Negatives (FN) : {fn:4d}  [Unpredicted leavers - blind spots]")
    print("-" * 70)
    print(" DISCRIMINATIVE & ACCURACY METRICS:")
    print(f"   ROC-AUC Area Under Curve : {roc_auc:.4f}  (Discrimination capability)")
    print(f"   PR-AUC Average Precision : {pr_auc:.4f}  (Precision-Recall trade-off)")
    print(f"   Overall Accuracy         : {acc*100:.2f}%")
    print(f"   Recall / Sensitivity     : {rec*100:.2f}%  (Coverage of true turnover)")
    print(f"   Precision / PPV          : {prec*100:.2f}%  (Reliability of turnover alerts)")
    print(f"   Specificity (TNR)        : {spec*100:.2f}%")
    print(f"   Balanced F1-Score        : {f1:.4f}")
    print(f"   Brier Score (Calibration): {brier:.4f}  (Lower is better, 0 = perfect)")
    print("=" * 70)

    # Top true positives preview
    tp_sample = merged[(merged["Attrition"].str.lower() == "yes") & (merged["Retention_Action_Required"].str.upper() == "YES")]
    print("\n[+] SAMPLE TOP CORRECTLY PREDICTED FLIGHT-RISK EMPLOYEES (TRUE POSITIVES):")
    print(f"{'EmployeeID':<12} | {'Dept':<25} | {'Role':<25} | {'Risk %':<8} | {'GroundTruth':<12}")
    print("-" * 90)
    for _, row in tp_sample.head(6).iterrows():
        print(f"{row['EmployeeID']:<12} | {str(row['Department'])[:25]:<25} | {str(row['JobRole'])[:25]:<25} | {row['Predicted_Attrition_Risk_Pct']:>5.1f}%  | {row['Attrition']:<12}")

    # Top false negatives preview if any
    fn_sample = merged[(merged["Attrition"].str.lower() == "yes") & (merged["Retention_Action_Required"].str.upper() == "NO")]
    if len(fn_sample) > 0:
        print(f"\n[-] SAMPLE MISSED CASES (FALSE NEGATIVES - {len(fn_sample)} total):")
        print(f"{'EmployeeID':<12} | {'Dept':<25} | {'Role':<25} | {'Risk %':<8} | {'GroundTruth':<12}")
        print("-" * 90)
        for _, row in fn_sample.head(4).iterrows():
            print(f"{row['EmployeeID']:<12} | {str(row['Department'])[:25]:<25} | {str(row['JobRole'])[:25]:<25} | {row['Predicted_Attrition_Risk_Pct']:>5.1f}%  | {row['Attrition']:<12}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Ground-truth verification complete.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Match Kaizen predictions against ground truth.")
    parser.add_argument(
        "-p", "--predictions",
        type=str,
        default=None,
        help="Path to scored CSV exported from Kaizen Web UI or API. If omitted, scores test_workforce_blind.csv automatically."
    )
    parser.add_argument(
        "-t", "--truth",
        type=str,
        default=str(DEFAULT_TRUTH_PATH),
        help="Path to ground truth CSV (default: data/test_workforce_ground_truth.csv)"
    )
    parser.add_argument(
        "-b", "--blind",
        type=str,
        default=str(DEFAULT_BLIND_PATH),
        help="Path to blind CSV (default: data/test_workforce_blind.csv)"
    )
    args = parser.parse_args()

    truth_path = Path(args.truth)
    if not truth_path.exists():
        print(f"[ERROR] Ground truth file not found: {truth_path}")
        sys.exit(1)
    truth_df = pd.read_csv(truth_path)

    if args.predictions:
        scored_path = Path(args.predictions)
        if not scored_path.exists():
            # Check user's Downloads folder as a convenience
            downloads_path = Path.home() / "Downloads" / args.predictions
            if downloads_path.exists():
                scored_path = downloads_path
            else:
                print(f"[ERROR] Scored predictions file not found: {args.predictions}")
                sys.exit(1)
        print(f"[*] Loading scored predictions from: {scored_path.resolve()}")
        scored_df = pd.read_csv(scored_path)
    else:
        blind_path = Path(args.blind)
        if not blind_path.exists():
            print(f"[ERROR] Blind test dataset not found: {blind_path}")
            sys.exit(1)
        # Try scoring via live API first, fallback to engine
        try:
            scored_df = score_via_api(blind_path)
        except Exception as e:
            print(f"[!] Server API connection failed ({e}). Falling back to local model bundle...")
            scored_df = score_via_engine(blind_path)

    match_and_evaluate(scored_df, truth_df)


if __name__ == "__main__":
    main()
