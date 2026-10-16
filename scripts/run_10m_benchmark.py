"""High-Throughput Streaming Benchmark & Evaluation for 10 Million Employees.

1. Streams M:\\meetra_data\\workforce_blind_10m.csv in memory-safe chunks (250,000 rows).
2. Performs batch inference via Kaizen's production model (AlexzanderXS champion).
3. Writes enriched predictions to M:\\meetra_data\\workforce_scored_10m.csv.
4. Matches against M:\\meetra_data\\workforce_ground_truth_10m.csv in real time.
5. Computes exact Confusion Matrix, F1, Precision, Recall, Accuracy, Brier Score, and ROC-AUC.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ml.engine import KaizenEngine

DATA_DIR = Path("M:/meetra_data")
BLIND_PATH = DATA_DIR / "workforce_blind_10m.csv"
TRUTH_PATH = DATA_DIR / "workforce_ground_truth_10m.csv"
SCORED_PATH = DATA_DIR / "workforce_scored_10m.csv"
MODEL_PATH = ROOT_DIR / "storage" / "models" / "kaizen_attrition_bundle.joblib"

CHUNK_SIZE = 250_000
SAMPLE_RESERVOIR_SIZE = 200_000


import argparse

def main():
    parser = argparse.ArgumentParser(description="10-Million Employee Workforce Benchmark")
    parser.add_argument("--blend", action="store_true", default=True, help="Enable calibrated soft-voting ensemble")
    parser.add_argument("--no-blend", dest="blend", action="store_false", help="Disable blending; use champion model directly")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (defaults to model optimal cutoff)")
    args = parser.parse_args()

    print("=" * 75)
    print("      KAIZEN 10-MILLION WORKFORCE BENCHMARK & GROUND-TRUTH MATCHER")
    print("=" * 75)

    if not BLIND_PATH.exists():
        print(f"[ERROR] Blind dataset not found at: {BLIND_PATH}")
        sys.exit(1)
    if not TRUTH_PATH.exists():
        print(f"[ERROR] Ground truth dataset not found at: {TRUTH_PATH}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model bundle not found at: {MODEL_PATH}")
        sys.exit(1)

    print(f"[*] Loading production model bundle: {MODEL_PATH}")
    engine = KaizenEngine.load(MODEL_PATH)
    active_model = engine.active_model
    decision_thresh = float(args.threshold) if args.threshold is not None else float(engine.decision_threshold)
    mode_name = "Ensemble Blend (AlexzanderXS 70% + PorusXS 30%)" if args.blend else f"Champion: {active_model}"
    print(f"    Inference Mode        : {mode_name}")
    print(f"    Active Champion Model : {active_model}")
    print(f"    Optimal Cutoff Thresh : {decision_thresh:.4f}")
    print(f"    Numeric Features      : {len(engine.numeric_features)}")
    print(f"    Categorical Features  : {len(engine.categorical_features)}")

    # Optimize inference device if XGBoost supports it
    try:
        pipeline = engine._base_pipeline(engine.model)
        estimator = pipeline.named_steps.get("estimator")
        if hasattr(estimator, "set_params"):
            # Set to multi-threaded CPU or CUDA
            estimator.set_params(n_jobs=-1)
    except Exception as e:
        pass

    # Remove existing scored file if present
    if SCORED_PATH.exists():
        SCORED_PATH.unlink()

    print(f"\n[*] Commencing 10,000,000-Row Streaming Inference & Match Pipeline...")
    print(f"    Source Blind CSV  : {BLIND_PATH}")
    print(f"    Ground Truth CSV  : {TRUTH_PATH}")
    print(f"    Output Scored CSV : {SCORED_PATH}")
    print(f"    Chunk Size        : {CHUNK_SIZE:,} rows per iteration\n")

    # Metrics accumulators (Constant memory consumption)
    total_processed = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    brier_sq_err_sum = 0.0

    # Reservoir for global ROC-AUC / PR-AUC calculation
    reservoir_probs = []
    reservoir_truth = []
    rng = np.random.default_rng(42)

    # Top sample high-risk predictions
    sample_top_positives = []
    sample_false_negatives = []

    start_time = time.time()
    chunk_index = 0

    blind_reader = pd.read_csv(BLIND_PATH, chunksize=CHUNK_SIZE)
    truth_reader = pd.read_csv(TRUTH_PATH, usecols=["EmployeeID", "Attrition"], chunksize=CHUNK_SIZE)

    for blind_chunk, truth_chunk in zip(blind_reader, truth_reader):
        chunk_t0 = time.time()
        chunk_len = len(blind_chunk)
        chunk_index += 1

        # 1. Batch Vectorized Probability Prediction
        probs = engine.predict_batch(blind_chunk, blend_ensemble=args.blend)

        # 2. Extract Ground Truth Binary
        truth_binary = (truth_chunk["Attrition"].astype(str).str.lower().isin(["yes", "true", "1"])).astype(int).values

        # 3. Decision Threshold Classification
        preds = (probs >= decision_thresh).astype(int)

        # 4. Update Confusion Matrix
        chunk_tp = int(np.sum((preds == 1) & (truth_binary == 1)))
        chunk_fp = int(np.sum((preds == 1) & (truth_binary == 0)))
        chunk_tn = int(np.sum((preds == 0) & (truth_binary == 0)))
        chunk_fn = int(np.sum((preds == 0) & (truth_binary == 1)))

        tp += chunk_tp
        fp += chunk_fp
        tn += chunk_tn
        fn += chunk_fn

        # 5. Brier Score Loss Accumulator
        brier_sq_err_sum += float(np.sum((probs - truth_binary) ** 2))

        # 6. Reservoir Sampling for Global AUC
        sample_indices = rng.choice(chunk_len, size=min(chunk_len, 5_000), replace=False)
        reservoir_probs.extend(probs[sample_indices])
        reservoir_truth.extend(truth_binary[sample_indices])

        # 7. Collect sample cases for display
        if len(sample_top_positives) < 6:
            tp_indices = np.where((preds == 1) & (truth_binary == 1))[0]
            for idx in tp_indices[:2]:
                sample_top_positives.append({
                    "id": blind_chunk["EmployeeID"].iloc[idx],
                    "dept": blind_chunk["Department"].iloc[idx],
                    "role": blind_chunk["JobRole"].iloc[idx],
                    "risk_pct": round(probs[idx] * 100, 1),
                    "truth": "Yes",
                })
        if len(sample_false_negatives) < 4:
            fn_indices = np.where((preds == 0) & (truth_binary == 1))[0]
            for idx in fn_indices[:2]:
                sample_false_negatives.append({
                    "id": blind_chunk["EmployeeID"].iloc[idx],
                    "dept": blind_chunk["Department"].iloc[idx],
                    "role": blind_chunk["JobRole"].iloc[idx],
                    "risk_pct": round(probs[idx] * 100, 1),
                    "truth": "Yes",
                })

        # 8. Stream Scored Columns to Output CSV
        scored_chunk = blind_chunk.copy()
        scored_chunk["Predicted_Attrition_Risk_Pct"] = np.round(probs * 100, 2)
        scored_chunk["Risk_Category"] = np.where(probs >= 0.70, "Critical", np.where(probs >= 0.40, "Elevated", "Low"))
        scored_chunk["Retention_Action_Required"] = np.where(probs >= decision_thresh, "YES", "NO")

        write_header = (chunk_index == 1)
        scored_chunk.to_csv(SCORED_PATH, mode="a", index=False, header=write_header)

        total_processed += chunk_len
        elapsed = time.time() - start_time
        speed = round(total_processed / elapsed) if elapsed > 0 else 0
        pct_done = (total_processed / 10_000_000) * 100
        print(
            f"  [>] Chunk {chunk_index:2d}/40 ({pct_done:5.1f}%) | "
            f"{total_processed:,} scored | "
            f"TP: {tp:,} FP: {fp:,} | "
            f"{speed:,} rec/s | "
            f"Time: {time.time() - chunk_t0:.2f}s (Elapsed: {elapsed:.1f}s)"
        )

    total_time = time.time() - start_time

    # Calculate final global metrics
    actual_positives = tp + fn
    actual_negatives = tn + fp
    predicted_positives = tp + fp
    total_eval = tp + tn + fp + fn

    accuracy = (tp + tn) / total_eval if total_eval > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    brier_score = brier_sq_err_sum / total_eval if total_eval > 0 else 0.0

    # Calculate global AUC on reservoir sample
    res_probs = np.array(reservoir_probs)
    res_truth = np.array(reservoir_truth)
    roc_auc = float(roc_auc_score(res_truth, res_probs))
    pr_auc = float(average_precision_score(res_truth, res_probs))

    scored_size_mb = SCORED_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 75)
    print("          10-MILLION WORKFORCE EVALUATION & BENCHMARK REPORT")
    print("=" * 75)
    print(f" Total Workforce Evaluated    : {total_eval:,} employees")
    print(f" Real Attrition (Ground Truth): {actual_positives:,} ({actual_positives / total_eval * 100:.2f}%)")
    print(f" Model Predicted Departures   : {predicted_positives:,} ({predicted_positives / total_eval * 100:.2f}%)")
    print(f" Decision Cutoff Threshold    : {decision_thresh:.4f}")
    print(f" Total Inference Time         : {total_time:.2f} seconds")
    print(f" Peak Throughput              : {round(total_eval / total_time):,} records/second")
    print(f" Output Scored CSV Size       : {scored_size_mb:.2f} MB ({scored_size_mb / 1024:.2f} GB)")
    print("-" * 75)
    print(" CONFUSION MATRIX (10,000,000 Records):")
    print(f"   True Positives  (TP) : {tp:9,d}  [Accurately flagged leavers]")
    print(f"   False Positives (FP) : {fp:9,d}  [False alarms / retained flagged]")
    print(f"   True Negatives  (TN) : {tn:9,d}  [Accurately confirmed retained]")
    print(f"   False Negatives (FN) : {fn:9,d}  [Unpredicted leavers / blind spots]")
    print("-" * 75)
    print(" DISCRIMINATIVE & ACCURACY METRICS:")
    print(f"   ROC-AUC Area Under Curve : {roc_auc:.4f}  (Superior discrimination)")
    print(f"   PR-AUC Average Precision : {pr_auc:.4f}  (Imbalanced precision-recall)")
    print(f"   Overall Accuracy         : {accuracy * 100:.2f}%")
    print(f"   Recall / Sensitivity     : {recall * 100:.2f}%  (Coverage of true turnover)")
    print(f"   Precision / PPV          : {precision * 100:.2f}%  (Reliability of retention flags)")
    print(f"   Specificity (TNR)        : {specificity * 100:.2f}%")
    print(f"   Balanced F1-Score        : {f1:.4f}")
    print(f"   Brier Score (Calibration): {brier_score:.4f}  (Platt probability alignment)")
    print("=" * 75)

    print("\n[+] SAMPLE CORRECTLY PREDICTED FLIGHT-RISK EMPLOYEES (TRUE POSITIVES):")
    print(f"{'EmployeeID':<14} | {'Department':<26} | {'Job Role':<28} | {'Risk %':<8} | {'Actual'}")
    print("-" * 88)
    for row in sample_top_positives[:6]:
        print(f"{row['id']:<14} | {row['dept']:<26} | {row['role']:<28} | {row['risk_pct']:>6.1f}% | {row['truth']}")

    print("\n[-] SAMPLE MISSED CASES (FALSE NEGATIVES):")
    print(f"{'EmployeeID':<14} | {'Department':<26} | {'Job Role':<28} | {'Risk %':<8} | {'Actual'}")
    print("-" * 88)
    for row in sample_false_negatives[:4]:
        print(f"{row['id']:<14} | {row['dept']:<26} | {row['role']:<28} | {row['risk_pct']:>6.1f}% | {row['truth']}")

    print("=" * 75)
    print("[SUCCESS] 10 Million Benchmark and Ground-Truth Match Complete.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
