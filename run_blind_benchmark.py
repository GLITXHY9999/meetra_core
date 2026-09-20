"""
Run Blind Benchmark Suite for Meetra Core / Kaizen.
Evaluates the pre-trained champion model against an unseen 1,000,000-employee blind dataset
without retraining, comparing predictions to the ground-truth answer key.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Set stdout encoding for Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence loky worker warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from app.services.model_service import get_model_service
from app.ml.engine import KaizenEngine
from app.services.data_pipeline import POSITIVE_LABELS

DATA_DIR = Path(__file__).parent / "data"


def run_benchmark(tag: str = "1m", chunk_size: int = 250_000) -> dict:
    blind_file = DATA_DIR / f"workforce_blind_{tag}.csv"
    truth_file = DATA_DIR / f"workforce_ground_truth_{tag}.csv"

    if not blind_file.exists() or not truth_file.exists():
        print(f"[ERROR] Benchmark datasets not found for tag '{tag}'.")
        print(f"   Looked for: {blind_file}")
        print(f"   Please run `python generate_blind_benchmark_1m.py` first.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"[+] EXECUTING BLIND OUT-OF-SAMPLE BENCHMARK ({tag.upper()})")
    print(f"=======================================================")
    print(f"Blind Dataset       : {blind_file.name}")
    print(f"Ground Truth Dataset: {truth_file.name}")

    # 1. Load Pre-trained Champion Model
    print("\n[*] Loading persisted production model bundle (Zero retraining)...")
    t_load = time.time()
    service = get_model_service()
    if not service.is_trained:
        print("[ERROR] No trained model bundle found in storage/models/.")
        sys.exit(1)

    engine = service.engine
    print(f"   Champion Model   : {engine.active_model}")
    print(f"   Model Version    : {engine.version}")
    print(f"   Decision Cutoff  : {engine.decision_threshold:.4f}")
    print(f"   Loaded in        : {time.time() - t_load:.3f}s")

    # 2. Extract Ground Truth Labels
    print("\n[*] Loading ground truth labels...")
    t_truth = time.time()
    # Read only the Attrition column to minimize memory overhead
    truth_df = pd.read_csv(truth_file, usecols=["EmployeeID", "Attrition"])
    y_raw = truth_df["Attrition"].astype(str).str.strip().str.lower()
    y_true = y_raw.isin(POSITIVE_LABELS).astype(int).to_numpy()
    total_samples = len(y_true)
    actual_churn_count = int(np.sum(y_true == 1))
    print(f"   Total Records    : {total_samples:,}")
    print(f"   Actual Churn     : {actual_churn_count:,} ({actual_churn_count / total_samples * 100:.2f}%)")
    print(f"   Ground Truth read: {time.time() - t_truth:.2f}s")

    # 3. Pure Batch Inference on Blind Dataset (Vectorized Chunks)
    print(f"\n[*] Scoring blind dataset via vectorized batch inference ({chunk_size:,} rows/chunk)...")
    all_probs = []
    t_score_start = time.perf_counter()

    chunks_processed = 0
    for chunk in pd.read_csv(blind_file, chunksize=chunk_size):
        probs_chunk = engine.predict_batch(chunk)
        all_probs.append(probs_chunk)
        chunks_processed += 1
        scored_so_far = sum(len(p) for p in all_probs)
        print(f"   Scored chunk {chunks_processed}: {scored_so_far:,} / {total_samples:,} employees...")

    scoring_duration = time.perf_counter() - t_score_start
    probabilities = np.concatenate(all_probs)
    throughput = total_samples / scoring_duration if scoring_duration > 0 else 0

    print(f"\n[OK] Scoring Complete!")
    print(f"   Inference Duration: {scoring_duration:.3f}s")
    print(f"   Throughput Rate   : {throughput:,.0f} employees/sec")

    # 4. Rigorous Out-of-Sample Metrics Evaluation
    print(f"\n[*] Computing out-of-sample evaluation metrics...")
    threshold = float(engine.decision_threshold)
    metrics = KaizenEngine._evaluate(pd.Series(y_true), probabilities, engine.active_model, threshold=threshold)

    cm = metrics["confusion"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    # 5. Executive Display
    print("\n" + "=" * 65)
    print(f"[REPORT] BLIND BENCHMARK AUDIT REPORT ({total_samples:,} EMPLOYEES)")
    print("=" * 65)
    print(f"Champion Model       : {metrics['model_name']} ({metrics['algorithm']})")
    print(f"Decision Threshold   : {metrics['decision_threshold']:.4f}")
    print(f"Scoring Latency      : {scoring_duration:.2f} seconds ({throughput:,.0f} rows/sec)")
    print("-" * 65)
    print(f"Out-of-Sample Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy     : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"ROC-AUC Score         : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC Score          : {metrics['pr_auc']:.4f}")
    print(f"F1-Score              : {metrics['f1']:.4f}")
    print(f"Precision             : {metrics['precision'] * 100:.2f}%")
    print(f"Recall (Sensitivity)  : {metrics['recall'] * 100:.2f}%")
    print(f"MCC                   : {metrics['mcc']:.4f}")
    print(f"Brier Score (Loss)    : {metrics['brier_score']:.4f}")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print(f"                 Predicted STAY       Predicted LEAVE")
    print(f"Actual STAY   :  TN = {tn:,} ({tn/(tn+fp)*100:.1f}%)        FP = {fp:,} ({fp/(tn+fp)*100:.1f}%)")
    print(f"Actual LEAVE  :  FN = {fn:,} ({fn/(fn+tp)*100:.1f}%)        TP = {tp:,} ({tp/(fn+tp)*100:.1f}%)")
    print("=" * 65)

    return {
        "metrics": metrics,
        "scoring_duration": scoring_duration,
        "throughput": throughput,
        "total_samples": total_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blind Benchmark Suite")
    parser.add_argument("--size", type=str, default="1m", choices=["100k", "1m"], help="Benchmark dataset size")
    args = parser.parse_args()

    run_benchmark(tag=args.size)
