"""Fast streaming evaluation of threshold tradeoffs across the 10,000,000 employee predictions."""
import time
import pandas as pd
import numpy as np

TRUTH_PATH = "M:/meetra_data/workforce_ground_truth_10m.csv"
SCORED_PATH = "M:/meetra_data/workforce_scored_10m.csv"
CHUNK_SIZE = 500_000

thresholds = [0.40, 0.45, 0.47, 0.49, 0.50, 0.52, 0.55, 0.60]

counts = {t: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for t in thresholds}
total_eval = 0
actual_pos = 0

t0 = time.time()
print(f"[*] Reading and evaluating 10M rows in streaming chunks of {CHUNK_SIZE:,}...")

truth_reader = pd.read_csv(TRUTH_PATH, usecols=["Attrition"], chunksize=CHUNK_SIZE)
scored_reader = pd.read_csv(SCORED_PATH, usecols=["Predicted_Attrition_Risk_Pct"], chunksize=CHUNK_SIZE)

for t_chunk, s_chunk in zip(truth_reader, scored_reader):
    y_true = (t_chunk["Attrition"].astype(str).str.lower().isin(["yes", "true", "1"])).astype(int).values
    probs = s_chunk["Predicted_Attrition_Risk_Pct"].values / 100.0
    
    total_eval += len(y_true)
    actual_pos += int(np.sum(y_true))
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        counts[t]["tp"] += int(np.sum((preds == 1) & (y_true == 1)))
        counts[t]["fp"] += int(np.sum((preds == 1) & (y_true == 0)))
        counts[t]["tn"] += int(np.sum((preds == 0) & (y_true == 0)))
        counts[t]["fn"] += int(np.sum((preds == 0) & (y_true == 1)))

actual_neg = total_eval - actual_pos
elapsed = time.time() - t0

print(f"\n[+] Completed in {elapsed:.1f}s across {total_eval:,} employees (Turnover: {actual_pos:,} / {actual_pos/total_eval*100:.2f}%):")
print("-" * 105)
print(f"{'Cutoff':<8} | {'True Pos (TP)':<13} | {'False Pos (FP)':<14} | {'False Neg (FN)':<14} | {'Recall':<8} | {'Precision':<9} | {'F1-Score':<8} | {'Accuracy':<8}")
print("-" * 105)

for t in thresholds:
    c = counts[t]
    rec = c["tp"] / actual_pos
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    acc = (c["tp"] + c["tn"]) / total_eval
    print(f"  {t:.2f}   | {c['tp']:13,d} | {c['fp']:14,d} | {c['fn']:14,d} | {rec*100:6.2f}% |  {prec*100:6.2f}%  |  {f1:.4f}  | {acc*100:6.2f}%")

print("-" * 105)
