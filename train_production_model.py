"""Train a Kaizen model bundle from an approved, labelled historical cohort.

This command intentionally does not claim that a benchmark/synthetic file is
production data.  The saved result records the supplied provenance string so a
reviewer can distinguish a demonstration run from a deployment candidate.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from app.ml.engine import KaizenEngine
from app.services.data_pipeline import prepare_training_data

def main():
    parser = argparse.ArgumentParser(description="Train Kaizen on approved historical attrition data.")
    parser.add_argument("--data", required=True, help="CSV containing an Attrition-style target column")
    parser.add_argument("--target", default="Attrition", help="Target column name")
    parser.add_argument("--provenance", required=True, help="Short description of the data owner, period, and approval")
    args = parser.parse_args()
    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"[ERROR] Ground truth data not found at {csv_path}")
        return

    print(f"[*] Loading approved training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    X, y, telemetry = prepare_training_data(df, target_column=args.target)

    print("[*] Training and calibrating PorusXS and AlexzanderXS...")
    t0 = time.time()
    engine = KaizenEngine(task_type="classification", target_col="Attrition")
    engine.fit(X, y)
    engine.training_summary = {
        "source": "approved historical CSV",
        "provenance": args.provenance,
        "total_rows": telemetry["total_rows"],
        "attrition_rate": telemetry["attrition_rate"],
        "target": telemetry["target"],
        "dropped_columns": telemetry["dropped_columns"],
        "deployment_note": "Requires time-separated validation and governance approval before operational use.",
    }
    t1 = time.time()

    bundle_path = Path("storage/models/kaizen_attrition_bundle.joblib")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(bundle_path)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Model bundle persisted to {bundle_path}")
    print(f"  Training Time      : {t1 - t0:.2f} seconds")
    print(f"  Champion Model     : {engine.active_model}")
    print(f"  Holdout ROC-AUC    : {engine.metrics['roc_auc']:.4f}")
    print(f"  Holdout PR-AUC     : {engine.metrics['pr_auc']:.4f}")
    print(f"  Holdout F1-Score   : {engine.metrics['f1']:.4f}")
    print(f"  Holdout Accuracy   : {engine.metrics['accuracy']*100:.2f}%")
    print(f"  Decision Threshold : {engine.decision_threshold:.3f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
