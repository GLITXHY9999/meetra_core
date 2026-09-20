# Meetra Core / Kaizen: Master Architecture & Engineering Roadmap

## 1. Project Overview & Current State
Meetra Core (Kaizen v3.0.0) is an enterprise-grade automated machine learning (AutoML) and workforce retention risk scoring platform.

### Core Stack
- **Backend**: FastAPI with async threadpool execution (`starlette.concurrency.run_in_threadpool`) to prevent event-loop freezing during heavy ML workloads.
- **ML Engine**: Dual-tournament architecture:
  - **AlexzanderXS (Champion)**: Histogram-based Gradient Boosting with stratified out-of-fold threshold optimization.
  - **PorusXS (Challenger)**: Balanced Random Forest with ensemble bagging.
- **Persisted Production Model**: `storage/models/kaizen_attrition_bundle.joblib` (Champion model trained on 1,000,000 workforce records).
- **Test Suite**: 13 automated unit tests in `tests/` (`13 passed in 83s`).

---

## 2. 1-Million Blind Out-of-Sample Benchmark Certification
Evaluated on **1,000,000 unseen employees** generated with an independent seed (`seed=999`), scored with **zero retraining**:
- **Dataset A (Blind Input)**: `data/workforce_blind_1m.csv` (131.1 MB, all 23 features, no `Attrition` column).
- **Dataset B (Ground Truth)**: `data/workforce_ground_truth_1m.csv` (134.1 MB, actual `Attrition` answer key).
- **Out-of-Sample Accuracy**: **97.79%**
- **Balanced Accuracy**: **95.82%**
- **ROC-AUC**: **0.9969**
- **PR-AUC**: **0.9833**
- **F1-Score**: **0.9262**
- **Precision**: **92.24%**
- **Recall (Sensitivity)**: **93.01%** (Caught 138,676 out of 149,102 true departures)
- **Specificity**: **98.6%** (Only 1.4% false alarms across 850,898 stayers)
- **Scoring Throughput**: **63,458 employees/second** (1,000,000 rows scored in **15.76 seconds** on CPU)

---

## 3. Hardware Transition: NVIDIA GeForce GTX 1650 (4GB VRAM)
The project is transitioning to a dedicated laptop equipped with an **NVIDIA GeForce GTX 1650** (Turing architecture, 896 CUDA cores, 4 GB VRAM).

### Why the GTX 1650 is Ideal for Meetra Core
- Tabular data (1M rows) is ~130MB in memory. Unlike 70B generative LLMs which require 16GB+ VRAM, tabular datasets fit comfortably inside 4GB VRAM.
- The GTX 1650 supports **GPU-accelerated XGBoost / LightGBM** (`tree_method='hist'`, `device='cuda'`), speeding up training by 5x to 15x.
- It supports **Deep Tabular Neural Networks** (e.g. PyTorch TabNet / Multi-Layer Perceptrons) with batch sizes of 1,024 to 4,096.

### Verification on NVIDIA Laptop
```powershell
# 1. Verify NVIDIA Driver & CUDA
nvidia-smi

# 2. Verify Python & Virtual Environment
python --version  # Recommended: 3.11 or 3.12
```

---

## 4. Next Phase: Model Laboratory 2.0 (Industrial & Paper-Grade)
The user requires an industrial-grade research observatory (comparable to Weights & Biases, MLflow, and DataRobot):

1. **Dual-Model Overlaid Observatory**:
   - High-resolution canvas/SVG rendering **both AlexzanderXS and PorusXS curves simultaneously** on the same graph.
   - Interactive hover HUD displaying Cutoff Threshold $\tau$, True Positive Rate (TPR), False Positive Rate (FPR), and F1.
2. **Precision-Recall (PR) Curve Integration**:
   - Since turnover is an imbalanced problem (~15%), the PR curve is mathematically the gold standard over ROC.
3. **Calibration & Reliability Curve (Brier Decomposition)**:
   - Evaluates expected calibration error (ECE) across probability deciles to prove predictions are well-calibrated.
4. **Dynamic Threshold & Financial Cost Matrix**:
   - Interactive slider ($0.00 \to 1.00$) dynamically updating the Confusion Matrix, F1, and business ROI in dollars:
     $$\text{Net ROI} = (\text{Retained Talent} \times \text{Avg Replacement Cost}) - (\text{False Alarms} \times \text{Intervention Cost})$$
5. **Algorithmic Fairness / EEOC Compliance Monitor**:
   - Checks parity and disparate impact across demographic subgroups (e.g., gender, age brackets) to ensure compliance with enterprise hiring/retention regulations.
6. **Academic Publication Rigor**:
   - Bootstrapped 95% Confidence Intervals for all key metrics ($AUC \pm 0.002$).
   - SHAP (Shapley Additive exPlanations) reason codes for global and local interpretability.

---

## 5. Quick Run Commands
```powershell
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run the 100k blind benchmark (1.6s)
python run_blind_benchmark.py --size 100k

# Run the 1M blind benchmark (15.7s)
python run_blind_benchmark.py --size 1m

# Run the automated test suite
pytest
```
