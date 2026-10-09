# Meetra Core (Kaizen v3.0) — Enterprise Workforce Retention Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EEOC 80% Compliant](https://img.shields.io/badge/EEOC%20Title%20VII-Compliant-brightgreen)](https://www.eeoc.gov/)
[![XGBoost CUDA](https://img.shields.io/badge/XGBoost-GPU%20Hist%20Accelerated-76B900?logo=nvidia)](https://xgboost.readthedocs.io/)

Meetra Core (Kaizen v3.0) is a high-throughput, calibrated machine-learning platform for **enterprise workforce flight-risk prediction, algorithmic fairness auditing, and game-theoretic retention intelligence**.

Designed for Fortune 500 scale, Meetra Core processes **1,000,000 workforce records in ~11 seconds (87,894 records/sec)** on consumer GPU hardware while maintaining **0.9972 ROC-AUC**, rigorous Murphy Brier probability calibration, and automated Title VII / EEOC Four-Fifths compliance.

---

## ⚡ One-Click Cloud Execution (Google Colab)

Run the complete 1-Million benchmark, TreeSHAP attribution, and fairness auditor on a free cloud GPU (NVIDIA T4):

👉 [**Open in Google Colab**](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)

---

## 🏆 Key Capabilities & Innovations

### 1. Dual-Tournament AutoML Engine
* **AlexzanderXS (Champion)**: GPU-accelerated XGBoost with lossguide tree growth, L2 regularization, early stopping, and automatic zero-context CPU transition for threadpool safety.
* **PorusXS (Challenger)**: Multi-threaded Balanced Random Forest (`n_jobs=-1`) with sub-sample class weighting.
* **Out-of-Fold (OOF) Stratified Cutoff Optimization**: Decision thresholds ($\tau$) tuned on training folds to maximize harmonic composite accuracy without holdout leakage.

### 2. 1-Million Out-of-Sample Blind Benchmark
* Evaluated on **1,000,000 unseen employees** (`seed=999`) with zero retraining:
  * **Scoring Throughput**: **87,894 rows/sec** (scored in **11.38 seconds** on an NVIDIA GTX 1650).
  * **Out-of-Sample Accuracy**: **97.89%**
  * **ROC-AUC**: **0.9972** | **PR-AUC**: **0.9850**
  * **Sensitivity (Recall)**: **92.82%** (Intercepted 138,401 / 149,102 departures).
  * **Specificity**: **98.78%** (Only 1.2% false alarms across 850,000 stayers).
  * **Brier Score (Loss)**: **0.0196** (Murphy 3-component decomposition).

### 3. Algorithmic Fairness & EEOC 80% Four-Fifths Rule Auditor
* Automated regulatory compliance across **Gender** (Male, Female, Non-Binary) and **ADEA Age Cohorts** ($<40$ vs $\ge 40$ years).
* Calculates **Adverse Impact Ratio (AIR)**, **Demographic Parity Gap ($\Delta_{DP}$)**, **Equal Opportunity Gap ($\Delta_{TPR}$)**, and **Predictive Equality Gap ($\Delta_{FPR}$)**.
* Embedded legal risk advisories for compliance with 29 C.F.R. § 1607.4D and the EU AI Act.

### 4. TreeSHAP Game-Theoretic Explainability & Turnover Contagion
* **Exact Game-Theoretic Attribution**: Verified mathematical additivity ($\sigma(\phi_0 + \sum \phi_j) \equiv \hat{P}_{\text{model}}$ down to $\epsilon = 9.45 \times 10^{-11}$).
* **Team Turnover Contagion Detector**: Sociometric non-linear hazard acceleration model ($M_{\text{contagion}} \in [1.00, 1.45]$) based on *Felps et al. (2008, Journal of Applied Psychology)*.

---

## 💻 Run Locally

### 1. Clone & Set Up Environment
```powershell
git clone https://github.com/GLITXHY9999/meetra_core.git
cd meetra_core

# Create virtual environment (Python 3.11 or 3.12 recommended)
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install xgboost
```

### 2. Start the Live Application
```powershell
python run.py
```
Open **`http://localhost:8000`** in your browser.

### 3. Run Benchmark Suite
```powershell
# Fast 100k test (1.2s)
python run_blind_benchmark.py --size 100k

# Full 1-Million test (11.4s)
python run_blind_benchmark.py --size 1m

# Automated Test Suite (16 passed)
pytest -v
```

---

## 📚 Academic Research Reference

For researchers and engineers, complete mathematical derivations, Murphy Brier decomposition proofs, and literature citations are documented in:
* [`docs/RESEARCH_PAPER_DOSSIER.md`](docs/RESEARCH_PAPER_DOSSIER.md)
* [`ROADMAP.md`](ROADMAP.md)

---

## ⚖️ Responsible AI Notice

Meetra Core is a decision-support platform designed to prioritize confidential stay interviews and proactive compensation reviews. Under EEOC Title VII and GDPR principles, risk scores should never be used as the sole determinant for adverse employment actions.
