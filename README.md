# Meetra Core (Kaizen v3.1) — Enterprise Workforce Retention Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Disparity Diagnostic](https://img.shields.io/badge/EEOC%20Title%20VII-Disparity%20Diagnostic-brightgreen)](https://www.eeoc.gov/)
[![Project Report PDF](https://img.shields.io/badge/Project%20Report-32--Page%20PDF-red?logo=adobe-acrobat-reader)](https://github.com/GLITXHY9999/meetra_core/blob/main/Kaizen_Final_Project_Report.pdf)
[![XGBoost Calibrated](https://img.shields.io/badge/XGBoost-GPU%20Calibrated%20Ensemble-76B900?logo=nvidia)](https://xgboost.readthedocs.io/)

> ### 🎓 Official Academic Course Dossier & Laboratory Submission
> - **Unified Google Colab Master Notebook**: [meetra_core_colab.ipynb](https://github.com/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb) — *Runs 100% inside Google Colab, covering Experiments 1.1–1.4, baseline evaluations (=43, TN=22, FP=6, FN=7, TP=8$), 10-fold CV leaderboard, and Capstone tournament.*
> - **Official Final Project Report (PDF)**: [Kaizen_Final_Project_Report.pdf](https://github.com/GLITXHY9999/meetra_core/blob/main/Kaizen_Final_Project_Report.pdf) — *Complete, unabridged 32-page report fully synchronized with the Colab EDA and validation figures.*

Meetra Core (Kaizen v3.1) is a high-throughput, calibrated machine-learning decision-support platform for **proactive employee attrition analysis, flight-risk prediction, algorithmic disparity auditing, and retention intelligence**.

Designed for enterprise scale, Meetra Core processes millions of workforce records via vectorized streaming inference while maintaining rigorous Platt probability calibration, TreeSHAP game-theoretic explainability, and automated demographic disparity screening.

---

## ⚡ One-Click Cloud Execution (Google Colab Master Notebook)

* **Official Unified Master Laboratory Notebook (Exp 1.1 — 1.4 & Capstone):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)  
  A single, end-to-end, self-contained Google Colab dossier unifying the complete semester curriculum for **AI Applications Lab (`26ECP-102`)**:
  * **Experiment 1.1:** Data Pre-processing, NumPy arrays, Pandas DataFrames & multi-panel Matplotlib charts.
  * **Experiment 1.2:** Exploratory Data Analysis (EDA) on `employee_data.csv` with mandatory visualizations and Pearson correlation heatmaps.
  * **Experiment 1.3:** Supervised learning foundations, stratified 80:20 train-test splitting, and full computation of all 7 classification performance metrics.
  * **Experiment 1.4:** Multi-model classification tournament comparing 13 diverse algorithms under 10-Fold Stratified Cross-Validation.
  * **Capstone Production Engine:** PorusXS vs. AlexzanderXS dual-tournament showdown, OOF threshold tuning ($	au^* = 0.25$), Murphy Brier calibration, 1-million employee GPU benchmark, TreeSHAP game theory, Title VII EEOC 80% four-fifths fairness audit, and an interactive in-notebook Gradio dashboard!


---

## 🏆 Key Capabilities & Innovations

### 1. Dual-Tournament AutoML Engine with Calibrated Stacking
* **AlexzanderXS (Champion)**: GPU-accelerated XGBoost with lossguide tree growth, cost-sensitive `scale_pos_weight`, L2 regularization, and Platt probability calibration.
* **PorusXS (Challenger)**: Multi-threaded Balanced Random Forest (`n_jobs=-1`) with sub-sample class weighting.
* **Calibrated Soft-Voting Stacking**: Blends 70% AlexzanderXS and 30% PorusXS calibrated probabilities to balance turnover capture and false alarm reduction.
* **Out-of-Fold (OOF) Stratified Cutoff Optimization**: Decision thresholds ($\tau$) tuned on training folds to maximize balanced composite accuracy without holdout leakage.

### 2. High-Throughput Streaming & 10-Million Employee Benchmark
* Evaluated across **10,000,000 workforce records** on disk using streaming memory-safe chunks (<600 MB RAM):
  * **Scoring Throughput**: **36,000+ to 57,000+ records/sec**.
  * **ROC-AUC**: **0.9800** | **PR-AUC**: **0.9190**.
  * **Recall / Turnover Coverage**: **79.65%** (1,321,044 departures accurately intercepted).
  * **Brier Calibration Score**: **0.0417** (Platt-aligned probabilities).

### 3. Algorithmic Fairness & Disparity Diagnostic Auditor
* Automated diagnostic screening across **Gender** (Male, Female, Non-Binary) and **ADEA Age Cohorts** ($<40$ vs $\ge 40$ years).
* Calculates **Adverse Impact Ratio (AIR)**, **Demographic Parity Gap ($\Delta_{DP}$)**, **Equal Opportunity Gap ($\Delta_{TPR}$)**, and **Predictive Equality Gap ($\Delta_{FPR}$)**.
* Embedded ethical guidelines: requires at least 30 observations per subgroup and does not claim legal compliance certification without qualified counsel review.

### 4. Protected-Attribute Exclusion
* Age, gender, race, marital status, and other protected attributes are strictly excluded from predictive model inputs and retained solely for held-out disparity monitoring.

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
```

### 2. Start the Live Application
```powershell
python run.py
```
Open **`http://localhost:8000`** in your browser.

### 3. Run Benchmark Suite
```powershell
# 10M streaming benchmark and ground-truth matcher
python scripts/run_10m_benchmark.py

# Independent blind test on external cohort
python scripts/test_real_life_blind.py

# Automated Test Suite (16 passed)
pytest -v
```

---

## 📋 Data Governance & Real-World Deployment Gate

Before operational use in a company:
* Bring a legally approved, de-identified longitudinal HR cohort with a documented snapshot date and future outcome window.
* See [the data governance guide](docs/DATA_GOVERNANCE.md) for required validation, privacy, and human-review controls.
* Model scores serve strictly as decision support to prioritize voluntary, supportive stay-interviews and compensation reviews.

---

## 📚 Academic Research Reference

For researchers and engineers, complete mathematical derivations, Murphy Brier decomposition proofs, and literature citations are documented in:
* [`docs/RESEARCH_PAPER_DOSSIER.md`](docs/RESEARCH_PAPER_DOSSIER.md)
* [`ROADMAP.md`](ROADMAP.md)

---

## ⚖️ Responsible AI Notice

Meetra Core is an evidence-based decision-support system designed to identify retention friction before unwanted turnover occurs. Under EEOC Title VII and GDPR principles, risk scores should never be used as the sole determinant for adverse employment actions.
