# College Lab Practical Report
# Experiment - 1.4: Supervised Classification Algorithm Comparison

**Course:** Data Science & Machine Learning Laboratory  
**Objective:** 1.4 Build a classification model by using different machine learning algorithms.  
**Dataset:** Real-World Employee Retention & Flight-Risk Analytics Dataset (`employee_data.csv`)  
**Records Analyzed:** 215 Corporate Personnel Records | **Features:** 32 Attributes | **Target:** Binary Attrition (`Yes` = Flight Risk, `No` = Retained)  
**GitHub Public Source:** [GLITXHY9999/meetra_core/employee_data.csv](https://github.com/GLITXHY9999/meetra_core/blob/main/employee_data.csv)  
**Colab Interactive Runner:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)

---

## 1. Key Concepts in Supervised Learning

### 1.1 Features and Labels
- **Features ($X$):** Independent variables that act as input to the model. In this workforce study, $X$ contains 32 demographic, compensatory, and operational parameters:
  - Demographic & Family: `Age`, `Gender`, `MaritalStatus`, `Higher_Education`
  - Operational & Role: `Department`, `JobRole`, `JobLevel`, `JobInvolvement`, `Mode_of_work`, `Job_mode`
  - Workload & Environment: `OverTime`, `DistanceFromHome`, `BusinessTravel`, `Work_accident`, `Absenteeism`, `Leaves`
  - Career Progression: `TotalWorkingYears`, `YearsAtCompany`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `TrainingTimesLastYear`
  - Compensation & Performance: `MonthlyIncome`, `PercentSalaryHike`, `StockOptionLevel`, `PerformanceRating`
- **Labels ($Y$):** Dependent target variable that the model is tasked with predicting. Here, $Y \in \{0, 1\}$ denotes employee retention status (`0` = Retained, `1` = Attrition / Flight Risk).

### 1.2 Training, Validation, and Testing Framework
- **Cross-Validation Protocol:** Evaluated using **10-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`). Stratification enforces that each fold maintains identical positive/negative class prevalence (~34.9% positive attrition, ~65.1% negative retention), eliminating sampling variance and fold selection bias.
- **Objective:** Minimize out-of-sample generalization error and maximize flight-risk recall to ensure timely talent retention interventions.

---

## 2. Ingested Dataset Overview & Hygiene

### 2.1 Structural Attributes
- **Total Records:** 215 corporate employee records
- **Data Completeness:** **0 missing values across all active predictive attributes** (100% operational feature completeness)
- **Categorical Encoding:** One-hot encoded via `pd.get_dummies(drop_first=True)`
- **Feature Scaling:** Continuous parameters standardized to zero-mean and unit-variance ($\mu = 0, \sigma = 1$) via `StandardScaler()`:
  $$z = \frac{x - \mu}{\sigma}$$

### 2.2 Descriptive Statistics Summary (Key Features)

| Feature | Description | Mean ($\mu$) | Std Dev ($\sigma$) | Min | Median ($Q_2$) | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Age`** | Employee age in years | 34.27 | 10.11 | 18.0 | 34.0 | 60.0 |
| **`MonthlyIncome`** | Monthly compensation ($) | \$4,565.10 | \$4,063.31 | \$1,009.00 | \$2,853.00 | \$19,613.00 |
| **`TotalWorkingYears`** | Total career experience (years) | 7.11 | 7.59 | 0.0 | 4.0 | 34.0 |
| **`DistanceFromHome`** | Commute distance in miles | 9.24 | 8.05 | 1.0 | 7.0 | 29.0 |
| **`YearsAtCompany`** | Tenure with company (years) | 0.80 | 0.40 | 0.0 | 1.0 | 1.0 |

---

## 3. Supervised Classification Algorithms Compared

A total of **13 classification algorithms** were benchmarked under identical cross-validation folds and scoring pipelines:

1. **Parametric & Linear Classifiers:**
   - Logistic Regression (`max_iter=1000`)
   - Ridge Classifier
   - Linear Discriminant Analysis (LDA)
2. **Non-Linear & Non-Parametric Classifiers:**
   - Decision Tree Classifier (`random_state=42`)
   - K-Nearest Neighbors Classifier (`K=5`)
   - Support Vector Classifier (SVC with RBF kernel, `probability=True`)
   - Quadratic Discriminant Analysis (QDA)
   - Gaussian Naive Bayes
3. **Ensemble & Boosting Classifiers:**
   - Random Forest Classifier (`random_state=42`)
   - Extra Trees Classifier (`random_state=42`)
   - Gradient Boosting Classifier (`random_state=42`)
   - AdaBoost Classifier (`random_state=42`)
4. **Baseline Sanity Model:**
   - Dummy Classifier (`strategy='most_frequent'`)

---

## 4. Evaluation Metrics Formulation

To provide a rigorous 360-degree performance evaluation, the following seven evaluation metrics were recorded across every fold:

1. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
3. **Precision:** $\frac{TP}{TP + FP}$
4. **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **ROC-AUC:** Area under the True Positive Rate vs False Positive Rate curve across all thresholds.
6. **Cohen's Kappa ($\kappa$):** Inter-rater reliability taking chance agreement into account.
7. **Matthews Correlation Coefficient (MCC):** Balanced correlation metric robust against class imbalances:
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

---

## 5. Experimental Results & Model Comparison

The final sorted performance comparison across all 13 algorithms evaluated via 10-Fold Stratified Cross-Validation on the employee dataset:

| Model | Accuracy | AUC | Recall (Sensitivity) | Precision | F1 Score | Kappa ($\kappa$) | MCC | Execution Time (Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 0.4227 | 0.7254 | **0.9607** | 0.3733 | 0.5367 | 0.0711 | 0.1642 | 0.2089 |
| **Quadratic Discriminant Analysis (QDA)** | 0.4500 | 0.6276 | **0.8125** | 0.3725 | 0.5087 | 0.0589 | 0.0738 | 0.4724 |
| **Logistic Regression** | **0.7589** | **0.8000** | **0.6589** | **0.6732** | **0.6579** | **0.4740** | **0.4816** | 0.4279 |
| **Ridge Classifier** | **0.7498** | **0.8080** | **0.6321** | 0.6496 | **0.6314** | **0.4460** | **0.4527** | 0.2722 |
| **Linear Discriminant Analysis (LDA)** | **0.7450** | **0.8080** | **0.6321** | 0.6396 | **0.6276** | 0.4374 | 0.4435 | 0.4440 |
| **AdaBoost Classifier** | 0.6881 | 0.7357 | 0.5625 | 0.5649 | 0.5563 | 0.3200 | 0.3240 | 2.2642 |
| **Decision Tree Classifier** | 0.6571 | 0.6143 | 0.4714 | 0.5198 | 0.4867 | 0.2338 | 0.2370 | 0.1973 |
| **Gradient Boosting Classifier** | 0.7076 | 0.7221 | 0.4714 | 0.5970 | 0.5119 | 0.3165 | 0.3266 | 2.6812 |
| **Support Vector Machine (SVC)** | 0.7448 | 0.7732 | 0.4446 | **0.7138** | 0.5382 | 0.3821 | 0.4046 | 0.4347 |
| **Random Forest Classifier** | 0.7307 | 0.7716 | 0.4446 | 0.6472 | 0.5157 | 0.3512 | 0.3642 | 3.7326 |
| **Extra Trees Classifier** | 0.7171 | 0.7469 | 0.4179 | 0.6144 | 0.4858 | 0.3165 | 0.3276 | 2.8471 |
| **K-Nearest Neighbors (KNN)** | 0.7071 | 0.6849 | 0.3339 | 0.6950 | 0.4443 | 0.2741 | 0.3133 | 0.3690 |
| **Dummy Classifier** *(Baseline)* | 0.6515 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2851 |

---

## 6. Visual Performance Models

### 6.1 Multi-Metric Benchmark Comparison
![Model Comparison Benchmark](classification_insights/01_model_comparison_benchmark.png)
- **Observations:** Logistic Regression and regularized linear models (Ridge, LDA) attain the highest balanced performance across Accuracy (75.89%), ROC-AUC (0.8000), and F1 Score (0.6579).

### 6.2 Receiver Operating Characteristic (ROC) Trajectory
![ROC Comparison](classification_insights/02_roc_curves.png)
- **Observations:** Logistic Regression sustains an AUC of **0.800**, substantially outperforming the uninformative baseline (0.500).

### 6.3 Diagnostic Confusion Matrix
![Confusion Matrix](classification_insights/03_confusion_matrix.png)
- **Observations:** In Logistic Regression evaluated out-of-fold, the majority of true departures are correctly identified while keeping false alarms contained.

### 6.4 Key Feature Coefficients (Attrition Drivers)
![Feature Coefficients](classification_insights/04_feature_coefficients.png)
- **Primary Attrition Catalysts (Positive Weights):** Mandatory `OverTime`, high commute distances (`DistanceFromHome`), and specific volatile roles.
- **Primary Retention Anchors (Negative Weights):** Higher tenure (`TotalWorkingYears`, `YearsAtCompany`), stock option incentives (`StockOptionLevel`), and satisfaction metrics (`JobSatisfaction`, `JobInvolvement`).

---

## 7. Lab Task Detailed Answers

### Step 1: Load any classification dataset (downloaded from Kaggle)
- Ingested the **IBM HR Analytics Employee Attrition Dataset** (`employee_data.csv`), containing 215 employee records across 32 structural features and a binary target (`Attrition`).

### Step 2: Perform Classification
- Encoded categorical variables into dummy indicators, applied `StandardScaler` normalization, and executed a 10-fold stratified cross-validation split across 13 diverse supervised algorithms.

### Step 3: Compare the performance of different algorithms
1. **Best Operational Balance:** **Logistic Regression** achieved the highest overall balance (**75.89% Accuracy, 80.00% ROC-AUC, 65.89% Recall, 0.6579 F1 Score**).
2. **Top Regularized Models:** **Ridge Classifier** and **LDA** matched at **80.80% ROC-AUC** and **74.98% Accuracy**.
3. **Ensemble Models:** **Random Forest** achieved **77.16% AUC** and **73.07% Accuracy**, while **AdaBoost** captured **56.25% Recall**.
4. **Naive Bayes Trade-Off:** **Gaussian Naive Bayes** captured 96.07% of departures but with high false positives (Accuracy dropped to 42.27%), illustrating the trade-off of naive feature independence assumptions.
5. **Baseline Comparison:** The **Dummy Classifier** (guessing `No`) scored 0.0000 Recall and 0.5000 AUC, showing zero operational value.

### Step 4: Which performance metric is essential for the dataset used and why?
- **Essential Metric: Recall (Sensitivity / Flight-Risk Coverage)**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Critical Organizational & Business Justification:**
  1. **Asymmetric Cost of False Negatives (FN):** In corporate talent management, a False Negative occurs when an employee who is on the verge of resigning is classified as stable. When they abruptly depart, the organization incurs substantial replacement and hiring costs (**50% to 150% of annual salary**), knowledge drain, and severe workflow disruption.
  2. **Acceptable Cost of False Positives (FP):** In contrast, a False Positive occurs when a loyal employee is flagged as a flight risk. The organizational intervention is positive and supportive (e.g., conducting a stay interview, checking workload balance, offering training). It does not harm the employee or damage operations.
  3. **Multi-Metric Guardrails:** While Recall is the vital primary metric, it must be paired with **ROC-AUC** and **F1-Score** to ensure high coverage is driven by true predictive discernment rather than indiscriminately flagging all employees.

---

## 8. Interactive Execution Link

The complete notebook is fully runnable on Google Colab with one click:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/meetra_core_colab.ipynb)

---
*Report formatted strictly to university laboratory practical standards.*
