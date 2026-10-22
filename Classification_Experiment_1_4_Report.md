# College Lab Practical Report
# Experiment - 1.4: Supervised Classification Algorithm Comparison

**Course:** Data Science & Machine Learning Laboratory  
**Objective:** 1.4 Build a classification model by using different machine learning algorithms.  
**Dataset:** Cleveland Heart Disease Diagnostic Dataset (`heart.csv`)  
**Records Analyzed:** 303 Clinical Observations | **Features:** 13 Diagnostic Predictors | **Target:** Binary Heart Disease (`0` = Healthy, `1` = Heart Disease)  
**GitHub Public Source:** [GLITXHY9999/meetra_core/heart.csv](https://github.com/GLITXHY9999/meetra_core/blob/main/heart.csv)  
**Colab Interactive Runner:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/Experiment_1_4_Classification.ipynb)

---

## 1. Key Concepts in Supervised Learning

### 1.1 Features and Labels
- **Features ($X$):** Independent variables that act as input to the model. In this cardiovascular study, $X$ contains 13 physiological and clinical biomarkers:
  - Demographic: `age`, `sex`
  - Symptoms: chest pain type (`cp`)
  - Vitals & Labs: resting blood pressure (`trestbps`), serum cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`)
  - Stress Test Parameters: maximum heart rate achieved (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), ST slope (`slope`)
  - Diagnostic Imaging: fluoroscopy vessel count (`ca`), thalassemia test (`thal`)
- **Labels ($Y$):** Dependent target variable that the model is tasked with predicting. Here, $Y \in \{0, 1\}$ denotes absence (`0`) or presence (`1`) of ischemic heart disease.

### 1.2 Training, Validation, and Testing Framework
- **Cross-Validation Protocol:** Evaluated using **10-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`). Stratification enforces that each fold maintains identical positive/negative class prevalence (~54.5% positive, ~45.5% negative), eliminating sampling variance and fold selection bias.
- **Objective:** Minimize out-of-sample generalization error and maximize diagnostic recall to ensure zero undetected high-risk cardiac patients.

---

## 2. Ingested Dataset Overview & Hygiene

### 2.1 Structural Attributes
- **Total Records:** 303 patient profiles
- **Missing Values:** **0 null values** across all 14 features (100% operational completeness)
- **Feature Scaling:** Continuous parameters standardized to zero-mean and unit-variance ($\mu = 0, \sigma = 1$) via `StandardScaler()`:
  $$z = \frac{x - \mu}{\sigma}$$

### 2.2 Descriptive Statistics Summary

| Feature | Description | Mean ($\mu$) | Std Dev ($\sigma$) | Min | Median ($Q_2$) | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`age`** | Age in years | 54.37 | 9.08 | 29.0 | 55.0 | 77.0 |
| **`trestbps`** | Resting blood pressure (mm Hg) | 131.62 | 17.54 | 94.0 | 130.0 | 200.0 |
| **`chol`** | Serum cholesterol (mg/dl) | 246.26 | 51.83 | 126.0 | 240.0 | 564.0 |
| **`thalach`** | Maximum heart rate achieved (bpm) | 149.65 | 22.91 | 71.0 | 153.0 | 202.0 |
| **`oldpeak`** | ST depression induced by exercise | 1.04 | 1.16 | 0.0 | 0.8 | 6.2 |

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

The final sorted performance comparison across all 13 algorithms evaluated via 10-Fold Stratified Cross-Validation:

| Model | Accuracy | AUC | Recall (Sensitivity) | Precision | F1 Score | Kappa ($\kappa$) | MCC | Execution Time (Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** *(Baseline)* | 0.5445 | 0.5000 | **1.0000** | 0.5445 | 0.7050 | 0.0000 | 0.0000 | 0.1574 |
| **Linear Discriminant Analysis (LDA)** | **0.8345** | **0.9000** | **0.9154** | 0.8098 | **0.8579** | 0.6617 | 0.6729 | 0.2206 |
| **Ridge Classifier** | **0.8345** | **0.8995** | **0.9154** | 0.8098 | **0.8579** | 0.6617 | 0.6729 | 0.3612 |
| **Logistic Regression** | **0.8345** | 0.8954 | **0.8971** | 0.8194 | **0.8555** | **0.6630** | **0.6688** | 0.5238 |
| **Support Vector Machine (SVC)** | 0.8280 | 0.8944 | 0.8860 | 0.8214 | 0.8492 | 0.6505 | 0.6599 | 0.5718 |
| **K-Nearest Neighbors (KNN)** | 0.8246 | 0.8806 | 0.8794 | 0.8177 | 0.8450 | 0.6444 | 0.6520 | 0.4726 |
| **Extra Trees Classifier** | **0.8380** | **0.9162** | 0.8559 | **0.8595** | 0.8518 | **0.6734** | **0.6859** | 2.1956 |
| **AdaBoost Classifier** | 0.8149 | 0.8624 | 0.8559 | 0.8235 | 0.8331 | 0.6262 | 0.6408 | 1.9334 |
| **Random Forest Classifier** | 0.8117 | 0.9027 | 0.8504 | 0.8224 | 0.8289 | 0.6197 | 0.6340 | 3.0052 |
| **Quadratic Discriminant Analysis (QDA)** | 0.8216 | 0.8928 | 0.8485 | 0.8387 | 0.8394 | 0.6383 | 0.6474 | 0.1985 |
| **Gradient Boosting Classifier** | 0.7952 | 0.8850 | 0.8375 | 0.8015 | 0.8153 | 0.5856 | 0.5941 | 2.6224 |
| **Gaussian Naive Bayes** | 0.8078 | 0.8900 | 0.8368 | 0.8291 | 0.8271 | 0.6109 | 0.6219 | 0.4393 |
| **Decision Tree Classifier** | 0.7455 | 0.7438 | 0.7695 | 0.7703 | 0.7663 | 0.4868 | 0.4924 | 0.5344 |

---

## 6. Visual Performance Models

### 6.1 Multi-Metric Benchmark Comparison
![Model Comparison Benchmark](classification_insights/01_model_comparison_benchmark.png)
- **Observations:** Linear and regularized models (LDA, Ridge, Logistic Regression) consistently yield the highest Recall scores (>89.7%), making them premier candidates for high-sensitivity medical screening. Extra Trees demonstrates highest overall ROC-AUC (0.916).

### 6.2 Receiver Operating Characteristic (ROC) Trajectory
![ROC Comparison](classification_insights/02_roc_curves.png)
- **Observations:** Top models exhibit steep True Positive curves near the origin, sustaining TPR $> 85\%$ at FPR $< 18\%$.

### 6.3 Diagnostic Confusion Matrix
![Confusion Matrix](classification_insights/03_confusion_matrix.png)
- **Observations:** In Logistic Regression (evaluated via 10-fold cross validation out-of-fold predictions), only **17 False Negatives** occurred out of 165 true heart disease patients, yielding an empirical clinical sensitivity of ~89.7%.

### 6.4 Feature Importance / Standardized Coefficients
![Feature Coefficients](classification_insights/04_feature_coefficients.png)
- **Top Positive Predictors of Heart Disease:** Chest pain type (`cp`), maximum heart rate (`thalach`), and ST slope (`slope`).
- **Top Protective/Negative Indicators:** Number of major fluoroscopy vessels (`ca`), exercise-induced angina (`exang`), ST depression (`oldpeak`), and sex (`sex`).

---

## 7. Lab Task Detailed Answers

### Step 1: Load any classification dataset (downloaded from Kaggle)
- Ingested the **Cleveland Heart Disease Dataset** comprising 303 rows and 13 clinical biomarkers with zero missing observations.

### Step 2: Perform Classification
- Preprocessed features with z-score standardization (`StandardScaler`) and partitioned into 10 stratified validation folds.
- Executed training and evaluation across 13 distinct machine learning algorithms.

### Step 3: Compare the performance of different algorithms
1. **Best Sensitivity / Recall:** **Linear Discriminant Analysis** and **Ridge Classifier** achieved **91.54% Recall**, capturing 9 out of 10 true heart disease patients.
2. **Best Overall Discriminator:** **Extra Trees Classifier** achieved the peak **ROC-AUC of 0.9162** and highest Accuracy (**83.80%**).
3. **Best Operational Balance:** **Logistic Regression** achieved **83.45% Accuracy, 89.71% Recall, 0.8954 AUC**, and **0.8555 F1 Score** with minimal training overhead (0.52s).
4. **Weakest Supervised Model:** An unpruned single **Decision Tree** suffered from variance and overfitting, yielding lowest performance (74.55% Accuracy, 0.7438 AUC).
5. **Baseline Comparison:** The **Dummy Classifier** trivially produced 1.0000 Recall by always guessing the majority class, but produced 0.5000 AUC and 0.0000 Cohen's Kappa, proving that Recall alone without calibration or AUC is non-viable.

### Step 4: Which performance metric is essential for the dataset used and why?
- **Essential Metric: Recall (Sensitivity / True Positive Rate)**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Critical Clinical Justification:**
  1. **Catastrophic Impact of False Negatives (FN):** In cardiovascular screening, a False Negative represents a sick patient erroneously told they are disease-free. Left untreated, coronary ischemia or severe arterial blockages can progress to sudden myocardial infarction (heart attack) or fatality.
  2. **Acceptable Overhead of False Positives (FP):** A False Positive represents a healthy patient flagged for further testing. The clinical cost is merely a secondary non-invasive test (such as a 2D echocardiogram, stress thallium test, or cardiac CT), causing temporary concern but zero clinical harm or mortality risk.
  3. **Multi-Metric Validation Requirement:** While Recall is the essential diagnostic objective, it must always be supported by **ROC-AUC** and **F1-Score** to confirm that high sensitivity is achieved through true algorithmic discrimination rather than indiscriminately predicting positive outcomes.

---

## 8. Interactive Execution Link

The complete notebook is fully runnable on Google Colab with one click:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GLITXHY9999/meetra_core/blob/main/Experiment_1_4_Classification.ipynb)

---
*Report formatted strictly to university laboratory practical standards.*
