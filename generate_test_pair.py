"""Generate 1,000-employee paired blind testing dataset for Kaizen Model Laboratory.

Outputs:
1. data/test_workforce_blind.csv: 1,000 records containing employee features only (NO Attrition column)
   Ready to be uploaded to Kaizen Web UI or API.
2. data/test_workforce_ground_truth.csv: Identical 1,000 records with the real Attrition ground truth label.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BLIND_PATH = DATA_DIR / "test_workforce_blind.csv"
TRUTH_PATH = DATA_DIR / "test_workforce_ground_truth.csv"

def generate_test_pair(n_rows: int = 1_000, seed: int = 42):
    rng = np.random.default_rng(seed)

    departments = [
        "Engineering & Technology",
        "Sales & Revenue",
        "Research & Development",
        "Customer Operations",
        "Marketing & Growth",
        "Finance & Accounting",
        "Human Resources & People",
        "Supply Chain & Logistics",
        "Legal & Compliance",
        "Product Management",
    ]
    dept_weights = [0.30, 0.20, 0.15, 0.11, 0.06, 0.05, 0.04, 0.04, 0.02, 0.03]

    role_map = {
        "Engineering & Technology": [
            "Software Engineer", "Senior Software Engineer", "DevOps Engineer",
            "Data Engineer", "Systems Architect", "Engineering Manager"
        ],
        "Sales & Revenue": [
            "Account Executive", "Sales Representative", "Enterprise Sales Director",
            "Business Development Rep", "Sales Operations Lead"
        ],
        "Research & Development": [
            "Research Scientist", "Staff Scientist", "Lab Technician",
            "R&D Team Lead", "Product Innovation Director"
        ],
        "Customer Operations": [
            "Customer Success Manager", "Support Specialist", "Implementation Lead",
            "Client Solutions Architect"
        ],
        "Marketing & Growth": [
            "Product Marketing Manager", "Growth Lead", "Content Strategist", "Brand Manager"
        ],
        "Finance & Accounting": [
            "Financial Analyst", "Senior Accountant", "FP&A Manager", "Finance Director"
        ],
        "Human Resources & People": [
            "HR Business Partner", "Talent Acquisition Lead", "People Operations Specialist", "HR Director"
        ],
        "Supply Chain & Logistics": [
            "Supply Chain Analyst", "Logistics Coordinator", "Procurement Manager", "Warehouse Lead"
        ],
        "Legal & Compliance": [
            "Corporate Counsel", "Compliance Specialist", "Contracts Manager"
        ],
        "Product Management": [
            "Associate Product Manager", "Product Manager", "Principal PM", "Group PM"
        ],
    }

    base_income_by_level = {1: 3500, 2: 6300, 3: 10500, 4: 15800, 5: 22500}

    age = rng.integers(21, 62, n_rows)
    gender = rng.choice(["Female", "Male", "Non-Binary"], n_rows, p=[0.47, 0.51, 0.02])
    dept = rng.choice(departments, n_rows, p=dept_weights)
    job_roles = [rng.choice(role_map[d]) for d in dept]
    job_level = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.38, 0.33, 0.16, 0.08, 0.05])
    monthly_income = np.array([base_income_by_level[lvl] for lvl in job_level]) + rng.integers(-400, 1500, n_rows)

    overtime = rng.choice(["Yes", "No"], n_rows, p=[0.28, 0.72])
    total_working_years = np.clip(age - 21 - rng.integers(0, 3, n_rows), 0, 40)
    years_at_company = np.clip(rng.integers(0, total_working_years + 1), 0, 30)
    years_with_curr_manager = np.clip(rng.integers(0, years_at_company + 1), 0, 15)
    years_since_last_promo = np.clip(rng.integers(0, years_at_company + 1), 0, 12)

    job_sat = rng.choice([1, 2, 3, 4], n_rows, p=[0.18, 0.22, 0.35, 0.25])
    env_sat = rng.choice([1, 2, 3, 4], n_rows, p=[0.16, 0.22, 0.35, 0.27])
    work_life = rng.choice([1, 2, 3, 4], n_rows, p=[0.12, 0.25, 0.53, 0.10])
    stock_level = rng.choice([0, 1, 2, 3], n_rows, p=[0.42, 0.38, 0.14, 0.06])
    marital = rng.choice(["Single", "Married", "Divorced"], n_rows, p=[0.34, 0.46, 0.20])
    travel = rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n_rows, p=[0.70, 0.18, 0.12])
    distance = rng.integers(1, 40, n_rows)
    num_companies = rng.integers(0, 9, n_rows)
    education = rng.choice(["Bachelors", "Masters", "Doctorate", "Associate Degree"], n_rows, p=[0.55, 0.30, 0.05, 0.10])
    perf_rating = rng.choice([2, 3, 4], n_rows, p=[0.10, 0.76, 0.14])
    percent_hike = rng.integers(11, 25, n_rows)
    training_times = rng.integers(0, 7, n_rows)

    # Real-world turnover risk scoring equation
    risk_score = (
        2.8 * (overtime == "Yes")
        + 2.1 * (job_sat == 1)
        + 1.7 * (env_sat == 1)
        + 1.5 * (stock_level == 0)
        + 1.4 * (marital == "Single")
        + 1.3 * (travel == "Travel_Frequently")
        + 1.2 * (work_life == 1)
        + 0.04 * distance
        - 0.05 * (age - 35)
        - 0.0003 * (monthly_income - 7000)
        - 1.3 * (job_level >= 3)
        - 0.8 * (years_at_company >= 5)
        - 0.6 * (years_with_curr_manager >= 3)
    )

    cutoff = np.percentile(risk_score, 83.5)  # ~16.5% turnover rate
    noise = rng.normal(0, 0.25, n_rows)
    attrition = np.where((risk_score + noise) > cutoff, "Yes", "No")

    emp_ids = [f"EMP-{10001 + i}" for i in range(n_rows)]

    features = {
        "EmployeeID": emp_ids,
        "Age": age,
        "Gender": gender,
        "Department": dept,
        "JobRole": job_roles,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "YearsWithCurrManager": years_with_curr_manager,
        "YearsSinceLastPromotion": years_since_last_promo,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "WorkLifeBalance": work_life,
        "StockOptionLevel": stock_level,
        "MaritalStatus": marital,
        "BusinessTravel": travel,
        "DistanceFromHome": distance,
        "NumCompaniesWorked": num_companies,
        "Education": education,
        "PerformanceRating": perf_rating,
        "PercentSalaryHike": percent_hike,
        "TrainingTimesLastYear": training_times,
        "OverTime": overtime,
    }

    # 1. Blind file (NO Attrition)
    blind_df = pd.DataFrame(features)
    blind_df.to_csv(BLIND_PATH, index=False)

    # 2. Ground Truth file (Identical features + Attrition)
    truth_df = pd.DataFrame(features)
    truth_df["Attrition"] = attrition
    truth_df.to_csv(TRUTH_PATH, index=False)

    n_attrition = (truth_df["Attrition"] == "Yes").sum()
    pct_attrition = (n_attrition / n_rows) * 100

    print("================================================================")
    print(f"[OK] Generated Test Datasets ({n_rows:,} Employees)")
    print(f"Blind CSV (Upload to Kaizen) : {BLIND_PATH.resolve()}")
    print(f"  - Columns: {len(blind_df.columns)} (NO Attrition column)")
    print(f"Ground Truth CSV             : {TRUTH_PATH.resolve()}")
    print(f"  - Columns: {len(truth_df.columns)} (Includes 'Attrition': {n_attrition} Yes / {n_rows - n_attrition} No [{pct_attrition:.1f}%])")
    print("================================================================")

if __name__ == "__main__":
    generate_test_pair()
