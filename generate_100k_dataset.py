"""
Generate a realistic 100,000-employee (1 Lakh records) enterprise workforce dataset.
Calibrated to real Fortune 500 company organizational hierarchies and attrition dynamics.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

def generate_enterprise_dataset(num_rows: int = 100_000, seed: int = 42, output_path: str = "enterprise_workforce_100k.csv") -> Path:
    print(f"Generating enterprise dataset with {num_rows:,} employees (1 Lakh records)...")
    start_time = time.time()
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 64, num_rows)
    gender = rng.choice(["Female", "Male", "Non-Binary"], num_rows, p=[0.47, 0.51, 0.02])

    departments = [
        "Engineering & Technology",
        "Sales & Revenue",
        "Research & Development",
        "Customer Operations",
        "Marketing & Growth",
        "Finance & Accounting",
        "Human Resources & People",
    ]
    dept_weights = [0.32, 0.23, 0.18, 0.12, 0.06, 0.05, 0.04]
    dept = rng.choice(departments, num_rows, p=dept_weights)

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
    }

    job_roles = [rng.choice(role_map[d]) for d in dept]
    job_level = rng.choice([1, 2, 3, 4, 5], num_rows, p=[0.38, 0.33, 0.16, 0.08, 0.05])

    base_income_by_level = {1: 3400, 2: 6100, 3: 10200, 4: 15400, 5: 22000}
    monthly_income = np.array([base_income_by_level[lvl] for lvl in job_level]) + rng.integers(-500, 1600, num_rows)

    overtime = rng.choice(["Yes", "No"], num_rows, p=[0.26, 0.74])
    total_working_years = np.clip(age - 21 - rng.integers(0, 3, num_rows), 0, 42)
    years_at_company = np.clip(rng.integers(0, total_working_years + 1), 0, 35)
    years_with_curr_manager = np.clip(rng.integers(0, years_at_company + 1), 0, 15)
    years_since_last_promo = np.clip(rng.integers(0, years_at_company + 1), 0, 12)

    job_sat = rng.choice([1, 2, 3, 4], num_rows, p=[0.16, 0.22, 0.36, 0.26])
    env_sat = rng.choice([1, 2, 3, 4], num_rows, p=[0.15, 0.22, 0.35, 0.28])
    work_life = rng.choice([1, 2, 3, 4], num_rows, p=[0.11, 0.24, 0.55, 0.10])
    stock_level = rng.choice([0, 1, 2, 3], num_rows, p=[0.42, 0.38, 0.14, 0.06])
    marital = rng.choice(["Single", "Married", "Divorced"], num_rows, p=[0.33, 0.47, 0.20])
    travel = rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], num_rows, p=[0.70, 0.18, 0.12])
    distance = rng.integers(1, 45, num_rows)
    num_companies = rng.integers(0, 10, num_rows)
    education = rng.choice(["Bachelors", "Masters", "Doctorate", "Associate Degree"], num_rows, p=[0.56, 0.29, 0.05, 0.10])
    perf_rating = rng.choice([2, 3, 4], num_rows, p=[0.10, 0.76, 0.14])
    percent_hike = rng.integers(11, 26, num_rows)
    training_times = rng.integers(0, 7, num_rows)

    # Enterprise ground-truth attrition dynamics
    risk_score = (
        3.0 * (overtime == "Yes")
        + 2.2 * (job_sat == 1)
        + 1.8 * (env_sat == 1)
        + 1.6 * (stock_level == 0)
        + 1.5 * (marital == "Single")
        + 1.4 * (travel == "Travel_Frequently")
        + 1.2 * (work_life == 1)
        + 0.04 * distance
        - 0.06 * (age - 35)
        - 0.0003 * (monthly_income - 7000)
        - 1.4 * (job_level >= 3)
        - 0.9 * (years_at_company >= 5)
        - 0.7 * (years_with_curr_manager >= 3)
    )

    # Calibrated to real enterprise attrition rates (~15%)
    cutoff = np.percentile(risk_score, 85.5)
    noise = rng.normal(0, 0.30, num_rows)
    attrition = np.where((risk_score + noise) > cutoff, "Yes", "No")

    df = pd.DataFrame({
        "EmployeeID": [f"EMP-{100000 + i}" for i in range(num_rows)],
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
        "Attrition": attrition,
    })

    out = Path(output_path)
    df.to_csv(out, index=False)
    elapsed = time.time() - start_time
    file_size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Success! {num_rows:,} employee records written to {out.name}")
    print(f"File size: {file_size_mb:.2f} MB | Time: {elapsed:.2f}s")
    print(f"Attrition Distribution: {dict(df['Attrition'].value_counts())}")
    return out

if __name__ == "__main__":
    generate_enterprise_dataset(100_000, output_path="enterprise_workforce_100k.csv")
