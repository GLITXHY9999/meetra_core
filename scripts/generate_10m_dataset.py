"""Generate 10 Million Paired Employee Dataset on Drive M: in Streaming Chunks.

Outputs:
1. M:\meetra_data\workforce_blind_10m.csv (10,000,000 rows, employee features only, NO Attrition)
2. M:\meetra_data\workforce_ground_truth_10m.csv (10,000,000 rows, identical features + Attrition)
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_DIR = Path("M:/meetra_data")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

BLIND_PATH = TARGET_DIR / "workforce_blind_10m.csv"
TRUTH_PATH = TARGET_DIR / "workforce_ground_truth_10m.csv"

TOTAL_ROWS = 10_000_000
CHUNK_SIZE = 500_000
NUM_CHUNKS = TOTAL_ROWS // CHUNK_SIZE

DEPARTMENTS = np.array([
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
])
DEPT_WEIGHTS = np.array([0.30, 0.20, 0.15, 0.11, 0.06, 0.05, 0.04, 0.04, 0.02, 0.03])

ROLE_MAP = {
    "Engineering & Technology": np.array([
        "Software Engineer", "Senior Software Engineer", "DevOps Engineer",
        "Data Engineer", "Systems Architect", "Engineering Manager"
    ]),
    "Sales & Revenue": np.array([
        "Account Executive", "Sales Representative", "Enterprise Sales Director",
        "Business Development Rep", "Sales Operations Lead"
    ]),
    "Research & Development": np.array([
        "Research Scientist", "Staff Scientist", "Lab Technician",
        "R&D Team Lead", "Product Innovation Director"
    ]),
    "Customer Operations": np.array([
        "Customer Success Manager", "Support Specialist", "Implementation Lead",
        "Client Solutions Architect"
    ]),
    "Marketing & Growth": np.array([
        "Product Marketing Manager", "Growth Lead", "Content Strategist", "Brand Manager"
    ]),
    "Finance & Accounting": np.array([
        "Financial Analyst", "Senior Accountant", "FP&A Manager", "Finance Director"
    ]),
    "Human Resources & People": np.array([
        "HR Business Partner", "Talent Acquisition Lead", "People Operations Specialist", "HR Director"
    ]),
    "Supply Chain & Logistics": np.array([
        "Supply Chain Analyst", "Logistics Coordinator", "Procurement Manager", "Warehouse Lead"
    ]),
    "Legal & Compliance": np.array([
        "Corporate Counsel", "Compliance Specialist", "Contracts Manager"
    ]),
    "Product Management": np.array([
        "Associate Product Manager", "Product Manager", "Principal PM", "Group PM"
    ]),
}

BASE_INCOME_BY_LEVEL = np.array([0, 3500, 6300, 10500, 15800, 22500])


def generate_chunk(rng: np.random.Generator, start_idx: int, n_rows: int):
    dept_indices = rng.choice(len(DEPARTMENTS), n_rows, p=DEPT_WEIGHTS)
    depts = DEPARTMENTS[dept_indices]

    job_roles = np.empty(n_rows, dtype=object)
    for d_idx, d_name in enumerate(DEPARTMENTS):
        mask = (dept_indices == d_idx)
        count = int(np.sum(mask))
        if count > 0:
            roles = ROLE_MAP[d_name]
            job_roles[mask] = rng.choice(roles, count)

    job_level = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.38, 0.33, 0.16, 0.08, 0.05])
    monthly_income = BASE_INCOME_BY_LEVEL[job_level] + rng.integers(-400, 1500, n_rows)

    age = rng.integers(21, 62, n_rows)
    gender = rng.choice(["Female", "Male", "Non-Binary"], n_rows, p=[0.47, 0.51, 0.02])
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

    emp_ids = [f"EMP-{start_idx + i}" for i in range(n_rows)]

    features = {
        "EmployeeID": emp_ids,
        "Age": age,
        "Gender": gender,
        "Department": depts,
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

    df_blind = pd.DataFrame(features)
    df_truth = pd.DataFrame(features)
    df_truth["Attrition"] = attrition
    return df_blind, df_truth


def main():
    print(f"[*] Starting 10-Million Employee Dataset Generation on M:\\meetra_data")
    print(f"    Total Target Rows: {TOTAL_ROWS:,}")
    print(f"    Chunk Size       : {CHUNK_SIZE:,} rows per batch ({NUM_CHUNKS} batches)")
    print(f"    Blind File       : {BLIND_PATH}")
    print(f"    Ground Truth File: {TRUTH_PATH}")

    # Remove existing files if any
    if BLIND_PATH.exists():
        BLIND_PATH.unlink()
    if TRUTH_PATH.exists():
        TRUTH_PATH.unlink()

    rng = np.random.default_rng(42)
    start_time = time.time()
    total_attrition = 0

    for chunk_idx in range(NUM_CHUNKS):
        c_t0 = time.time()
        start_emp_id = 10_000_001 + (chunk_idx * CHUNK_SIZE)
        df_blind, df_truth = generate_chunk(rng, start_emp_id, CHUNK_SIZE)

        attrition_count = int((df_truth["Attrition"] == "Yes").sum())
        total_attrition += attrition_count

        write_header = (chunk_idx == 0)
        df_blind.to_csv(BLIND_PATH, mode="a", index=False, header=write_header)
        df_truth.to_csv(TRUTH_PATH, mode="a", index=False, header=write_header)

        elapsed = time.time() - start_time
        pct_done = ((chunk_idx + 1) / NUM_CHUNKS) * 100
        print(
            f"  [+] Chunk {chunk_idx + 1:2d}/{NUM_CHUNKS} ({pct_done:5.1f}%) | "
            f"{(chunk_idx + 1) * CHUNK_SIZE:,} rows written | "
            f"Chunk Time: {time.time() - c_t0:.2f}s | "
            f"Elapsed: {elapsed:.1f}s"
        )

    total_time = time.time() - start_time
    blind_size_mb = BLIND_PATH.stat().st_size / (1024 * 1024)
    truth_size_mb = TRUTH_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" [SUCCESS] 10 Million Employee Datasets Generated Successfully!")
    print(f"  Total Rows Generated : {TOTAL_ROWS:,}")
    print(f"  Total True Leavers   : {total_attrition:,} ({total_attrition / TOTAL_ROWS * 100:.2f}%)")
    print(f"  Total Retained       : {TOTAL_ROWS - total_attrition:,}")
    print(f"  Total Time Taken     : {total_time:.2f} seconds ({round(TOTAL_ROWS / total_time):,} rows/sec)")
    print(f"  Blind CSV Size       : {blind_size_mb:.2f} MB ({blind_size_mb / 1024:.2f} GB)")
    print(f"  Ground Truth CSV Size: {truth_size_mb:.2f} MB ({truth_size_mb / 1024:.2f} GB)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
