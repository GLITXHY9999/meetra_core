"""
Generate paired 1-Million-Employee Benchmark Datasets:
1. workforce_blind_1m.csv: Pure user data (features only, NO 'Attrition' column)
2. workforce_ground_truth_1m.csv: Paired dataset with the actual 'Attrition' ground truth
3. workforce_blind_100k.csv & workforce_ground_truth_100k.csv: Lightweight 100k pair for instant web UI testing
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Set stdout encoding for Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence loky worker warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

DATA_DIR = Path(__file__).parent / "data"


def generate_benchmark_pair(
    total_rows: int = 1_000_000,
    chunk_size: int = 250_000,
    seed: int = 999,
    tag: str = "1m",
) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    blind_path = DATA_DIR / f"workforce_blind_{tag}.csv"
    truth_path = DATA_DIR / f"workforce_ground_truth_{tag}.csv"

    if blind_path.exists():
        blind_path.unlink()
    if truth_path.exists():
        truth_path.unlink()

    print(f"\n=======================================================")
    print(f"[+] GENERATING {total_rows:,} BENCHMARK PAIR ({tag.upper()})")
    print(f"   Seed: {seed} (Out-of-sample unseen enterprise population)")
    print(f"   Blind File : {blind_path.name} (Features only, NO labels)")
    print(f"   Truth File : {truth_path.name} (Complete ground truth)")
    print(f"=======================================================")

    t0 = time.time()
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
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    total_written = 0
    attrition_counts = {"Yes": 0, "No": 0}

    for chunk_idx in range(num_chunks):
        c_rows = min(chunk_size, total_rows - total_written)
        emp_offset = total_written

        age = rng.integers(20, 64, c_rows)
        gender = rng.choice(["Female", "Male", "Non-Binary"], c_rows, p=[0.47, 0.51, 0.02])
        dept = rng.choice(departments, c_rows, p=dept_weights)
        job_roles = [rng.choice(role_map[d]) for d in dept]
        job_level = rng.choice([1, 2, 3, 4, 5], c_rows, p=[0.38, 0.33, 0.16, 0.08, 0.05])
        monthly_income = np.array([base_income_by_level[lvl] for lvl in job_level]) + rng.integers(-500, 1600, c_rows)

        overtime = rng.choice(["Yes", "No"], c_rows, p=[0.26, 0.74])
        total_working_years = np.clip(age - 21 - rng.integers(0, 3, c_rows), 0, 42)
        years_at_company = np.clip(rng.integers(0, total_working_years + 1), 0, 35)
        years_with_curr_manager = np.clip(rng.integers(0, years_at_company + 1), 0, 15)
        years_since_last_promo = np.clip(rng.integers(0, years_at_company + 1), 0, 12)

        job_sat = rng.choice([1, 2, 3, 4], c_rows, p=[0.16, 0.22, 0.36, 0.26])
        env_sat = rng.choice([1, 2, 3, 4], c_rows, p=[0.15, 0.22, 0.35, 0.28])
        work_life = rng.choice([1, 2, 3, 4], c_rows, p=[0.11, 0.24, 0.55, 0.10])
        stock_level = rng.choice([0, 1, 2, 3], c_rows, p=[0.42, 0.38, 0.14, 0.06])
        marital = rng.choice(["Single", "Married", "Divorced"], c_rows, p=[0.33, 0.47, 0.20])
        travel = rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], c_rows, p=[0.70, 0.18, 0.12])
        distance = rng.integers(1, 45, c_rows)
        num_companies = rng.integers(0, 10, c_rows)
        education = rng.choice(["Bachelors", "Masters", "Doctorate", "Associate Degree"], c_rows, p=[0.56, 0.29, 0.05, 0.10])
        perf_rating = rng.choice([2, 3, 4], c_rows, p=[0.10, 0.76, 0.14])
        percent_hike = rng.integers(11, 26, c_rows)
        training_times = rng.integers(0, 7, c_rows)

        # Multi-factor turnover ground truth calibrated to Fortune 100 enterprise turnover
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

        cutoff = np.percentile(risk_score, 85.2)
        noise = rng.normal(0, 0.28, c_rows)
        attrition = np.where((risk_score + noise) > cutoff, "Yes", "No")

        emp_ids = [f"BENCH-{2000000 + emp_offset + i}" for i in range(c_rows)]

        feature_dict = {
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

        # 1. Blind dataframe: features ONLY, NO Attrition column
        blind_chunk = pd.DataFrame(feature_dict)
        header = (chunk_idx == 0)
        blind_chunk.to_csv(blind_path, mode="a", header=header, index=False)

        # 2. Ground truth dataframe: features + Attrition column
        truth_dict = dict(feature_dict)
        truth_dict["Attrition"] = attrition
        truth_chunk = pd.DataFrame(truth_dict)
        truth_chunk.to_csv(truth_path, mode="a", header=header, index=False)

        total_written += c_rows
        counts = pd.Series(attrition).value_counts()
        attrition_counts["Yes"] += int(counts.get("Yes", 0))
        attrition_counts["No"] += int(counts.get("No", 0))

        print(f"  Processed chunk {chunk_idx + 1}/{num_chunks}: {total_written:,} / {total_rows:,} rows written...")

    elapsed = time.time() - t0
    blind_mb = blind_path.stat().st_size / (1024 * 1024)
    truth_mb = truth_path.stat().st_size / (1024 * 1024)

    print(f"\n[OK] Generation Complete ({elapsed:.2f}s)!")
    print(f"   Blind CSV       : {blind_path} ({blind_mb:.1f} MB)")
    print(f"   Ground Truth CSV: {truth_path} ({truth_mb:.1f} MB)")
    print(f"   Total Records   : {total_written:,}")
    print(f"   Ground Truth Attrition: {attrition_counts['Yes']:,} Yes / {attrition_counts['No']:,} No ({attrition_counts['Yes']/total_written*100:.2f}%)")

    return blind_path, truth_path


if __name__ == "__main__":
    # Generate 100k fast pair for immediate testing
    generate_benchmark_pair(total_rows=100_000, chunk_size=100_000, seed=999, tag="100k")

    # Generate 1M benchmark pair for executive out-of-sample certification
    generate_benchmark_pair(total_rows=1_000_000, chunk_size=250_000, seed=999, tag="1m")
