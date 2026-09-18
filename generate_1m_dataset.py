"""
Generate a realistic 1,000,000-employee (10 Lakh records) enterprise workforce dataset.
Calibrated to Global Fortune 100 enterprise hierarchies, realistic compensation bands,
and multi-factorial turnover dynamics.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd


def generate_enterprise_1m_dataset(
    total_rows: int = 1_000_000,
    chunk_size: int = 250_000,
    seed: int = 42,
    output_path: str = "enterprise_workforce_1m.csv",
) -> Path:
    print(f"Generating 1-Million-Record Enterprise Dataset ({total_rows:,} employees / 10 Lakhs)...")
    start_time = time.time()
    rng = np.random.default_rng(seed)
    out = Path(output_path)

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

    if out.exists():
        out.unlink()

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

        # Multi-factor turnover ground truth calibrated to Fortune 100 benchmarks
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

        chunk_df = pd.DataFrame({
            "EmployeeID": [f"EMP-{1000000 + emp_offset + i}" for i in range(c_rows)],
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

        header = (chunk_idx == 0)
        chunk_df.to_csv(out, mode="a", header=header, index=False)
        total_written += c_rows
        counts = chunk_df["Attrition"].value_counts()
        attrition_counts["Yes"] += int(counts.get("Yes", 0))
        attrition_counts["No"] += int(counts.get("No", 0))

        print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {total_written:,} / {total_rows:,} rows written...")

    elapsed = time.time() - start_time
    file_size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nCompleted! 1-Million Dataset written to '{out.name}'.")
    print(f"Total Rows: {total_written:,} | File Size: {file_size_mb:.2f} MB | Time Elapsed: {elapsed:.2f}s")
    print(f"Attrition Breakdown: {attrition_counts} (Rate: {(attrition_counts['Yes'] / total_written) * 100:.2f}%)")
    return out


if __name__ == "__main__":
    generate_enterprise_1m_dataset(1_000_000, output_path="enterprise_workforce_1m.csv")
