"""ROI math for the retention simulator — pulled out of the route handler
so it can be unit tested and tuned without touching HTTP concerns."""
from __future__ import annotations

TURNOVER_REPLACEMENT_MULTIPLE = 1.5  # SHRM-style rule of thumb: ~1.5x annual salary
RETENTION_INTERVENTION_RATE = 0.08  # assumed cost of a targeted retention program


def compute_financials(monthly_income: float) -> dict[str, float]:
    annual_salary = max(monthly_income, 0) * 12
    turnover_cost = round(annual_salary * TURNOVER_REPLACEMENT_MULTIPLE)
    retention_cost = round(annual_salary * RETENTION_INTERVENTION_RATE)
    return {
        "turnover_replacement_cost": turnover_cost,
        "retention_intervention_cost": retention_cost,
        "net_capital_saved": turnover_cost - retention_cost,
    }
