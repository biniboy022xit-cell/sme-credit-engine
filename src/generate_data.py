"""Generate SYNTHETIC SME data for the credit engine demo.

Everything here is simulated. The 'default' label comes from a hand-written
risk formula plus noise, so a model trained on it only learns that formula.
It is for demonstrating the pipeline, not for real lending decisions.
"""
import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
N = 3000

rng = np.random.default_rng(SEED)
Faker.seed(SEED)
fake = Faker()

SECTORS = ["retail", "agriculture", "transport", "food_services", "services", "manufacturing"]
SECTOR_P = [0.28, 0.18, 0.12, 0.16, 0.16, 0.10]
SECTOR_RISK = {
    "retail": 0.0, "agriculture": 0.3, "transport": 0.15,
    "food_services": 0.25, "services": -0.1, "manufacturing": 0.05,
}


def make_businesses(n: int) -> pd.DataFrame:
    sector = rng.choice(SECTORS, n, p=SECTOR_P)
    years = rng.integers(0, 16, n)

    revenue = rng.lognormal(mean=9.3, sigma=0.7, size=n)      # monthly revenue (ZMW)
    expenses = revenue * rng.uniform(0.55, 1.0, n)
    margin = (revenue - expenses) / revenue
    volatility = rng.uniform(0.05, 0.6, n)                    # month-to-month revenue swing
    revenue_trend = rng.normal(0, 0.15, n)                    # 3-month revenue change
    late_payments = rng.poisson(1.2, n)                       # late payments, last 12 months

    existing_debt = revenue * rng.uniform(0, 6, n)
    loan_amount = revenue * rng.uniform(1, 8, n)
    collateral = loan_amount * rng.uniform(0, 1.5, n)
    cash_buffer = revenue * rng.uniform(0, 3, n)
    cash_buffer_days = cash_buffer / (expenses / 30)

    debt_ratio = (existing_debt + loan_amount) / (revenue * 12)
    collateral_cover = collateral / loan_amount

    # Hidden risk formula (the "ground truth" of this simulation)
    z = (
        -2.0
        + 2.5 * debt_ratio
        - 3.0 * margin
        + 1.5 * volatility
        - 1.5 * revenue_trend
        + 0.35 * late_payments
        - 0.08 * years
        - 0.01 * cash_buffer_days
        - 0.6 * collateral_cover
        + np.array([SECTOR_RISK[s] for s in sector])
        + rng.normal(0, 0.5, n)
    )
    p_default = 1 / (1 + np.exp(-z))
    defaulted = (rng.random(n) < p_default).astype(int)

    return pd.DataFrame({
        "business_id": [f"SME{i:05d}" for i in range(n)],
        "business_name": [fake.company() for _ in range(n)],
        "sector": sector,
        "years_operating": years,
        "monthly_revenue": revenue.round(2),
        "monthly_expenses": expenses.round(2),
        "revenue_volatility": volatility.round(3),
        "revenue_trend_3m": revenue_trend.round(3),
        "late_payments_12m": late_payments,
        "existing_debt": existing_debt.round(2),
        "loan_amount": loan_amount.round(2),
        "collateral_value": collateral.round(2),
        "cash_buffer_days": cash_buffer_days.round(1),
        "defaulted": defaulted,
    })


if __name__ == "__main__":
    df = make_businesses(N)
    df.to_csv("data/smes.csv", index=False)
    print(f"Saved {len(df)} rows to data/smes.csv")
    print(f"Default rate: {df['defaulted'].mean():.1%}")
    print(df.head(3).T)
