"""Load the synthetic SME CSV into PostgreSQL."""
import pandas as pd

from db import get_engine, init_schema

COLS = [
    "business_id", "business_name", "sector", "years_operating", "monthly_revenue",
    "monthly_expenses", "revenue_volatility", "revenue_trend_3m", "late_payments_12m",
    "existing_debt", "loan_amount", "collateral_value", "cash_buffer_days", "defaulted",
]

if __name__ == "__main__":
    init_schema()
    df = pd.read_csv("data/smes.csv")[COLS]
    df["defaulted"] = df["defaulted"].astype(bool)

    engine = get_engine()
    df.to_sql("businesses", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"Loaded {len(df)} businesses into PostgreSQL.")
