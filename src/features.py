"""Shared feature engineering so training and scoring use identical logic."""
import pandas as pd

NUMERIC = [
    "years_operating", "revenue_volatility", "revenue_trend_3m",
    "late_payments_12m", "cash_buffer_days",
    "margin", "debt_ratio", "collateral_cover",
]
CATEGORICAL = ["sector"]
TARGET = "defaulted"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["margin"] = (out["monthly_revenue"] - out["monthly_expenses"]) / out["monthly_revenue"]
    out["debt_ratio"] = (out["existing_debt"] + out["loan_amount"]) / (out["monthly_revenue"] * 12)
    out["collateral_cover"] = out["collateral_value"] / out["loan_amount"]
    return out
