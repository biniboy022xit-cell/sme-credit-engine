"""Score a business from PostgreSQL and save the result as an assessment row."""
import json

import pandas as pd
from sqlalchemy import text

from db import get_engine
from scoring import score_business

RAW_COLS = [
    "business_id", "business_name", "sector", "years_operating", "monthly_revenue",
    "monthly_expenses", "revenue_volatility", "revenue_trend_3m", "late_payments_12m",
    "existing_debt", "loan_amount", "collateral_value", "cash_buffer_days",
]


def get_business(business_id: str) -> dict:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM businesses WHERE business_id = :bid"),
        engine, params={"bid": business_id},
    )
    if df.empty:
        raise ValueError(f"No business with id {business_id}")
    return df.iloc[0][RAW_COLS].to_dict()


def save_assessment(business_id: str, result: dict) -> int:
    engine = get_engine()
    reasons = {"raising_risk": result["raising_risk"], "lowering_risk": result["lowering_risk"]}
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO assessments
                    (business_id, probability_of_default, risk_band,
                     suggested_action, warnings, reasons)
                VALUES
                    (:bid, :p, :band, :action, :warnings, :reasons)
                RETURNING id
            """),
            {
                "bid": business_id,
                "p": result["probability_of_default"],
                "band": result["risk_band"],
                "action": result["suggested_action"],
                "warnings": json.dumps(result["warnings"]),
                "reasons": json.dumps(reasons),
            },
        )
        return row.scalar_one()


def score_and_save(business_id: str) -> dict:
    raw = get_business(business_id)
    result = score_business(raw)
    assessment_id = save_assessment(business_id, result)
    result["assessment_id"] = assessment_id
    result["business_name"] = raw["business_name"]
    return result


if __name__ == "__main__":
    result = score_and_save("SME00001")
    print(f"{result['business_name']}: {result['probability_of_default']:.1%} "
          f"-> {result['risk_band']} (assessment #{result['assessment_id']})")
