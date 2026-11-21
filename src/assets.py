"""Asset register: collateral tracking for the 'vault' side of the project.

Deliberately simple (CRUD over a table). No valuation modelling here —
that would need real market data this project doesn't have.
"""
from sqlalchemy import text

from db import get_engine

CATEGORIES = ["equipment", "vehicle", "property", "inventory", "other"]


def register_asset(business_id: str, description: str, category: str, value: float) -> int:
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO assets (business_id, description, category, value)
                VALUES (:bid, :desc, :cat, :val)
                RETURNING id
            """),
            {"bid": business_id, "desc": description, "cat": category, "val": value},
        )
        return row.scalar_one()


def list_assets(business_id: str):
    engine = get_engine()
    import pandas as pd
    return pd.read_sql(
        text("SELECT * FROM assets WHERE business_id = :bid ORDER BY registered_at"),
        engine, params={"bid": business_id},
    )


def total_collateral(business_id: str) -> float:
    df = list_assets(business_id)
    return float(df["value"].sum()) if not df.empty else 0.0


def coverage_vs_loan(business_id: str, loan_amount: float) -> float:
    """Ratio of registered collateral to the loan amount. >1 means fully covered."""
    total = total_collateral(business_id)
    return round(total / loan_amount, 2) if loan_amount else 0.0


if __name__ == "__main__":
    bid = "SME00001"
    register_asset(bid, "Delivery van, 2019 Toyota Hiace", "vehicle", 85000)
    register_asset(bid, "Warehouse inventory", "inventory", 22000)
    df = list_assets(bid)
    print(df[["description", "category", "value"]])
    print(f"Total registered collateral: ZMW {total_collateral(bid):,.2f}")
