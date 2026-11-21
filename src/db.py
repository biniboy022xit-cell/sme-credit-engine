"""Database connection and schema setup for the SME credit engine."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', 'postgres')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'sme_credit')}"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    business_id         TEXT PRIMARY KEY,
    business_name        TEXT NOT NULL,
    sector                TEXT NOT NULL,
    years_operating       INTEGER NOT NULL,
    monthly_revenue       NUMERIC(14, 2) NOT NULL,
    monthly_expenses      NUMERIC(14, 2) NOT NULL,
    revenue_volatility    NUMERIC(6, 3) NOT NULL,
    revenue_trend_3m      NUMERIC(6, 3) NOT NULL,
    late_payments_12m     INTEGER NOT NULL,
    existing_debt         NUMERIC(14, 2) NOT NULL,
    loan_amount           NUMERIC(14, 2) NOT NULL,
    collateral_value       NUMERIC(14, 2) NOT NULL,
    cash_buffer_days       NUMERIC(8, 1) NOT NULL,
    defaulted              BOOLEAN,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS assessments (
    id                      SERIAL PRIMARY KEY,
    business_id             TEXT NOT NULL REFERENCES businesses(business_id),
    probability_of_default  NUMERIC(6, 4) NOT NULL,
    risk_band               TEXT NOT NULL,
    suggested_action        TEXT NOT NULL,
    warnings                JSONB NOT NULL DEFAULT '[]',
    reasons                 JSONB NOT NULL DEFAULT '{}',
    assessed_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS assets (
    id             SERIAL PRIMARY KEY,
    business_id    TEXT NOT NULL REFERENCES businesses(business_id),
    description    TEXT NOT NULL,
    category       TEXT NOT NULL,
    value          NUMERIC(14, 2) NOT NULL,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assessments_business ON assessments(business_id);
CREATE INDEX IF NOT EXISTS idx_assets_business ON assets(business_id);
"""


def get_engine():
    return create_engine(DB_URL)


def init_schema():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(SCHEMA))
    print("Schema is up to date.")


if __name__ == "__main__":
    init_schema()
