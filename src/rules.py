"""Rule-based early warnings. These are transparent heuristics, not model output."""

# Thresholds are illustrative, chosen for the demo. A real lender would set
# them with credit specialists and test them on real portfolio data.
RULES = [
    ("LOW_CASH_RUNWAY", lambda r: r["cash_buffer_days"] < 30,
     "Cash buffer covers under 30 days of expenses."),
    ("FALLING_REVENUE", lambda r: r["revenue_trend_3m"] < -0.15,
     "Revenue has dropped more than 15% over the last 3 months."),
    ("REPEAT_LATE_PAYMENTS", lambda r: r["late_payments_12m"] >= 3,
     "Three or more late payments in the last 12 months."),
    ("HIGH_DEBT_LOAD", lambda r: r["debt_ratio"] > 0.8,
     "Debt plus the new loan exceeds 80% of annual revenue."),
    ("THIN_COLLATERAL", lambda r: r["collateral_cover"] < 0.5,
     "Collateral covers less than half of the loan."),
    ("THIN_MARGIN", lambda r: r["margin"] < 0.05,
     "Profit margin is below 5%."),
]


def check_warnings(row) -> list[dict]:
    """Return a list of triggered warnings for one business (dict or Series)."""
    return [
        {"code": code, "message": message}
        for code, test, message in RULES
        if test(row)
    ]
