import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rules import check_warnings

HEALTHY = {
    "cash_buffer_days": 90, "revenue_trend_3m": 0.02, "late_payments_12m": 0,
    "debt_ratio": 0.3, "collateral_cover": 1.0, "margin": 0.25,
}


def codes(row):
    return {w["code"] for w in check_warnings(row)}


def test_healthy_business_has_no_warnings():
    assert codes(HEALTHY) == set()


def test_low_cash_runway_triggers():
    assert "LOW_CASH_RUNWAY" in codes({**HEALTHY, "cash_buffer_days": 12})


def test_falling_revenue_triggers():
    assert "FALLING_REVENUE" in codes({**HEALTHY, "revenue_trend_3m": -0.30})


def test_repeat_late_payments_triggers_at_three():
    assert "REPEAT_LATE_PAYMENTS" not in codes({**HEALTHY, "late_payments_12m": 2})
    assert "REPEAT_LATE_PAYMENTS" in codes({**HEALTHY, "late_payments_12m": 3})


def test_multiple_warnings_can_fire_together():
    bad = {**HEALTHY, "cash_buffer_days": 5, "margin": 0.01, "collateral_cover": 0.1}
    assert {"LOW_CASH_RUNWAY", "THIN_MARGIN", "THIN_COLLATERAL"} <= codes(bad)
