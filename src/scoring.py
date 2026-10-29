"""Score one business: risk probability, band, plain-language reasons, warnings."""
import joblib
import numpy as np
import pandas as pd

from features import CATEGORICAL, NUMERIC, add_features
from rules import check_warnings

LOW_CUTOFF = 0.15    # below this: low risk
HIGH_CUTOFF = 0.30   # at or above this: high risk (see src/thresholds.py)

READABLE = {
    "debt_ratio": "debt load relative to annual revenue",
    "late_payments_12m": "late payments in the last 12 months",
    "margin": "profit margin",
    "cash_buffer_days": "cash buffer",
    "years_operating": "years in operation",
    "collateral_cover": "collateral cover",
    "revenue_volatility": "revenue volatility",
    "revenue_trend_3m": "3-month revenue trend",
}

_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load("models/credit_model.joblib")
    return _model


def band(p: float) -> str:
    if p < LOW_CUTOFF:
        return "LOW"
    if p < HIGH_CUTOFF:
        return "MEDIUM"
    return "HIGH"


def suggested_action(risk_band: str) -> str:
    return {
        "LOW": "Eligible for standard review",
        "MEDIUM": "Needs a closer manual review",
        "HIGH": "Escalate to a credit officer before any decision",
    }[risk_band]


def score_business(raw: dict) -> dict:
    """raw: one business with the raw columns from data/smes.csv."""
    model = _load()
    df = add_features(pd.DataFrame([raw]))
    X = df[NUMERIC + CATEGORICAL]

    p = float(model.predict_proba(X)[0, 1])

    # Per-business explanation: standardised value x coefficient (log-odds)
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = prep.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    contrib = Xt[0] * clf.coef_[0]
    names = prep.get_feature_names_out()

    reasons_up, reasons_down = [], []
    for name, c in sorted(zip(names, contrib), key=lambda t: abs(t[1]), reverse=True):
        key = name.split("__", 1)[1]
        label = READABLE.get(key, key.replace("sector_", "sector: "))
        (reasons_up if c > 0 else reasons_down).append((label, round(float(c), 2)))

    risk_band = band(p)
    return {
        "probability_of_default": round(p, 3),
        "risk_band": risk_band,
        "suggested_action": suggested_action(risk_band),
        "raising_risk": reasons_up[:3],
        "lowering_risk": reasons_down[:3],
        "warnings": check_warnings(df.iloc[0]),
    }


if __name__ == "__main__":
    sample = pd.read_csv("data/smes.csv").iloc[1].to_dict()
    sample.pop("defaulted", None)
    result = score_business(sample)
    print(f"{sample['business_name']} ({sample['business_id']})")
    print(f"Risk: {result['probability_of_default']:.1%} -> {result['risk_band']}")
    print(f"Action: {result['suggested_action']}")
    print("Raising risk:", result["raising_risk"])
    print("Lowering risk:", result["lowering_risk"])
    print("Warnings:")
    for w in result["warnings"]:
        print(f"  [{w['code']}] {w['message']}")
