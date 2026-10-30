"""Streamlit dashboard for the synthetic SME credit risk engine."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

import scoring  # noqa: E402
from features import CATEGORICAL, NUMERIC, add_features  # noqa: E402
from rules import check_warnings  # noqa: E402

st.set_page_config(page_title="SME Credit Risk Engine", layout="wide")
st.title("SME Credit Risk Engine")
st.warning(
    "Demo only. All businesses and outcomes are synthetic, and the model learned "
    "from a simulated risk formula. It must not be used for real lending decisions."
)

RAW_COLS = [
    "business_id", "business_name", "sector", "years_operating", "monthly_revenue",
    "monthly_expenses", "revenue_volatility", "revenue_trend_3m", "late_payments_12m",
    "existing_debt", "loan_amount", "collateral_value", "cash_buffer_days",
]
BAND_COLOURS = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/smes.csv")


@st.cache_data
def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    model = scoring._load()
    feats = add_features(df)
    out = df.copy()
    out["risk"] = model.predict_proba(feats[NUMERIC + CATEGORICAL])[:, 1]
    out["band"] = out["risk"].apply(scoring.band)
    out["warnings"] = [len(check_warnings(row)) for _, row in feats.iterrows()]
    return out


def show_result(result: dict) -> None:
    band = result["risk_band"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of default", f"{result['probability_of_default']:.1%}")
    c2.markdown(f"**Risk band**\n\n:{BAND_COLOURS[band]}[**{band}**]")
    c3.markdown(f"**Suggested action**\n\n{result['suggested_action']}")

    left, right = st.columns(2)
    with left:
        st.subheader("What is raising risk")
        for label, val in result["raising_risk"]:
            st.write(f"- {label} ({val:+.2f})")
        st.subheader("What is lowering risk")
        for label, val in result["lowering_risk"]:
            st.write(f"- {label} ({val:+.2f})")
    with right:
        st.subheader("Early warnings")
        if result["warnings"]:
            for w in result["warnings"]:
                st.error(w["message"])
        else:
            st.success("No warning rules triggered.")

    items = result["raising_risk"] + result["lowering_risk"]
    chart_df = pd.DataFrame(items, columns=["factor", "effect"])
    fig = px.bar(chart_df, x="effect", y="factor", orientation="h",
                 title="Top factors (log-odds contribution; positive raises risk)")
    st.plotly_chart(fig, width="stretch")


data = load_data()
tab1, tab2, tab3 = st.tabs(["Assess a business", "Enter a business", "Portfolio overview"])

with tab1:
    choice = st.selectbox(
        "Choose a business",
        data["business_id"] + " - " + data["business_name"],
    )
    row = data[data["business_id"] == choice.split(" - ")[0]].iloc[0]
    st.caption(
        f"Sector: {row['sector']} | Monthly revenue: ZMW {row['monthly_revenue']:,.0f} | "
        f"Loan requested: ZMW {row['loan_amount']:,.0f}"
    )
    show_result(scoring.score_business(row[RAW_COLS].to_dict()))

with tab2:
    st.write("Type in a business and see how the engine reacts.")
    a, b, c = st.columns(3)
    sector = a.selectbox("Sector", sorted(data["sector"].unique()))
    years = a.number_input("Years operating", 0, 50, 5)
    revenue = a.number_input("Monthly revenue (ZMW)", 1000.0, 1_000_000.0, 12000.0, step=500.0)
    expenses = b.number_input("Monthly expenses (ZMW)", 500.0, 1_000_000.0, 9000.0, step=500.0)
    volatility = b.slider("Revenue volatility", 0.0, 1.0, 0.3)
    trend = b.slider("3-month revenue trend", -0.5, 0.5, 0.0)
    late = c.number_input("Late payments (12 months)", 0, 12, 1)
    debt = c.number_input("Existing debt (ZMW)", 0.0, 5_000_000.0, 20000.0, step=1000.0)
    loan = c.number_input("Loan requested (ZMW)", 1000.0, 5_000_000.0, 50000.0, step=1000.0)
    collateral = c.number_input("Collateral value (ZMW)", 0.0, 5_000_000.0, 40000.0, step=1000.0)
    buffer_days = st.slider("Cash buffer (days of expenses)", 0, 200, 45)
    manual = {
        "sector": sector, "years_operating": years, "monthly_revenue": revenue,
        "monthly_expenses": expenses, "revenue_volatility": volatility,
        "revenue_trend_3m": trend, "late_payments_12m": late, "existing_debt": debt,
        "loan_amount": loan, "collateral_value": collateral, "cash_buffer_days": buffer_days,
    }
    show_result(scoring.score_business(manual))

with tab3:
    scored = score_portfolio(data)
    counts = scored["band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Low risk", int(counts["LOW"]))
    m2.metric("Medium risk", int(counts["MEDIUM"]))
    m3.metric("High risk", int(counts["HIGH"]))
    st.plotly_chart(
        px.histogram(scored, x="risk", nbins=40, title="Distribution of predicted risk"),
        width="stretch",
    )
    st.subheader("Highest-risk businesses")
    st.dataframe(
        scored.sort_values("risk", ascending=False)
        .head(15)[["business_id", "business_name", "sector", "risk", "band", "warnings"]],
        width="stretch",
    )
