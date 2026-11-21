"""Streamlit dashboard for the SME credit risk engine, backed by PostgreSQL."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent / "src"))

import scoring  # noqa: E402
from db import get_engine  # noqa: E402
from db_scoring import get_business, save_assessment, score_and_save  # noqa: E402
from features import CATEGORICAL, NUMERIC, add_features  # noqa: E402
from rules import check_warnings  # noqa: E402

st.set_page_config(page_title="SME Credit Risk Engine", layout="wide")
st.title("SME Credit Risk Engine")
st.warning(
    "Demo only. All businesses and outcomes are synthetic, and the model learned "
    "from a simulated risk formula. It must not be used for real lending decisions."
)

BAND_COLOURS = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}


@st.cache_resource
def engine():
    return get_engine()


def load_businesses() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM businesses ORDER BY business_id", engine())


def load_history(business_id: str) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT assessed_at, probability_of_default, risk_band, suggested_action, warnings
            FROM assessments WHERE business_id = :bid ORDER BY assessed_at DESC
        """),
        engine(), params={"bid": business_id},
    )


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


data = load_businesses()
tab1, tab2, tab3, tab4 = st.tabs(
    ["Assess a business", "Enter a business", "Portfolio overview", "Ask the assistant"]
)

with tab1:
    choice = st.selectbox(
        "Choose a business",
        data["business_id"] + " - " + data["business_name"],
    )
    bid = choice.split(" - ")[0]
    row = data[data["business_id"] == bid].iloc[0]
    st.caption(
        f"Sector: {row['sector']} | Monthly revenue: ZMW {row['monthly_revenue']:,.0f} | "
        f"Loan requested: ZMW {row['loan_amount']:,.0f}"
    )
    if st.button("Score and save to database", key="score_existing"):
        result = score_and_save(bid)
        st.session_state["last_result"] = result
    if "last_result" in st.session_state and st.session_state["last_result"].get("business_name") == row["business_name"]:
        show_result(st.session_state["last_result"])

    st.divider()
    st.subheader(f"Assessment history for {row['business_name']}")
    hist = load_history(bid)
    if hist.empty:
        st.info("No assessments saved yet for this business. Click the button above.")
    else:
        hist_display = hist.copy()
        hist_display["probability_of_default"] = hist_display["probability_of_default"].map("{:.1%}".format)
        st.dataframe(
            hist_display[["assessed_at", "probability_of_default", "risk_band", "suggested_action"]],
            width="stretch",
        )

with tab2:
    st.write("Type in a business, score it, and optionally save the result.")
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
    result = scoring.score_business(manual)
    show_result(result)
    st.caption("This result is not saved unless the business exists in the database.")

with tab3:
    scored = pd.read_sql("""
        SELECT b.business_id, b.business_name, b.sector,
               a.probability_of_default AS risk, a.risk_band AS band
        FROM businesses b
        LEFT JOIN LATERAL (
            SELECT probability_of_default, risk_band
            FROM assessments WHERE business_id = b.business_id
            ORDER BY assessed_at DESC LIMIT 1
        ) a ON true
    """, engine())
    assessed = scored.dropna(subset=["risk"])
    st.caption(f"{len(assessed)} of {len(scored)} businesses have a saved assessment.")
    if assessed.empty:
        st.info("No assessments saved yet. Score some businesses in the first tab.")
    else:
        counts = assessed["band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Low risk", int(counts["LOW"]))
        m2.metric("Medium risk", int(counts["MEDIUM"]))
        m3.metric("High risk", int(counts["HIGH"]))
        st.plotly_chart(
            px.histogram(assessed, x="risk", nbins=40, title="Distribution of saved risk scores"),
            width="stretch",
        )
        st.subheader("Highest-risk assessed businesses")
        st.dataframe(
            assessed.sort_values("risk", ascending=False).head(15),
            width="stretch",
        )

with tab4:
    st.write(
        "Ask about any business ID (e.g. SME00001). The assistant calls the same "
        "risk model and asset register as the other tabs — it does not invent numbers, "
        "and it does not approve or decline anything itself."
    )
    question = st.text_input(
        "Your question",
        placeholder="Assess SME00001 and tell me if I should be worried.",
    )
    if st.button("Ask", key="ask_agent"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            try:
                from agent import run_agent
                with st.spinner("Calling the model and checking the database..."):
                    answer = run_agent(question)
                st.markdown(answer)
            except KeyError:
                st.error(
                    "No GROQ_API_KEY found. Add one to your .env file to use this tab "
                    "(see .env.example)."
                )
            except Exception as e:
                st.error(f"The assistant could not complete the request: {e}")
