import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "portfolio_data" not in st.session_state:
    st.session_state.portfolio_data = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Finance Assistant")
    st.caption("Powered by GPT-4o-mini + LangGraph")
    st.divider()
    st.markdown("**Quick questions to try:**")
    st.markdown("- What is an ETF?")
    st.markdown("- Analyze my portfolio")
    st.markdown("- What is the price of AAPL?")
    st.markdown("- How long to save $50,000?")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Portfolio Dashboard", "📈 Market Data"])


# ── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab1:
    st.header("Chat with your Finance Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything about finance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from agents.orchestrator import chat
                    portfolio = st.session_state.portfolio_data
                    goals = portfolio.get("goals") if portfolio else None
                    response = chat(
                        query=prompt,
                        session_id=st.session_state.session_id,
                        portfolio=portfolio,
                        goals=goals,
                    )
                except Exception as e:
                    response = f"⚠️ Error: {e}\n\nMake sure your `.env` file has valid API keys and the FAISS index is built (`python knowledge_base/ingest.py`)."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ── Tab 2: Portfolio Dashboard ─────────────────────────────────────────────────
with tab2:
    st.header("Portfolio Dashboard")

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload your portfolio JSON", type="json")
        if uploaded:
            st.session_state.portfolio_data = json.load(uploaded)
    with col_sample:
        if st.button("📂 Load Sample Portfolio"):
            sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "sample_portfolio.json")
            with open(sample_path) as f:
                st.session_state.portfolio_data = json.load(f)

    if st.session_state.portfolio_data:
        with st.spinner("Fetching live prices..."):
            try:
                from agents.portfolio_agent import analyze
                data = analyze(st.session_state.portfolio_data)
            except Exception as e:
                st.error(f"Portfolio analysis error: {e}")
                data = None

        if data:
            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Portfolio Value", f"${data['total_portfolio_value']:,.2f}")
            k2.metric("Net Worth", f"${data['net_worth']:,.2f}")
            gl = data['total_gain_loss']
            k3.metric("Total Gain/Loss", f"${gl:,.2f}", delta=f"${gl:,.2f}")
            k4.metric("Total Liabilities", f"${data['total_liabilities']:,.2f}")

            st.divider()
            left, right = st.columns(2)

            with left:
                st.subheader("Asset Allocation")
                alloc = data["allocation_pct"]
                fig_pie = px.pie(
                    names=list(alloc.keys()),
                    values=list(alloc.values()),
                    title="Allocation by Asset Class (%)",
                    hole=0.3,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with right:
                st.subheader("Holdings Performance")
                positions = data["positions"]
                df = pd.DataFrame(positions)
                fig_bar = px.bar(
                    df,
                    x="symbol",
                    y="gain_loss",
                    color="gain_loss",
                    color_continuous_scale=["red", "green"],
                    title="Gain / Loss per Holding ($)",
                    labels={"gain_loss": "Gain/Loss ($)", "symbol": "Symbol"},
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Positions Table")
            df_display = df[["symbol", "shares", "avg_cost", "current_price", "market_value", "gain_loss", "gain_loss_pct", "asset_class"]]
            df_display.columns = ["Symbol", "Shares", "Avg Cost", "Price", "Market Value", "Gain/Loss $", "Gain/Loss %", "Asset Class"]
            st.dataframe(df_display, use_container_width=True)

            if data.get("goals"):
                st.subheader("Goals Progress")
                from agents.goal_planning_agent import analyze_goals
                goals_data = analyze_goals(data["goals"])
                for g in goals_data:
                    pct = min(g["percent_complete"], 100)
                    st.markdown(f"**{g['name']}** — ${g['current']:,.0f} / ${g['target']:,.0f} ({pct}%) — ETA: {g['time_estimate']}")
                    st.progress(pct / 100)
    else:
        st.info("Upload a portfolio JSON file or load the sample portfolio to see your dashboard.")


# ── Tab 3: Market Data ─────────────────────────────────────────────────────────
with tab3:
    st.header("Market Data")

    symbol_input = st.text_input("Enter a stock symbol (e.g. AAPL, MSFT, TSLA)", value="AAPL").upper().strip()
    period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=0)

    if st.button("🔍 Fetch Data") and symbol_input:
        col_quote, col_news = st.columns([1, 2])

        with col_quote:
            with st.spinner(f"Fetching {symbol_input}..."):
                try:
                    from agents.market_data_agent import get_quote
                    quote = get_quote(symbol_input)
                    st.metric("Price", f"${quote['price']:,.2f}")
                    if quote.get("change") is not None:
                        st.metric("Change", f"${quote['change']:,.2f}", delta=quote.get("change_pct", ""))
                    if quote.get("volume"):
                        st.metric("Volume", f"{quote['volume']:,}")
                except Exception as e:
                    st.error(f"Quote error: {e}")

        with col_news:
            with st.spinner("Fetching news..."):
                try:
                    from agents.market_data_agent import get_news
                    news = get_news(symbol_input)
                    if news:
                        st.subheader("Recent News")
                        for item in news[:5]:
                            title = item.get("title", "")
                            url = item.get("url", "#")
                            sentiment = item.get("overall_sentiment_label", "")
                            st.markdown(f"- [{title}]({url}) *{sentiment}*")
                    else:
                        st.info("No recent news found.")
                except Exception as e:
                    st.warning(f"News unavailable: {e}")

        with st.spinner("Fetching price history..."):
            try:
                from agents.market_data_agent import get_history
                history = get_history(symbol_input, period)
                if history:
                    df_hist = pd.DataFrame(history)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=df_hist["date"],
                        y=df_hist["close"],
                        mode="lines",
                        name=symbol_input,
                        line=dict(color="#00b4d8", width=2),
                    ))
                    fig_line.update_layout(
                        title=f"{symbol_input} — {period} Price History",
                        xaxis_title="Date",
                        yaxis_title="Close Price ($)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.error(f"History error: {e}")
