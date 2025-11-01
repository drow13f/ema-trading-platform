import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

# =============================================================================
# EMA BULL/BEAR RESEARCH & TRADING PLATFORM – WITH REGIME BUTTONS
# =============================================================================
st.set_page_config(page_title="EMA Platform", layout="wide")
st.title("EMA Bull/Bear Research & Trading Platform")
st.markdown("**Screener • Heatmap • Backtest • Filters • Regime Buttons**")

# -------------------------------------------------------------------------
# WATCHLIST
# -------------------------------------------------------------------------
WATCHLISTS_FILE = "watchlists.json"
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL"]

if not os.path.exists(WATCHLISTS_FILE):
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump({"default": DEFAULT_SYMBOLS}, f)

with open(WATCHLISTS_FILE) as f:
    watchlists = json.load(f)

SYMBOLS = watchlists.get("default", DEFAULT_SYMBOLS)

# Sidebar – Watchlist + Regime Buttons
st.sidebar.header("Watchlist")
new_symbols = st.sidebar.text_area("Edit (comma)", ", ".join(SYMBOLS), height=100)
if st.sidebar.button("Save Watchlist"):
    watchlists["default"] = [s.strip().upper() for s in new_symbols.split(",") if s.strip()]
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(watchlists, f)
    st.sidebar.success("Saved!")
    st.rerun()

st.sidebar.header("Regime Filters")
# Regime buttons (color-coded with markdown)
if st.sidebar.button("🟢 Strong Bull"):
    st.session_state.regime_filter = "STRONG BULL"
if st.sidebar.button("🟡 Bull"):
    st.session_state.regime_filter = "BULL"
if st.sidebar.button("🟠 Weak Bull"):
    st.session_state.regime_filter = "WEAK BULL"
if st.sidebar.button("⚪ Neutral/Alert"):
    st.session_state.regime_filter = "NEUTRAL/ALERT"
if st.sidebar.button("🟤 Weak Bear"):
    st.session_state.regime_filter = "WEAK BEAR"
if st.sidebar.button("🔴 Bear"):
    st.session_state.regime_filter = "BEAR"
if st.sidebar.button("⚫ Strong Bear"):
    st.session_state.regime_filter = "STRONG BEAR"
if st.sidebar.button("❌ Clear Filter"):
    st.session_state.regime_filter = None

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Heatmap", "Backtest", "Filters"])

# -------------------------------------------------------------------------
# SAFE YFINANCE (retry + fallback)
# -------------------------------------------------------------------------
def safe_download(symbol, period="1y", interval="1d", retries=2):
    for _ in range(retries):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                quiet=True,
            )
            if not df.empty and len(df) >= 50:
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

# -------------------------------------------------------------------------
# TAB 1 – SCREENER (with regime filter from buttons)
# -------------------------------------------------------------------------
with tab1:
    @st.cache_data(ttl=60)
    def screen_symbols(symbols):
        results = []

        def compute(sym):
            df = safe_download(sym)
            if df.empty:
                return None

            close = df["Close"]
            ema10 = close.ewm(span=10, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            rsi_val = RSIIndicator(close).rsi().iloc[-1]

            e10 = ema10.iloc[-1]
            e20 = ema20.iloc[-1]
            e50 = ema50.iloc[-1]

            if pd.isna(e10) or pd.isna(e20) or pd.isna(e50):
                return None

            regime = (
                "STRONG BULL" if e10 > e20 > e50 else
                "WEAK BULL" if e10 > e20 else
                "STRONG BEAR" if e10 < e20 < e50 else
                "WEAK BEAR" if e10 < e20 else
                "SIDEWAYS"
            )

            return {
                "symbol": sym,
                "regime": regime,
                "price": round(close.iloc[-1], 2),
                "rsi": round(rsi_val, 1) if not pd.isna(rsi_val) else "N/A",
            }

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(compute, sym) for sym in symbols]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        return pd.DataFrame(results) if results else pd.DataFrame()

    df_screen = screen_symbols(SYMBOLS)

    if not df_screen.empty:
        # Apply regime filter from buttons
        if "regime_filter" in st.session_state and st.session_state.regime_filter:
            df_screen = df_screen[df_screen["regime"] == st.session_state.regime_filter]
            st.write(f"**Filtered by {st.session_state.regime_filter}**")
        st.dataframe(df_screen, use_container_width=True)
    else:
        st.warning("No live data – showing demo")
        demo = pd.DataFrame([
            {"symbol": "SPY", "regime": "STRONG BULL", "price": 580.50, "rsi": 68.2},
            {"symbol": "AAPL", "regime": "WEAK BULL", "price": 232.10, "rsi": 55.4},
            {"symbol": "QQQ", "regime": "STRONG BULL", "price": 495.30, "rsi": 70.1},
        ])
        st.dataframe(demo, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 2 – HEATMAP
# -------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-Timeframe Heatmap")

    @st.cache_data(ttl=300)
    def get_heatmap(symbols):
        tfs = {"1d": ("2y", "1d"), "1wk": ("10y", "1wk")}
        data = {}

        for sym in symbols:
            row = {}
            for label, (p, i) in tfs.items():
                df = safe_download(sym, period=p, interval=i)
                if df.empty:
                    row[label] = "N/A"
                    continue
                close = df["Close"]
                e10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
                e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
                e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
                if pd.isna(e10):
                    row[label] = "N/A"
                    continue
                row[label] = "BULL" if e10 > e20 > e50 else "BEAR" if e10 < e20 < e50 else "SIDE"
            data[sym] = row
        return pd.DataFrame.from_dict(data, orient="index")

    hm_df = get_heatmap(SYMBOLS)

    def color(val):
        return "green" if val == "BULL" else "red" if val == "BEAR" else "lightgray"

    fig = go.Figure(data=go.Heatmap(
        z=[[color(v) for v in row] for row in hm_df.values],
        x=hm_df.columns, y=hm_df.index,
        text=hm_df.values, texttemplate="%{text}",
        colorscale="RdYlGn", showscale=False
    ))
    fig.update_layout(height=200 + len(SYMBOLS)*35)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 3 – BACKTEST
# -------------------------------------------------------------------------
with tab3:
    symbol_bt = st.selectbox("Symbol", SYMBOLS or ["SPY"])
    start = st.date_input("Start", pd.to_datetime("2023-01-01"))
    capital = st.number_input("Capital ($)", 1000, 1000000, 10000)

    if st.button("Run"):
        df_bt = safe_download(symbol_bt, start=start)
        if df_bt.empty:
            st.error("No data. Try SPY.")
        else:
            df_bt["EMA10"] = df_bt["Close"].ewm(span=10, adjust=False).mean()
            df_bt["EMA20"] = df_bt["Close"].ewm(span=20, adjust=False).mean()
            df_bt["EMA50"] = df_bt["Close"].ewm(span=50, adjust=False).mean()

            position = 0  # shares
            cash = capital
            equity = []

            for i in range(50, len(df_bt)):
                e10 = df_bt["EMA10"].iloc[i]
                e20 = df_bt["EMA20"].iloc[i]
                e50 = df_bt["EMA50"].iloc[i]
                price = df_bt["Close"].iloc[i]

                if pd.isna(e10) or pd.isna(e20) or pd.isna(e50):
                    continue

                if e10 > e20 > e50 and cash > 0:
                    position = cash / price
                    cash = 0
                elif e10 < e20 < e50 and position > 0:
                    cash = position * price
                    position = 0

                value = cash + position * price
                equity.append(value)

            final = cash + position * df_bt["Close"].iloc[-1]
            ret_pct = (final - capital) / capital * 100

            col1, col2 = st.columns(2)
            col1.metric("Final", f"${final:,.0f}")
            col2.metric("Return", f"{ret_pct:+.1f}%")

            eq_series = pd.Series(equity, index=df_bt.index[50:])
            st.line_chart(eq_series)

# -------------------------------------------------------------------------
# TAB 4 – FILTERS
# -------------------------------------------------------------------------
with tab4:
    st.subheader("Filter Screener")
    rsi_min = st.slider("Min RSI", 0, 100, 30)
    rsi_max = st.slider("Max RSI", 0, 100, 70)

    if "df_screen" in locals() and not df_screen.empty:
        filtered = df_screen[
            (df_screen["rsi"] != "N/A") &
            (df_screen["rsi"].astype(float).between(rsi_min, rsi_max))
        ]
        st.write(f"**{len(filtered)} matches**")
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Run **Screener** tab first.")

st.caption("Data: Yahoo Finance • Built with Streamlit • 100% Stable")
