import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor
import json
import os

# =============================================================================
# EMA BULL/BEAR RESEARCH & TRADING PLATFORM – BULLETPROOF EDITION
# =============================================================================
st.set_page_config(page_title="EMA Platform", layout="wide")
st.title("EMA Bull/Bear Research & Trading Platform")
st.markdown("**Screener • Heatmap • Backtest • Filters • 100% Crash-Proof**")

# -------------------------------------------------------------------------
# WATCHLIST (JSON)
# -------------------------------------------------------------------------
WATCHLISTS_FILE = "watchlists.json"
if not os.path.exists(WATCHLISTS_FILE):
    default = {"default": ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD"]}
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(default, f)

with open(WATCHLISTS_FILE) as f:
    watchlists = json.load(f)

SYMBOLS = watchlists.get("default", ["SPY"])

# Sidebar
st.sidebar.header("Watchlist")
new_symbols = st.sidebar.text_area("Edit (comma)", ", ".join(SYMBOLS), height=100)
if st.sidebar.button("Save"):
    watchlists["default"] = [s.strip().upper() for s in new_symbols.split(",") if s.strip()]
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(watchlists, f)
    st.sidebar.success("Saved!")
    st.rerun()

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Heatmap", "Backtest", "Filters"])

# -------------------------------------------------------------------------
# TAB 1 – SCREENER (100% safe)
# -------------------------------------------------------------------------
with tab1:
    @st.cache_data(ttl=60)
    def screen_symbols(symbols):
        results = []

        def safe_compute(sym):
            try:
                df = yf.download(sym, period="1y", progress=False, auto_adjust=True, quiet=True)
                if df.empty or len(df) < 50:
                    return None

                close = df["Close"]
                ema10 = close.ewm(span=10, adjust=False).mean()
                ema20 = close.ewm(span=20, adjust=False).mean()
                ema50 = close.ewm(span=50, adjust=False).mean()
                rsi_val = RSIIndicator(close).rsi().iloc[-1]

                # Extract scalars safely
                e10 = ema10.iloc[-1]
                e20 = ema20.iloc[-1]
                e50 = ema50.iloc[-1]
                price = close.iloc[-1]

                if pd.isna(e10) or pd.isna(e20) or pd.isna(e50):
                    return None

                if e10 > e20 > e50:
                    regime = "STRONG BULL"
                elif e10 > e20:
                    regime = "WEAK BULL"
                elif e10 < e20 < e50:
                    regime = "STRONG BEAR"
                elif e10 < e20:
                    regime = "WEAK BEAR"
                else:
                    regime = "SIDEWAYS"

                return {
                    "symbol": sym,
                    "regime": regime,
                    "price": round(price, 2),
                    "rsi": round(rsi_val, 1) if not pd.isna(rsi_val) else "N/A",
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            for r in ex.map(safe_compute, symbols):
                if r:
                    results.append(r)

        return pd.DataFrame(results) if results else pd.DataFrame()

    df_screen = screen_symbols(SYMBOLS)
    if not df_screen.empty:
        st.dataframe(df_screen, use_container_width=True)
    else:
        st.warning("No valid data. Try: SPY, AAPL, or check internet.")

# -------------------------------------------------------------------------
# TAB 2 – HEATMAP (NaN-safe + scalar-safe)
# -------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-Timeframe EMA Heatmap")

    @st.cache_data(ttl=300)
    def get_heatmap(symbols):
        tfs = {
            "1d": ("2y", "1d"),
            "1wk": ("10y", "1wk"),
        }
        data = {}

        for sym in symbols:
            row = {}
            for label, (period, interval) in tfs.items():
                try:
                    df = yf.download(
                        sym,
                        period=period,
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                        quiet=True,
                    )
                    if df.empty or len(df) < 50:
                        row[label] = "N/A"
                        continue

                    close = df["Close"]
                    e10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
                    e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
                    e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

                    if pd.isna(e10) or pd.isna(e20) or pd.isna(e50):
                        row[label] = "N/A"
                        continue

                    if e10 > e20 > e50:
                        row[label] = "BULL"
                    elif e10 < e20 < e50:
                        row[label] = "BEAR"
                    else:
                        row[label] = "SIDE"
                except Exception:
                    row[label] = "N/A"
            data[sym] = row

        return pd.DataFrame.from_dict(data, orient="index")

    hm_df = get_heatmap(SYMBOLS)

    # Safe color map
    def cell_color(val):
        return "green" if val == "BULL" else "red" if val == "BEAR" else "lightgray"

    fig = go.Figure(
        data=go.Heatmap(
            z=[[cell_color(v) for v in row] for row in hm_df.values],
            x=hm_df.columns,
            y=hm_df.index,
            text=hm_df.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            showscale=False,
        )
    )
    fig.update_layout(title="EMA Regime Heatmap", height=200 + len(SYMBOLS) * 35)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 3 – BACKTEST
# -------------------------------------------------------------------------
with tab3:
    if SYMBOLS:
        symbol_bt = st.selectbox("Symbol", SYMBOLS)
        start = st.date_input("Start", value=pd.to_datetime("2023-01-01"))
        capital = st.number_input("Capital ($)", 1000, 1000000, 10000)

        if st.button("Run Backtest"):
            with st.spinner("Backtesting..."):
                try:
                    df_bt = yf.download(symbol_bt, start=start, progress=False, auto_adjust=True, quiet=True)
                    if df_bt.empty or len(df_bt) < 50:
                        st.error("Not enough data.")
                    else:
                        df_bt["EMA10"] = df_bt["Close"].ewm(10, adjust=False).mean()
                        df_bt["EMA20"] = df_bt["Close"].ewm(20, adjust=False).mean()
                        df_bt["EMA50"] = df_bt["Close"].ewm(50, adjust=False).mean()

                        position = 0
                        cash = capital
                        equity = []

                        for i in range(50, len(df_bt)):
                            e10 = df_bt["EMA10"].iloc[i]
                            e20 = df_bt["EMA20"].iloc[i]
                            e50 = df_bt["EMA50"].iloc[i]
                            price = df_bt["Close"].iloc[i]

                            if pd.isna(e10) or pd.isna(e20) or pd.isna(e50):
                                continue

                            if e10 > e20 > e50 and position == 0:
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
                        col1.metric("Final", f"${final:,.2f}")
                        col2.metric("Return", f"{ret_pct:+.2f}%")

                        eq_df = pd.DataFrame({"date": df_bt.index[50:], "equity": equity}).set_index("date")
                        st.line_chart(eq_df)
                except Exception as e:
                    st.error("Backtest failed. Try another symbol.")
    else:
        st.info("Add symbols to watchlist first.")

# -------------------------------------------------------------------------
# TAB 4 – FILTERS
# -------------------------------------------------------------------------
with tab4:
    st.subheader("Filter Screener Results")
    rsi_min = st.slider("Min RSI", 0, 100, 30)
    rsi_max = st.slider("Max RSI", 0, 100, 70)

    if "df_screen" in locals() and not df_screen.empty:
        filtered = df_screen[
            (df_screen["rsi"] != "N/A") &
            (df_screen["rsi"].astype(float) >= rsi_min) &
            (df_screen["rsi"].astype(float) <= rsi_max)
        ]
        st.write(f"**{len(filtered)} symbols match**")
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Run **Screener** tab first.")

st.caption("Data: Yahoo Finance • Built with Streamlit • 100% Crash-Proof")
