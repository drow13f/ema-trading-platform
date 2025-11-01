import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor
import json
import os

# =============================================================================
# EMA BULL/BEAR RESEARCH & TRADING PLATFORM
# =============================================================================
st.set_page_config(page_title="EMA Trading Platform", layout="wide")
st.title("EMA Bull/Bear Research & Trading Platform")
st.markdown("**Multi-Timeframe Screener | RSI/Volume Filters | Backtest P&L | Watchlists**")

# -------------------------------------------------------------------------
# Simple JSON watchlist (no DB needed)
# -------------------------------------------------------------------------
WATCHLISTS_FILE = "watchlists.json"
if not os.path.exists(WATCHLISTS_FILE):
    default = {"default": ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD"]}
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(default, f)

with open(WATCHLISTS_FILE, "r") as f:
    watchlists = json.load(f)

SYMBOLS = watchlists.get("default", ["SPY"])

# Sidebar – watchlist editor
st.sidebar.header("Watchlist")
new_symbols = st.sidebar.text_area(
    "Edit (comma-separated)", value=", ".join(SYMBOLS), height=100
)
if st.sidebar.button("Save Watchlist"):
    watchlists["default"] = [
        s.strip().upper() for s in new_symbols.split(",") if s.strip()
    ]
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(watchlists, f)
    st.sidebar.success("Saved!")
    st.rerun()

# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Heatmap", "Backtest", "Filters"])

# -------------------------------------------------------------------------
# TAB 1 – SCREENER (EMA + RSI)
# -------------------------------------------------------------------------
with tab1:
    @st.cache_data(ttl=60)
    def screen_symbols(symbols):
        results = []

        def compute(sym):
            try:
                df = yf.download(sym, period="1y", progress=False, auto_adjust=True)
                if df.empty or len(df) < 50:
                    return None

                close = df["Close"]
                ema10 = close.ewm(span=10).mean()
                ema20 = close.ewm(span=20).mean()
                ema50 = close.ewm(span=50).mean()
                rsi = RSIIndicator(close).rsi().iloc[-1]

                e10 = ema10.iloc[-1]
                e20 = ema20.iloc[-1]
                e50 = ema50.iloc[-1]

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
                    "price": round(close.iloc[-1], 2),
                    "rsi": round(rsi, 1),
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            for r in ex.map(compute, symbols):
                if r:
                    results.append(r)

        return pd.DataFrame(results)

    df_screen = screen_symbols(SYMBOLS)
    if not df_screen.empty:
        st.dataframe(df_screen, use_container_width=True)
    else:
        st.info("Add symbols to your watchlist!")

# -------------------------------------------------------------------------
# TAB 2 – MULTI-TIMEFRAME HEATMAP (fixed scalar comparison)
# -------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-TF EMA Regime Heatmap")

    @st.cache_data(ttl=300)
    def get_heatmap(symbols):
        tfs = {"1d": ("1y", "1d"), "1wk": ("5y", "1wk")}
        data = {}

        for sym in symbols:
            row = {}
            for tf, (period, interval) in tfs.items():
                df = yf.download(sym, period=period, interval=interval, progress=False)
                if df.empty or len(df) < 50:
                    row[tf] = "N/A"
                    continue

                close = df["Close"]
                e10 = close.ewm(10).mean().iloc[-1]
                e20 = close.ewm(20).mean().iloc[-1]
                e50 = close.ewm(50).mean().iloc[-1]

                if e10 > e20 > e50:
                    row[tf] = "BULL"
                elif e10 < e20 < e50:
                    row[tf] = "BEAR"
                else:
                    row[tf] = "SIDE"
            data[sym] = row

        return pd.DataFrame.from_dict(data, orient="index")

    hm_df = get_heatmap(SYMBOLS)

    # Simple color map
    def color(val):
        return "green" if val == "BULL" else "red" if val == "BEAR" else "gray"

    fig = go.Figure(
        data=go.Heatmap(
            z=[[color(v) for v in row] for row in hm_df.values],
            x=hm_df.columns,
            y=hm_df.index,
            text=hm_df.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            showscale=False,
        )
    )
    fig.update_layout(title="EMA Regime Heatmap", height=150 + len(SYMBOLS) * 30)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 3 – BACKTEST P&L
# -------------------------------------------------------------------------
with tab3:
    symbol_bt = st.selectbox("Symbol", SYMBOLS, key="bt")
    start = st.date_input("Start", value=pd.to_datetime("2023-01-01"))
    capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000)

    if st.button("Run Backtest"):
        df_bt = yf.download(symbol_bt, start=start, progress=False, auto_adjust=True)
        if df_bt.empty:
            st.error("No data for this period.")
        else:
            df_bt["EMA10"] = df_bt["Close"].ewm(10).mean()
            df_bt["EMA20"] = df_bt["Close"].ewm(20).mean()
            df_bt["EMA50"] = df_bt["Close"].ewm(50).mean()

            position = 0
            cash = capital
            equity = []

            for i in range(50, len(df_bt)):
                e10 = df_bt["EMA10"].iloc[i]
                e20 = df_bt["EMA20"].iloc[i]
                e50 = df_bt["EMA50"].iloc[i]

                if e10 > e20 > e50 and position == 0:  # enter long
                    position = cash / df_bt["Close"].iloc[i]
                    cash = 0
                elif e10 < e20 < e50 and position > 0:  # exit long
                    cash = position * df_bt["Close"].iloc[i]
                    position = 0

                value = cash + position * df_bt["Close"].iloc[i]
                equity.append(value)

            final = cash + position * df_bt["Close"].iloc[-1]
            ret_pct = (final - capital) / capital * 100

            col1, col2 = st.columns(2)
            col1.metric("Final Portfolio", f"${final:,.2f}")
            col2.metric("Return", f"{ret_pct:+.2f}%")

            eq_df = pd.DataFrame(
                {"date": df_bt.index[50:], "equity": equity}
            ).set_index("date")
            st.line_chart(eq_df)

# -------------------------------------------------------------------------
# TAB 4 – RSI + VOLUME FILTERS (example)
# -------------------------------------------------------------------------
with tab4:
    st.subheader("Filter Settings (applied to Screener)")
    rsi_min = st.slider("Min RSI", 0, 100, 30)
    rsi_max = st.slider("Max RSI", 0, 100, 70)
    vol_mult = st.slider("Volume > x SMA(20)", 0.5, 3.0, 1.5)

    # Show filtered version of the screener dataframe
    if "df_screen" in locals() and not df_screen.empty:
        filtered = df_screen[
            (df_screen["rsi"] >= rsi_min)
            & (df_screen["rsi"] <= rsi_max)
        ]
        st.write(f"**{len(filtered)} symbols pass filters**")
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Run the Screener tab first.")

st.caption("Data: Yahoo Finance • Built with Streamlit")
