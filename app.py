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
# EMA BULL/BEAR RESEARCH & TRADING PLATFORM – FIXED & STABLE
# =============================================================================
st.set_page_config(page_title="EMA Platform", layout="wide")
st.title("EMA Bull/Bear Research & Trading Platform")
st.markdown("**Screener • Heatmap • Backtest • Filters • Zero Errors**")

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
# TAB 1 – SCREENER
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
            ema50 = close
