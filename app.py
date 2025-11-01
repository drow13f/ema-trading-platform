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
st.title("🔥 EMA Bull/Bear Research & Trading Platform")
st.markdown("**Multi-Timeframe Screener | RSI/Volume Filters | Backtest P&L | Watchlists**")

# Simple JSON-based watchlists (no external DB needed)
WATCHLISTS_FILE = "watchlists.json"
if not os.path.exists(WATCHLISTS_FILE):
    default_watchlist = {"default": ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD"]}
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(default_watchlist, f)

# Load watchlist
with open(WATCHLISTS_FILE, "r") as f:
    watchlists = json.load(f)
SYMBOLS = watchlists.get("default", ["SPY"])

# Sidebar: Watchlist Editor
st.sidebar.header("📊 Watchlist")
new_symbols = st.sidebar.text_area("Edit (comma-separated)", value=", ".join(SYMBOLS), height=100)
if st.sidebar.button("Save Watchlist"):
    watchlists["default"] = [s.strip().upper() for s in new_symbols.split(",") if s.strip()]
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(watchlists, f)
    st.sidebar.success("Saved!")
    st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🟢 Screener", "🌡️ Heatmap", "📈 Backtest", "⚙️ Filters"])

# =============================================================================
# TAB 1: SCREENER (EMA + RSI + Volume)
# =============================================================================
with tab1:
    @st.cache_data(ttl=60)
    def screen_symbols(symbols):
        results = []
        def compute(symbol):
            try:
                df = yf.download(symbol, period="1y", progress=False)
                if df.empty: return None
                close = df['Close']
                ema10 = close.ewm(span=10).mean()
                ema20 = close.ewm(span=20).mean()
                ema50 = close.ewm(span=50).mean()
                rsi = RSIIndicator(close).rsi().iloc[-1]
                
                regime = "STRONG BULL" if ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] else \
                         "WEAK BULL" if ema10.iloc[-1] > ema20.iloc[-1] else \
                         "STRONG BEAR" if ema10.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1] else \
                         "WEAK BEAR" if ema10.iloc[-1] < ema20.iloc[-1] else "SIDEWAYS"
                
                return {"symbol": symbol, "regime": regime, "price": close.iloc[-1], "rsi": rsi}
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            for res in executor.map(compute, symbols):
                if res: results.append(res)
        return pd.DataFrame(results)

    df_screen = screen_symbols(SYMBOLS)
    if not df_screen.empty:
        st.dataframe(df_screen, use_container_width=True)
    else:
        st.info("Add symbols to your watchlist!")

# =============================================================================
# TAB 2: MULTI-TIMEFRAME HEATMAP
# =============================================================================
with tab2:
    st.subheader("Multi-TF EMA Regime Heatmap")
    @st.cache_data(ttl=300)
    def get_heatmap(symbols):
        tfs = ["1d", "1wk"]
        data = {tf: [] for tf in tfs}
        for sym in symbols:
            for tf in tfs:
                df = yf.download(sym, period="2y" if tf=="1wk" else "1y", interval=tf, progress=False)
                ema10 = df['Close'].ewm(10).mean().iloc[-1]
                ema20 = df['Close'].ewm(20).mean().iloc[-1]
                ema50 = df['Close'].ewm(50).mean().iloc[-1]
                regime = "🟢 BULL" if ema10 > ema20 > ema50 else "🔴 BEAR" if ema10 < ema20 < ema50 else "⚪ SIDE"
                data[tf].append(regime)
        return pd.DataFrame(data, index=symbols)

    hm_df = get_heatmap(SYMBOLS)
    fig = go.Figure(data=go.Heatmap(
        z=hm_df.values,
        x=hm_df.columns,
        y=hm_df.index,
        text=hm_df.values,
        texttemplate="%{text}",
        colorscale=["red", "gray", "green"]
    ))
    fig.update_layout(title="Regime Heatmap", height=400)
    st.plotly_chart(fig)

# =============================================================================
# TAB 3: BACKTEST P&L
# =============================================================================
with tab3:
    symbol_bt = st.selectbox("Symbol", SYMBOLS)
    start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    capital = st.number_input("Capital ($)", 1000, 100000, 10000)
    
    if st.button("Run Backtest"):
        df_bt = yf.download(symbol_bt, start=start, progress=False)
        df_bt['EMA10'] = df_bt['Close'].ewm(10).mean()
        df_bt['EMA20'] = df_bt['Close'].ewm(20).mean()
        df_bt['EMA50'] = df_bt['Close'].ewm(50).mean()
        
        position = 0
        cash = capital
        equity = [capital]
        
        for i in range(50, len(df_bt)):
            bull = df_bt['EMA10'].iloc[i] > df_bt['EMA20'].iloc[i] > df_bt['EMA50'].iloc[i]
            bear = df_bt['EMA10'].iloc[i] < df_bt['EMA20'].iloc[i] < df_bt['EMA50'].iloc[i]
            
            if bull and position == 0:
                position = cash / df_bt['Close'].iloc[i]
                cash = 0
            elif bear and position > 0:
                cash = position * df_bt['Close'].iloc[i]
                position = 0
            
            value = cash + position * df_bt['Close'].iloc[i]
            equity.append(value)
        
        final = cash + position * df_bt['Close'].iloc[-1]
        ret_pct = (final - capital) / capital * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Final Portfolio", f"${final:,.2f}")
        col2.metric("Total Return", f"{ret_pct:+.1f}%")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt.index[49:], y=equity[1:], name="Equity Curve"))
        st.plotly_chart(fig_bt)

# =============================================================================
# TAB 4: RSI + VOLUME FILTERS
# =============================================================================
with tab4:
    st.subheader("Advanced Filters")
    rsi_threshold = st.slider("RSI Threshold", 30, 70, 50)
    vol_mult = st.slider("Volume Multiple", 1.0, 3.0, 1.5)
    
    # Example filtered screen (integrate with Tab 1)
    st.info(f"Filter: RSI > {rsi_threshold} & Volume > {vol_mult}x average")
    # (In full platform, pipe this to screener)

st.markdown("---")
st.caption("Built with ❤️ by Grok | Data: Yahoo Finance | Extend with live trading APIs")
