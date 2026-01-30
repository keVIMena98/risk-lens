# app.py
import streamlit as st
import pandas as pd
from datetime import datetime

import risklens.config as config
from risklens.data_interface import (
    get_available_symbols,
    ingest_symbol,
    load_asset_prices,
    load_returns_matrix,
    run_strategies_to_equity_curves
)
from risklens.analytics import (
    construct_portfolio_returns,
    returns_to_equity,
    summarize_series,
    summarize_assets_and_portfolio,
    rolling_volatility,
    rolling_sharpe
)
from risklens.plots import (
    plot_equity_curves,
    plot_drawdown,
    plot_rolling_metric,
    plot_risk_return_scatter
)

# Use real backtester import if available
try:
    from backtester.strategy import BaseStrategy
except ImportError:
    # If using local running where imports might fail or mocking
    class BaseStrategy: pass

# --- Dummy Strategy Classes for Demo ---
class MomentumStrategy(BaseStrategy):
    pass
class MeanReversionStrategy(BaseStrategy):
    pass

# --- Page Config ---
st.set_page_config(page_title="RiskLens", layout="wide")

# --- 7.3.1 Title ---
st.title("RiskLens – Risk & Portfolio Analytics Dashboard")
st.markdown("""
This dashboard integrates **QuantStream** (Market Data), **PyTradeLab** (Backtesting), and **FactorLab** (Risk Metrics) 
to provide a unified view of portfolio risk and performance.
""")

# --- 7.2 Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. Data Selection
st.sidebar.subheader("Data Selection")
available_assets = get_available_symbols()
if not available_assets:
    # Fallback to config defaults if no data files found
    available_assets = config.DEFAULT_SYMBOLS
default_selection = available_assets[:3] if len(available_assets) >= 3 else available_assets
selected_assets = st.sidebar.multiselect(
    "Assets",
    options=available_assets,
    default=default_selection
)

# Add New Asset
st.sidebar.subheader("Add New Asset")
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Binance (Crypto)", "Yahoo Finance (Stocks)", "Alpha Vantage (Stocks)"],
    key="data_source"
)
new_symbol = st.sidebar.text_input(
    "Symbol", 
    placeholder="AAPL, NVDA, btcusdt...",
    key="new_symbol"
)
ingest_days = st.sidebar.number_input("Days of history", value=30, min_value=1, max_value=365, key="ingest_days")

# Show API key input for Alpha Vantage
av_api_key = ""
if "Alpha Vantage" in data_source:
    av_api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password", key="av_api_key")

if st.sidebar.button("Ingest Asset"):
    if new_symbol:
        # Map UI selection to source code
        source_map = {
            "Binance (Crypto)": "binance",
            "Yahoo Finance (Stocks)": "yfinance",
            "Alpha Vantage (Stocks)": "alphavantage"
        }
        source = source_map[data_source]
        
        with st.spinner(f"Ingesting {new_symbol} from {data_source}..."):
            success, msg = ingest_symbol(new_symbol, ingest_days, source=source, api_key=av_api_key)
        if success:
            st.sidebar.success(msg)
            st.cache_data.clear()  # Clear cache to refresh asset list
            st.rerun()
        else:
            st.sidebar.error(msg)
    else:
        st.sidebar.warning("Please enter a symbol")


col_d1, col_d2 = st.sidebar.columns(2)
start_date_input = col_d1.date_input("Start Date", pd.to_datetime(config.DEFAULT_START_DATE))
end_date_input = col_d2.date_input("End Date", datetime.today())
# handle empty end date logic if needed, but date_input always returns object
end_date = end_date_input.strftime("%Y-%m-%d") if end_date_input else ""
s_date = start_date_input.strftime("%Y-%m-%d")

timeframe = st.sidebar.selectbox("Timeframe", [config.DEFAULT_TIMEFRAME])

# 2. Portfolio Weights
st.sidebar.subheader("Portfolio Weights")
raw_weights = {}
for asset in selected_assets:
    raw_weights[asset] = st.sidebar.number_input(f"Weight: {asset}", value=1.0/len(selected_assets) if selected_assets else 0.0, step=0.05)

if st.sidebar.button("Normalize Weights"):
    total = sum(raw_weights.values())
    if total > 0:
        st.session_state['norm_weights_trigger'] = True
    else:
        st.sidebar.error("Total weight is 0")
        
norm_weights = raw_weights
if 'norm_weights_trigger' in st.session_state and st.session_state['norm_weights_trigger']:
    # Simple display below
    if sum(raw_weights.values()) > 0:
        total = sum(raw_weights.values())
        norm_weights = {k: v/total for k, v in raw_weights.items()}
        st.sidebar.markdown("**Normalized Weights:**")
        st.sidebar.json(norm_weights)
    else:
        pass

# 3. Benchmark
use_benchmark = st.sidebar.checkbox("Use equal-weight portfolio as benchmark")

# 4. Risk Settings
st.sidebar.subheader("Risk Settings")
periods_per_year = st.sidebar.number_input("Periods per year", value=config.DEFAULT_PERIODS_PER_YEAR)
rolling_window = st.sidebar.number_input("Rolling window (days)", value=config.DEFAULT_ROLLING_WINDOW)
initial_capital = st.sidebar.number_input("Initial capital", value=config.DEFAULT_INITIAL_CAPITAL)

# 5. Strategies
include_strategies = st.sidebar.checkbox("Include strategy equity curves from PyTradeLab")
selected_strategies_labels = []
if include_strategies:
    selected_strategies_labels = st.sidebar.multiselect(
        "Select Strategies",
        ["Momentum_20_BTC", "Momentum_60_ETH", "MeanReversion_5_SOL"],
        default=["Momentum_20_BTC"]
    )

# --- Main Page Logic ---

if not selected_assets:
    st.info("Please select at least one asset.")
    st.stop()

# 2. Data Loading
@st.cache_data
def get_market_data(assets, start, end, tf):
    return load_returns_matrix(assets, start, end, tf)
    
@st.cache_data
def get_strategy_curves(specs_tuple, start, end, tf, init_cap, fee):
    specs_list = []
    required_syms = set()
    for (name, s_type, window, sym) in specs_tuple:
        cls = MomentumStrategy if s_type == 'Momentum' else MeanReversionStrategy
        specs_list.append((name, cls, {'window': window}, sym))
        required_syms.add(sym)
        
    p_map = load_asset_prices(list(required_syms), start, end, tf)
    return run_strategies_to_equity_curves(specs_list, p_map, init_cap, fee)

with st.spinner("Loading Data..."):
    # Load Asset Returns (Wide DF)
    returns_df = get_market_data(selected_assets, s_date, end_date, timeframe)
    
    if returns_df.empty:
        st.error(f"No data found for selected assets in range {s_date} to {end_date}. Try extending range or verifying market_pipeline.")
        st.stop()
        
    combined_returns = returns_df.copy()
    strat_curves_map = {}
    
    # Run strategies if requested
    if include_strategies and selected_strategies_labels:
        # Build Specs for loading
        prep_specs = []
        for label in selected_strategies_labels:
            if "Momentum_20_BTC" == label:
                prep_specs.append((label, "Momentum", 20, "BTCUSD"))
            elif "Momentum_60_ETH" == label:
                prep_specs.append((label, "Momentum", 60, "ETHUSD"))
            elif "MeanReversion_5_SOL" == label:
                prep_specs.append((label, "MeanReversion", 5, "SOLUSD"))
        
        specs_t = tuple(prep_specs)
        strat_curves_map = get_strategy_curves(specs_t, s_date, end_date, timeframe, initial_capital, 10.0)
        
        for s_name, s_curve in strat_curves_map.items():
            if not s_curve.empty:
                 s_ret = s_curve.pct_change().fillna(0.0)
                 # Reindex using logic
                 s_ret = s_ret.reindex(returns_df.index).fillna(0.0)
                 combined_returns[s_name] = s_ret

# 3. Portfolio Construction
# Use raw weights because user might not have clicked normalize.
# We normalize them implicitly during calculation in analytics if we pass them.
# Our construct_portfolio_returns does normalize them.
try:
    port_returns = construct_portfolio_returns(combined_returns, raw_weights)
except ValueError as e:
    st.error(f"Portfolio Construction Error: {e}")
    st.stop()
except Exception as e:
    # Just in case key error if asset has no data
    st.error(f"Data Mismatch: {e}")
    st.stop()

port_equity = returns_to_equity(port_returns, initial_capital)

benchmark_equity = pd.Series(dtype=float)
bench_ret = pd.Series(dtype=float)
if use_benchmark:
    # Equal weight of selected assets
    eq_w = {k: 1.0 for k in selected_assets} 
    try:
        bench_ret = construct_portfolio_returns(returns_df, eq_w) # Use asset-only DF
        benchmark_equity = returns_to_equity(bench_ret, initial_capital)
    except:
        pass

# 4. Key Metrics Summary
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
try:
    p_stats = summarize_series(port_equity, port_returns, periods_per_year)
    col1.metric("Cumulative Return", f"{p_stats['total_return']:.2%}")
    col2.metric("Annualized Return", f"{p_stats['annualized_return']:.2%}")
    col3.metric("Volatility", f"{p_stats['volatility']:.2%}")
    col4.metric("Sharpe Ratio", f"{p_stats['sharpe_ratio']:.2f}")
    col5.metric("Max Drawdown", f"{p_stats['max_drawdown']:.2%}")
except Exception as e:
    st.error(f"Error calculating metrics: {e}")

if use_benchmark and not benchmark_equity.empty:
    b_stats = summarize_series(benchmark_equity, bench_ret, periods_per_year)
    col1.metric("Benchmark Total", f"{b_stats['total_return']:.2%}")
    col2.metric("Benchmark Ann.", f"{b_stats['annualized_return']:.2%}")
    col3.metric("Benchmark Vol", f"{b_stats['volatility']:.2%}")
    col4.metric("Benchmark Sharpe", f"{b_stats['sharpe_ratio']:.2f}")
    col5.metric("Benchmark DD", f"{b_stats['max_drawdown']:.2%}")

# 5. Charts
st.subheader("Charts")

# Equity
curves_dict = {"Portfolio": port_equity}
if not benchmark_equity.empty:
    curves_dict["Benchmark"] = benchmark_equity

fig_eq = plot_equity_curves(curves_dict, "Equity Curves")
st.plotly_chart(fig_eq)

# Drawdown
fig_dd = plot_drawdown(port_equity, "Portfolio Drawdown")
st.plotly_chart(fig_dd)

# Rolling
r_vol = rolling_volatility(port_returns, rolling_window)
r_sharpe = rolling_sharpe(port_returns, rolling_window) # assuming 0RF

st.plotly_chart(plot_rolling_metric(r_vol, f"Rolling Volatility ({rolling_window}d)"))
st.plotly_chart(plot_rolling_metric(r_sharpe, f"Rolling Sharpe ({rolling_window}d)"))

# 6. Risk-Return View
st.subheader("Risk-Return Analysis")
summary_df = summarize_assets_and_portfolio(
    combined_returns, 
    port_returns, 
    port_equity, 
    periods_per_year
)
st.dataframe(summary_df)

fig_sc = plot_risk_return_scatter(summary_df, "Risk-Return Profile")
st.plotly_chart(fig_sc)
