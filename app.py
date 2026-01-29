import streamlit as st
import pandas as pd
from datetime import datetime
import risklens.config as cfg
import risklens.data_interface as di
import risklens.analytics as an
import risklens.plots as pl

# Page Config
st.set_page_config(layout="wide", page_title=cfg.APP_TITLE)

st.title(cfg.APP_TITLE)
st.markdown(f"**{cfg.APP_DESCRIPTION}**")

if cfg.USE_MOCKS:
    st.warning("⚠️ QuantStream/PyTradeLab not found. Using MOCK data for demonstration.")

# -----------------
# 1. SIDEBAR
# -----------------
st.sidebar.header("Configuration")

# Data Selection
available_symbols = cfg.DEFAULT_SYMBOLS.copy()

# Allow user to add custom symbols
custom_symbol = st.sidebar.text_input("Add Custom Symbol (e.g. SPY, AAPL)")
if custom_symbol:
    custom_symbol = custom_symbol.upper().strip()
    if custom_symbol not in available_symbols:
        available_symbols.append(custom_symbol)

selected_symbols = st.sidebar.multiselect("Select Assets", available_symbols, default=available_symbols[:3])

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(cfg.DEFAULT_START_DATE))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Strategy Selection (Optional)
use_strategies = st.sidebar.checkbox("Include Strategies (PyTradeLab)")
selected_strategies = []
if use_strategies:
    # In a real app, we'd fetch these dynamically
    avail_strats = ["ma_cross_BTCUSDT", "breakout_ETHUSDT", "ma_cross_SOLUSDT"]
    selected_strategies = st.sidebar.multiselect("Select Strategies", avail_strats, default=avail_strats[:1])

all_keys = selected_symbols + selected_strategies

if not all_keys:
    st.info("Please select at least one asset or strategy.")
    st.stop()

# Load Data
@st.cache_data
def get_data(syms, strats, s, e):
    # Assets
    prices = di.load_asset_prices(syms, str(s), str(e))
    returns_df = di.compute_asset_returns(prices)
    
    # Strategies
    strat_curves, strat_stats = di.load_strategy_equity_curves(strats, str(s), str(e))
    
    # Merge strategy returns
    if strat_curves:
        strat_returns = pd.DataFrame({name: c.pct_change().fillna(0) for name, c in strat_curves.items()})
        if not returns_df.empty:
            returns_df = returns_df.join(strat_returns, how='inner')
        else:
            returns_df = strat_returns
            
    return returns_df, strat_stats

try:
    returns_df, strat_stats = get_data(selected_symbols, selected_strategies, start_date, end_date)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if returns_df.empty:
    st.warning("No data found for selected range.")
    st.stop()

# Weights
st.sidebar.subheader("Portfolio Weights")
raw_weights = {}
for asset in all_keys:
    # Default equal weight just for UI cleanliness
    default_w = 1.0 / len(all_keys)
    raw_weights[asset] = st.sidebar.slider(f"Weight: {asset}", 0.0, 1.0, default_w, 0.05)

# Benchmark
use_benchmark = st.sidebar.checkbox("Compare to Benchmark")
bench_symbol = None
if use_benchmark:
    bench_symbol = st.sidebar.selectbox("Benchmark Asset", selected_symbols)

# Risk Settings
with st.sidebar.expander("Risk Settings"):
    rf_rate = st.number_input("Risk Free Rate", value=cfg.DEFAULT_RISK_FREE_RATE)
    periods = st.number_input("Periods/Year", value=cfg.DEFAULT_PERIODS_PER_YEAR)
    window = st.number_input("Rolling Window", value=cfg.DEFAULT_ROLLING_WINDOW)

# -----------------
# 2. LOGIC
# -----------------
# Construct Portfolio
port_returns = an.construct_portfolio_returns(returns_df, raw_weights)
port_equity = an.returns_to_equity(port_returns, cfg.DEFAULT_INITIAL_CAPITAL)

# Metrics
metrics = {
    "Total Return": an.total_return(port_equity),
    "Ann. Return": an.annualized_return(port_equity, periods),
    "Volatility": an.volatility(port_returns, periods),
    "Sharpe": an.sharpe_ratio(port_returns, rf_rate, periods),
    "Max DD": an.max_drawdown(port_equity)
}

# -----------------
# 3. VISUALIZATION
# -----------------

# Top Row Metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return", f"{metrics['Total Return']:.2%}")
c2.metric("Ann. Return", f"{metrics['Ann. Return']:.2%}")
c3.metric("Volatility", f"{metrics['Volatility']:.2%}")
c4.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
c5.metric("Max Drawdown", f"{metrics['Max DD']:.2%}")

# Equity Chart
st.subheader("Performance")
curves_to_plot = {"Portfolio": port_equity}

if use_benchmark and bench_symbol and bench_symbol in returns_df.columns:
    bench_ret = returns_df[bench_symbol]
    bench_eq = an.returns_to_equity(bench_ret, cfg.DEFAULT_INITIAL_CAPITAL)
    curves_to_plot[f"Benchmark ({bench_symbol})"] = bench_eq

fig_eq = pl.plot_equity_curves(curves_to_plot)
st.plotly_chart(fig_eq, use_container_width=True)

# Drawdown
st.subheader("Drawdown Profile")
dd_series = an.drawdown_series(port_equity)
fig_dd = pl.plot_drawdown(dd_series)
st.plotly_chart(fig_dd, use_container_width=True)

# Rolling Metrics
c_roll1, c_roll2 = st.columns(2)
with c_roll1:
    roll_vol = an.rolling_volatility(port_returns, window, periods)
    st.plotly_chart(pl.plot_rolling_metric(roll_vol, "Volatility"), use_container_width=True)
with c_roll2:
    roll_sharpe = an.rolling_sharpe(port_returns, window, rf_rate, periods)
    st.plotly_chart(pl.plot_rolling_metric(roll_sharpe, "Sharpe"), use_container_width=True)

# Risk/Return Scatter
st.subheader("Risk vs Return Landscape")
# Calculate metrics for individual assets for comparison
asset_metrics = []
# Add Portfolio
asset_metrics.append({
    "Name": "Portfolio", 
    "Return": metrics["Ann. Return"], 
    "Volatility": metrics["Volatility"], 
    "Sharpe": metrics["Sharpe"]
})

for asset in returns_df.columns:
    s_ret = returns_df[asset]
    s_eq = an.returns_to_equity(s_ret)
    asset_metrics.append({
        "Name": asset,
        "Return": an.annualized_return(s_eq, periods),
        "Volatility": an.volatility(s_ret, periods),
        "Sharpe": an.sharpe_ratio(s_ret, rf_rate, periods)
    })
    
df_scatter = pd.DataFrame(asset_metrics)
fig_scat = pl.plot_risk_return_scatter(df_scatter)
st.plotly_chart(fig_scat, use_container_width=True)

# -----------------
# 4. ADVANCED ANALYTICS
# -----------------
st.header("Advanced Analytics")
tab1, tab2 = st.tabs(["Correlation", "Strategy Details"])

with tab1:
    st.subheader("Asset Correlation Matrix")
    if not returns_df.empty:
        corr_matrix = an.compute_correlation_matrix(returns_df)
        fig_corr = pl.plot_correlation_matrix(corr_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data for correlation.")

with tab2:
    st.subheader("Strategy Performance Metrics")
    if strat_stats:
        # Create a DataFrame for nicer display
        # stats is Dict[str, Dict[str, float]]
        stats_df = pd.DataFrame(strat_stats).T
        st.dataframe(stats_df.style.format("{:.2f}"))
    else:
        st.info("No active strategies selected.")
