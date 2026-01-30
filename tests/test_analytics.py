# tests/test_analytics.py
import pytest
import pandas as pd
import numpy as np
from risklens.analytics import (
    construct_portfolio_returns,
    returns_to_equity,
    summarize_series,
    summarize_assets_and_portfolio
)

def test_construct_portfolio_returns():
    # Setup
    dates = pd.date_range("2023-01-01", periods=5)
    data = {
        "A": [0.01, 0.01, 0.01, 0.01, 0.01], # Constant 1%
        "B": [0.02, 0.02, 0.02, 0.02, 0.02]  # Constant 2%
    }
    df = pd.DataFrame(data, index=dates)
    
    # 50/50 weights
    weights = {"A": 0.5, "B": 0.5}
    
    port_ret = construct_portfolio_returns(df, weights)
    
    # Expected: 1.5% daily
    assert np.allclose(port_ret, 0.015)
    
    # Check normalization (weights 1, 1 -> 0.5, 0.5)
    weights2 = {"A": 1.0, "B": 1.0}
    port_ret2 = construct_portfolio_returns(df, weights2)
    assert np.allclose(port_ret2, 0.015)

def test_returns_to_equity():
    returns = pd.Series([0.1, 0.1], index=pd.date_range("2023-01-01", periods=2))
    equity = returns_to_equity(returns, 100)
    
    # T0 (prepended) = 100
    # T1 = 110
    # T2 = 121
    # Check last value
    assert np.isclose(equity.iloc[-1], 121.0)
    # Check length (original 2 + 1 initial) = 3
    # returns_to_equity logic adds one initial point
    assert len(equity) == 3 

def test_summarize_series():
    equity = pd.Series([100, 110, 121], index=pd.date_range("2023-01-01", periods=3))
    returns = pd.Series([0.1, 0.1], index=pd.date_range("2023-01-02", periods=2))
    
    summary = summarize_series(equity, returns, periods_per_year=252)
    
    assert "total_return" in summary
    assert np.isclose(summary["total_return"], 0.21)

def test_summarize_assets_and_portfolio():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "A": [0.01]*5, 
        "B": [0.02]*5
    }, index=dates)
    
    weights = {"A": 0.5, "B": 0.5}
    p_ret = construct_portfolio_returns(df, weights)
    p_eq = returns_to_equity(p_ret, 1000)
    
    # We pass p_eq, not recalculate inside necessarily, but function logic 
    # uses returns_to_equity for assets, takes Portfolio equity as arg.
    summary_df = summarize_assets_and_portfolio(df, p_ret, p_eq, 252)
    
    # Rows: Portfolio + 2 Assets = 3
    assert len(summary_df) == 3
    assert "name" in summary_df.columns
    assert "PORTFOLIO" in summary_df["name"].values
