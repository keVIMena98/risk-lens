# tests/test_analytics.py
import pytest
import pandas as pd
import numpy as np
from risklens.analytics import (
    construct_portfolio_returns,
    returns_to_equity,
    total_return,
    annualized_return,
    volatility,
    sharpe_ratio,
    max_drawdown
)

@pytest.fixture
def simple_returns():
    # 5 days of 1% return
    return pd.Series([0.01] * 5)

@pytest.fixture
def mixed_returns():
    # -10%, +10%, -10%, +10%
    return pd.Series([-0.1, 0.1, -0.1, 0.1])

def test_construct_portfolio_returns():
    # 2 assets, equal weights
    df = pd.DataFrame({
        "A": [0.01, 0.02],
        "B": [0.03, 0.04]
    })
    weights = {"A": 0.5, "B": 0.5}
    port = construct_portfolio_returns(df, weights)
    
    expected = pd.Series([0.02, 0.03]) # (1+3)/2, (2+4)/2
    pd.testing.assert_series_equal(port, expected)

def test_returns_to_equity(simple_returns):
    eq = returns_to_equity(simple_returns, 100)
    # 100 * 1.01^5
    final = eq.iloc[-1]
    assert final == pytest.approx(100 * (1.01**5))

def test_total_return(simple_returns):
    eq = returns_to_equity(simple_returns, 100)
    tot = total_return(eq)
    assert tot == pytest.approx((1.01**5) - 1)

def test_max_drawdown():
    # 100 -> 50 (50% loss) -> 75 -> 40 (46.6% loss from 75, but max is from peak 100)
    # wait, global peak is 100. low is 40. DD is (40-100)/100 = -0.6
    
    eq = pd.Series([100, 50, 75, 40])
    dd = max_drawdown(eq)
    assert dd == pytest.approx(-0.6)

def test_sharpe_ratio():
    # Flat return of 1% daily, 0 volatility -> Sharpe should be infinite in theory 
    # but our code divides by std.
    rets = pd.Series([0.01, 0.01, 0.01])
    # std is 0
    assert sharpe_ratio(rets) == 0.0 # Code handles div by zero returning 0
    
    # Simple case
    rets2 = pd.Series([0.01, -0.01]) # mean 0
    # Sharpe should be 0 since mean excess is 0 (assuming risk free 0)
    assert sharpe_ratio(rets2, risk_free_rate=0) == 0.0
