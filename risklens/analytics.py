# risklens/analytics.py
import pandas as pd
import numpy as np

def construct_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float]
) -> pd.Series:
    """
    Compute portfolio return series based on weights.
    Weights are normalized to sum to 1.0.
    """
    available_assets = asset_returns.columns.intersection(weights.keys())
    if available_assets.empty:
        return pd.Series(0.0, index=asset_returns.index)
        
    # Extract relevant weights and normalize
    w_vec = np.array([weights[a] for a in available_assets])
    total_w = w_vec.sum()
    if total_w == 0:
        return pd.Series(0.0, index=asset_returns.index)
    
    w_vec = w_vec / total_w
    
    # Dot product: returns matrix @ weights
    port_returns = asset_returns[available_assets].dot(w_vec)
    return port_returns

def returns_to_equity(
    returns: pd.Series,
    initial_capital: float = 100_000.0
) -> pd.Series:
    """Convert returns to equity curve."""
    if returns.empty:
        return pd.Series([initial_capital])
        
    cumulative = (1 + returns).cumprod()
    equity = initial_capital * cumulative
    
    # Try to infer start date to prepend initial capital
    # If index is Datetime, subtract 1 period
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        # Infer freq
        delta = returns.index[1] - returns.index[0]
        start_date = returns.index[0] - delta
    elif isinstance(returns.index, pd.DatetimeIndex):
        start_date = returns.index[0] - pd.Timedelta(days=1)
    else:
        # RangeIndex or other
        start_date = returns.index[0] - 1 if isinstance(returns.index, pd.RangeIndex) else 0

    start_series = pd.Series([initial_capital], index=[start_date])
    return pd.concat([start_series, equity])


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) - 1

def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    tot_ret = total_return(equity)
    n_periods = len(equity)
    if n_periods < 2:
        return 0.0
    
    # Geometric mean
    years = n_periods / periods_per_year
    if years == 0: return 0.0
    
    return (1 + tot_ret) ** (1 / years) - 1

def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    # Adjust Rf for the period
    rf_daily = risk_free_rate / periods_per_year
    excess_ret = returns - rf_daily
    
    mean_excess = excess_ret.mean()
    std_excess = excess_ret.std()
    
    if std_excess == 0:
        return 0.0
        
    return (mean_excess / std_excess) * np.sqrt(periods_per_year)

def max_drawdown(equity: pd.Series) -> float:
    """Calculate Max Drawdown (as a positive number representing the % loss)."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min() # This is negative, e.g. -0.20 for 20% DD

def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max

def rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252
) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods_per_year)

def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of returns.
    """
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()

def rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.Series:
    rf_daily = risk_free_rate / periods_per_year
    excess = returns - rf_daily
    
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    
    return (roll_mean / roll_std) * np.sqrt(periods_per_year)
