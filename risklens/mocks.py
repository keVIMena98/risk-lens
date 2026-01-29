# risklens/mocks.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_geometric_brownian_motion(
    symbol: str,
    start: str,
    end: str,
    initial_price: float = 100.0,
    mu: float = 0.1,    # Drift
    sigma: float = 0.4, # Volatility
    freq: str = "D"
) -> pd.DataFrame:
    """Simulate realistic price paths (Geometric Brownian Motion)."""
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Generate date range
    dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    T = len(dates)
    dt = 1/365 if freq == 'D' else 1/252 # Approximation
    
    # Stochastic component
    np.random.seed(sum(ord(c) for c in symbol)) # Deterministic seed per symbol
    dW = np.random.normal(0, np.sqrt(dt), T)
    
    # Drift and Diffusion
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW
    
    # Log returns
    log_returns = drift + diffusion
    log_returns[0] = 0 # No change on day 0
    
    # Prices
    price_path = initial_price * np.exp(np.cumsum(log_returns))
    
    # Build OHLCV (Synthetic)
    # High/Low/Open derived somewhat arbitrarily around close for realism
    df = pd.DataFrame(index=dates)
    df['close'] = price_path
    noise = np.random.normal(0, 0.005, T)
    df['open'] = df['close'].shift(1).fillna(initial_price) * (1 + noise)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, T)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, T)))
    df['volume'] = np.random.randint(1000, 1000000, T)
    
    return df

class MockQuantStream:
    @staticmethod
    def load_ohlcv(symbol: str, start: str, end: str, timeframe: str = "1D"):
        # Custom params per fake symbol for variety
        mu = 0.05
        sigma = 0.3
        price = 100.0
        
        if "BTC" in symbol:
            sigma = 0.6; mu = 0.2; price = 30000
        elif "ETH" in symbol:
            sigma = 0.7; mu = 0.25; price = 2000
        elif "SPY" in symbol:
            sigma = 0.15; mu = 0.08; price = 400
        
        return generate_geometric_brownian_motion(symbol, start, end, price, mu, sigma)

class MockPyTradeLab:
    @staticmethod
    def list_strategies():
        return ["TrendFollowing_BTC", "MeanReversion_ETH", "Momentum_CryptoBasket"]
        
    @staticmethod
    def run_strategy_mock(name: str, start: str, end: str):
        # Generate an equity curve that looks like a strategy (smoother, simulated alpha)
        # Using the GBM generator but treating "Close" as "Equity"
        df = generate_geometric_brownian_motion(
            name, start, end, initial_price=100000, mu=0.15, sigma=0.25
        )
        return df['close']
