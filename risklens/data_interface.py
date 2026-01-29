# risklens/data_interface.py
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from risklens.config import USE_MOCKS
from risklens.mocks import MockQuantStream, MockPyTradeLab

def load_asset_prices(
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = "1D"
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols.
    Returns: Dict[symbol, DataFrame]
    """
    data = {}
    
    # Calculate days from start to now for fetch_ohlcv
    start_dt = pd.to_datetime(start)
    days_needed = (datetime.now() - start_dt).days + 5 # Buffer
    
    # Normalize timeframe
    tf = timeframe.lower()
    
    for sym in symbols:
        if USE_MOCKS:
            df = MockQuantStream.load_ohlcv(sym, start, end, timeframe)
        else:
            df = pd.DataFrame()
            # 1. Try QuantStream (Binance/Crypto)
            try:
                from market_pipeline.ingest import fetch_ohlcv
                # QuantStream expects specific format often, but let's try
                # It logs errors internally usually, we catch exceptions here
                qs_df = fetch_ohlcv(sym, interval=tf, days=days_needed)
                
                if not qs_df.empty:
                    if 'timestamp' in qs_df.columns:
                        qs_df = qs_df.set_index('timestamp')
                    qs_df.index = pd.to_datetime(qs_df.index)
                    df = qs_df
            except Exception:
                pass
            
            # 2. Fallback to PyTradeLab (yfinance) if empty
            if df.empty:
                try:
                    from backtester.data_loader import DataLoader
                    loader = DataLoader() # no specific dir needed for fetch
                    # yfinance is good for stocks (SPY, AAPL) and crypto (BTC-USD)
                    # Maps "BTCUSDT" -> might fail in yfinance, expects "BTC-USD" often
                    # But if user types "SPY", it works.
                    yf_sym = sym
                    if sym.endswith("USDT"): # naive conversion for yf fallback
                        yf_sym = sym.replace("USDT", "-USD")
                    
                    yf_df = loader.fetch(yf_sym, source="yfinance", start=start, end=end)
                    if not yf_df.empty:
                        if 'timestamp' in yf_df.columns:
                            yf_df = yf_df.set_index('timestamp')
                        yf_df.index = pd.to_datetime(yf_df.index)
                        df = yf_df
                except Exception as e:
                    print(f"Error loading {sym} from yfinance: {e}")
            
            data[sym] = df
        
    return data

def compute_asset_returns(
    prices: Dict[str, pd.DataFrame],
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Combine individual asset DataFrames into a single wide DataFrame of returns.
    Aligns by index (inner join).
    """
    if not prices:
        return pd.DataFrame()
    
    # Extract series and align
    series_list = {}
    for sym, df in prices.items():
        if df.empty: continue
        if price_col in df.columns:
            # Drop duplicates just in case
            s = df[price_col]
            s = s[~s.index.duplicated(keep='first')]
            series_list[sym] = s
            
    if not series_list:
        return pd.DataFrame()
        
    prices_df = pd.DataFrame(series_list)
    prices_df.dropna(inplace=True) 
    
    # Calculate simple returns
    returns_df = prices_df.pct_change().fillna(0.0)
    
    return returns_df

def load_strategy_equity_curves(
    strategy_names: List[str],
    start: str,
    end: str
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict]]:
    """
    Load equity curves and performance statistics.
    Returns: (curves_dict, stats_dict)
    """
    curves = {}
    stats = {}
    
    for name in strategy_names:
        if USE_MOCKS:
            curves[name] = MockPyTradeLab.run_strategy_mock(name, start, end)
            stats[name] = {"Win Rate": 0.55, "Profit Factor": 1.5, "Trades": 42} # Mock stats
        else:
            try:
                # Parse: "ma_cross_BTCUSDT"
                parts = name.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                strat_key, symbol = parts
                
                # 1. Fetch Data
                # Try fetch from shared loader logic if possible, but keep it simple here
                from market_pipeline.ingest import fetch_ohlcv
                start_dt = pd.to_datetime(start)
                days = (datetime.now() - start_dt).days + 5
                
                df_raw = fetch_ohlcv(symbol, interval="1d", days=days)
                if df_raw.empty:
                    # Try fallback to mocked/yfinance if user selected stock?
                    # For now, strict on fetch_ohlcv or add simple fallback
                    pass
                    
                # 2. Setup PyTradeLab Loader
                from backtester.data_loader import DataLoader
                from backtester.engine import BacktestEngine
                from backtester.strategy import get_strategy_class
                
                loader = DataLoader() 
                if not df_raw.empty:
                    loader.load_dataframe(symbol, df_raw)
                else:
                     # Attempt fallback fetch immediately into loader if supported
                     pass

                # 3. Init Strategy
                StratCls = get_strategy_class(strat_key)
                if not StratCls:
                    print(f"Strategy class {strat_key} not found")
                    continue
                    
                strategy = StratCls(symbol)
                
                # 4. Run Engine
                engine = BacktestEngine(loader, initial_capital=100_000.0)
                result = engine.run(strategy, symbol, start=start, end=end)
                
                if result.equity_curve is not None and not result.equity_curve.empty:
                    curves[name] = result.equity_curve['total_equity']
                    stats[name] = result.metrics if result.metrics else {}
                    
            except Exception as e:
                print(f"Error running strategy {name}: {e}")
                
    return curves, stats
