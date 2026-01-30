# risklens/data_interface.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime

# --- Imports from Local Repositories ---
# Mapped based on inspection of directories:
# quantstream -> market_pipeline (ingest.fetch_ohlcv)
# pytradelab -> backtester

try:
    from market_pipeline.transform import load_raw_files
except ImportError:
    load_raw_files = None

try:
    from backtester.engine import BacktestEngine
    # Strategies might be in backtester.strategy or similar
    from backtester.strategy import BaseStrategy 
except ImportError:
    BacktestEngine = None
    BaseStrategy = None


def get_available_symbols() -> List[str]:
    """
    Scan the market_pipeline RAW_DIR and return unique symbols that have data files.
    
    Returns:
        Sorted list of symbol names (uppercase, e.g., ['BTCUSDT', 'ETHUSDT'])
    """
    try:
        from market_pipeline.config import RAW_DIR
    except ImportError:
        return []
    
    symbols = set()
    for filepath in RAW_DIR.glob("*_*.csv"):
        # Extract symbol from filename like "btcusdt_20260129.csv"
        name = filepath.stem  # "btcusdt_20260129"
        symbol = name.rsplit("_", 1)[0].upper()  # "BTCUSDT"
        symbols.add(symbol)
    return sorted(symbols)


def ingest_from_binance(symbol: str, days: int = 30) -> tuple[bool, str]:
    """
    Ingest a crypto symbol from Binance via market_pipeline.
    
    Args:
        symbol: Trading pair symbol (e.g., 'btcusdt')
        days: Number of days of historical data to fetch
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from market_pipeline.ingest import ingest_market_data
    except ImportError:
        return False, "market_pipeline not available"
    
    try:
        results = ingest_market_data(symbols=[symbol.lower()], days=days)
        if results.get(symbol.lower()):
            return True, f"Successfully ingested {symbol.upper()} from Binance"
        return False, f"Failed to ingest {symbol.upper()} - symbol may not exist on Binance"
    except Exception as e:
        return False, f"Binance error: {str(e)}"


def ingest_from_yfinance(symbol: str, days: int = 30) -> tuple[bool, str]:
    """
    Ingest a stock symbol from Yahoo Finance.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'NVDA')
        days: Number of days of historical data to fetch
        
    Returns:
        Tuple of (success, message)
    """
    try:
        import yfinance as yf
    except ImportError:
        return False, "yfinance not installed. Run: pip install yfinance"
    
    try:
        from market_pipeline.config import RAW_DIR
    except ImportError:
        return False, "market_pipeline not available"
    
    try:
        from datetime import datetime, timezone
        
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=f"{days}d")
        
        if df.empty:
            return False, f"No data returned for {symbol.upper()} - symbol may not exist"
        
        # Normalize to match Binance format
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Add metadata
        df['_symbol'] = symbol.lower()
        df['_venue'] = 'yfinance'
        df['_ingested_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save to RAW_DIR
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{symbol.lower()}_{date_str}.csv"
        filepath = RAW_DIR / filename
        df.to_csv(filepath, index=False)
        
        return True, f"Successfully ingested {symbol.upper()} from Yahoo Finance"
    except Exception as e:
        return False, f"Yahoo Finance error: {str(e)}"


def ingest_from_alphavantage(symbol: str, api_key: str, days: int = 30) -> tuple[bool, str]:
    """
    Ingest a stock symbol from Alpha Vantage.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'NVDA')
        api_key: Alpha Vantage API key
        days: Number of days of historical data to fetch (not directly used, fetches full history)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        import requests
    except ImportError:
        return False, "requests not installed"
    
    try:
        from market_pipeline.config import RAW_DIR
    except ImportError:
        return False, "market_pipeline not available"
    
    try:
        from datetime import datetime, timezone
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol.upper()}&outputsize=compact&apikey={api_key}"
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            error_msg = data.get("Note", data.get("Error Message", "Unknown error"))
            return False, f"Alpha Vantage error: {error_msg}"
        
        ts_data = data["Time Series (Daily)"]
        
        # Convert to DataFrame
        df = pd.DataFrame(ts_data).T
        df = df.reset_index()
        df = df.rename(columns={
            'index': 'timestamp',
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Limit to requested days
        df = df.head(days)
        
        # Add metadata
        df['_symbol'] = symbol.lower()
        df['_venue'] = 'alphavantage'
        df['_ingested_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save to RAW_DIR
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{symbol.lower()}_{date_str}.csv"
        filepath = RAW_DIR / filename
        df.to_csv(filepath, index=False)
        
        return True, f"Successfully ingested {symbol.upper()} from Alpha Vantage"
    except Exception as e:
        return False, f"Alpha Vantage error: {str(e)}"


def ingest_symbol(symbol: str, days: int = 30, source: str = "binance", api_key: str = "") -> tuple[bool, str]:
    """
    Ingest a symbol from the specified data source.
    
    Args:
        symbol: Trading pair or stock ticker
        days: Number of days of historical data
        source: Data source - 'binance', 'yfinance', or 'alphavantage'
        api_key: API key (required for Alpha Vantage)
        
    Returns:
        Tuple of (success, message)
    """
    if source == "binance":
        return ingest_from_binance(symbol, days)
    elif source == "yfinance":
        return ingest_from_yfinance(symbol, days)
    elif source == "alphavantage":
        if not api_key:
            return False, "Alpha Vantage requires an API key"
        return ingest_from_alphavantage(symbol, api_key, days)
    else:
        return False, f"Unknown data source: {source}"


def load_asset_prices(
    symbols: List[str], 
    start_date: str, 
    end_date: str, 
    timeframe: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols using market_pipeline.transform.load_raw_files.
    Reads locally ingested CSVs.
    """
    if load_raw_files is None:
        raise ImportError("market_pipeline package not found in PYTHONPATH.")

    data_map = {}
    
    for sym in symbols:
        try:
            # load_raw_files(symbol) reads all CSVs for that symbol from RAW_DIR
            df = load_raw_files(sym)
            
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                # Ensure timezone aware/naive consistency if needed. 
                # Usually standardizing to tz-naive or UTC is good.
                # If df['timestamp'] is UTC, strip for easier plotting/comp if risklens is simple
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
                
                df = df.set_index('timestamp').sort_index()
                
                # Filter to requested range
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    try:
                        e_dt = pd.to_datetime(end_date)
                        if pd.notnull(e_dt):
                            df = df[df.index <= e_dt]
                    except:
                        pass
                
                if not df.empty:
                    data_map[sym] = df
        except Exception as e:
            print(f"Error loading {sym}: {e}")
            
    # Align indices
    if not data_map:
        return {}
        
    common = None
    for sym, df in data_map.items():
        if common is None:
            common = df.index
        else:
            common = common.intersection(df.index)
            
    if common is not None and not common.empty:
        for sym in data_map:
            data_map[sym] = data_map[sym].loc[common]
            
    return data_map

def load_returns_matrix(
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d"
) -> pd.DataFrame:
    """
    Compute returns matrix from fetched prices.
    """
    prices_map = load_asset_prices(symbols, start_date, end_date, timeframe)
    
    if not prices_map:
        return pd.DataFrame()
        
    ret_dict = {}
    for sym, df in prices_map.items():
        if 'close' in df.columns:
            ret_dict[sym] = df['close'].pct_change()
            
    if not ret_dict:
        return pd.DataFrame()
        
    return pd.DataFrame(ret_dict).dropna()

def run_strategies_to_equity_curves(
    strategy_specs: List[Tuple[str, type, dict, str]],
    prices_map: Dict[str, pd.DataFrame],
    initial_capital: float = 100000.0,
    fee_bps: float = 10.0
) -> Dict[str, pd.Series]:
    """
    Run backtester strategies.
    specs: (Name, Class, Params, Symbol)
    """
    if BacktestEngine is None:
        return {}
        
    curves = {}
    
    for (name, cls, params, symbol) in strategy_specs:
        if symbol not in prices_map:
            continue
            
        data = prices_map[symbol]
        
        try:
            # Check BacktestEngine signature from imported class if possible, or assume standard
            # engine = BacktestEngine(data, strategy_class, ...)
            # We initialize the strategy instance usually?
            # Or engine takes class?
            # pytradelab typically: engine = BacktestEngine(data) -> engine.run(strategy)
            # OR engine = BacktestEngine(data, strategy_class, params...)
            # Let's assume the latter based on previous prompts.
            
            # Note: The prompt code snippet for pytradelab showed:
            # engine = BacktestEngine(data, cls, params, initial_capital, fee_bps)
            # We'll stick to that pattern but use real class.
            
            engine = BacktestEngine(
                data, 
                cls, 
                params, 
                initial_capital, 
                fee_bps
            )
            results = engine.run()
            
            # Extract equity curve
            if hasattr(results, 'equity_curve'):
                curves[name] = results.equity_curve
            elif isinstance(results, dict) and 'equity_curve' in results:
                curves[name] = results['equity_curve']
            elif hasattr(results, 'portfolio'):
                 # maybe results.portfolio.equity_curve?
                 if hasattr(results.portfolio, 'equity_curve'):
                     curves[name] = results.portfolio.equity_curve
                     
        except Exception as e:
            print(f"Backtest failed for {name}: {e}")
            
    return curves
