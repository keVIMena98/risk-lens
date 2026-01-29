# risklens/config.py
import os

# App Settings
APP_TITLE = "RiskLens"
APP_DESCRIPTION = "Risk & Portfolio Analytics Dashboard"
VERSION = "0.1.0"

# Defaults
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_PERIODS_PER_YEAR = 252  # 365 for crypto, but 252 is standard finance default
DEFAULT_RISK_FREE_RATE = 0.04   # 4%
DEFAULT_ROLLING_WINDOW = 30

# Feature Flags
# Auto-detect if libraries are installed, otherwise use Mocks
try:
    import market_pipeline
    import backtester
    USE_MOCKS = False
except ImportError:
    USE_MOCKS = True
