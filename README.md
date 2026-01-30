# RiskLens – Risk & Portfolio Analytics Dashboard

RiskLens is a comprehensive risk and portfolio analytics dashboard designed to orchestrate three key internal libraries: **QuantStream** (Market Data), **PyTradeLab** (Backtesting), and **FactorLab** (Factor Analysis). It provides a unified Streamlit interface for quantitative researchers to analyze assets, backtest strategies, and monitor portfolio risk in real-time.

## Features

-   **Unified Data Pipeline**: Seamlessly loads normalized market data (OHLCV) via `QuantStream`.
-   **Strategy Orchestration**: Runs backtests on-the-fly using `PyTradeLab` engines and visualizes equity curves.
-   **Advanced Risk Analytics**: Computes robust risk metrics (Sharpe, Volatility, Drawdown) using `FactorLab`.
-   **Interactive Dashboard**:
    -   Dynamic asset selection and weighting.
    -   Compare portfolio performance against benchmarks.
    -   Visualize rolling risk metrics and risk-return scatter plots.
    -   Advanced correlation analysis and strategy performance details.

## Project Structure

```
risk-lens/
├── risklens/
│   ├── config.py           # Configuration constants (Dates, Defaults)
│   ├── data_interface.py   # Wrappers for QuantStream & PyTradeLab
│   ├── analytics.py        # Portfolio logic & FactorLab integration
│   ├── plots.py            # Plotly visualization components
├── tests/                  # Unit tests
├── app.py                  # Main Streamlit Dashboard application
├── pyproject.toml          # Project metadata & dependencies
└── README.md               # This documentation
```

## Setup & Installation

1.  **Prerequisites**: Ensure you have Python 3.11+ installed.
2.  **Dependencies**: RiskLens depends on local repositories (`quantstream`, `pytradelab`, `factorlab`). Ensure these are accessible or installed in your environment.
    ```bash
    # Example: Installing local dev dependencies
    pip install -e ../quantstream
    pip install -e ../pytradelab
    pip install -e ../factorlab
    ```
3.  **Installation**:
    ```bash
    git clone https://github.com/your-org/risk-lens.git
    cd risk-lens
    pip install -e .
    ```

## Usage

Run the dashboard locally:

```bash
streamlit run app.py
```

### Dashboard Workflow
1.  **Select Assets**: Choose from the default crypto universe (BTC, ETH, SOL) or add custom tickers.
2.  **Configure Portfolio**: specific weights or normalize equal weights.
3.  **Select Strategies**: Toggle "Include Strategies" to overlay Momentum or Mean Reversion backtests (powered by `PyTradeLab`) on your portfolio chart.
4.  **Analyze**: View key metrics, drawdowns, rolling volatility, and correlation matrices.

## Testing

Run the test suite to verify analytics logic:

```bash
pytest tests/
```
