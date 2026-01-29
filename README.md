# RiskLens

**Risk & Portfolio Analytics Dashboard**

RiskLens is an interactive dashboard built with Python and Streamlit for constructing portfolios, analyzing risk, and visualizing performance. It acts as a unified frontend for:
- **QuantStream**: Market data ingestion.
- **PyTradeLab**: Backtesting strategies.

## Features

- **Portfolio Construction**: Combine assets and strategies with custom weights.
- **Risk Metrics**: Calculate Sharpe Ratio, Volatility, Max Drawdown, and Annualized Returns.
- **Interactive Visualizations**:
    - Equity Curves (vs Benchmark)
    - Underwater/Drawdown Plots
    - Rolling Volatility & Sharpe
    - Risk/Return Scatter Matrix
- **Scenario Analysis**: Adjust weights and risk parameters on the fly.

## Project Structure

```text
risklens/
  __init__.py
  data_interface.py   # Data loading abstraction (mocks fallback)
  analytics.py        # Financial math & metrics
  plots.py            # Plotly visualizations
  mocks.py            # Simulation generation
  config.py           # Settings
app.py                # Main Streamlit application
tests/                # Unit tests
```

## Installation

1. Clone the repository.
2. Install local dependencies (QuantStream and PyTradeLab):
   ```bash
   pip install -e /path/to/quantstream
   pip install -e /path/to/pytradelab
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard locally:

```bash
streamlit run app.py
```

## Testing

Run unit tests:

```bash
pytest tests/
```
