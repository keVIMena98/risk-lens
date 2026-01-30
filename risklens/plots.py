# risklens/plots.py
import plotly.graph_objects as go
import pandas as pd
from typing import Dict

def plot_equity_curves(
    equity_curves: Dict[str, pd.Series],
    title: str,
) -> go.Figure:
    """
    Create a line chart with one trace per equity curve.
    Index is assumed to be datetime.
    """
    fig = go.Figure()
    
    for name, series in equity_curves.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=name
        ))
        
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity")
    return fig

def plot_drawdown(
    equity: pd.Series,
    title: str,
) -> go.Figure:
    """
    Compute drawdown series from equity and plot it as an area/line chart.
    """
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tozeroy',
        name='Drawdown'
    ))
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown %")
    return fig

def plot_rolling_metric(
    metric_series: pd.Series,
    title: str,
) -> go.Figure:
    """
    Generic function to plot a rolling metric series (volatility or Sharpe).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metric_series.index,
        y=metric_series.values,
        mode='lines',
        name=title
    ))
    fig.update_layout(title=title, xaxis_title="Date")
    return fig

def plot_risk_return_scatter(
    metrics_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """
    Scatter plot with x = volatility, y = annualized_return,
    one point per row; color or marker distinguishes PORTFOLIO vs assets.
    """
    fig = go.Figure()
    
    # Separate Portfolio from Assets for styling
    if 'name' not in metrics_df.columns:
         # Safety if caller provides older df format
         pass
    
    # We assume 'name', 'volatility', 'annualized_return' are present per analytics 5.4
    portfolio = metrics_df[metrics_df['name'] == 'PORTFOLIO']
    assets = metrics_df[metrics_df['name'] != 'PORTFOLIO']
    
    if not assets.empty:
        fig.add_trace(go.Scatter(
            x=assets['volatility'],
            y=assets['annualized_return'],
            mode='markers+text',
            text=assets['name'],
            textposition="top center",
            name='Assets',
            marker=dict(size=10, color='blue')
        ))
        
    if not portfolio.empty:
        fig.add_trace(go.Scatter(
            x=portfolio['volatility'],
            y=portfolio['annualized_return'],
            mode='markers+text',
            text=portfolio['name'],
            textposition="top center",
            name='Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (Ann.)",
        yaxis_title="Return (Ann.)"
    )
    return fig
