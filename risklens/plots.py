# risklens/plots.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "Equity Curves"
) -> go.Figure:
    """Compare multiple equity curves."""
    df_list = []
    for name, s in curves.items():
        sub_df = s.to_frame(name="Value")
        sub_df["Asset"] = name
        df_list.append(sub_df)
    
    if not df_list:
        return go.Figure()
        
    combined = pd.concat(df_list)
    fig = px.line(combined, y="Value", color="Asset", title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Equity Value")
    return fig

def plot_drawdown(
    drawdown_series: pd.Series,
    title: str = "Underwater Plot (Drawdown)"
) -> go.Figure:
    """Area chart for drawdown."""
    fig = px.area(
        x=drawdown_series.index, 
        y=drawdown_series.values,
        title=title
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
    return fig

def plot_rolling_metric(
    metric: pd.Series,
    metric_name: str,
    title: str = None
) -> go.Figure:
    if title is None: title = f"Rolling {metric_name}"
    
    fig = px.line(x=metric.index, y=metric.values, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title=metric_name)
    return fig

def plot_risk_return_scatter(
    metrics_df: pd.DataFrame,
    title: str = "Risk vs Return"
) -> go.Figure:
    """
    Expects DataFrame with columns: ['Name', 'Return', 'Volatility', 'Sharpe']
    """
    fig = px.scatter(
        metrics_df, 
        x='Volatility', 
        y='Return', 
        color='Sharpe',
        text='Name',
        size=[10]*len(metrics_df),
        title=title,
        color_continuous_scale='Bluered'
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title="Annualized Volatility", 
        yaxis_title="Annualized Return"
    )
    return fig

def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """Heatmap of correlations."""
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale='RdBu_r', # Red(neg) to Blue(pos)
        zmin=-1, 
        zmax=1
    )
    return fig
