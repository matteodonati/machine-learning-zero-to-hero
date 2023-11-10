import plotly.graph_objects as go

def _create_scatter_plot(x, y, color, marker_symbol=None):
    """
    Creates a Scatter object.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker_symbol=marker_symbol,
        marker=dict(
            color=color,
            line=dict(width=1),
        ),
    )

def create_plot():
    """
    Creates a plot.
    """
    fig = go.Figure()
    fig.update_xaxes(showgrid=True, showticklabels=False)
    fig.update_yaxes(showgrid=True, showticklabels=False)
    return fig

def add_data_to_plot(fig, X, y, marker_symbol='circle'):
    """
    Adds data to a plot.
    """
    scatter = _create_scatter_plot(X[:, 0], X[:, 1], y, marker_symbol=marker_symbol)
    fig.add_trace(scatter)