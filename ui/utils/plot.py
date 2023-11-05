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

def plot_data(X, y, name, marker_symbol='circle'):
    """
    Plots the data.
    """
    fig = go.Figure()
    scatter = _create_scatter_plot(X[:, 0], X[:, 1], y, marker_symbol=marker_symbol)
    fig.add_trace(scatter)
    fig.update_xaxes(showgrid=True, showticklabels=False)
    fig.update_yaxes(showgrid=True, showticklabels=False)
    return fig 