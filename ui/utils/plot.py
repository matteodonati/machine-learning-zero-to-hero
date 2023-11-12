import numpy as np
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
            colorscale=['rgb(0, 0, 255)', 'rgb(255, 0, 0)'],
            line=dict(width=1),
        ),
    )

def create_plot():
    """
    Creates a plot.
    """
    fig = go.Figure()
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def add_data_to_plot(fig, X, y, marker_symbol='circle'):
    """
    Adds data to a plot.
    """
    scatter = _create_scatter_plot(X[:, 0], X[:, 1], y, marker_symbol=marker_symbol)
    fig.add_trace(scatter)

def add_decision_boundary(fig_train, fig_test, X, model):
    """
    Adds a classification decision boundary to the plots.
    """

    h = 0.01
    min1, max1 = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    min2, max2 = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x1_grid = np.arange(min1, max1, h)
    x2_grid = np.arange(min2, max2, h)
    xx, yy = np.meshgrid(x1_grid, x2_grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    y_pred = np.array(model.predict(grid))
    zz = y_pred.reshape(xx.shape)
    boundary = go.Heatmap(
        x=xx[0], 
        y=x2_grid, 
        z=zz, 
        colorscale=['rgb(128, 128, 255)', 'rgb(255, 128, 128)'], 
        showscale=False
    )
    fig_train.add_trace(boundary)
    fig_test.add_trace(boundary)