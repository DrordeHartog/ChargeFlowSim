import pandas as pd
import plotly.graph_objects as go

def visualize_charges(df, radius):
    fig = go.Figure()

    # Adding sphere surface
    u = 2 * pd.np.pi * pd.np.outer(pd.np.ones(100), pd.np.linspace(0, 1, 100))
    v = pd.np.pi * pd.np.outer(pd.np.linspace(0, 1, 100), pd.np.ones(100))
    x = radius * pd.np.cos(u) * pd.np.sin(v)
    y = radius * pd.np.sin(u) * pd.np.sin(v)
    z = radius * pd.np.cos(v)

    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Blues'))

    # Adding charges at each time step
    for _, charge in df.iterrows():
        fig.add_trace(
            go.Scatter3d(
                x=charge['x'],
                y=charge['y'],
                z=charge['z'],
                mode='markers',
                marker=dict(color=charge['color'], size=charge['size'], line=dict(color='black', width=0.5)),
                name='Charge',
            )
        )

    # Setting layout properties
    fig.update_layout(
        title='Charges Visualization',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                center=dict(x=0, y=0, z=0)
            ),
        ),
    )

    # Show the plot
    fig.show()

# Example usage
df = pd.DataFrame({
    'x': [[0.5, 0.6, 0.7], [-0.3, -0.2, -0.1], [0.1, 0.2, 0.3], [-0.2, -0.1, 0.0]],  # x-coordinate of charges at different time steps
    'y': [[0.2, 0.3, 0.4], [-0.1, 0.0, 0.1], [0.6, 0.7, 0.8], [-0.5, -0.4, -0.3]],  # y-coordinate of charges at different time steps
    'z': [[-0.1, 0.0, 0.1], [0.4, 0.5, 0.6], [-0.3, -0.2, -0.1], [0.2, 0.3, 0.4]],  # z-coordinate of charges at different time steps
    'color': ['red', 'green', 'blue', 'magenta'],  # color of charges
    'size': [5, 5, 5, 5]  # size of charges
})

radius = 1.0  # radius of the sphere

visualize_charges(df, radius)
