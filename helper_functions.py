import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, colors, ticker as tk
from matplotlib.lines import Line2D
import random
import seaborn as sns


def generate_random_theta_phi(dim: int):
    ''' generate a random direction in two or three dimensions.
    :param dim = number of dimensions in the system'''
    if dim == 2:
        two_pi = 2 * math.pi
        theta = random.uniform(0, two_pi)  # Generate random theta between 0 and 2pi
        return theta
    elif dim == 3:
        theta = random.uniform(0, math.pi)  # Generate random theta between 0 and pi
        phi = random.uniform(0, 2 * math.pi)  # Generate random phi between 0 and 2pi
        return theta, phi
    else:
        raise ValueError("Number of dimensions must be 2 or 3.")


def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = 0
    return x, y, z


def cartesian_to_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def get_random_velocity(v: float, dim: int):
    """Get a velocity vector of magnitude v and a random direction
    :param v = magnitude of velocity vector
    :param dim = dimensions of system"""

    if dim == 2:
        theta = generate_random_theta_phi(dim)
        return polar_to_cartesian(v, theta)
    if dim == 3:
        theta, phi = generate_random_theta_phi(dim)
        return spherical_to_cartesian(v, theta, phi)


def generate_dataframe(distribution):
    n = len(distribution)
    df = pd.DataFrame(
        {
            'id': range(1, n + 1),
            'x_pos': [distribution[i][0] for i in range(n)],
            'y_pos': [distribution[i][1] for i in range(n)],
            'z_pos': [distribution[i][2] for i in range(n)],
            'cycle': [0 for _ in range(n)],
            'in_sphere': True
        }
    )
    return df


def update_dataframe(df, charges):
    for charge in charges:
        df.loc[len(df)] = {
            'id': charge.index,
            'x_pos': charge.x,
            'y_pos': charge.y,
            'z_pos': charge.z,
            'effective_field_direction': (charge.efx, charge.efy, charge.efz),
            'in_sphere': charge.get_in_sphere(),
            'cycle': df['cycle'].max() + 1
        }


def create_paths_graph(paths):
    # Create a figure and axes using Seaborn
    fig, ax = plt.subplots(figsize=(15, 6))
    labels = [path[0] for path in paths]
    paths = [path[1] for path in paths]

    # Set the colors for the lines and the legend
    line_colors = sns.color_palette(n_colors=len(paths))
    legend_handles = []

    # Plot the lines
    for i, path in enumerate(paths):
        x = path['x_pos']
        y = path['y_pos']
        sns.lineplot(x=x, y=y, ax=ax, color=line_colors[i])
        legend_handles.append(Line2D([0], [0], color=line_colors[i], lw=1.5))

    ax.xaxis.set_major_formatter(tk.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(tk.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    # Add arrow and label
    arrow_props = dict(arrowstyle='->', linewidth=1.5, color='red')
    plt.annotate('Field direction', xy=(0.6, 1.02), xytext=(0.5, 1.02),
                 xycoords='axes fraction', textcoords='axes fraction',
                 ha='right', va='center', arrowprops=arrow_props)

    ax.set_xlim(ax.get_xlim()[::-1])

    # Set a title and a legend for the graph
    ax.set_title('Optional paths for charge movement in electrical field', pad=20)
    ax.legend(legend_handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1),
              facecolor='white', edgecolor='black', framealpha=1, frameon=True)

    return ax
