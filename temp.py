'''helper functions'''
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, colors, ticker as tk

import random
import seaborn as sns



def generate_random_theta_phi(dim: int):
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
    if dim == 2:
        theta = generate_random_theta_phi(dim)
        return polar_to_cartesian(v, theta)
    if dim == 3:
        theta, phi = generate_random_theta_phi(dim)
        return spherical_to_cartesian(v, theta, phi)


def generate_dataframe(distribution, n: int = 200):
    df = pd.DataFrame(
        {
            'id': range(1, n + 1),
            'x_pos': [distribution[i][0] for i in range(n)],
            'y_pos': [distribution[i][1] for i in range(n)],
            'z_pos': [distribution[i][2] for i in range(n)],
            'effective_field_direction': None,
            'cycle': [0 for _ in range(n)],
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
            'cycle': df['cycle'].max() + 1
        }


def create_paths_graph(paths):
    # Create a figure and axes using Seaborn
    fig, ax = plt.subplots()
    labels = paths.keys()
    # Plot the lines
    for speed in paths:
        df = paths[speed]
        x = df['x_pos']
        y = df['y_pos']
        sns.lineplot(x=x, y=y, ax=ax)

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

    # Set a title for the graph
    ax.set_title('Optional paths for charge movement in electrical field', pad=20)
    ax.legend(labels, loc='upper right')

    return ax


def plot_charge_path(df, graph, color):
    x = df['x_pos'].to_numpy()
    y = df['y_pos'].to_numpy()

    graph.plot(x, y, color=colors.to_rgba(color), label='charge path')
    return


'''problem a'''
from scipy.constants import e, electron_mass
import numpy as np
import pandas as pd
import charge as ch
import helper_functions as hf
import matplotlib.pyplot as plt
import seaborn as sns

c = pd.DataFrame()


electric_field = [30, 0, 0]  # V/m
time_tao = 10**(-15)  # s
time_intervals = 100

v = 0.002  # m/s
dim = 2
paths = {}
# create 3 options for the charge's path in the field
for _ in range(3):
    # initialize a new path
    charge = ch.Charge(0, 0, 0, 0, -e, electron_mass)
    initial_position = (0, 0, 0)
    data = hf.generate_dataframe([initial_position], 1)

    # generate movement over 100 time intervals
    for i in range(time_intervals):
        # create movement parameters
        charge.calculate_electric_field([], electric_field)
        charge.update_motion(time_tao)
        velocity_vec = hf.get_random_velocity(v, dim)
        # calculate change in position during interval and save the new position
        charge.update_position(velocity_vec, time_tao)
        hf.update_dataframe(data, [charge])
    #calculate drift speed
    drift_speed = (data['x_pos'].max() - data['x_pos'].min())/time_intervals*time_tao
    drift_speed = "{:.2e}".format(drift_speed)
    # add the charge's path to the list
    paths[drift_speed] = data

# plot the 3d graph of the charge's path
path_graph = hf.create_paths_graph(paths)
plt.grid(True)
sns.set_palette("Set2")
sns.set_theme(style='darkgrid')
# plot the graph of the
plt.show()

'''problem b'''

from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape

sphere = shape.Sphere(1, 3, [])
n = 200
time_tao = 10**(-3)  # s
time_intervals = 10
sphere.distribute_charges_3d(n, -e, electron_mass)
# df = hf.generate_dataframe(sphere.distribution)
for i in range(time_intervals):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges, [0, 0, 0])
    for charge in sphere.charges:
        charge.update_motion(time_tao)
    sphere.check_charges_in_sphere()

sphere.reset_distribution()
sphere.project_distribution()
sphere.visualize_charges()

'''shape'''
import helper_functions as hf
import charge as ch
import math
import types
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

import plotly.graph_objects as go
import pandas as pd


class Sphere:
    def __init__(self, radius: float, dim: int, free_charge: list,
                 center=(0, 0, 0)):
        self.center = center
        self.radius = radius
        self.charges = free_charge
        self.dim = dim
        self.distribution = []

    def check_charges_in_sphere(self):
        for charge in self.charges:
            if not self.in_sphere(charge):
                charge.correct_r_to_radius(self.radius, self.dim)

    # def correct_position(self, charge: ch.Charge):
    #     # correct location
    #     charge.correct_r_to_radius(self.radius, self.dim)

    def in_sphere(self, charge):
        if self.radius >= math.sqrt(charge.x**2 + charge.y**2 + charge.z**2):
            return True
        else:
            return False

    def distribute_charges_2d(self, n, q, m):

        radius = self.radius  # Radius of the circle

        # Generate points within a square
        side_length = int(np.ceil(np.sqrt(n)))  # Convert to integer
        x = np.linspace(-radius, radius, side_length)
        y = np.linspace(-radius, radius, side_length)
        x_grid, y_grid = np.meshgrid(x, y)

        # Keep points within the circle
        mask = np.sqrt(x_grid**2 + y_grid**2) <= radius
        selected_points = np.column_stack((x_grid[mask], y_grid[mask]))

        if selected_points.shape[0] < n:
            # Reset the circle with a denser set of points
            point_cloud = []
            while len(point_cloud) < n:
                theta = 2 * np.pi * np.random.rand()
                r = radius * np.sqrt(np.random.rand())
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                point_cloud.append([x, y])
            point_cloud = np.array(point_cloud)
        else:
            # Randomly sample n points from the selected points
            indices = np.random.choice(selected_points.shape[0], n, replace=False)
            point_cloud = selected_points[indices]

        charges = []
        for i, (x, y) in enumerate(point_cloud):
            charges.append(ch.Charge(x, y, 0, i, q, m))  # Z-coordinate is 0 for 2D

        self.charges = charges
        self.distribution = point_cloud
        self.project_distribution_2d()

    def distribute_charges_3d(self, n, q, m):
        dim_len = int((n * 3)**(1 / 3))  # Adjusting dim_len based on n
        spacing = 2 / dim_len  # Spacing between points
        point_cloud = np.mgrid[-1:1:spacing, -1:1:spacing, -1:1:spacing]\
            .reshape(3, -1).T

        point_radius = np.linalg.norm(point_cloud, axis=1)  # Calculate the
        # distance from the origin for each point
        in_points = point_radius < self.radius  # Boolean array indicating
        # if the point is inside the sphere

        selected_points = point_cloud[in_points][:n]  # Selecting n points
        # within the sphere
        charges = []
        counter = 0
        if self.dim == 3:
            for x, y, z in selected_points:
                charges.append(ch.Charge(x, y, z, counter, q, m))
                counter += 1
        self.charges = charges
        self.distribution = selected_points
        self.project_distribution()

    def reset_distribution(self):
        for i in range(len(self.charges)):
            self.distribution[i][0] = self.charges[i].x
            self.distribution[i][1] = self.charges[i].y
            if self.dim == 3:
                self.distribution[i][2] = self.charges[i].z

    def project_distribution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the wireframe sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.radius * np.outer(np.sin(u), np.sin(v))
        if self.dim == 3:
            z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.2)

        # Plot the charge distribution
        ax.scatter(self.distribution[:, 0], self.distribution[:, 1], self.distribution[:, 2])

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Charge Distribution')

        # Set viewing angles
        angles = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        for angle in angles:
            ax.view_init(elev=30, azim=angle)  # Set the elevation and azimuth angles
            plt.draw()
            plt.pause(0.5)  # Pause for a short interval to display the plot
        plt.show()


    def visualize_charges(self):
        fig = go.Figure()

        # Plot the wireframe sphere
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = self.radius * np.outer(np.cos(u), np.sin(v))
        # y = self.radius * np.outer(np.sin(u), np.sin(v))
        # if self.dim == 3:
        #     z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        # ax.plot_surface(x, y, z, color='gray', alpha=0.2)

        # Adding sphere surface
        u = 2 * pd.np.pi * pd.np.outer(pd.np.ones(100), pd.np.linspace(0, 1, 100))
        v = pd.np.pi * pd.np.outer(pd.np.linspace(0, 1, 100), pd.np.ones(100))
        x = self.radius * pd.np.cos(u) * pd.np.sin(v)
        y = self.radius * pd.np.sin(u) * pd.np.sin(v)
        z = self.radius * pd.np.cos(v)

        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Blues'))

        # ax.scatter(self.distribution[:, 0], self.distribution[:, 1], self.distribution[:, 2])

        # Adding charges
        fig.add_trace(
            go.Scatter3d(
                x=self.distribution[:0],
                y=self.distribution[:1],
                z=self.distribution[:2],
                mode='markers',
                marker=dict(color='red', size=5, line=dict(color='black', width=0.5)),
                name='Charge',
            )
        )

        # Setting layout properties
        for _, charge in df.iterrows():
            trace = go.Scatter3d(
                x=[charge['x']],
                y=[charge['y']],
                z=[charge['z']],
                mode='markers',
                marker=dict(
                    color=charge['color'],
                    size=charge['size'],
                    symbol='circle'
                )
            )
            traces.append(trace)

        # Show the plot
        fig.show()


    def project_distribution_2d(self):
        # Plot the shape (e.g., a circle)
        fig, ax = plt.subplots()
        shape = plt.Circle((0, 0), self.radius, color='gray', fill=False)
        ax.add_artist(shape)

        # Plot the charge distribution
        distribution_x = self.distribution[:, 0]
        distribution_y = self.distribution[:, 1]
        ax.scatter(distribution_x, distribution_y)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Charge Distribution')

        # Set aspect ratio and limits based on the shape
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)

        plt.show()





