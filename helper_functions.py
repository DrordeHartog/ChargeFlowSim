import math
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def generate_random_theta_phi(dim: int):
    if dim == 2:
        two_pi = 2*math.pi
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
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = 0
    return x, y, z


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
            'cycle': [0 for _ in range (n)],
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
                'cycle': df['cycle'].max()+1
            }


def generate_evenly_distributed_positions(din, shape, num_electrons, radius=1.0):
    positions = []

    if shape == 'sphere':
        if din == 2:
            raise ValueError("Sphere shape is not applicable for 2 dimensions.")
        positions = generate_electron_coordinates(num_electrons, radius)
    elif shape == 'disc':
        if din != 2:
            raise ValueError("Disc shape is only applicable for 2 dimensions.")
        positions = generate_points_in_disc(num_electrons, radius)
    elif shape == 'square':
        if din != 2:
            raise ValueError("Square shape is only applicable for 2 dimensions.")
        positions = generate_points_in_square(num_electrons)
    else:
        raise ValueError("Invalid shape specified.")

    return positions


def generate_electron_coordinates(n: int = 200, r: int = 1):
    coordinates = []
    increment = math.pi * (3 - math.sqrt(5))  # Increment for golden angle

    for i in range(n):
        y = ((i / float(n - 1)) * 2) - 1  # y goes from -1 to 1
        radius_at_y = math.sqrt(1 - y * y)  # Radius at current y

        theta = i * increment  # Golden angle increment

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y

        coordinates.append((x * r, y * r, z * r))

    return coordinates


def generate_points_in_disc(num_points, r):
    positions = []
    for _ in range(num_points):
        theta = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(0, 1)) * radius

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        positions.append((x, y))

    return positions


def generate_points_in_square(num_points):
    side_length = int(math.sqrt(num_points))
    step_size = 1.0 / side_length

    positions = []
    for i in range(side_length):
        for j in range(side_length):
            x = (i + 0.5) * step_size
            y = (j + 0.5) * step_size
            positions.append((x, y))

    return positions


def plot_charge_path(df, dim=3):

    ax = plt.figure().add_subplot(projection = str(dim) + 'd')
    x = df['x_pos'].to_numpy()
    y = df['y_pos'].to_numpy()
    z = df['z_pos'].to_numpy()

    ax.plot(x, y, zs=z, zdir=z, label = 'charge path')
