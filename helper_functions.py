import random
import math
import pandas as pd
import numpy as np


def generate_random_theta_phi(num_dimensions):
    if num_dimensions == 2:
        theta = random.uniform(0, 2 * math.pi)  # Generate random theta between 0 and 2pi
        return theta
    elif num_dimensions == 3:
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


def generate_dataframe(n=200):
    data = {
        'id': range(1, num_rows + 1),
        'x_pos': np.random.uniform(low=-10, high=10, size=num_rows),
        'y_pos': np.random.uniform(low=-10, high=10, size=num_rows),
        'z_pos': np.random.uniform(low=-10, high=10, size=num_rows),
        'effective_field_direction': None,
        'cycle': np.random.randint(low=1, high=100, size=num_rows),
        'timestamp': [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(num_rows)]
    }
