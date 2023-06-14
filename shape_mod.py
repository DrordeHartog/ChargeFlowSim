import helper_functions as hf
import charge as ch
import math
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import types
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation


class Square:
    dim: int

    def __init__(self, radius: float, dim: int, free_charge: list,
                 center=(0, 0, 0)):
        self.center = center
        self.radius = radius
        self.charges = free_charge
        self.dim = dim
        self.distribution = []
        self.percent_in_sphere = []

    def distribute_charges(self, n, q, m):
        """Distributes n charges of mass m and charge q within a cube.
        When done, displays the final plot."""
        charges = []
        counter = 0
        selected_points = []

        while len(charges) < n:
            point_cloud = np.random.uniform(-1, 1, size=(n, self.dim))

            if self.dim == 2:
                in_points = np.abs(point_cloud[:, 0]) < 1  # Points are inside the square in the x-axis

            new_points = point_cloud[in_points]

            if self.dim == 2:
                for i in range(min(n - len(charges), len(new_points))):
                    x, y = new_points[i]
                    # z = np.random.uniform(-1, 1)  # Assign random z-coordinate within the cube
                    z = 0
                    charges.append(ch.Charge(x, y, z, counter, q, m))
                    counter += 1
                    selected_points.append([x, y, z])

        self.charges = charges
        self.distribution = np.array(selected_points)


    def return_charges_to_square(self):
        """ Checks all that all charges are within sphere. if not calls
        charge method to return to sphere"""
        for charge in self.charges:
            if self.radius < math.fabs(charge.x):
                charge.x = np.sign(charge.x)*self.radius
            if self.radius < math.fabs(charge.y):
                charge.y = np.sign(charge.y) * self.radius

    def recalc_distribution(self):
        """" recalculates distribution nparray locations"""
        for i in range(len(self.charges)):
            self.distribution[i][0] = self.charges[i].x
            self.distribution[i][1] = self.charges[i].y
            if self.dim == 3:
                self.distribution[i][2] = self.charges[i].z

    def project_distribution_2d(self):
        """Plots and displays points charges in a 2D square."""
        self.recalc_distribution()

        # Plot the shape (e.g., a square)
        fig, ax = plt.subplots()
        shape = plt.Rectangle((-self.radius, -self.radius), 2 * self.radius, 2 * self.radius, color='gray', fill=False)
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

        def project_distribution_2d(self):
            """Plots and displays points charges in a 2D square."""
            self.recalc_distribution()

            # Plot the shape (e.g., a square)
            fig, ax = plt.subplots()
            shape = plt.Rectangle((-self.radius, -self.radius), 2 * self.radius, 2 * self.radius, color='gray',
                                  fill=False)
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


class Sphere:
    def __init__(self, radius: float, dim: int, free_charge: list,
                 center=(0, 0, 0)):
        self.center = center
        self.radius = radius
        self.charges = free_charge
        self.dim = dim
        self.distribution = []
        self.percent_in_sphere = [100.0]

    def return_charges_to_sphere(self):
        """ Checks all that all charges are within sphere. if not calls
        charge method to return to sphere"""
        for charge in self.charges:
            if self.radius < math.sqrt(charge.x ** 2 + charge.y ** 2 + charge.z ** 2):
                charge.correct_r_to_radius(self.radius, self.dim)
                charge.set_in_sphere(False)
            else:
                charge.set_in_sphere(True)
        self.calculate_percentage_in_sphere()

    def calculate_percentage_in_sphere(self):
        count_in_sphere = sum(1 for charge in self.charges if charge.get_in_sphere())
        percentage = count_in_sphere / len(self.charges) * 100
        self.percent_in_sphere.append(percentage)


    def distribute_charges(self, n, q, m):
        """" distributes n charges of mass m and charge q within itself
        randomly and uniformly. when done, displays final plot"""
        charges = []
        counter = 0
        selected_points = []

        while len(charges) < n:
            point_cloud = np.random.uniform(-1, 1, size=(n, self.dim))
            point_radius = np.linalg.norm(point_cloud, axis=1)
            in_points = point_radius < self.radius
            new_points = point_cloud[in_points]

            if self.dim == 3:
                for i in range(min(n - len(charges), len(new_points))):
                    x, y, z = new_points[i]
                    charges.append(ch.Charge(x, y, z, counter, q, m))
                    counter += 1
                    selected_points.append([x, y, z])
            if self.dim == 2:
                for i in range(min(n - len(charges), len(new_points))):
                    x, y = new_points[i]
                    charges.append(ch.Charge(x, y, 0, counter, q, m))
                    counter += 1
                    selected_points.append([x, y, 0])

        self.charges = charges
        self.distribution = np.array(selected_points)
        # if self.dim == 3:
        #     self.project_distribution_3d()
        # if self.dim == 2:
        #     self.project_distribution_2d()

    def recalc_distribution(self):
        """" recalculates distribution nparray locations"""
        for i in range(len(self.charges)):
            self.distribution[i][0] = self.charges[i].x
            self.distribution[i][1] = self.charges[i].y
            if self.dim == 3:
                self.distribution[i][2] = self.charges[i].z

    def project_distribution_3d(self):
        """"plots and displays points inside a 3d sphere"""
        self.recalc_distribution()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the wireframe sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.radius * np.outer(np.sin(u), np.sin(v))
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


    def project_distribution_2d(self):
        """"plots and displays points charges in a 2d circle"""
        self.recalc_distribution()
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

    def print_charges_inside_volume(self):
        """"iterates over all charges in charge list and prints charges that
         are not on the shell of shape"""
        for charge in self.charges:
            if hf.cartesian_to_spherical(charge.x, charge.y, charge.z)[0] \
                    < 0.99:
                print(charge)


    def project_distribution_2d_2(self):
        """"plots and displays points charges in a 2d circle"""
        # Set Seaborn style
        sns.set(style='whitegrid')

        # Create the plot
        fig, ax = plt.subplots()

        # Plot the shape (e.g., a circle)
        shape = plt.Circle((0, 0), self.radius, color='gray', fill=False)
        ax.add_artist(shape)

        # Plot the charge distribution
        distribution_x = self.distribution[:, 0]
        distribution_y = self.distribution[:, 1]
        ax.scatter(distribution_x, distribution_y, color='blue', alpha=0.7)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Charge Distribution')

        # Set aspect ratio and limits based on the shape
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)

        plt.show()

    def visualise(self):
        # Adding sphere surface
        fig = go.Figure()
        u = 2 * pd.np.pi * pd.np.outer(pd.np.ones(100), pd.np.linspace(0, 1, 100))
        v = pd.np.pi * pd.np.outer(pd.np.linspace(0, 1, 100), pd.np.ones(100))
        x = self.radius * pd.np.cos(u) * pd.np.sin(v)
        y = self.radius * pd.np.sin(u) * pd.np.sin(v)
        z = self.radius * pd.np.cos(v)

        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Blues'))

        # Create a scatter trace
        fig.add_trace(
            go.Scatter3d(x=self.distribution[:, 0],
                y=self.distribution[:, 1],
            z=self.distribution[:, 2],
            mode='markers',
            marker=dict(
                color='blue',
                size=5,
                symbol='circle'
            )
        )
        )

        # Create the layout
        layout = go.Layout(
            title='Charges Final Location',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            showlegend=False
        )


        # Display the figure
        fig.show()

    def plot_percentage_in_sphere(self):
        # Create a list of cycles for the x-axis
        cycles = list(range(len(self.percent_in_sphere)))

        # Create a DataFrame with cycles and percentage in sphere
        data = {'Cycle': cycles, 'Percentage in Sphere': self.percent_in_sphere}
        df = pd.DataFrame(data)

        # Plot the data using Seaborn
        sns.set(style='whitegrid')
        sns.lineplot(data=df, x='Cycle', y='Percentage in Sphere')

        # Set labels and title
        plt.xlabel('Cycle')
        plt.ylabel('Percentage in Sphere')
        plt.title('Percentage of Charges in Sphere over Time')

        # Show the plot
        plt.show()

    def calculate_charge_density(self, n=100):
        """Calculates the charge density as a function of radius and plots the data."""
        radii = np.linspace(0, self.radius, n)  # Generate radii from 0 to sphere radius
        densities = []  # List to store the charge densities

        for radius in radii:
            # Count the number of charges within the given radius
            count_in_radius = sum(1 for charge in self.charges if charge.get_in_sphere() <= self.radius)

            # Calculate the charge density at the given radius
            if self.dim ==2:
                density = count_in_radius / (np.pi * radius ** 2)
            elif self.dim == 3:
                density = count_in_radius / (np.pi * radius ** 2)
            densities.append(density)

        # Plot the charge density
        sns.set(style='whitegrid')
        plt.plot(radii, densities)
        plt.xlabel('Radius')
        plt.ylabel('Charge Density')
        plt.title('Charge Density as a Function of Radius')
        plt.show()
