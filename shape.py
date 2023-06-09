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





