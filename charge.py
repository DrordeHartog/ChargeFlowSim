from scipy.constants import e
from scipy.constants import electron_mass
import helper_functions as hf
import math


class Charge:
    def __init__(self, x: float, y: float, z: float, index: int, q: float = -e,
                 mass: float = electron_mass):
        self.q = q
        self.m = mass
        self.x = x
        self.y = y
        self.z = z
        self.efx = 0
        self.efy = 0
        self.efz = 0
        self.in_sphere = 1
        self.index = index

# getters and setters
    def get_position(self):
        return self.x, self.y, self.z

    def get_charge(self):
        return self.q

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_charge(self, q):
        self.q = q

    def set_mass(self, m):
        self.m = m

    def set_in_sphere(self, in_sphere_value):
        self.in_sphere = in_sphere_value

    def get_in_sphere(self):
        return self.in_sphere

    def get_radius(self):
        return math.sqrt(self.x**2+self.y**2+self.z**2)

    def calculate_electric_field(self, charges: list, external_field=(0, 0, 0)):
        """ Calculate the electric field on a charge from all charges in the
         system and an external electric field..
         :param charges = list of charges of class Charge
         :param external_field = external electric field, default value 0 """
        k = 8.9875517923 * 10**9  # Coulomb's constant
        self.efx, self.efy, self.efz = external_field
        #  remove our charge from calculations
        if charges[self.index].index == self.index:
            other_charges = charges[:self.index] + charges[self.index+1:]
        else:
            other_charges = charges
        # use superposition and sum up field contributions
        for charge in other_charges:
            dx = self.x - charge.x
            dy = self.y - charge.y
            dz = self.z - charge.z

            r_squared = dx**2 + dy**2 + dz**2
            r_cubed = r_squared**(3/2)

            field_x = k * charge.q * dx / r_cubed
            field_y = k * charge.q * dy / r_cubed
            field_z = k * charge.q * dz / r_cubed

            self.efx += field_x
            self.efy += field_y
            self.efz += field_z
        return self.efx, self.efy, self.efz

    # move under effect of external field
    def update_motion(self, tao):
        """ Using equations of mechanics, this function calculates the movement
        of a charge under the effect of its given electric field int a time
        increment of tao."""
        self.x += self.q*self.efx*(tao**2)/(2*self.m)
        self.y += self.q*self.efy*(tao**2)/(2*self.m)
        self.z += self.q*self.efz*(tao**2)/(2*self.m)

    def update_position(self, velocity, tao):
        """ Move under effect of drude collision velocity """
        self.x += velocity[0] * tao
        self.y += velocity[1] * tao
        self.z += velocity[2] * tao

    def correct_r_to_radius(self, radius, dim):
        """Returns a charge back into a circular shape in the radial direction.
         this function assumes the center of the shape to be (0,0,0).
        :param radius = radius of shape
        :param dim = number of dimensions of shape."""
        if dim == 3:
            r, theta, phi = hf.cartesian_to_spherical(self.x, self.y, self.z)
            r = radius
            x, y, z = hf.spherical_to_cartesian(r, theta, phi)
        elif dim == 2:
            r, theta = hf.cartesian_to_polar(self.x, self.y)
            r = radius
            x, y, z = hf.polar_to_cartesian(r, theta)
        self.set_position(x, y, z)

    def __str__(self):
        x_rounded = round(self.x, 3)
        y_rounded = round(self.y, 3)
        z_rounded = round(self.z, 3)
        radius_rounded = round(hf.cartesian_to_spherical(self.x, self.y,
                                                         self.z)[0], 3)
        return f"Point charge {self.index} at radius {radius_rounded} from" \
               f" the center at point ({x_rounded}, {y_rounded}, {z_rounded})"\
               f" with charge {self.q} coulomb and mass {self.m} kg"
