import helper_functions as hf
import charge as ch
import math
import types


class Sphere:
    def __init__(self, radius: float, dim: int, free_charge: list,
                 center=(0, 0, 0)):
        self.center = center
        self.radius = radius
        self.charges = free_charge
        self.dim = dim

    def check_charges_in_sphere(self):
        for charge in self.charges:
            if not self.in_sphere(charge):
                self.correct_position(charge)

    def correct_position(self, charge: ch.Charge):
        # correct location
        charge.corect_radius_to_one(self.radius)

    def in_sphere(self, charge):
        if self.radius >= math.sqrt(charge.x**2 + charge.y**2 + charge.y**2):
            return true
        else:
            return false

    def distribute_charges_in_sphere(self, num_charges, charge_value,
                                     charge_mass):
        charges = []
    
        # Calculate the number of charges per concentric shell
        num_shells = int(math.sqrt(num_charges))
    
        # Calculate the number of charges in the last incomplete shell
        remaining_charges = num_charges - num_shells**2
    
        # Calculate the angular spacing for each charge in a shell
        theta_spacing = math.pi / num_shells
        phi_spacing = 2 * math.pi / num_shells
    
        # Distribute charges in complete shells
        for i in range(num_shells):
            theta = (i + 0.5) * theta_spacing
            num_phi = 2 * i + 1
    
            for j in range(num_phi):
                phi = j * phi_spacing
                self.distribute_charge_in_shell(theta, phi, j, charges,
                                                charge_value, charge_mass)
    
        # Distribute remaining charges in the last incomplete shell
        for i in range(remaining_charges):
            theta = (num_shells + 0.5) * theta_spacing
            phi = i * phi_spacing
            self.distribute_charge_in_shell(theta, phi, i, charges, charge_value,
                                            charge_mass)
    
        self.charges = charges
        return

    def distribute_charge_in_shell(self, theta, phi, index, charges: list,
                                   charge_value, charge_mass):
        x = self.radius * math.sin(theta) * math.cos(phi)
        y = self.radius * math.sin(theta) * math.sin(phi)
        z = self.radius * math.cos(theta)

        cur_charge = ch.Charge(x, y, z, index, charge_value, charge_mass)
        charges.append(cur_charge)