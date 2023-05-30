import helper_functions as hf
import charge
import math


class Sphere:
    def __init__(self, radius: float, dim: int, free_charge=[],
                 center=(0, 0, 0)):
        self.center = center
        self.radius = radius
        self.charges = free_charge
        self.dim = dim

    def check_charges_in_sphere(self):
        for charge in self.charges:
            if not self.in_sphere(charge):
                self.correct_position(charge)

    def correct_position(self, charge: charge.Charge):
        pass

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
                x = self.radius * math.sin(theta) * math.cos(phi)
                y = self.radius * math.sin(theta) * math.sin(phi)
                z = self.radius * math.cos(theta)
    
                charge = charge.Charge(x, y, z, charge_value, charge_mass)
                charges.append(charge)
    
        # Distribute remaining charges in the last incomplete shell
        for i in range(remaining_charges):
            theta = (num_shells + 0.5) * theta_spacing
            phi = i * phi_spacing
            x = self.radius * math.sin(theta) * math.cos(phi)
            y = self.radius * math.sin(theta) * math.sin(phi)
            z = self.radius * math.cos(theta)
    
            charge = charge.Charge(x, y, z, charge_value, charge_mass)
            charges.append(charge)
    
        self.charges = charges
        return
