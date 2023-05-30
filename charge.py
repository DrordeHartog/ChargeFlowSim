from scipy.constants import e
from scipy.constants import electron_mass

class Charge:
    def __init__(self, x: float, y: float, z: float, index: int, q: float = e,
                 mass: float = electron_mass):
        self.q = q
        self.m = mass
        self.x = x
        self.y = y
        self.z = z
        self.efx = 0
        self.efy = 0
        self.efz = 0
        self.index = index

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


    def calculate_electric_field(self, charges: list, external_field):
        k = 8.9875517923 * 10**9  # Coulomb's constant
        self.efx = external_field[0]
        self.efy = external_field[1]
        self.efz = external_field[2]

#remove our charge from calcs
        other_charges = charges[:self.index-1] + charges[self.index+1:]
        for charge in other_charges:
            dx = charge.x - self.x
            dy = charge.y - self.y
            dz = charge.z - self.z

            r_squared = dx**2 + dy**2 + dz**2
            r_cubed = r_squared**(3/2)

            field_x = k * charge.q * dx / r_cubed
            field_y = k * charge.q * dy / r_cubed
            field_z = k * charge.q * dz / r_cubed

            self.efx += field_x
            self.efy += field_y
            self.efz += field_z
        return self.efx, self.efy, self.efz

    def update_motion(self, time):
        self.x += self.q*self.efx*(time**2)/(2*self.m)
        self.y += self.q*self.efy*(time**2)/(2*self.m)
        self.z += self.q*self.efz*(time**2)/(2*self.m)

    def update_position(self, velocity, time):
        self.x += velocity[0] * time
        self.y += velocity[1] * time
        self.z += velocity[2] * time


    def __str__(self):
        return f"Point charge at ({self.x}, {self.y}, {self.z}) with charge" \
               f" {self.q} and mass {self.m}"
