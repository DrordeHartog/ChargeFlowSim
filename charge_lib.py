class Charge:
    def __init__(self, q: float, mass:float, x:float, y:float, z:float , index:int):
        self.q = q
        self.m = mass
        self.x = x
        self.y = y
        self.z = z
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

    def calculate_electric_field(self, charges: list):
        k = 8.9875517923 * 10**9  # Coulomb's constant
        total_field_x = 0
        total_field_y = 0
        total_field_z = 0
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

            total_field_x += field_x
            total_field_y += field_y
            total_field_z += field_z

        return total_field_x, total_field_y, total_field_z

    def update_motion(self, electric_field, time):
        self.x += self.q*electric_field[0]*(time**2)/(2*self.m)
        self.y += self.q*electric_field[1]*(time**2)/(2*self.m)
        self.z += self.q*electric_field[2]*(time**2)/(2*self.m)

    def update_position(self, velocity, time):
        self.x += velocity[0] * time
        self.y += velocity[1] * time
        self.z += velocity[2] * time


