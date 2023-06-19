from scipy.constants import e
from scipy.constants import electron_mass
import shape as shape

# initialize variables
sphere = shape.Sphere(1, 2, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass)
time_intervals = 900

# run simulation
for i in range(time_intervals):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.return_charges_to_sphere()

sphere.project_distribution_2d()
sphere.calculate_charge_density()
