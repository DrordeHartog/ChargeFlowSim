from scipy.constants import e
from scipy.constants import electron_mass
import shape as shape


# initialize variables
sphere = shape.Sphere(1, 3, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass)
time_intervals = 1000
# run simulation
for i in range(time_intervals):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.return_charges_to_sphere()
    print(sum(charge.in_sphere for charge in sphere.charges))

sphere.recalc_distribution()
sphere.project_distribution_3d()
sphere.plot_percentage_in_sphere()
