import pandas as pd
from scipy.constants import e
from scipy.constants import electron_mass
import helper_functions as hf
import shape_mod as shape


# intialize variables
sphere = shape.Sphere(1, 3, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass)
df = hf.generate_dataframe(sphere.distribution)
df['in_sphere'] = 1
sphere.visualise()
time_intervals = 900
# time_intervals = 10
# run simulation
for i in range(time_intervals):
    print(i)
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.return_charges_to_sphere()

# sphere.project_distribution_3d()
sphere.recalc_distribution()
sphere.visualise()
# sphere.print_charges_inside_volume()
sphere.plot_percentage_in_sphere()