
from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape


# intialize variables
sphere = shape.Sphere(1, 2, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass)
# df = hf.generate_dataframe(sphere.distribution)

# run simulation
for i in range(200):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.return_charges_to_sphere()

sphere.project_distribution_2d()
