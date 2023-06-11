
from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape

sphere = shape.Sphere(1, 2, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass, 2)
# df = hf.generate_dataframe(sphere.distribution)
for i in range(200):
    # O(n^2)
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    # O(n)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.check_charges_in_sphere()

sphere.recalc_distribution()
sphere.project_distribution_2d()
