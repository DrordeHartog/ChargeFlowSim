
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
sphere.distribute_charges_2d(n, -e, electron_mass)
# df = hf.generate_dataframe(sphere.distribution)
for i in range(100):
    # O(n^2)
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges, [0, 0, 0])
    # O(n)
    for charge in sphere.charges:
        charge.update_motion(tao)
    sphere.check_charges_in_sphere()

sphere.reset_distribution()
sphere.project_distribution_2d()
