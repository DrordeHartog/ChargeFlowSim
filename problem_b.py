
from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape


sphere = shape.Sphere(1, 3, [])
n = 200
time_tao = 10**(-3)  # s
sphere.distribute_charges_in_sphere(n, e, electron_mass)
for i in range(100):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges, [0, 0, 0])
    for charge in sphere.charges:
        charge.update_motion(time_tao)
    sphere.check_charges_in_sphere()

