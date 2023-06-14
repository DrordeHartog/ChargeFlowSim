
from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape_mod as shape


# intialize variables
square = shape.Square(1, 2, [])
n = 200
tao = 10**(-3)  # s
square.distribute_charges(n, -e, electron_mass)
square.project_distribution_2d()
time_intervals = 100
# df = hf.generate_dataframe(sphere.distribution)

# run simulation
for i in range(time_intervals):
    # print(i)
    for charge in square.charges:
        charge.calculate_electric_field(square.charges)
    for charge in square.charges:
        charge.update_motion(tao)
    square.return_charges_to_square()

square.project_distribution_2d()
