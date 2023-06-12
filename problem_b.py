import pandas as pd
from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge
import helper_functions as hf
import shape


# intialize variables
sphere = shape.Sphere(1, 3, [])
n = 200
tao = 10**(-3)  # s
sphere.distribute_charges(n, -e, electron_mass)
# df = hf.generate_dataframe(sphere.distribution)
sphere.project_distribution_3d()
sphere.visualise()
data = hf.generate_dataframe(sphere.distribution)
# run simulation
for i in range(100):
    for charge in sphere.charges:
        charge.calculate_electric_field(sphere.charges)
    for charge in sphere.charges:
        charge.update_motion(tao)
    hf.update_dataframe(data, sphere.charges)
    sphere.return_charges_to_sphere()

sphere.project_distribution_3d()
sphere.visualise()
sphere.print_charges_inside_volume()
