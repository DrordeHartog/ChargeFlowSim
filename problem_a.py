from numpy import linspace, meshgrid
from scipy.constants import e
from scipy.constants import electron_mass
import pandas as pd
import charge as ch
import helper_functions as hf
import shape

c = pd.DataFrame()


electric_field = [30, 0, 0]  # V/m
time_tao = 10**(-15)  # s
v = 0.002  # m/s
dim = 2
charge = ch.Charge(0, 0, 0, 0, -e, electron_mass)
charges = [charge]
initial_position = (0, 0, 0)
data = hf.generate_dataframe([initial_position], 1)
for i in range(100):
    charge.calculate_electric_field(charges, electric_field)
    charge.update_motion(time_tao)
    velocity_vec = hf.get_random_velocity(v, dim)
    charge.update_position(velocity_vec, time_tao)
    # update dataframe here (charge.x, charge.y, charge.z, time = i)
    hf.update_dataframe(data, [charge])

# plot the 3d graph of the charge's path
path_graph = hf.plot_charge_path(data)
# plot the graph og the
hf.plt.show()

