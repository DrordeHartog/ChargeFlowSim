from scipy.constants import e, electron_mass
import pandas as pd
import charge as ch
import helper_functions as hf
import seaborn as sns
import matplotlib.pyplot as plt

c = pd.DataFrame()

# initialize variables
time_tao = 10**(-15)  # s
time_intervals = 100

electric_field = [30, 0, 0]  # V/m
v = 0.002  # m/s
dim = 2
charge = ch.Charge(0, 0, 0, 0, -e, electron_mass)
charges = [charge]
initial_position = (0, 0, 0)

paths = []


# create 3 options for the charge's path in the field
for i in range(3):
    # initialize a new path
    charge = ch.Charge(0, 0, 0, 0, -e, electron_mass)
    charges = [charge]
    initial_position = (0, 0, 0)
    data = hf.generate_dataframe([initial_position])

    # generate movement over 100 time intervals
    for j in range(time_intervals):
        # create movement parameters
        charge.calculate_electric_field(charges, electric_field)
        charge.update_motion(time_tao)
        velocity_vec = hf.get_random_velocity(v, dim)
        charge.update_position(velocity_vec, time_tao)
        hf.update_dataframe(data, [charge])
    # calculate drift speed
    v_drift = data['x_pos'].min() / (time_intervals * time_tao)
    v_drift = "{:.2e}".format(v_drift)
    # add the charge's path to the list
    paths.append((v_drift, data))

# plot the 3d graph of the charge's path
path_graph = hf.create_paths_graph(paths)
plt.grid(True)
sns.set_palette("Set2")
sns.set_theme(style='darkgrid')
plt.show()
