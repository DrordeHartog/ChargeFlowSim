from scipy.constants import e, electron_mass
import pandas as pd
import charge as ch
import helper_functions as hf
import matplotlib.pyplot as plt
import seaborn as sns


#initialize variables
electric_field = [30, 0, 0]  # V/m
time_tao = 10**(-15)  # s
time_intervals = 100

v = 0.002  # m/s
dim = 2
paths = {}
# create 3 options for the charge's path in the field
for _ in range(3):
    # initialize a new path
    charge = ch.Charge(0, 0, 0, 0, -e, electron_mass)
    charges = [charge]
    initial_position = (0, 0, 0)
    data = hf.generate_dataframe([initial_position], 1)

    # generate movement over 100 time intervals
    for i in range(time_intervals):
        # create movement parameters
        charge.calculate_electric_field(charges, electric_field)
        charge.update_motion(time_tao)
        velocity_vec = hf.get_random_velocity(v, dim)
        charge.update_position(velocity_vec, time_tao)
        hf.update_dataframe(data, [charge])

    #calculate drift speed
    drift_speed = data['x_pos'].min()/(time_intervals*time_tao)
    drift_speed = "{:.2e}".format(drift_speed)
    print(drift_speed)
    # add the charge's path to the list
    paths[drift_speed] = data

# plot the 3d graph of the charge's path
path_graph = hf.create_paths_graph(paths)
plt.grid(True)
sns.set_palette("Set2")
sns.set_theme(style='darkgrid')
# plot the graph of the
plt.show()
