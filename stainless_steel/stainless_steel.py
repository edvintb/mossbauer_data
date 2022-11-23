import numpy as np
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from scipy import constants as const
from decimal import Decimal
import math

# Use 

gamma_energy = 14.4 * 1000 * const.electron_volt
print(const.Planck)
period = const.Planck / gamma_energy
print('Period: {}'.format(period))
wavelength = const.Planck * const.speed_of_light / gamma_energy
print('Wavelength: {}'.format(wavelength))
print('Period 2: {}'.format(wavelength / const.speed_of_light))

scope_file = '../T0001.CSV'

scope_time, pulse_channel, transducer_channel = np.loadtxt(scope_file, dtype=float, delimiter=',', usecols=[0, 1, 3], skiprows=16, unpack=True)

pulse_height = np.max(pulse_channel)
pulse_time = scope_time[np.argmax(pulse_channel)]

# Plot data from oscilloscope
plot_stop_index = 100000
scope_fig, scope_ax = plt.subplots()
scope_ax.plot(scope_time[:plot_stop_index], pulse_channel[:plot_stop_index] * 0.1)
scope_ax.plot(scope_time[:plot_stop_index], transducer_channel[:plot_stop_index])
scope_ax.scatter(pulse_time, pulse_height, color='red')
scope_fig.show()

file_ending = ".txt"

# Extract data from all files
file_list = [file for file in os.listdir() if file.endswith(file_ending) and file != 'stainless_steel_30um_96h_96.txt']

time_array, count_array = np.loadtxt('stainless_steel_30um_96h_96.txt', delimiter='\t', skiprows=5, unpack=True)

total_count = np.zeros(count_array.shape)

for file in file_list:
  count_array = np.loadtxt(file, delimiter='\t', skiprows=5, usecols=2, unpack=True)
  total_count += count_array

# Define velocity computation from time
def velocity(time):
  amplitude_of_voltage=1.39944948e-01;
  vert_shift_voltage=1.49262092e-03; 
  freq_of_voltage=2.71754996e+02; # Hz
  phase_shift=5.69070343e-41;
  velocity_voltage_ratio=0.080523086265892;
  return velocity_voltage_ratio * amplitude_of_voltage*np.cos(freq_of_voltage*time + phase_shift) + vert_shift_voltage

# Velocity decimal
def velocity_decimal(time):
  amplitude_of_voltage=Decimal(1.39944948e-01);
  vert_shift_voltage=Decimal(1.49262092e-03); 
  freq_of_voltage=Decimal(2.71754996e+02); # Hz
  phase_shift=Decimal(5.69070343e-41);
  velocity_voltage_ratio=Decimal(0.080523086265892);
  return Decimal(velocity_voltage_ratio * amplitude_of_voltage * Decimal(math.cos(freq_of_voltage*time + phase_shift)) + vert_shift_voltage)

# Set time = 0 to the time where the pulse is sent
def create_acc_array(time_array, period):
  new_velocity_array = np.zeros(len(time_array))
  acc_array = np.zeros(len(time_array))
  for index, time in enumerate(time_array):
    new_velocity_array[index] = (velocity_decimal(Decimal(time) + Decimal(pulse_time)) + velocity_decimal(Decimal(time) + Decimal(pulse_time) + Decimal(period))) / 2
    acc_array[index] = (velocity_decimal(Decimal(time) + Decimal(pulse_time)) - velocity_decimal(Decimal(time) + Decimal(pulse_time) + Decimal(period))) / Decimal(period)
  print(acc_array)
  return new_velocity_array

velocity_array = velocity(time_array + pulse_time)
velocity_1, velocity_2 = np.split(velocity_array, 2)
count_1, count_2 = np.split(total_count, 2)

vel_acc_array = create_acc_array(time_array, period)

fig, ax = plt.subplots()

ax.plot(velocity_1, count_1, label='1st iteration')
ax.plot(velocity_2, count_2, label='2nd iteration')
ax.plot(vel_acc_array, total_count, label='With acceleration')
ax.set_title('Counts vs. Velocity')
ax.set_yscale("linear")
ax.set_xscale("linear")
ax.set_xlabel('Velocity')
ax.set_ylabel('# Counts')
ax.legend(framealpha=1, shadow=True, loc='best')
ax.grid(alpha=0.25)
# Save .png of plot to folder with file
fig.show()
# plt.pause(0.001)
input("hit[enter] to close all plots")
plt.close('all')

# total_number_of_elements=length(time_for_voltage);

exit()

# regex to extract relevant values from xrdml files
reg_start_angle = "<startPosition>(.*)</startPosition>"
reg_end_angle = "<endPosition>(.*?)</endPosition>"
reg_counts = "<counts unit=\"counts\">(.*?)</counts>"

# Map all files to their data
for folder in xrdml_folder_file_dict:
  file_data_dict = {}
  for file in xrdml_folder_file_dict[folder]:
    with open(folder + "/" + file + file_ending, 'r') as xrdml_file:
        # Read file data
        file_text = xrdml_file.read()
        # Extract relevant values
        counts = re.findall(reg_counts, file_text)
        start_angle_list = re.findall(reg_start_angle, file_text)
        end_angle_list = re.findall(reg_end_angle, file_text)
        count_list = [float(count) for count in counts[0].split(' ')]
        start_angle = float(start_angle_list[0])
        end_angle = float(end_angle_list[0])
        angle_list = np.linspace(start=start_angle, stop=end_angle, num=len(count_list))
        # Put data in dictionary
        file_data_dict[file] = (angle_list, count_list)
  xrdml_folder_file_dict[folder] = file_data_dict

# Sort folders in desired order
doping_type = 'Nd'
def folder_sort(folder_name):
  doping_reg = '{}[0-9]+'.format(doping_type)
  doping = re.findall(doping_reg, folder_name)
  return doping

sorted_folder_dict = sorted(xrdml_folder_file_dict, key=folder_sort)

# Create comparison figures
total_fig, total_ax = plt.subplots()
total_fig_title = 'XRD Comparison'
total_ax.set_yscale('log')
total_ax.set_xscale('linear')
total_ax.set_xlabel('Angle (${}^\circ$)')
total_ax.set_ylabel('Intensity (counts/second)')
total_ax.set_title(total_fig_title)
total_ax.grid(alpha=0.25)
for peak in interesting_peaks:
  total_ax.axvline(interesting_peaks[peak], 0, 1, label=peak, ls='--', lw='0.5')

# Plot the data from each file in sorted order
for folder in sorted_folder_dict:
  for file in xrdml_folder_file_dict[folder]:
    angle_values = xrdml_folder_file_dict[folder][file][0]
    intensity_values = xrdml_folder_file_dict[folder][file][1]

    fig, ax = plt.subplots()
    ax.plot(angle_values, intensity_values, label=folder[:5])

    # Only plot vertical lines in xrd files
    if 'tth' in file:
      total_ax.plot(angle_values, intensity_values, label=folder[:5])
      for peak in interesting_peaks:
        ax.axvline(interesting_peaks[peak], 0, 1, label=peak, ls='--', lw='0.5')

    
    ax.set_title(file)
    ax.set_yscale("log")
    ax.set_xscale("linear")
    ax.set_xlabel('Angle (${}^\circ$)')
    ax.set_ylabel('Intensity (counts/second)')
    ax.legend(framealpha=1, shadow=True)
    ax.grid(alpha=0.25)
    # Save .png of plot to folder with file
    fig.savefig("{}/{}.png".format(folder, file))
    fig.show()
    # plt.pause(0.001)

# Label both by sample ID and by doping

total_ax.legend(loc='best')
total_fig.savefig('{}_{}.png'.format(total_fig_title, doping_type))
total_fig.show()

input("hit[enter] to close all plots")
plt.close('all')