import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
from scope_fit_11_22 import scope_fit
import scipy.constants as const
import re

# Get all files in folder
folder_name = "FeF2_84.6K/"
file_ending = '.txt'
file_list = [file for file in os.listdir(folder_name) if file.endswith(file_ending)]

# Get data from highest numbered file
highest_numbered_file = sorted(file_list, key=lambda file: int(re.findall('[0-9]+.txt', file)[0][:-4])).pop()
print(highest_numbered_file)
time_arr, total_count_arr = np.loadtxt(folder_name + highest_numbered_file, delimiter='\t', skiprows=5, unpack=True)

# Aggregate data from all the files
# for file in file_list:
#   count_array = np.loadtxt(folder_name + file, delimiter='\t', skiprows=5, usecols=2, unpack=True)
#   total_count_arr += count_array

# Plot data to check where fits need to happen
fig1, ax1 = plt.subplots()
ax1.scatter(time_arr, total_count_arr, label='Raw counts', marker='.')
ax1.set_title('Counts vs. Time')
ax1.set_yscale("linear")
ax1.set_xscale("linear")
ax1.set_xlabel('Time')
ax1.set_ylabel('# Counts')
ax1.legend(framealpha=1, shadow=True, loc='best')
ax1.grid(alpha=0.25)
fig1.savefig('{}/{}_counts_time.png'.format(folder_name[:-1], folder_name[:-1]))
# fig1.show()
# plt.pause(0.001)
# input("hit[enter] to close all plots")
# plt.close('all')
# exit()

# Define peaks and their approximate positions
number_peaks = 4
fit_range = np.asarray([(0.0025, 0.0075), (0.0030, 0.0080), (0.0150, 0.0200), (0.0150, 0.0200)])
approximate_positions = np.asarray([0.005, 0.006, 0.0174, 0.0180])
approximate_amplitude = np.asarray([-20000, -20000, -20000, -20000])
approximate_width = np.asarray([0.0005, 0.0005, 0.0005, 0.0005])
approximate_offset = 14500 * np.ones(number_peaks)
param_matrix = np.transpose(np.vstack((approximate_positions, approximate_amplitude, approximate_width, approximate_offset)))

# Define and fit Lorentzian
def lorentzian(data, E0, amplitude, Gamma, offset):
  return amplitude * (Gamma / ((data - E0)**2 + (Gamma / 2)**2)) + offset

opt_param_matrix = np.zeros(np.shape(param_matrix))
cov_matrix = np.zeros(shape=(np.shape(param_matrix)[0], np.shape(param_matrix)[1], np.shape(param_matrix)[1]))
chi2_matrix = np.zeros(shape=(number_peaks, 2))
for index in range(number_peaks):
  time_to_fit = [time for time in time_arr if time >= fit_range[index][0] and time <= fit_range[index][1]]
  start_index = np.where(time_arr == time_to_fit[0])[0][0]
  stop_index =  np.where(time_arr == time_to_fit[-1])[0][0] + 1
  counts_to_fit = total_count_arr[start_index:stop_index]
  count_std_dev = np.sqrt(counts_to_fit) # Poission process has variance lambda
  opt_param_matrix[index], cov_matrix[index] = optimize.curve_fit(lorentzian, time_to_fit, counts_to_fit, param_matrix[index], sigma=count_std_dev)
  chi2 = np.sum(np.square(np.divide(np.subtract(counts_to_fit, lorentzian(time_to_fit, *opt_param_matrix[index])), count_std_dev)))
  deg_of_freedom = len(time_to_fit) - np.shape(param_matrix)[1]
  reduced_chi2 = chi2 / deg_of_freedom
  chi2_matrix[index] = [chi2, reduced_chi2]

# Plot data to check fits
fig2, ax2 = plt.subplots()
ax2.errorbar(time_arr, total_count_arr, yerr = np.sqrt(total_count_arr), capsize=2, label='Count Data', fmt='.')
time_points = np.linspace(min(time_arr), max(time_arr), 10000)
for index in range(number_peaks):
  ax2.plot(time_points, lorentzian(time_points, *opt_param_matrix[index]), label=r'$\chi^2$: {:.2g}, $\chi^2_{{\nu}}$: {:.2g}'.format(*chi2_matrix[index]))
ax2.set_title('Counts vs. Time')
ax2.set_yscale("linear")
ax2.set_xscale("linear")
ax2.set_xlabel('Time')
ax2.set_ylabel('# Counts')
ax2.legend(framealpha=1, shadow=True, loc='best')
ax2.grid(alpha=0.25)
fig2.savefig('{}/{}_counts_time_fit.png'.format(folder_name[:-1], folder_name[:-1]))
# fig2.show()
# input("hit[enter] to close all plots")
# plt.close('all')

# Get velocity for peak through difference
num_distinct_peaks = int(number_peaks/2)
(full_amp, full_ang_freq, full_phase, full_shift), (simple_amp, simple_ang_freq)  = scope_fit.cos_params_from_data(plot=True)
velocity_voltage_ratio=0.080523086265892;
peak_velocity_matrix = np.zeros((num_distinct_peaks, 5))
print('Opt param matrix[0]: \n{}'.format(opt_param_matrix))
for index in range(num_distinct_peaks):
  left_peak_time = opt_param_matrix[index][0]
  left_peak_uncertain = np.sqrt(np.diag(cov_matrix[index]))[2]
  right_peak_time = opt_param_matrix[-index - 1][0]
  right_peak_uncertain = np.sqrt(np.diag(cov_matrix[-index - 1]))[2]
  print('Left peak time: {}, right peak time: {}'.format(left_peak_time, right_peak_time))
  time_diff = right_peak_time - left_peak_time
  time_diff_min = (right_peak_time - right_peak_uncertain) - (left_peak_time + left_peak_uncertain)
  time_diff_max = (right_peak_time + right_peak_uncertain) - (left_peak_time - left_peak_uncertain)
  print('time diff: {}, min: {}, max: {}'.format(time_diff, time_diff_min, time_diff_max))
  full_voltage_from_diff = scope_fit.voltage_from_time_diff(time_diff, full_amp, full_ang_freq)
  simple_voltage_from_diff = scope_fit.voltage_from_time_diff(time_diff, simple_amp, simple_ang_freq)
  simple_voltage_max = scope_fit.voltage_from_time_diff(time_diff_min, simple_amp, simple_ang_freq)
  simple_voltage_min = scope_fit.voltage_from_time_diff(time_diff_max, simple_amp, simple_ang_freq)
  simple_velocity_uncertainty = velocity_voltage_ratio * abs(simple_voltage_max - simple_voltage_min) / 2
  print('Velocity uncertainty: {}'.format(simple_velocity_uncertainty))
  # Using the peak for the two fits
  full_peak_velocity = full_voltage_from_diff * velocity_voltage_ratio
  simple_peak_velocity = simple_voltage_from_diff * velocity_voltage_ratio
  # print('velocity_uncertain: {:.2f}'.format(right_peak_uncertain + left_peak_uncertain))
  peak_velocity_matrix[index] = [full_peak_velocity, simple_peak_velocity, left_peak_time, right_peak_time, simple_velocity_uncertainty]

# Fit the full and simple trig function to these velocities
time_points = np.hstack((peak_velocity_matrix[:, 2], peak_velocity_matrix[:, 3]))
full_velocity_points = np.hstack((peak_velocity_matrix[:, 0], peak_velocity_matrix[:, 0]))
simple_velocity_points = np.hstack((peak_velocity_matrix[:, 1], peak_velocity_matrix[:, 1]))
velocity_error = np.hstack((peak_velocity_matrix[:, 4], peak_velocity_matrix[:, 4]))

def full_velocity(time, phase, shift):
  return velocity_voltage_ratio * full_amp * np.cos(full_ang_freq * time + phase) + shift

full_param_list, _ = optimize.curve_fit(full_velocity, time_points, full_velocity_points, p0=[0, 0])
full_chi2 = np.sum(np.square(np.divide(np.subtract(full_velocity_points, full_velocity(time_points, *full_param_list)), velocity_error)))

def simple_velocity(time, phase):
  return velocity_voltage_ratio * simple_amp * np.cos(simple_ang_freq * time + phase)

simple_phase, _ = optimize.curve_fit(simple_velocity, time_points, simple_velocity_points, p0=[-0.1], sigma=velocity_error)
simple_chi2 = np.sum(np.square(np.divide(np.subtract(simple_velocity_points, simple_velocity(time_points, *simple_phase)), velocity_error)))
simple_chi2_no_phase = np.sum(np.square(np.divide(np.subtract(simple_velocity_points, simple_velocity(time_points, 0)), velocity_error)))

# print('Simple phase: {}\nVelocity (phase, shift): {}'.format(*simple_phase, full_param_list))
# print('Simple chi2: {}\n simple chi2 no phase: {}\nVel chi2: {}'.format(simple_chi2, simple_chi2_no_phase, full_chi2))

# Check fit to these points
fig3, ax3 = plt.subplots()

# Plot the velocity-time points given by the difference function
ax3.errorbar(x=time_points, y=full_velocity_points, yerr=velocity_error, label='Velocity from diff', color='red', fmt='.', capsize=2)
ax3.hlines(peak_velocity_matrix[:, 0], peak_velocity_matrix[:, 2], peak_velocity_matrix[:, 3], ls='dashed', lw=0.7)
ax3.errorbar(x=time_points, y=simple_velocity_points, yerr=velocity_error, label='Velocity from diff', color='green', fmt='.', capsize=2)
ax3.hlines(peak_velocity_matrix[:, 1], peak_velocity_matrix[:, 2], peak_velocity_matrix[:, 3], ls='dashed', lw=0.7)

# Plot the functions
time_points = np.linspace(min(time_arr), max(time_arr), 10000)
ax3.plot(time_points, full_velocity(time_points, *full_param_list), label=r'Full voltage fit $\chi^2$ = {:.4f}'.format(full_chi2))
ax3.plot(time_points, simple_velocity(time_points, simple_phase), label=r'Simple voltage fit $\chi^2$ = {:.4f}'.format(simple_chi2))

ax3.set_title('Velocity vs. Time')
ax3.set_yscale("linear")
ax3.set_xscale("linear")
ax3.set_xlabel('Time')
ax3.set_ylabel('Velocity')
ax3.legend(framealpha=1, shadow=True, loc='best')
ax3.grid(alpha=0.25)
# Save .png of plot to folder with file
fig3.savefig('{}/{}_vel_phase_check.png'.format(folder_name[:-1], folder_name[:-1]))
# fig3.show()
# input("hit[enter] to close all plots")
# plt.close('all')

# Map time to energy shift through velocity
gamma_energy = 14.41 * 1000 * const.electron_volt
def energy_shift_from_time(time):
  velocity = simple_velocity(time, *simple_phase)
  shift = (velocity / const.speed_of_light) * gamma_energy
  return shift / const.electron_volt

energy_shift_list = energy_shift_from_time(time_arr)

fig4, ax4 = plt.subplots()
ax4.plot(energy_shift_from_time(time_arr) , total_count_arr)
ax4.set_title(r'Count vs. $\Delta E$')
ax4.set_yscale("linear")
ax4.set_xscale("linear")
ax4.set_xlabel(r'$\Delta E (eV)$')
ax4.set_ylabel('Count')
ax4.legend(framealpha=1, shadow=True, loc='best')
ax4.grid(alpha=0.25)
# Save .png of plot to folder with file
fig4.savefig('{}/{}_energy_shift_vs_count.png'.format(folder_name[:-1], folder_name[:-1]))
fig4.show()
input("hit[enter] to close all plots")
plt.close('all')

exit()

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