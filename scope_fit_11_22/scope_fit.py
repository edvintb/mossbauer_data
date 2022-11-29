import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize

def cos_params_from_data(plot=False):
  data_path = '/Users/edvinberhan/Desktop/physics191/Mossbauer Spectroscopy/mossbauer_data/scope_fit_11_22/scope_data_11-22.csv'
  time_arr, ch1_arr, ch2_arr = np.loadtxt(data_path, dtype=float, skiprows=16, delimiter=',', usecols=[0, 1, 3], unpack=True)
  def cos_function(data, amp, ang_freq, phase, shift):
    return amp * np.cos(ang_freq * data + phase) + shift
  def simple_cos(data, amp, ang_freq):
    return amp * np.cos(ang_freq * data)
  full_popt, _ = optimize.curve_fit(cos_function, time_arr, ch2_arr, p0=[0.2, 300 , -1, 0])
  simple_popt, _ = optimize.curve_fit(simple_cos, time_arr, ch2_arr, p0=[0.2, 270])
  if plot:
    print('Velocity fit: \n amp: {} \n ang_freq: {}\n phase: {} \n shift: {} \n freq: {}'.format(*full_popt, full_popt[1]/(2 * np.pi)))
    print('Simple velocity fit: \n amp: {} \n ang_freq: {} \n freq: {}'.format(*simple_popt, simple_popt[1]/(2 * np.pi)))
    fig, ax = plt.subplots()
    # ax.plot(time_arr, ch1_arr * 0.1)
    ax.scatter(time_arr, ch2_arr, marker='.')
    time_spaced = np.linspace(min(time_arr), max(time_arr), 10000)
    ax.plot(time_spaced, cos_function(time_spaced, *full_popt), lw = '0.8', color='orange', label='Full cos')
    ax.plot(time_spaced, simple_cos(time_spaced, *simple_popt), color='red', label='Simple cos')
    ax.legend(loc='upper left')
    # print(cos_function(0, *popt))
    fig.show()
    # input("hit [enter] to close plots")
    # plt.close(fig)
    
  return (full_popt, simple_popt)

def voltage_from_time_diff(time_diff, amp, ang_freq):
  voltage = amp * np.cos(0.5 * (2 * np.pi - ang_freq * time_diff))

  fig, ax = plt.subplots()

  ax.plot()


  return voltage

