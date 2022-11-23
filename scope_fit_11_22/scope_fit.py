import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize

def cos_params_from_data(plot=False):
  data_path = '/Users/edvinberhan/Desktop/physics191/Mossbauer Spectroscopy/mossbauer_data/scope_fit_11_22/scope_data_11-22.csv'
  time_arr, ch1_arr, ch2_arr = np.loadtxt(data_path, dtype=float, skiprows=16, delimiter=',', usecols=[0, 1, 3], unpack=True)
  def cos_function(data, amp, ang_freq, phase, shift):
    return amp * np.cos(ang_freq * data + phase) + shift
  popt, _ = optimize.curve_fit(cos_function, time_arr, ch2_arr, p0=[0.2, 300 , 0, 0])
  print('Velocity fit: \n amp: {} \n ang_freq: {}\n phase: {} \n shift: {} \n freq: {}'.format(*popt, popt[1]/(2 * np.pi)))
  if plot:
    
    fig, ax = plt.subplots()
    ax.plot(time_arr, ch1_arr * 0.1)
    ax.plot(time_arr, ch2_arr)
    time_spaced = np.linspace(min(time_arr), max(time_arr), 10000)
    ax.plot(time_spaced, cos_function(time_spaced, *popt), lw = '0.8')
    fig.show()
    input("hit [enter] to close plots")
    plt.close(fig)
    
  return popt

def voltage_from_time_diff(time_diff, amp, ang_freq):
  return amp * np.cos(0.5 * (2 * np.pi - ang_freq * time_diff))

