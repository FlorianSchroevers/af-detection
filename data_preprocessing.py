# -*- coding: utf-8 -*-
""" File: data_loading.py
    Handles the preprocessing of the data.
Authors: Florian Schroevers, Flavio Miceli
Functions:
    savitzky_golay
        smooth data
    fourier
        perform fourier analysis
    generate_dataframe
        generates a dataframe with all extracted features
    fourier_straighten
        straightens an ecg
"""
import data_generator as dgen
from helpers import progress_bar
import data_convert as dcon

import biosppy.signals.ecg as biosppy_ecg
import biosppy.signals.tools as biosppy_tools

import pandas as pd
import numpy as np
import math
import os
import time
import warnings

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.interpolate as interp

def savitzky_golay(y, window_size=51, order=4, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    Note: this code was copied from:
    https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def get_fourier_coefficients(signal, resolution=600):
    """ function: get_fourier_coefficients
    perform fourier series analysis on a signal and return coefficients
    for a function that approximate the wave
    assumes the data is modelled by an even function (f(x) = f(-x)).
    the function on this page is followed (the second one under cosine series)
    https://en.wikipedia.org/wiki/Fourier_sine_and_cosine_series.
    the function that reconstructs this wave is: reconstruct_wave
    Args:
        signal : np.ndarray
            a 1d array containing the signal to perform the analysis on
        resolution : int [optional, default: 600]
            the amount of coefficients to return. the higher the resolution the
            higher the fidelity of the reconstructed wave
    Returns:
        coefficients : np.ndarray
            a 1d array containing the coefficients that model the given data
    """
    # only works if array is 1d
    assert signal.ndim == 1
    linspace = np.linspace(0, 1, len(signal)).reshape((-1, 1))
    pi_n = np.pi * np.arange(resolution).reshape((1, -1))
    tmp1 = np.cos(linspace @ pi_n)
    tmp2 = signal.reshape((-1,1)) * tmp1
    coefficients = (2/len(signal)) * np.sum(tmp2, axis=0)
    return coefficients

def reconstruct_wave(coefficients, signal_length):
    """ function: reconstruct wave
    reconstructs a wave using a given set of coefficients using the function as
    seen on this page (first equation under cosine series)
    https://en.wikipedia.org/wiki/Fourier_sine_and_cosine_series
    Args:
        coefficients : np.ndarray
            a 1d array of coefficients to reconstruct wave from, such as those
            given by the function get_fourier_coefficients
        signal_length : int
            the target length of the reconstructed array
    Returns:
        reconstruction : np.ndarray
            a 1d array of length signal_length that approximates the data of
            which the coefficients were extracted
    """
    x = np.arange(signal_length).reshape((-1,1))
    n = np.arange(len(coefficients)).reshape((1,-1))
    tmp1 = np.cos(((x @ n) * np.pi) / signal_length)
    coefficients[0]  = coefficients[0] / 2
    tmp2 = tmp1 * coefficients
    reconstruction = np.sum(tmp2, axis=1)
    return reconstruction

def fourier_straighten(signal, resolution=20):
    """ function: fourier_straighten
    straighten an ecg using a low resolution to obtain ecg baseline, and then
    straightening the ecg by setting the baseline to zero
    Args:
        signal : np.ndarray
            the ecg signal to straighten
        resolution : int [optional, default: 20]
            the resolution of the baseline. if set to high important details
            from the ecg will be removed, recommended to keep under 30
    Returns:
        corrected_ecg : np.ndarray
            the straightened array
    """
    assert signal.ndim == 1
    c = get_fourier_coefficients(signal, resolution)
    baseline = reconstruct_wave(coefficients=c, signal_length=signal.shape[0])

    corrected_ecg = np.subtract(signal, baseline)
    return corrected_ecg

def preprocess_data(data_x, smooth_window_size=51, smooth_order=4, fourier_baseline_resolution=20, verbosity=False):
    """ function: preprocess_data
    preprocess the data by smoothing and straightening.
    Args:
        data_x : np.ndarray
            the data to preprocess.
    Returns:
        p_data_x : np.ndarray
            preprocessed data
    """
    assert data_x.ndim == 3

    if verbosity:
        print("Preprocessing data...")
        start = time.time()

    p_data_x = np.empty(shape=data_x.shape)

    for i, ecg in enumerate(data_x):
        for channel in range(ecg.shape[1]):
            prepped_channel = savitzky_golay(
                ecg[:, channel],
                window_size = smooth_window_size,
                order = smooth_order
            )
            prepped_channel = fourier_straighten(
                prepped_channel,
                resolution = fourier_baseline_resolution
            )
            p_data_x[i, :, channel] = prepped_channel
        if verbosity:
            progress_bar("Processed", i, data_x.shape[0])
    if verbosity:
        print('\nDone, took ' + str(round(time.time() - start, 1)) + ' seconds')
    return p_data_x

def pulse_scale(pulse, target_size):
    """ function: pulse_scale
    scales an array to a given length, using 1d linear interpolation
    Args:
        pulse : np.ndarray
            the array to scale
        target_size : int
            the size to scale to
    Returns:
        scaled_pulse : np.ndarray
            the scaled pulse
    """
    scaled_pulse = np.empty(shape=(target_size, pulse.shape[1]))
    for i in range(pulse.shape[1]):
        a = np.arange(pulse.shape[0])
        interp_function = interp.interp1d(a, pulse[:, i])
        b = np.linspace(0, pulse.shape[0]-1, target_size)
        scaled_pulse[:, i] = interp_function(b)

    return scaled_pulse

    # interp_function = interp.interp1d(np.arange(pulse.size), pulse)
    # return interp_function(np.linspace(0, pulse.size-1, target_size))


def extract_windows(data_x, data_y, pulse_size, fnames=[], verbosity=False):
    """ function : extract_windows
    extract all pulses from an ecg and scale them to a given size
    Args:
        data_x : np.ndarray
            an array of ECG's
        data_y : np.ndarray
            an array of targets of the ECG's
        pulse_size : int [optional, default: 80]
            the size to scale the pulses to
        exclude_first_channel : bool [optional, default: False]
            whether to extract pulses from the first channel.
            used when only rpeak information is needed from first channel
    Returns:
        pulse_data_x : np.ndarray
            an array of pulses
        pulse_data_y : np.ndarray
            an array of targets of the corresponding pulses
    """
    if verbosity:
        start = time.time()
        print("Extracting and scaling pulses from ECG's...")
    n_samples, n_points, n_channels = data_x.shape

    # if exclude_first_channel:
    #     n_channels = max(n_channels - 1, 1)
    pulses = np.empty(shape=(n_samples*25, pulse_size, n_channels))
    pulse_targets = np.empty(shape=(n_samples*25))
    pulse_n = 0

    new_fnames = []

    for i, ecg in enumerate(data_x):
        # We assume lead 0 is a lead where we can extract rpeaks
        rpeaks = get_rpeaks(ecg.T[0])

        ecg_start = 0
        for rpeak_n in range(1, len(rpeaks) - 1):
            pulse = ecg[rpeaks[rpeak_n]:rpeaks[rpeak_n + 1], :]

            try:
                pulses[pulse_n, :, :] = pulse_scale(pulse, pulse_size)
                pulse_n += 1
                pulse_targets[pulse_n] = data_y[i]
                if fnames:
                    new_fnames.append(fnames[i].split('.')[0] + "_" + str(ecg_start) + ".csv")
                ecg_start += 1
            except:
                pass

        ecg_start = 0
        if verbosity:
            progress_bar("Extracted pulses from ECG", i, n_samples)
    if verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')

    if len(fnames) > 0:
        return pulses[:pulse_n], pulse_targets[:pulse_n], new_fnames
    # make sure the data is of the correct length
    return pulses[:pulse_n], pulse_targets[:pulse_n]







    #     for channel, signal in enumerate(ecg.T):
    #         if len(rpeaks) == 0 or np.isnan(signal.min()):
    #             if len(fnames) > i:
    #                 warnings.warn("Faulty ECG found:: " + fnames[i])
    #             else:
    #                 warnings.warn("Faulty ECG found.")

    #             continue
    #         signal = signal[rpeaks[0]:rpeaks[-1]]

    #         rpeaks = np.subtract(rpeaks, rpeaks[0])

    #         # mean_rpeak_distance = int(np.mean(np.diff(rpeaks))/2) # fkhds
    #         for rpeak_n in range(1, len(rpeaks) - 1):
    #             pulse = signal[rpeaks[rpeak_n]:rpeaks[rpeak_n + 1]]


    #             # pulse_start_index = int((rpeaks[rpeak_n] - mean_rpeak_distance))
    #             # pulse_end_index = int((rpeaks[rpeak_n] + mean_rpeak_distance))
    #             # pulse = signal[pulse_start_index:pulse_end_index]
    #             # import matplotlib.pyplot as plt
    #             # plt.plot(pulse)
    #             # plt.show()

    #             # some pulses result in errors, we simpy ignore them
    #             # NOTE: this may cause bias in the model
    #             try:
    #                 # scale the data
    #                 pulse = pulse_scale(pulse, pulse_size)
    #             except:
    #                 continue

    #             # add to new array
    #             pulses[pulse_n, :, channel] = pulse
    #             pulse_targets[pulse_n] = data_y[i]

    #             if fnames:
    #                 new_fnames.append(fnames[i])

    #     pulse_n += 1
    #     i

    # if len(fnames) > 0:
    #     return pulses[:pulse_n], pulse_targets[:pulse_n], new_fnames
    # # make sure the data is of the correct length
    # return pulses[:pulse_n], pulse_targets[:pulse_n]

def get_rpeaks(ecg):
    """ function: get_rpeaks
    returns an array of indices of the r peaks in a given ecg
    Args:
        ecg : np.ndarray
            an ecg
    Returns:
        rpeaks : np.ndarray
            an array of indices of the r peaks
    """
    try:
        ecg = ecg[:, 0]
    except:
        pass

    filtered, _, _ = biosppy_tools.filter_signal(
        signal = ecg,
        ftype = 'FIR',
        band = 'bandpass',
        order = 150,
        frequency = [3, 45],
        sampling_rate = 500
    )
    rpeaks, = biosppy_ecg.hamilton_segmenter(
        signal = filtered,
        sampling_rate = 500
    )
    # correct R-peak locations
    rpeaks, = biosppy_ecg.correct_rpeaks(
        signal = filtered,
        rpeaks = rpeaks,
        sampling_rate = 500,
        tol = 0.05
    )

    return np.array(rpeaks)

if __name__ == "__main__":
    pass
