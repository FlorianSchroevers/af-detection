""" File: feature_extraction.py
    Handles the generation of the data from files.
Authors: Florian Schroevers, Christopher Madsen, Flavio Miceli
"""

import biosppy.signals.ecg as biosppy_ecg
import biosppy.signals.tools as biosppy_tools
import data_preprocessing as dprep
import data_generator as dgen
from helpers import progress_bar
from global_params import cfg

import time
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

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

def get_peak_offset(ecg, rpeaks):
    """ function: get_peak_offset

    calculate the mean of deviations from the horizontal location
    of r peaks of an ecg to their expected position

    Args:
        ecg : np.ndarray
            the ecg
        rpeaks : np.ndarray
            an array containing the indices of the r peaks
    Returns:
        peakdev : int
            The mean of deviations from the r peaks to their expected location
            if spread uniformly over the ecg
    """
    devs = []
    # Expected distance between R peaks
    m_peak_distance = len(ecg)/len(rpeaks)

    # Calculate the deviation of each true peak with the expected location
    for i in range(len(rpeaks)):
        devs.append(abs(rpeaks[i] - m_peak_distance * (i + 1/2) ))

    # Calculate the mean deviation
    peakdev = int(sum(devs)/len(rpeaks))
    return peakdev

def get_heartrate(rpeaks):
    """ function: get_heartrate

    calculate the heartrate of an ecg

    Args:
        rpeaks : np.ndarray
            an array containing the indices of the r peaks
    Returns:
        hr : int
            the heartrate of an ecg
    """
    _, hr = biosppy_tools.get_heart_rate(
        beats=rpeaks,
        sampling_rate=500,
        smooth=True,
        size=3
    )
    return np.mean(hr)

def get_ppeaks(ecg):
    """ function: get_ppeaks

    Detects P-peaks in an ECG by looking for max values inside windows between
    the R-peaks, but takes the T-peak into regard as well as missing P-peaks.
    Returns the amount of P-peaks and a number between 0 and 1 representing the
    relation between the mean of the P-peaks and the mean of the R-peaks.

    Args:
        data_x : np.ndarray
            3D array with ECG data (should be smoothed)

    Returns:
        mean_peak_height : 1D array
            array of the means of the p-peaks in ECGs
        n_ppeaks : 1D array
            array of the number of p-peaks in ECGs
    """
    r_peaks = get_rpeaks(ecg)
    p_list = []

    # Get the points in the middle of each two R-peaks
    windows = [np.int(np.around((r_peaks[i] + r_peaks[i+1])/2)) for i in range(len(r_peaks)-1)]

    # Get the windows between each halfpoint and a little bit before the next R-peak
    for j in range(len(windows)):
        window = ecg[windows[j]:r_peaks[j+1]-30]

        # If there is no larger value (T-peak) detected before the current window,
        # The current window is considered not to have a P-peak.

        # TODO: This currently works well enough to have a R to P ratio of 0.6 for healthy,
        #       and 0.4 for unhealthy ECG's. This should be 1.0 for healthy though.
        #       it currently works well enough as a feature, but it's not a solid P detector.
        if np.max(ecg[r_peaks[j]+30:windows[j]]) > np.max(window):
            p_list.append(np.max(window))

    # Mean ratio of P-peak height to R-peak height
    mean_peak_height = np.mean(p_list)/np.mean(ecg[r_peaks])

    # Number of found P-peaks
    n_ppeaks = len(p_list)
    return n_ppeaks, mean_peak_height


def show_correlation_heatmap(df):
    """ function: show_correlation_heatmap

    shows a correlation heatmap of a given dataframe

    Args:
        df : pd.DataFrame
            the dataframe to calculate the correlation matrix of
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.matshow(np.abs(corr), cmap=cm.Spectral_r)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);

    plt.show()

def load_extracted_features(fname):
    """ function : load_extracted_features

    load the extracted features from the location specified in global_params.py

    Args:
        fname : str
            the name of the file to load
    Returns:
        df : pd.DataFrame
            a dataframe with the features


    """
    return pd.read_csv(cfg.data_saving_location_fe + fname, index_col=0)

def extract_features(data_x, data_y, smooth=False, fourier_resolution=600, save_to_file=''):
    """ function: extract_features

    creates a dataframe from a given collection of ecgs containing the fourier
    series coefficients, heartrate and peak offset

    Args:
        data_x : np.ndarray
            a set of ecgs. must only contain one channel per ecg. can be 1d, 2d
            or 3d
        data_y : np.ndarray
            the targets of the ecg's
        smooth : bool [optional, default: False]
            whether to smooth the data
        fourier_resolution : int [optional, default: 600]
            the amount of coefficients to return. the higher the resolution the
            higher the fidelity of the reconstructed wave
        save_to_file : str [optional, default: '']
            saves to this file if the length of this string > 0, with this as
            filename
    Returns:
        df : pd.DataFrame
            a dataframe with the extracted features

    """
    if data_x.ndim == 1:
        data_x = np.expand_dims(data_x, 0)
    if data_x.ndim == 2:
        data_x = np.expand_dims(data_x, 0)
    if cfg.verbosity:
        print("Extracting fourier series coefficients and other features...")
        start = time.time()

    columns = ['par' + str(i) for i in range(fourier_resolution)]
    columns += ['target']
    # more features here
    df = pd.DataFrame(columns=columns)

    for i, ecg in enumerate(data_x):
        if smooth:
            ecg = dprep.savitzky_golay(ecg[:, 0], window_size=51, order=4)
        else:
            ecg = ecg[:, 0]

        coefficients = dprep.get_fourier_coefficients(ecg, fourier_resolution)
        target = np.array([data_y[i]])
        # rpeaks = get_rpeaks(ecg)
        # heartrate = np.array([get_heartrate(rpeaks)])
        # peak_offset = np.array([get_peak_offset(ecg, rpeaks)])
        # n_ppeaks, mean_ppeak = get_ppeaks(ecg)
        # n_ppeaks = np.array([n_ppeaks])
        # mean_ppeak = np.array([mean_ppeak])

        features = np.concatenate([
            coefficients, target
        ])

        df = df.append(pd.Series(features, index=columns), ignore_index=True)
        if cfg.verbosity:
            progress_bar("Extracting features from ECG/pulse", i, data_x.shape[0])
    if cfg.verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')

    if save_to_file:
        if '.csv' not in save_to_file:
            save_to_file = save_to_file + '.csv'
        df.to_csv(cfg.data_saving_location_fe + save_to_file)

    return df

if __name__ == "__main__":
    pass
