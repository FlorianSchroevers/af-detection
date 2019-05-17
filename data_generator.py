# -*- coding: utf-8 -*-
""" File: data_generator.py
    Handles the generation of the data from files.
Authors: Florian Schroevers
"""
import os
import re
import random
import warnings
import time

import numpy as np
from sklearn.preprocessing import scale, normalize

from helpers import progress_bar
# from global_params import cfg
import global_params

# from ecg import ECG

def get_filenames(location, extension, filters):
    fnames = []
    for fname in os.listdir(location):
        if fname.endswith(extension):
            for var, value in filters.items():
                if filename_info(fname, var) == value or filename_info(fname, var) in value:
                    fnames.append(fname)

    return fnames


def filename_info(fname, var, fmt="", sep="_"):
    """ Extract a given var from a filename
        Args:
            fname : str
                the filename to extractan attribute from
            var : str
                which attribute to extract (see filename format 
                in config.json)
            fmt : str [optional, default: ""]
                the formatting of the filename (which information is where)
            sep : str [optional, default: "_"]
                the seperation character in the filenames and fmt
        Returns:
            v : str
                the value of the requested variable from the file name
    """
    if fmt == "":
        fmt = global_params.cfg.fname_format

    vard = fmt.split(sep)
    fnamed = fname.split('.')[0].split(sep)

    try:
        v = fnamed[vard.index(var)]
    except IndexError:
        raise IOError("Filename does not conform to required format (please change filename or format in config.json)")
    return v

def get_data(cfg=None, n_files=None, split=False, channels=[], targets=[], 
            return_fnames=False, randomize_order=False, extension='.csv', 
            n_points=None, include_first_channel=False, location=None, 
            filename_fmt=None, filename_sep="_", verbosity=None, open_files=[]):
    """ function: get_data

    returns data in the directory specified in the helpers.py file

    Args:
        n_files : (Nonetype or int) [optional, default: None]
            the number of samples to return, return all available data if set to
            None
        extension : str [optional, default: '.csv']
            the extension (filtype) of the data. can be anything, as long as
            it's readable by np.loadtxt
        split : (bool or str) [optional, default: False]
            to split data 50/50 into healthy/non-healthy or not (only works if
            target is set to None)
            if set to 'max', the function will determine what the max amount of
            files is while keeping the ration 50/50 (will override n_files)
        channels : (Nonetype or np.array) [optional, default: None]
            indices of channels to return or None for all channels
        targets : (list) [optional, default: []]
            a list of conditions to return
        return_fnames : bool [optional, default: False]
            wheter to return a the filenames of the data
        randomize_order : bool [optional, default: True]
            whether to randomize the order of the data
        n_points : int [optional, default: None]
            the number of data points to exctract
        include_first_channel : bool [optional, default: False]
            whether to return an extra copy of the first channels
            (for determining rpeaks in data from other channels)
        unique_patients : bool [optional, default: False]
            whether to only use one ecg per patient to reduce bias
        location : str or Nonetype [optional, default: None]
            the location to load the data from. if None loads the processed 
            data location specified in the config

    Returns:
        data_x : np.ndarray
            the ecg data itself as a 3D array with shape
            (n_ecgs, ecg_len, n_channels)
        data_y : np.ndarray
            an array of target variables
        files : list [optional]
            a list of all files
    """
    if cfg == None:
        cfg = global_params.cfg

    # if delimiter == None:
    #     delimiter = cfg.delimiter

    if verbosity == None:
        verbosity = cfg.verbosity

    if verbosity:
        print("Assembling data from files...")
        start = time.time()

    if channels == []:
        channels = [x for x in range(cfg.n_channels)]

    n_channels = len(channels)

    if include_first_channel and 0 not in channels:
        channels = [0] + channels
        n_channels += 1

    if location == None:
        location = cfg.processed_data_location

    if targets == []:
        targets = cfg.targets

    # get a list of all filenames
    used_patients = []

    if len(open_files) == 0:
        filters = {}
        if targets:
            filters["TARGET"] = targets
        all_files = get_filenames(location, extension, filters)
    else:
        all_files = open_files

    # set number of files to all files if target number is not specified
    if type(n_files) != int or n_files != len(all_files):
        n_files = len(all_files)

    # handle the case where the data has to be split with specified amount
    if split != "max" and split:
        # all healthy files
        sr_files  = [f for f in all_files if filename_info(f, "TARGET") == "SR"]
        # all non-healthy files
        asr_files = [f for f in all_files if filename_info(f, "TARGET") != "SR"]

        try:
            # try to get a random sample of these files of the amount specified
            files = random.sample(sr_files, int(n_files/2))
            files += random.sample(asr_files, int(n_files/2))
        except ValueError:
            # if thats not possible, the max amount that can still be loaded
            # will be used.
            warnings.warn("Not enough files with given target for requested \
                    amount, continuing with lower amount to maintain split.")
            split = "max"
    # handle the case where as many files as possible have to be gotten but the
    # split must be maintained
    if split == "max":
        sr_files = []
        asr_files = []
        for f in all_files:
            # create lists of healthy and non-healthy files
            if filename_info(f, "TARGET") == "SR":
                # target is sinus rythm
                sr_files.append(f)
            else:
                asr_files.append(f)

        # check which of the two lists is smaller, and set this to the size of
        # the sample that has to be taken from both
        m_files = min([len(sr_files), len(asr_files)])
        # concatenate these samples
        files = random.sample(sr_files, m_files)
        files += random.sample(asr_files, m_files)
        # reset number of files (since the number was found by checking what
        # the max amount is without losing the 50/50 ratio)
        n_files = len(files)
    if not split:
        # if no split is required, just take a random subset of the data
        files = random.sample(all_files, n_files)

    if randomize_order:
        # specified by args
        np.random.shuffle(files)

    if len(files) != n_files:
        warnings.warn("The amount of files loaded is not the same as the amount requested")

    if n_points == None:
        n_points = cfg.n_points

    data_x = np.empty(shape=(n_files, n_points, n_channels))
    data_y = np.zeros(shape=(n_files, ))

    for i, fname in enumerate(files):

        ecg = np.loadtxt(
            location + fname,
            delimiter=cfg.delimiter,
            dtype=np.float32,
            usecols=channels,
            ndmin=2
        )

        if cfg.normalize_data:
            # specified by args
            # divide each value in the ecg by the max of its column
            ecg = ecg / np.amax(np.abs(ecg), axis=0)[None, :]

        data_x[i, :, :] = ecg
        # data_y[i] = 0 if filename_info(fname, "SEX") == "M" else 1
        data_y[i] = getattr(cfg, filename_info(fname, "TARGET")[:2])

        if verbosity:
            progress_bar("Load ECG", i, n_files)
    if verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')
    if return_fnames:
        # specified by args
        return data_x, data_y, files

    return data_x, data_y

def get_data_xml_format():
    all_data_x = np.load("data/npy/final_data.npy")
    all_data_y = np.load("data/npy/final_labels.npy")
    
    print(len(all_data_x))
    print(len(all_data_y))

    data = []

    for patient_id, ecgs in enumerate(all_data_x):
        for ecg_i in range(0, len(ecgs), 8):
            ecg = np.array(ecgs[ecg_i:ecg_i+8])
            data.append(ECG(ecg.T, all_data_y[patient_id], patient_id))

    return np.array(data)

def get_ECG_data(**kwargs):
    """ function: get_ECG_data

    Returns an array of loaded data saved in ECG classes

    See get_data for args
    """
    data_x, data_y, fnames = get_data(**kwargs)

    data = np.empty(shape=(data_x.shape[0]), dtype=ECG)
    for i in range(data_x.shape[0]):
        patient_id = filename_info(fnames[i], "ID")
        date = filename_info(fnames[i], "DATE")
        time = filename_info(fnames[i], "TIME")
        data[i] = ECG(data_x[i, :, :], data_y[i], patient_id, date, time)

    return data

if __name__ == "__main__":
    data = get_ECG_data(return_fnames=True, n_files=100, targets=["AF", "SR"])
    pass

