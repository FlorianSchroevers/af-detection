""" File: data_generator.py
    Handles the generation of the data from files.
Authors: Florian Schroevers
"""
from helpers import progress_bar
from global_params import cfg
import os
import numpy as np
import re
import random
from sklearn.preprocessing import scale, normalize

import time

def get_ids():
    """ function: get_ids

    returns a list of all patient id's in the dataset

    Args:
    Returns:
        ids : list
            a list of all patient id's in the dataset

    """
    ids = [fname[:7] for fname in os.listdir(cfg.data_loading_location)]
    return list(set(ids))

def get_ecg_by_id(patient_id, t=None):
    """ function: get_ecg_data_by_id

    returns the contents of an ecg file and its target (rythm), given a data id.
    if the time is not given, a random ecg will be chosen.

    Args:
        patient_id : str
            the id of the patient
        t : str or Nonetype [optional, default: None]
            the time at which the ecg was taken
    Returns:
        tuple (5000x8 np.array, int): a numpy array with the data and the target
                                      (0 if sinus rythm, 1 otherwise)
    """
    filedict = get_time_fname_mapping(patient_id)
    if not t:
        fname = filedict[random.choice(list(filedict.keys()))]
    else:
        fname = filedict[t]

    ecg = np.loadtxt(
        cfg.data_loading_location + fname,
        delimiter=',',
        dtype=int
    )
    target = 0 if 'SR' in fname else 1

    return ecg, target

def get_ecg_fnames(patient_id):
    """ function: get_ecg_fnames

    Returns a list of filenames which are ecg's for a given patient

    Args:
        patient_id : str
            the id of the patient
    Returns:
        fnames : list
            a list containing all filenames with ecg's of this patient
    """
    i = 0
    fnames = [f for f in os.listdir(cfg.data_loading_location) if re.match(patient_id, f)]
    return fnames

def get_times(patient_id):
    """ function: get_times

    Returns a list of times at which ecg's for a given patient are taken

    Args:
        patient_id : str
            the id of the patient
    Returns:
        times : list
            a list containing times at which ecg's of this patient are taken
    """
    files = get_ecg_fnames(patient_id)
    times = [f[14:16] + '-' + f[12:14] + '-' + f[8:12] + ' at ' +  f[17:19] + \
             ':' +  f[19:21] for f in files]
    return times

def get_gender(patient_id):
    """ function: get_gender

    Returns a the gender of the given patient

    Args:
        patient_id : str
            the id of the patient
    Returns:
        gender : str
            'M' if the patient is male, 'F' if female
    """
    files = get_ecg_fnames(patient_id)
    gender = 'M' if 'M' in files[0] else 'F'
    return gender

def get_time_rythm_mapping(patient_id):
    """ function: get_time_rythm_mapping

    Returns a dict mapping the times to the rythm at that time for a given
    patient

    Args:
        patient_id : str
            the id of the patient
    Returns:
        time_rythm_dict : dict
            keys are times ecg's were taken, values are rythms at
            those times
    """
    files = get_ecg_fnames(patient_id)
    times = [f[14:16] + '-' + f[12:14] + '-' + f[8:12] + ' at ' +  f[17:19] \
             + ':' +  f[19:21] for f in files]
    r = ['SR' if 'SR' in f else 'AF' for f in files]
    time_rythm_dict = {times[i]:r[i] for i in range(len(files))}
    return time_rythm_dict

def get_time_fname_mapping(data_id):
    """ function: get_time_fname_mapping

    Returns a dict mapping the times to the filename of the ecg taken at that
    time for a given patient

    Args:
        data_id : str
            the id of the patient
    Returns:
        times_fnames_dict : dict
            keys are times ecg's were taken, values are filenames of
            ecg's taken at those times
    """
    files = get_ecg_fnames(data_id)
    times = [f[14:16] + '-' + f[12:14] + '-' + f[8:12] + ' at ' +  f[17:19] \
             + ':' +  f[19:21] for f in files]
    times_fnames_dict = {times[i]:files[i] for i in range(len(files))}
    return times_fnames_dict

def get_feat_data(df):
    """ function: get_feat_data

    Args:
        df: a pandas dataframe with atleast 7 columns, 6 of which are not params
    Returns:
        features: part of the dataframe that makes up the processed features
                Shape is 6 + number of params by number of ECG's
        targets: targets that belong to those features
                Shape is 1 by number of ECG's
    """
    try:
        # Figures out the number of parameters
        # Assumes 6 columns in the dataframe to be something else!
        params = len(df.columns) - 6

        # Slice everything up to targets, and get targets separatly
        features = df.ix[:,'signal':'par'+str(params)]
        targets = df['target']
    except:
        print("Chris you idiot, you broke the dataframe! Contact Florian to fix it.")
        print("Love, -Flavio")
        print("PS it might have been me who broke stuff.")
        print("PPS if it was me, contact Florian aswell.")

    return features, targets

def get_data(n_files=None, split=False, channels=None, norm=False, exclude_targets=[], return_fnames=False, randomize_order=True, extension='.csv', n_points=5000):
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
        norm : (bool) [optional, default: False]
            normalize the channels
        exclude_targets : (list) [optional, default: []]
            a list of conditions not to return (0: healthy, 1: afib, 2: afl,
            3: svt, 4: unknown)
        return_fnames : bool [optional, default: False]
            wheter to return a the filenames of the data
        randomize_order : bool [optional, default: True]
            whether to randomize the order of the data
        n_points : int [optional, default: 5000]
            the number of data points to exctract
    Returns:
        data_x : np.ndarray
            the ecg data itself as a 3D array with shape
            (n_ecgs, ecg_len, n_channels)
        data_y : np.ndarray
            an array of target variables
        files : list [optional]
            a list of all files
    """
    if cfg.verbosity:
        print("Assembling data from files...")
        start = time.time()

    # handle the number of channels
    if channels == None:
        n_channels = 8
    else:
        n_channels = len(channels)

    """
    SR_ Sine Rythm
        Typical hearthbeat

    AF_ Atrial Fibrillation
        Absent P wave

    AFL Atrial Flutter
        Presence of "flutter waves" resembling p-waves or sawtooths

    SVT Supraventricular Tachycardia
        Paroxysmal SVT, narrow QRS and fast heart rhythm
    """

    # A dict mapping condition name in filename to condition id and other way
    # around
    y_dict = {
        'SR_': 0,
        'AF_': 1,
        'AFL': 2,
        'SVT': 3,
        'XX_': 4,
        0: 'SR_',
        1: 'AF_',
        2: 'AFL',
        3: 'SVT',
        4: 'XX_'
    }

    # get a list of all filenames
    all_files = [
        f for f in os.listdir(cfg.data_loading_location)
            if f.endswith(extension) and y_dict[f[24:27]] not in exclude_targets
        ]

    # set number of files to all files if target number is not specified
    if type(n_files) != int or n_files > len(all_files):
        n_files = len(all_files)

    # handle the case where the data has to be split with specified amount
    if split != "max" and split:
        # all healthy files
        sr_files  = [f for f in all_files if y_dict[f[24:27]] == 0]
        # all non-healthy files
        asr_files = [f for f in all_files if y_dict[f[24:27]] > 0]

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
            if y_dict[f[24:27]] == 0:
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

    assert len(files) == n_files

    data_x = np.empty(shape=(n_files, n_points, n_channels))
    data_y = np.zeros(shape=(n_files, ))

    for i, fname in enumerate(files):
        ecg = np.loadtxt(
            cfg.data_loading_location + fname,
            delimiter=',',
            dtype=int,
            usecols=channels,
            ndmin=2
        )

        if norm:
            # specified by args
            # divide each value in the ecg by the max of its column
            ecg = ecg / np.amax(ecg, axis=0)[None, :]

        data_x[i, :, :] = ecg

        # set target variable (by id, see ydict above)
        data_y[i] = y_dict[fname[24:27]]

        if cfg.verbosity:
            progress_bar("Load ECG", i, n_files)
    if cfg.verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')
    if return_fnames:
        # specified by args
        return data_x, data_y, files
    return data_x, data_y

def get_fe_data():
    """ function: get_fe_data

    get the data from exctracted features from an csv file

    Args:
    Returns:
        x : np.ndarray
            all features as a 2D array
        y : np.ndarray
            all targets as a 1D array

    """
    all_files = [f for f in os.listdir(cfg.data_saving_location_fe) if f.lower().endswith(".csv")]

    cols = [a for a in range(1, 87)]
    cols = np.concatenate((np.array([0, 1, 2, 3, 4, 5, 6, 7, 82]), np.arange(10, 62, 2)))
    df = pd.concat((pd.read_csv(data_loc + f, usecols=cols) for f in all_files))

    plt.matshow(df.corr())
    plt.show()

    x, y = df.iloc[:, 5:-1].values, df.iloc[:, -1].values
    x, y = x.astype(np.float32), y.astype(np.float32)
    x = np.nan_to_num(x)

    x = normalize(x, axis=0, norm='l1')

    return x, y


if __name__ == "__main__":
    pass
