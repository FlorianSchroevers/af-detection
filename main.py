# -*- coding: utf-8 -*-
""" File: main.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""
import keras.backend as K
from keras.models import load_model

import data_generator as dgen
import data_preprocessing as dprep
import neural_network as nn
from global_params import cfg
from data_convert import convert_file

import numpy as np
import os
import errno

import tkinter as tk
from tkinter import filedialog

np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_batch_prediction(directory, predictions_fname, predict_on_channel):
    if cfg.model_save_name not in os.listdir("./"):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg.model_save_name + " - run core.py once to generate this file.")
    else:
        model = load_model(cfg.model_save_name, custom_objects={'precision':nn.precision, 'recall':nn.recall})

    fnames = os.listdir(directory)
    n_files = len(fnames)

    data = np.empty(shape=(n_files, int(cfg.sampling_frequency * cfg.interval), 1))

    for i, fname in enumerate(fnames):
        if fname.lower().endswith(".csv"):
            print(i, end="\r")
            ecg = np.loadtxt(
                directory + fname,
                delimiter=cfg.delimiter,
                dtype=int,
                usecols=[predict_on_channel],
                ndmin=2
            )

            data[i, :, :] = ecg

    print("")

    data, _, new_fnames = dprep.extract_windows(data, np.empty(shape=(n_files,)), fnames=fnames)

    predictions = model.predict(np.squeeze(data)).ravel()


    prediction_dict = {}
    for fname, prediction in zip(new_fnames, predictions):
        p = np.round(np.mean(prediction))

        if fname in prediction_dict:
            prediction_dict[fname][0] += p
            prediction_dict[fname][1] += 1
        else:
            prediction_dict[fname] = [p, 1]

    with open(predictions_fname, 'w') as predictions_file:
        predictions_file.write("file,p_beats,total_beats\n")
        for fname, p in prediction_dict.items():
            predictions_file.write(fname + "," + str(p[0]) + "," + str(p[1]) + "\n")

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    file_path = convert_file(file_path)

    ecg = np.loadtxt(
        file_path,
        delimiter=',',
        dtype=int,
        usecols=cfg.leads,
        ndmin=2
    )
    ecg = np.expand_dims(ecg, 0)

    if cfg.model_save_name not in os.listdir("./"):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg.model_save_name + " - run core.py once to generate this file.")
    else:
        model = load_model(cfg.model_save_name, custom_objects={'precision':nn.precision, 'recall':nn.recall})

    data_x, _ = dprep.extract_windows(ecg, [0])
    data_x = np.squeeze(data_x)
    predictions = model.predict(data_x).ravel()
    fname = file_path.split("/")[-1]

    if sum(predictions) > (cfg.min_af_ratio_for_positive_prediction * len(predictions)):
        p = "AF"
    else:
        p = "SR"

    print("Prediction for file '{}': {} ({}/{})".format(fname, p, sum(predictions), len(predictions)))


if __name__ == "__main__":
    try:
        # make_batch_prediction("data/processed_data/", "predictions_sex.csv", 0)
        main()
    except KeyboardInterrupt:
        K.clear_session()
