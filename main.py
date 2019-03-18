""" File: main.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""
import keras.backend as K
from keras.models import load_model

import data_generator as dgen
import data_preprocessing as dprep
import feature_extraction as fextract
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

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    file_path = convert_file(file_path)

    ecg = np.loadtxt(
        file_path,
        delimiter=',',
        dtype=int,
        usecols=[cfg.lead],
        ndmin=2
    )
    ecg = np.expand_dims(ecg, 0)

    if cfg.model_save_name not in os.listdir("./"):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg.model_save_name + " - run core.py once to generate this file.")
    else:
        model = load_model(cfg.model_save_name, custom_objects={'precision':nn.precision, 'recall':nn.recall})

    data_x, _ = dprep.extract_windows(ecg, [0])
    data_x = np.squeeze(data_x)
    prediction = model.predict(data_x)
    fname = file_path.split("/")[-1]
    print("Prediction for file '{}': {} ({})".format(fname, "AF" if np.round(np.mean(prediction)) else "SR", np.mean(prediction)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        K.clear_session()
