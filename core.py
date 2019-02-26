""" File: core.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""

# TODO: test set no 50/50 split
# TODO: splitting patients in steaad of signals
# TODO: running multiple times with different tvt splits
# TODO: Lime for visualization (gradcom)

import keras.backend as K

import data_generator as dgen
import data_preprocessing as dprep
import feature_extraction as fextract
import neural_network as nn
from global_params import cfg
import numpy as np
import os

np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    data_x, data_y, fnames = dgen.get_data(
        verbose = cfg.verbosity,
        return_fnames = True,
        channels = np.array([cfg.lead]),
        norm = True,
        exclude_targets = [2, 3, 4]
    )
    data_x, data_y = dprep.extract_windows(data_x, data_y, verbose=cfg.verbosity)

    x_train, y_train, x_val, y_val, x_test, y_test = nn.prepare_train_val_data(data_x, data_y)
    x_train = np.squeeze(x_train["ecg_inp"])
    x_val = np.squeeze(x_val["ecg_inp"])
    x_test = np.squeeze(x_test["ecg_inp"])

    model = nn.ffnet(x_train.shape[1:])
    nn.train(model, x_train, y_train, x_val, y_val, batch_size=32, save=cfg.save_on_train)
    r = nn.eval(model, x_test, y_test)
    print(
        " loss:", r[0], '\n',
        "accuracy:", r[1], '\n',
        "precision:", r[2], '\n',
        "recall:", r[3],
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        K.clear_session()
