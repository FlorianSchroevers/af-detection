""" File: core.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""

# NOTE fixed: split p to p
# NOTE fixed: test set no 50/50 split
# NOTE: this didnt work to begin with, also fixed

# TODO: fix weird 0.0, 0.0 precision and recall

# TODO: splitting patients in stead of signals
# TODO: running multiple times with different tvt splits
# TODO: Lime for visualization (gradcom)

import keras.backend as K

import data_generator as dgen
import data_preprocessing as dprep
import neural_network as nn
from global_params import cfg
import numpy as np
import os

np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    data_x, data_y, fnames = dgen.get_data(
        return_fnames = True,
        channels = np.array([cfg.lead]),
        norm = True,
        exclude_targets = [2, 3, 4]
    )
    data_x, data_y = dprep.extract_windows(data_x, data_y)

    x_train, y_train, x_val, y_val, x_test, y_test = nn.prepare_train_val_data(
        data_x, 
        data_y, 
        tvt_split=cfg.tvt_split, 
        equal_split_test=cfg.equal_split_test
    )

    model = nn.ffnet((cfg.nn_input_size, ))
    nn.train(
        model, x_train, y_train, x_val, y_val, 
        batch_size=cfg.training_batch_size, 
        epochs=cfg.epochs, 
        save=cfg.save_on_train
    )
    r = nn.eval(model, x_test, y_test, batch_size=cfg.evaluation_batch_size)
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
