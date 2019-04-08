# -*- coding: utf-8 -*-
""" File: core.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""
import os
import re
import datetime

import numpy as np
import keras.backend as K
from tensorflow import logging

import data_generator as dgen
import data_preprocessing as dprep
import neural_network as nn
from global_params import cfg

# Make keras stfu
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.ERROR)

def write_log(lead, model, r):
    with open(cfg.log_location + "log_" + cfg.t + ".log", 'a') as log, open(cfg.log_location + "log_" + cfg.t + ".csv", 'a') as csvlog:
        log.write("="*65 + "\n")
        log.write("Log entry at: {}\n\n".format(cfg.t))

        log.write("Config:\n")
        log.write("lead\t\t{}\n".format(str(lead)))
        log.write("split\t\t{}\n".format(str(" ".join([str(s) for s in cfg.tvt_split]))))
        log.write("epochs\t\t{}\n".format(str(cfg.epochs)))
        log.write("_"*65 + "\n\n")

        log.write("Data:\n")
        log.write("One ecg per patient:\t{}\n".format(cfg.split_on))
        log.write("Training set size:\t\t{}\n".format(cfg.train_size))
        log.write("Validation set size:\t{}\n".format(cfg.validation_size))
        log.write("Test set size:\t\t\t{}\n".format(cfg.test_size))

        log.write("\nModel:\n")
        model.summary(print_fn=lambda x: log.write(x + "\n"))

        log.write("Results:\n")
        log.write("loss\t\taccuracy\tprecision\trecall\t\tAUC\t\t\tF1\n")
        log.write("{0:4.3f}\t\t{1:4.3f}\t\t{2:4.3f}\t\t{3:4.3f}\t\t{3:4.3f}\t\t{3:4.3f}".format(r[0], r[1], r[2], r[3], r[4], r[5]) + "\n")
        log.write("_"*65 + "\n\n")            

        csvlog.write(",".join([
            cfg.t, str(lead), 
            str(cfg.tvt_split[0]), str(cfg.tvt_split[1]), str(cfg.tvt_split[2]),
            str(cfg.epochs), str(cfg.split_on), 
            str(cfg.train_size), str(cfg.validation_size), str(cfg.test_size), 
            str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5])
        ]) + "\n")

def main():
    all_data_x, all_data_y, fnames = dgen.get_data(
        n_files=100,
        return_fnames = True,
        channels = np.array(range(cfg.n_channels)),
        norm = cfg.normalize_data,
        targets = cfg.targets,
        extension = "." + cfg.file_extension
    )

    if cfg.logging:
        with open(cfg.log_location + "log_" + cfg.t + ".csv", 'a') as csvlog:
            csvlog.write("t,lead,split_train,split_val,split_test,epochs,unique_patients,train_size,validation_size,test_size,loss,accuracy,precision,recall,AUC,F1\n")
    
    for lead in cfg.leads:
        data_x = all_data_x.copy()[:, :, [0, lead]]
        data_y = all_data_y.copy()

        data_x, data_y = dprep.extract_windows(
            data_x, 
            data_y, 
            exclude_first_channel = True, 
            fnames = fnames
        )

        x_train, y_train, x_val, y_val, x_test, y_test = nn.prepare_train_val_data(
            data_x, 
            data_y, 
            cfg.tvt_split, 
            split_on = cfg.split_on,
            patient_ids = [dgen.filename_info(f, "ID") for f in fnames]
        )

        cfg.train_size = x_train.shape[0]
        cfg.validation_size = x_val.shape[0]
        cfg.test_size = x_test.shape[0]

        model = nn.ffnet((cfg.nn_input_size, ))
        nn.train(
            model, x_train, y_train, x_val, y_val, 
            batch_size = cfg.training_batch_size, 
            epochs = cfg.epochs, 
            save = cfg.save_on_train
        )
        r = nn.eval(model, x_test, y_test, batch_size = cfg.evaluation_batch_size)
        r.append((2 * r[2] * r[3]) / (r[2] + r[3]))
        if cfg.verbosity:
            print(
                "loss\t\t", r[0],
                "\naccuracy\t", r[1],
                "\nprecision\t", r[2],
                "\nrecall\t\t", r[3],
                "\nAUC\t\t", r[4],
                "\nF1-score\t", r[5],
            )

        if cfg.logging:
            write_log(lead, model, r)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        K.clear_session()
