# -*- coding: utf-8 -*-
""" File: core.py
    Main file of the DeepLearningDoc project.
Authors: Florian Schroevers
"""
import os
import re
import datetime
import math

import numpy as np
import keras.backend as K
import tensorflow as tf
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
        log.write("Split on:\t{}\n".format(cfg.split_on))
        log.write("Training set size:\t\t{}\n".format(cfg.train_size))
        log.write("Validation set size:\t{}\n".format(cfg.validation_size))
        log.write("Test set size:\t\t\t{}\n".format(cfg.test_size))

        log.write("\nModel:\n")
        model.summary(print_fn=lambda x: log.write(x + "\n"))

        log.write("Results:\n")
        log.write("loss\taccuracy\tprecision\trecall\tROC-AUC\tPR-AUC\tF1\n")
        log.write("{0:4.3f}\t{1:4.3f}\t\t{2:4.3f}\t\t{3:4.3f}\t{3:4.3f}\t{3:4.3f}\t{3:4.3f}".format(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) + "\n")
        log.write("_"*65 + "\n\n")            

        csvlog.write(",".join([
            cfg.t, str(lead), 
            str(cfg.tvt_split[0]), str(cfg.tvt_split[1]), str(cfg.tvt_split[2]),
            str(cfg.epochs), str(cfg.split_on), 
            str(cfg.train_size), str(cfg.validation_size), str(cfg.test_size), 
            str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6])
        ]) + "\n")

def generator(location, fnames, lead):
    for fname in fnames:
        data_x, data_y = dgen.get_data(location=location, open_files=[fname], verbosity=False)
        yield data_x[:, :, lead], data_y

def main2():
    filters = {"TARGET": cfg.targets}
    fnames = dgen.get_filenames(cfg.pulse_data_location, '.' + cfg.file_extension, filters)

    train_idx, validation_idx, test_idx = nn.prepare_train_val_data(
        np.empty(shape=(len(fnames), )), 
        np.empty(shape=(len(fnames), )), 
        cfg.tvt_split,
        split_on="",
        patient_ids=[dgen.filename_info(fname, "ID") for fname in fnames],
        return_idx=True
    )

    train_fnames = np.array(fnames)[train_idx]
    validation_fnames = np.array(fnames)[validation_idx]
    test_fnames = np.array(fnames)[test_idx]

    print(len(train_fnames))
    print(len(validation_fnames))
    print(len(test_fnames))

    # print()
    # quit()

    for lead in cfg.leads:
        model = nn.ffnet((cfg.nn_input_size, ))
        model.fit_generator(
            generator(cfg.pulse_data_location, train_fnames, lead),
            steps_per_epoch = math.floor(len(train_fnames) / cfg.epochs),
            epochs = cfg.epochs,
            validation_data = generator(cfg.pulse_data_location, validation_fnames, lead),
            validation_steps = math.floor(len(validation_fnames) / cfg.epochs)
        )

        r = model.evaluate_generator(
            generator(cfg.pulse_data_location, test_fnames, lead),
            steps = len(test_fnames)
        )

        print(r)
        with open("results.txt", 'w') as fout:
            fout.write(",".join(list(r)) + "\n")


def run_training_session(all_data_x, all_data_y, model_save_name, fnames, lead):
    cfg.current_lead = lead
    cfg.model_save_name = "lead" + str(lead) + model_save_name

    data_x = all_data_x.copy()[:, :, (lead,)]
    data_y = all_data_y.copy()

    print(data_x.shape)

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
        epochs = cfg.epochs
    )

    r = nn.eval(model, x_test, y_test, batch_size = cfg.evaluation_batch_size)
    if cfg.verbosity:
        print(
            "loss\t\t", r[0],
            "\naccuracy\t", r[1],
            "\nprecision\t", r[2],
            "\nrecall\t\t", r[3],
            "\nROC-AUC\t\t", r[4],
            "\nPR-AUC\t\t", r[5],
            "\nF1-score\t", r[6],
        )

    # prediction_dict = nn.get_ecg_predictions(model, data_x, fnames)
    # af_r, acc = nn.find_best_af_ratio(prediction_dict)
    # af_r_f.write(str(af_r) + "\n")

    # print(af_r, acc)

    # act_r = nn.evaluate_model(ecg_data_x, ecg_data_y, ecg_fnames, model)

    if cfg.logging:
        write_log(lead, model, r)

def main():
    data_x, data_y, ecg_fnames = dgen.get_data(
        # n_files=10,
        location=cfg.pulse_data_location,
        return_fnames = True,
        channels = np.array(range(cfg.n_channels)),
        # norm = cfg.normalize_data,
        targets = cfg.targets,
        extension = "." + cfg.file_extension
    )

    # data_x, data_y, fnames = dprep.extract_windows(
    #     all_ecg_data_x, 
    #     all_ecg_data_y,
    #     cfg.nn_input_size,
    #     fnames = ecg_fnames,
    #     verbosity = cfg.verbosity
    # )

    if cfg.logging:
        with open(cfg.log_location + "log_" + cfg.t + ".csv", 'a') as csvlog:
            csvlog.write("t,lead,split_train,split_val,split_test,epochs,split_on,train_size,validation_size,test_size,loss,accuracy,precision,recall,ROC-AUC,PR-AUC,F1\n")

    model_save_name = cfg.model_save_name

    af_r_f = open('model/af_ratio_predictor.txt', 'w')
    for lead in cfg.leads:
        run_training_session(data_x, data_y, model_save_name, ecg_fnames, lead)

    af_r_f.close()


if __name__ == "__main__":
    try:
        sess = tf.Session(config=tf.ConfigProto())
        K.set_session(sess)
        main2()
    except KeyboardInterrupt:
        K.clear_session()
