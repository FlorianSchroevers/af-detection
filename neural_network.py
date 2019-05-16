# -*- coding: utf-8 -*-
""" File: core.py
    Implements the the neural network model and the data.
Authors: Florian Schroevers, Abel Oakley, Flavio Miceli
"""

from data_generator import get_data, filename_info
import data_preprocessing as dprep
from global_params import cfg
from helpers import progress_bar

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from sklearn.metrics import auc as sklearn_auc

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, concatenate, Conv1D
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
from keras import callbacks
from keras.regularizers import l1, l2

import matplotlib.pyplot as plt
import numpy as np

import time

def split_patients(data_x, data_y, patient_ids, tvt_split):
    # number of inputs
    data_len = data_x.shape[0]

    # make sure the tvt split adds up to one so all the data will be used
    tvt_split = cfg.tvt_split
    if np.sum(tvt_split) != 1:
        tvt_split = np.divide(np.array(tvt_split), np.sum(tvt_split))

    id_set = list(set(patient_ids))
    np.random.shuffle(id_set)

    n_train, n_validation, n_test = np.multiply(np.array(tvt_split), len(id_set)).astype(int)

    train_ids = id_set[:n_train]
    val_ids = id_set[n_train:n_train + n_validation]
    test_ids = id_set[n_train + n_validation:n_train + n_validation + n_test]

    train_idx, validation_idx, test_idx = [], [], []
    for i in range(data_len):
        if patient_ids[i] in train_ids:
            train_idx.append(train_ids)
        elif patient_ids[i] in val_ids:
            validation_idx.append(val_ids)
        elif patient_ids[i] in test_ids:
            test_idx.append(test_ids)


def prepare_train_val_data(data_x, data_y, tvt_split, split_on="", patient_ids=[], return_idx=False):
    """ function : prepare_train_val_data

    splits the data in a training, validation and test set, while maintaining a
    50/50 split of targets in all sets, so that the network won't learn to
    always predict one target

    Args:
        data_x : np.ndarray
            an array of input data
        data_y : np.ndarray
            an array of targets of the data
        tvt_split : list
            a list with three floats that represent the fraction of the size of
            the training, validation and test (tvt) sets respectively
        split_on : str [optional, default: ""]
            can be "", "patient_id" or "label"; on which property to split the 
            different sets on.
            if set to "patient_id", there will be no patient present in multiple sets
            if set to "label", each set will hold 50/50 ratio of healthy and unhealthy samples
            else, the split will be entirely random

    Returns:
        train_x : dict
            a dict containing the data of this set with input name as key and
            data as value
        train_y : np.ndarray
            an array with targets for this set
        validation_x : dict
            a dict containing the data of this set with input name as key and
            data as value
        validation_y : np.ndarray
            an array with targets for this set
        test_x : dict
            a dict containing the data of this set with input name as key and
            data as value
        test_y : np.ndarray
            an array with targets for this set

    """
    # number of inputs
    data_len = data_x.shape[0]

    # make sure the tvt split adds up to one so all the data will be used
    tvt_split = cfg.tvt_split
    if np.sum(tvt_split) != 1:
        tvt_split = np.divide(np.array(tvt_split), np.sum(tvt_split))

    if split_on == "label":
        # get the indices of healthy and unhealthy data
        healthy_idx = np.array([i for i in range(data_len) if data_y[i] == 0])
        unhealthy_idx = np.array([i for i in range(data_len) if data_y[i] != 0])

        # shuffle them so the function will return different ecg's every time
        np.random.shuffle(healthy_idx)
        np.random.shuffle(unhealthy_idx)

        # to make sure the split is 50/50, we maximize the amount of samples to the 
        # smallest of the lengths of the two sets
        data_len = min(len(healthy_idx), len(unhealthy_idx))

        # get the real size of the tvt sets based on the fractions specified by tvt_split
        n_train, n_validation, n_test = np.multiply(np.array(tvt_split), data_len).astype(int)

        # set the indices of the samples each of the tvt sets should take in
        # each set will hold about 50/50 healthy and unhealthy ecg's
        train_idx = np.concatenate([
            healthy_idx[:n_train],
            unhealthy_idx[:n_train]
        ])

        validation_idx = np.concatenate([
            healthy_idx[n_train:n_train + n_validation],
            unhealthy_idx[n_train:n_train + n_validation]
        ])

        if cfg.equal_split_test:
            test_idx = np.concatenate([
                healthy_idx[n_train + n_validation:n_train + n_validation + n_test],
                unhealthy_idx[n_train + n_validation:n_train + n_validation + n_test]
            ])
        else:
            test_idx = np.concatenate([
                healthy_idx[n_train + n_validation:],
                unhealthy_idx[n_train + n_validation:]
            ])

            np.random.shuffle(test_idx)
            test_idx = test_idx[:n_test]

    elif split_on == "patient_id":
        id_set = list(set(patient_ids))
        np.random.shuffle(id_set)

        # print(len(id_set))

        n_train, n_validation, n_test = np.multiply(np.array(tvt_split), len(id_set)).astype(int)

        # print(n_train, n_validation, n_test)

        train_ids = id_set[:n_train]
        val_ids = id_set[n_train:n_train + n_validation]
        test_ids = id_set[n_train + n_validation:n_train + n_validation + n_test]

        train_idx, validation_idx, test_idx = [], [], []
        for i in range(data_len):
            if patient_ids[i] in train_ids:
                train_idx.append(i)
            elif patient_ids[i] in val_ids:
                validation_idx.append(i)
            elif patient_ids[i] in test_ids:
                test_idx.append(i)

        train_idx = np.array(train_idx)
        validation_idx = np.array(validation_idx)
        test_idx = np.array(test_idx)

        # print(train_idx.shape)

    else:
        n_train, n_validation, n_test = np.multiply(np.array(tvt_split), data_len).astype(int)
        idx = [x for x in range(data_len)]
        np.random.shuffle(idx)
        train_idx = idx[:n_train]
        validation_idx = idx[n_train:n_train + n_validation]
        test_idx = idx[n_train + n_validation:n_train + n_validation + n_test]

    if return_idx:
        return train_idx, validation_idx, test_idx


    # apply these indices to the input data to get the actual ecg's
    ecg_train_x = data_x[train_idx, :, :]
    train_y = data_y[train_idx]

    ecg_validation_x = data_x[validation_idx, :, :]
    validation_y = data_y[validation_idx]

    ecg_test_x = data_x[test_idx, :, :]
    test_y = data_y[test_idx]

    # put them in a dictionary. we do this so we can give keras models multiple
    # inputs based on name (the key must be the name of the input layer)
    train_x = {"ecg_inp" : ecg_train_x}
    validation_x = {"ecg_inp" : ecg_validation_x}
    test_x = {"ecg_inp" : ecg_test_x}

    train_x = np.squeeze(train_x["ecg_inp"])
    validation_x = np.squeeze(validation_x["ecg_inp"])
    test_x = np.squeeze(test_x["ecg_inp"])

    return train_x, train_y, validation_x, validation_y, test_x, test_y

def ffnet(ecg_shape, summarize=False):
    """ function: ffnet

    returns model

    Args:
        ecg_shape : tuple
            the shape of the input
        summarize : bool [optional, default:False]
            whether to show a summary of the model
    Returns:
        model : keras.models.Model
            the model
    """
    layer_typedict = {
        "dense":Dense,
        "conv1d":Conv1D
    }

    # activation_typedict = {
    #     "leakyrelu":keras.layers.LeakyReLU,
    #     "conv1d":Conv1D
    # }


    ecg_input = Input(
        shape=(cfg.nn_input_size, ),
        name="ecg_inp"
    )
    net = ecg_input

    for layer in cfg.layers:
        net = layer_typedict[layer.type](
            layer.nodes,
            activation = layer.activation
            # kernel_regularizer = l2(layer.kernel_regularizer),
            # activity_regularizer = l1(layer.activity_regularizer)
        )(net)
        net = Dropout(layer.dropout)(net)

    output = layer_typedict[cfg.output_layer.type](
        1,
        activation=cfg.output_layer.activation
    )(net)

    model = Model(
        [ecg_input],
        output
    )

    opt = Adam(
        lr=cfg.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        decay=cfg.decay,
        amsgrad=False
    )

    model.compile(
        loss = "mse",
        optimizer = opt,
        metrics = [
            "accuracy",
            precision,
            recall
        ]
    )

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    if summarize:
        model.summary()

    return model

def precision(y_true, y_pred):
    """ function : precision

    Precision metric, only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """ function : recall

    Recall metric, only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def train(model, train_x, y_train, x_val, y_val, batch_size=32, epochs=32, plot=False):
    """ function : train

    fit data on a model and return the trained model

    Args:
        model : keras.models.Model
            the model to evaluate
        x_train : dict
            a dictionary mapping input names to actual data
        y_train : np.ndarray
            the targets of the train data
        x_val : dict
            a dictionary mapping input names to actual data
        y_val : np.ndarray
            the targets of the validation data
        batch_size : int [optional, default: 32]
            the size of the batches to be fed into the network
        epochs : int [optional, default: 32]
            the number of epochs (times to run the network)
    Returns:
        r : list
            list of the loss and metrics specified by the model after running
            the model on the test data
    """
    history = model.fit(
        x = train_x,
        y = y_train,
        callbacks = [callbacks.RemoteMonitor(root='http://localhost:9000')], 
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (x_val, y_val),
        verbose = int(cfg.verbosity)
    )

    if cfg.model_save_name != "":
        model.save("model/" + cfg.model_save_name)

    if cfg.save_images:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy (lead {})'.format(cfg.current_lead))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("images/accuracy_lead_{}_{}.png".format(cfg.current_lead, cfg.t))
        plt.clf()
        # plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss (lead {})'.format(cfg.current_lead))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("images/loss_lead_{}_{}.png".format(cfg.current_lead, cfg.t))
        plt.clf()
        # plt.show()
    return model

def eval(model, x_test, y_test, batch_size=32):
    """ function : eval

    evaluate the model on a test set (consisting of pulses)

    Args:
        model : keras.models.Model
            the model to evaluate
        x_test : dict
            a dictionary mapping input names to actual data
        y_test : np.ndarray
            the targets of the test data
        batch_size : int [optional, default: 32]
            the size of the batches to be fed into the network
    Returns:
        r : list
            list of the loss and metrics specified by the model after running
            the model on the test data
    """
    r = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=int(cfg.verbosity))

    predictions = model.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, predictions.ravel())
    fpr, tpr, thresholds_keras = roc_curve(y_test, predictions.ravel())

    # f1 = f1_score(y_test, predictions.ravel())
    f1 = (2 * r[2] * r[3]) / (r[2] + r[3])

    fpr_tpr_auc = sklearn_auc(fpr, tpr)
    pr_auc = sklearn_auc(recall, precision)

    r += [fpr_tpr_auc, pr_auc, f1]

    if cfg.save_images:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area = {:.3f}'.format(fpr_tpr_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (lead {})'.format(cfg.current_lead))
        plt.legend(loc='best')
        plt.savefig("images/roc_aoc_lead_{}_{}.png".format(cfg.current_lead, cfg.t))
        plt.clf()

        plt.figure(1)
        # plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(recall, precision, label='area = {:.3f}'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve (lead {})'.format(cfg.current_lead))
        plt.legend(loc='best')
        plt.savefig("images/pr_aoc_lead_{}_{}.png".format(cfg.current_lead, cfg.t))
        plt.clf()
    return r

def evaluate_model(data_x=[], targets=[], fnames=[], model=None):
    """ function : evaluate_model

    create an evaluation (accuracy) for classification based on a threshold
    at least 'threshold' pulses in an ecg must be classified as unhealthy for the
    whole ecg to be unhealthy

    Args:
        n_ecgs : int or Nonetype [optional, default: None]
            the number of ecg's to base accuracy on
        threshold : int [optional, default: 2]
            the number of pulses that need to be unhealthy for the ecg to be
            labeled as unhealthy
    Returns:
        accuracy : float
            the accuracy of the modeled tested on ecg's
    """

    if len(data_x) == 0:
        data_x, targets, fnames = dgen.get_data(
            return_fnames = True,
            channels = np.array([0]),
            norm = True,
            exclude_targets = [2, 3, 4]
        )

    if model == None:
        model = load_model(cfg.model_save_name, custom_objects={'precision':precision, 'recall':recall})

    # n_correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    predictions = []

    mse = 0

    if cfg.verbosity:
        print("Evaluating model with ECG's")
        start = time.time()

    for i, ecg in enumerate(data_x):
        # print(ecg.shape)
        pulse_data_x, pulse_data_y = dprep.extract_windows(
            np.expand_dims(ecg, axis=0), 
            np.array([targets[i]]),
            cfg.nn_input_size,
            exclude_first_channel = True
        )

        nn_pulse_data_x = {"ecg_inp": np.squeeze(pulse_data_x)}

        # preds = model.predict(nn_pulse_data_x)
        preds = [int(round(pred[0])) for pred in model.predict(nn_pulse_data_x)]
        pred = 1 if sum(preds) >= len(preds) * cfg.min_af_ratio_for_positive_prediction else 0

        mse += (targets[i] - pred)**2

        predictions.append(pred)

        if pred == 1 and targets[i] == 1:
            tp += 1
        elif pred == 0 and targets[i] == 0:
            tn += 1
        elif pred == 1 and targets[i] == 0:
            fp += 1
        elif pred == 0 and targets[i] == 0:
            fn += 1


        progress_bar("Evaluating ECG", i, data_x.shape[0])
    if cfg.verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')

    mse /= len(targets)

    ppv, tpr, thresholds_pr = precision_recall_curve(targets, predictions)
    fpr, tpr, thresholds_roc = roc_curve(targets, predictions)

    fpr_tpr_auc = sklearn_auc(fpr, tpr)
    tpr_ppv_auc = sklearn_auc(tpr, ppv)


    accuracy = (tp + tn)/len(targets)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    # specificity = tn/(fp + tn)
    f1 = (2 * tp)/(2 * tp + fp + fn)

    metrics = [mse, accuracy, precision, recall, fpr_tpr_auc, tpr_ppv_auc, f1]
    print(metrics)
    return metrics

def get_ecg_predictions(model, data, fnames):
    predictions = model.predict(np.squeeze(data)).ravel()

    prediction_dict = {}
    for fname, prediction in zip(fnames, predictions):
        p = np.round(np.mean(prediction))

        if fname in prediction_dict:
            prediction_dict[fname][0] += p
            prediction_dict[fname][1] += 1
        else:
            prediction_dict[fname] = [p, 1]

    return prediction_dict

def find_best_af_ratio(predictions):
    best_acc = 0

    for r in range(1, 20):
        r /= 20
        n_correct = 0
        n_total = len(predictions)


        for fname, v in predictions.items():
            correct_label = filename_info(fname, "TARGET")
            act_ratio = v[0]/v[1]

            if (r >= act_ratio and correct_label == "AF") or (r < act_ratio and correct_label == "SR"):
                n_correct += 1

        if n_correct/n_total > best_acc:
            best_acc = n_correct/n_total
            best_r = r

    return best_r, best_acc

if __name__ == "__main__":
    pass
