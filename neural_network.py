# -*- coding: utf-8 -*-
""" File: core.py
    Implements the the neural network model and the data.
Authors: Florian Schroevers, Abel Oakley, Flavio Miceli
"""

from data_generator import get_data, get_fe_data
from global_params import cfg

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, concatenate, Conv1D
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

import numpy as np

def prepare_train_val_data(data_x, data_y, feature_data=None, tvt_split=[0.6, 0.3, 0.1], equal_split_test=False, split_ids=[]):
    """ function : prepare_train_val_data

    splits the data in a training, validation and test set, while maintaining a
    50/50 split of targets in all sets, so that the network won't learn to
    always predict one target

    Args:
        data_x : np.ndarray
            an array of input data
        data_y : np.ndarray
            an array of targets of the data
        feature_data : pandas.DataFrame
            a dataframe with any additional extracted features
        tvt_split : list
            a list with three floats that represent the fraction of the size of
            the training, validation and test (tvt) sets respectively
        equal_split_test : bool
            whether to split the test set 50/50
        split_ids : np.array [optional, default: []]
            array containing ids to split the data by so each id occurs in only one 
            of the set
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

    np.random.shuffle(train_idx)
    np.random.shuffle(validation_idx)
    np.random.shuffle(test_idx)

    # we need to cut the amount of test samples short to ensure tvt split 
    # is still accurate (in case test set is not 50/50 SR/AF)
    test_idx = test_idx[:n_test]

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

    # do the same stuff for feature data if its given
    if type(feature_data) != type(None):
        # drop the targets, they won't be used to train on!
        feature_data = feature_data.drop('target', axis=1).values.astype(np.float32)

        fe_train_x = feature_data[train_idx]
        fe_validation_x = feature_data[validation_idx]
        fe_test_x = feature_data[test_idx]

        train_x["features_inp"] = fe_train_x
        validation_x["features_inp"] = fe_validation_x
        test_x["features_inp"] = fe_test_x

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

    MaxPooling 1D:
        Arguments:
            - pool_size: Integer, size of the max pooling windows.
            - strides: Integer, or None. Factor by which to downscale.
            - padding: One of 'valid' or 'same'

    Conv1D:
        Arguments:
            - filters: integer, the dimensionality of the output space.
            - kernel_size: An integer or tuple/list of a single integer,
              specifying the length of the 1D convolution window.
            - activation: Activation function to use.

    Dense:
        Arguments:
            - units: Positive integer, dimensionality of the output space.
            - activation: Activation function to use.

    Dropout:
        Arguments:
            - rate: float between 0 and 1. Fraction of the input units to drop.

    """
    layer_typedict = {
        "dense":Dense,
        "conv1d":Conv1D
    }

    ecg_input = Input(              # Input-Layer
        shape=(cfg.nn_input_size, ),            # Ecg input
        name="ecg_inp"
    )
    net = ecg_input

    for layer in cfg.layers:
        net = layer_typedict[layer.type](
            layer.nodes,
            activation = layer.activation
        )(net)
        net = Dropout(layer.dropout)(net)

    output = layer_typedict[cfg.output_layer.type](                 # Output-Layer
        1,                          # Dim Output Space: 1
        activation=cfg.output_layer.activation      # Activation Function: Sigmoid
    )(net)

    model = Model(                  # Create Model
        [ecg_input],
        output
    )

    opt = Adam(                     # Optimizer: Adam
        lr=cfg.learning_rate,
        beta_1=0.9,                 # Beta-1, 0 < Beta < 1: 0.9
        beta_2=0.999,               # Beta-2, 0 < Beta < 1: 0.999
        decay=cfg.decay,
        amsgrad=False               # Apply AMSGrad Variant: False
    )

    model.compile(                  # Compiler
        loss = "mse",               # Loss: MSE
        optimizer = opt,            # Optimizer: Opt
        metrics = [
            "accuracy",
            precision,
            recall
        ]
    )

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    if summarize:
        model.summary()                 # Model Summary

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

def train(model, train_x, y_train, x_val, y_val, batch_size=32, epochs=32, save=False, plot=False):
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
        save : bool [optional, default: False]
            whether to train the saved network
    Returns:
        r : list
            list of the loss and metrics specified by the model after running
            the model on the test data
    """
    history = model.fit(
            x = train_x,
            y = y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (x_val, y_val),
            verbose = int(cfg.verbosity)
        )

    if save:
        model.save(cfg.model_save_name)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
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
    return r

def evaluate_model(n_ecgs=None, threshold=2):
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
    data_x, targets, fnames = dgen.get_data(
        n_files = n_ecgs,
        return_fnames = True,
        channels = np.array([0]),
        norm = True,
        exclude_targets = [2, 3, 4]
    )
    model = load_model(cfg.model_save_name, custom_objects={'precision':precision, 'recall':recall})
    n_correct = 0

    if cfg.verbosity:
        print("Evaluating model with ECG's")
        start = time.time()

    for i, ecg in enumerate(data_x):
        pulse_data_x, pulse_data_y = dprep.extract_windows(np.expand_dims(ecg, axis=0), np.array([targets[i]]))

        nn_pulse_data_x = {"ecg_inp": np.squeeze(pulse_data_x)}

        preds = [int(round(pred[0])) for pred in model.predict(nn_pulse_data_x)]
        pred = 1 if sum(preds) >= threshold else 0
        if pred == int(targets[i]):
            n_correct += 1

        cfg.progress_bar("Evaluating ECG", i, data_x.shape[0])
    if cfg.verbosity:
        print('Done, took ' + str(round(time.time() - start, 1)) + ' seconds')

    accuracy = n_correct/len(targets)
    return accuracy

if __name__ == "__main__":
    pass
