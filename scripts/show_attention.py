import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, concatenate, Conv1D, merge, Multiply
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data_preprocessing as dprep
import data_generator as dgen
from global_params import cfg
import neural_network as nn
from core import write_log


def ffnet(ecg_shape, summarize=False):
    layer_typedict = {
        "dense":Dense,
        "conv1d":Conv1D
    }

    ecg_input = Input(              # Input-Layer
        shape=(cfg.nn_input_size, ),            # Ecg input
        name="ecg_inp"
    )
    net = ecg_input

    # for layer in cfg.layers:
    #     net = layer_typedict[layer.type](
    #         layer.nodes,
    #         activation = layer.activation
    #     )(net)
    #     net = Dropout(layer.dropout)(net)
    attention_probs = Dense(cfg.nn_input_size, activation='softmax', name='attention_vec')(ecg_input)
    # attention_mul = merge([ecg_input, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    attention_mul = Multiply()([ecg_input, attention_probs])
    attention_mul = Dense(64)(attention_mul)

    output = layer_typedict[cfg.output_layer.type](                 # Output-Layer
        1,                          # Dim Output Space: 1
        activation=cfg.output_layer.activation      # Activation Function: Sigmoid
    )(attention_mul)

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
            nn.precision,
            nn.recall
        ]
    )

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    if summarize:
        model.summary()                 # Model Summary

    return model

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def hsl(value):
    return (value**0.35, 0, 1-value**2, 1)

if cfg.logging:
    with open(cfg.log_location + "log_" + cfg.t + ".csv", 'a') as csvlog:
        csvlog.write("t,lead,split_train,split_val,split_test,epochs,unique_patients,train_size,validation_size,test_size,loss,accuracy,precision,recall,AUC,F1\n")

all_data_x, all_data_y, fnames = dgen.get_data(
    # n_files=100,
    return_fnames = True,
    channels = np.array(range(cfg.n_channels)),
    norm = cfg.normalize_data,
    targets = cfg.targets,
    extension = "." + cfg.file_extension
)

for lead in range(8):
    cfg.current_lead = lead

    data_x = all_data_x.copy()[:, :, (0, lead)]
    data_y = all_data_y.copy()

    data_x, data_y = dprep.extract_windows(
        data_x, 
        data_y, 
        exclude_first_channel = True, 
        fnames = fnames
    )
    # data_x, data_y = dprep.extract_windows(all_data_x[:, :, lead], all_data_y)

    # x_train, y_train, x_val, y_val, x_test, y_test = nn.prepare_train_val_data(
    #     data_x, 
    #     data_y, 
    #     tvt_split=cfg.tvt_split, 
    #     split_on="patient_ids"
    # )

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

    model = ffnet((cfg.nn_input_size, ))
    nn.train(
        model, x_train, y_train, x_val, y_val, 
        batch_size=cfg.training_batch_size, 
        epochs=cfg.epochs, 
        save=cfg.save_on_train
    )

    x_sine = np.array([x for i, x in enumerate(x_train) if y_train[i] == 0])
    x_af = np.array([x for i, x in enumerate(x_train) if y_train[i] == 1])
    x_sine = np.mean(x_sine, axis=0)
    x_af = np.mean(x_af, axis=0)

    a1 = get_activations(model, np.array([x_sine]),
                                       print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()

    a2 = get_activations(model, np.array([x_af]),
                                   print_shape_only=True,
                                   layer_name='attention_vec')[0].flatten()

    rr = "rr"

    half = 40
    a1 = np.concatenate([a1[half:], a1[:half]]) / max(a1)
    a2 = np.concatenate([a2[half:], a2[:half]]) / max(a2)
    x_sine = np.concatenate([x_sine[half:], x_sine[:half]]) / max(x_sine)
    x_af = np.concatenate([x_af[half:], x_af[:half]]) / max(x_af)

    # plt.plot(a1, c="g", label="Sinus attention", linestyle="--")
    plt.plot(x_sine, c="g", label="Sinus rythm")
    plt.title("Attention to Sinus rythm ecg (lead {})".format(lead))
    for i in range(1, 80):
        plt.fill_between([i-1, i], [x_sine[i-1], x_sine[i]], [-1, -1], color=[hsl(a1[i-1]), hsl(a1[i])])
    plt.legend()
    plt.savefig("images/attention_sr_lead_{}_{}_{}.png".format(lead, rr, cfg.t))

    plt.clf()

    # plt.plot(a2, c="r", label="AF attention", linestyle="--")
    plt.plot(x_af, c="r", label="Atrial firbrillation")
    plt.title("Attention to AF rythm ecg (lead {})".format(lead))
    for i in range(1, 80):
        plt.fill_between([i-1, i], [x_af[i-1], x_af[i]], [-1, -1], color=[hsl(a2[i-1]), hsl(a2[i])])
    plt.legend()
    plt.savefig("images/attention_af_lead_{}_{}_{}.png".format(lead, rr, cfg.t))

    a = get_activations(model, np.array([np.mean(x_train, axis=0)]),
                                       print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()

    plt.clf()


    flat = [1] * 80

    plt.plot(flat, c="g", label="Flat input")
    plt.title("Attention to flat input (lead {})".format(lead))
    for i in range(1, 80):
        plt.fill_between([i-1, i], [flat[i-1], flat[i]], [-1, -1], color=[hsl(a[i-1]), hsl(a[i])])
    plt.legend()
    plt.savefig("images/attention_flat_lead_{}_{}_{}.png".format(lead, rr, cfg.t))

    plt.clf()

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