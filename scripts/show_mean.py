""" File: show_mean.py
    Shows plots of the mean of the extracted pulses seperated in sinus rythms
    and atrial fibrillation labels.
Authors: Florian Schroevers
"""

import data_preprocessing as dprep
import data_generator as dgen
from global_params import cfg

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data_x, data_y, fnames = dgen.get_data(
    return_fnames = True,
    channels = np.array([cfg.lead]),
    norm = True,
    exclude_targets = [1, 2, 3, 4]
)
data_x_sine, _ = dprep.extract_windows(data_x, data_y)

data_x, data_y, fnames = dgen.get_data(
    return_fnames = True,
    channels = np.array([cfg.lead]),
    norm = True,
    exclude_targets = [0, 2, 3, 4]
)
data_x_afib, _ = dprep.extract_windows(data_x, data_y)

plt.plot(np.mean(data_x_sine, axis=0), c="g", label="Sinus rythm")
plt.plot(np.mean(data_x_afib, axis=0), c="r", label="Atrial firbrillation", linestyle="--")
plt.legend()
plt.show()


x1 = np.mean(data_x_sine, axis=0)
half = int(len(x1)/2)
x1 = np.concatenate([x1[half:], x1[:half]])
x2 = np.mean(data_x_afib, axis=0)
x2 = np.concatenate([x2[half:], x2[:half]])


plt.plot(x1, c="g", label="Sinus rythm")
plt.plot(x2, c="r", label="Atrial firbrillation", linestyle="--")
plt.legend()
plt.show()
