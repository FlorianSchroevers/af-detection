# -*- coding: utf-8 -*-
"""

This program reads the raw .ECG data obtained from the PHD projects.
It restructures the data to a .csv file.

Authors: Wim Pilkes and Florian Schroevers
"""
import os
from global_params import cfg
from collections import defaultdict
import xmltodict
import base64
import numpy as np
import pywt

import matplotlib.pyplot as plt
import data_generator as dgen
import data_preprocessing as dprep
from helpers import progress_bar

def convert_ecg_files():
    # loops through current directory
    for filename in os.listdir(cfg.raw_data_location):
        # checks if file is .ECG/.Ecg/.ecg/etc. file
        if filename.lower().endswith(".ecg"):
            # converts ECG file to CSV file, adds it to converted_files directory and deletes "mu" error
            new_filename = cfg.converted_data_location + filename[:-4] + ".csv"
            with open(cfg.raw_data_location + filename, errors="ignore") as fr, open(new_filename, 'w') as fw:
                data = fr.readlines()
                fw.writelines(data[1:])

def convert_file(file_path):
    if file_path.lower().endswith(".ecg"):
        path_components = file_path.split('/')
        new_file_path = cfg.converted_data_location + path_components[-1][:-4] + ".csv"
        with open(file_path, errors="ignore") as fr, open(new_file_path, 'w') as fw:
            data = fr.readlines()
            fw.writelines(data[1:])
    else:
        new_file_path = file_path

    return new_file_path

def get_xml_label(file_id):
    return "XX"

def get_xml_lead_data(filename):
    data_signal = []
    with open(filename, 'r') as fd:
        ecg_dict = xmltodict.parse(fd.read(), process_namespaces=True)
    mean_wave, leads = ecg_dict['RestingECG']['Waveform']
    patient_id = ecg_dict['RestingECG']['PatientDemographics']['PatientID']
    for k,i in enumerate(leads['LeadData'][:]):
        amp = float(leads['LeadData'][k]['LeadAmplitudeUnitsPerBit'])               
        b64_encoded = ''.join(i['WaveFormData'].split('\n'))
        decoded = base64.b64decode(b64_encoded)
        signal = np.frombuffer(decoded, dtype='int16')
        data_signal.append(np.expand_dims(signal*amp, axis=0))
    return patient_id, np.concatenate(data_signal).T

def convert_xmls():
    for filename in os.listdir(cfg.xml_data_location):
        # checks if file is .XML/.Xml/.xml/etc. file
        if filename.lower().endswith(".xml"):
            patient_id, data = get_xml_lead_data(cfg.xml_data_location + filename)
            
            fs = filename[:-4].split("_")
            file_id = "_".join([patient_id] + fs[1:4])
            label = get_xml_label(file_id)

            np.savetxt(cfg.converted_data_location + file_id + "_" + label + ".csv", data, delimiter=',', fmt="%d")

def save_wavelet_img(time, signal, scales, 
                 waveletname = 'cmor', 
                 cmap = plt.cm.binary, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)

    ax.invert_yaxis()

    plt.axis('off')
    plt.savefig(cfg.dwt_image_data_location + title + ".png")
    plt.close(fig)

def convert_dwt_images(lead):
    data_x, data_y, fnames = dgen.get_data(
        # n_files=1,
        targets = cfg.targets,
        return_fnames = True,
        channels = [lead],
        norm = True
    )

    for i, ecg in enumerate(data_x):
        title = fnames[i].split('.')[0]
        save_wavelet_img(
            [i for i in range(data_x.shape[1])], 
            ecg[:, 0], 
            np.arange(1, 128, 2),
            title = title
        )
        # used_fnames[fnames[i]] += 1
        progress_bar("Converting to DWT image", i, data_x.shape[0])

def save_data(data_x, data_y, location, fnames=[]):
    """ function : save_data

    saves the preprocessed data to the location specified in global_params.py
    saves the files as the original filename + 'preprocessed' if the oriinal
    filenames are given, otherwise as a generic name with target variable in it

    Args:
        data_x : np.ndarray
            the ecg data to save
        data_y : np.ndarray
            the targetsof the ecg's
        fnames : list [optional, default: []
            the filenames of the original files

    """
    for i, ecg in enumerate(data_x):
        if fnames:
            fname = fnames[i][:-4] + "_processed.csv"
        else:
            fname = "ecg_" + str(i) + "_" + str(data_y[i]) + ".csv"
        np.savetxt(
            location + fname,
            ecg,
            delimiter=',',
            fmt='%i'
        )
        print(i, end='\r')

def save_pulse_data(fmt='%.6f'):
    data, targets, fnames = dgen.get_data(
        # n_files=10,
        return_fnames=True, 
        norm=True, 
        targets=["AF", "SR"]
    )

    data, targets, fnames = dprep.extract_windows(
        data,
        targets,
        cfg.nn_input_size,
        fnames=fnames
    )

    for fname, pulse in zip(fnames, data):
        np.savetxt(
            cfg.pulse_data_location + fname, 
            pulse, 
            delimiter=',',
            fmt=fmt
        )

def convert_and_process():
    convert_ecgs()
    convert_xmls()
    data_x, data_y, fnames = dgen.get_data(return_fnames=True, location=cfg.converted_data_location)
    processed_data_x = dprep.preprocess_data(data_x)

    dprep.save_data(processed_data_x, data_y, cfg.processed_data_location, fnames)

if __name__ == "__main__":
    # data = get_xml_lead_data("data/XML_AFACT/MUSE_20170315_104900_38000.xml")
    # print(data.shape)
    # convert_xml()
    # convert_dwt_images(1)
    save_pulse_data()
