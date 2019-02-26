"""

This program reads the raw .ECG data obtained from the PHD projects.
It restructures the data to a .csv file.

Authors: Wim Pilkes and Florian Schroevers
"""
import os
from global_params import cfg

def convert_all():
    # loops through current directory
    for filename in os.listdir(cfg.raw_data_location):
        # checks if file is .ECG/.Ecg/.ecg/etc. file
        if filename.lower().endswith(".ecg"):
            # converts ECG file to CSV file, adds it to converted_files directory and deletes "mu" error
            new_filename = cfg.data_loading_location + filename[:-4] + ".csv"
            with open(cfg.raw_data_location + filename, errors="ignore") as fr, open(new_filename, 'w') as fw:
                data = fr.readlines()
                fw.writelines(data[1:])

def convert_file(file_path):
    if file_path.lower().endswith(".ecg"):
        path_components = file_path.split('/')
        new_file_path = cfg.data_loading_location + path_components[-1][:-4] + ".csv"
        with open(file_path, errors="ignore") as fr, open(new_file_path, 'w') as fw:
            data = fr.readlines()
            fw.writelines(data[1:])
    else:
        new_file_path = file_path

    return new_file_path

if __name__ == "__main__":
    convert_all()
