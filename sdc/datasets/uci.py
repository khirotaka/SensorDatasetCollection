import os
from typing import Tuple

import pandas as pd
import numpy as np

from . import core

BASE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/"


def load_har(raw: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    save_dir = core.SAVE_DIR + "/UCI HAR Dataset/"

    if not os.path.isdir(save_dir):
        url = BASE_URL + "00240/UCI%20HAR%20Dataset.zip"

        print("Downloading UCI HAR Dataset ...")
        core.fetch_dataset(url, extract=True)

    if raw:
        types = ["train", "test"]
        path_to_raw = "Inertial Signals/"

        def load_raw_data(file_type):
            datasets = [
                "total_acc_x_" + file_type + ".txt", "total_acc_y_" + file_type + ".txt",
                "total_acc_z_" + file_type + ".txt",
                "body_acc_x_" + file_type + ".txt", "body_acc_y_" + file_type + ".txt",
                "body_acc_z_" + file_type + ".txt",
                "body_gyro_x_" + file_type + ".txt", "body_gyro_y_" + file_type + ".txt",
                "body_gyro_z_" + file_type + ".txt"
            ]

            tmp = [save_dir + file_type + "/" + path_to_raw + i for i in datasets]

            X = core.load_files(tmp)
            return X

        raw_datasets = []

        for mode in types:
            data = load_raw_data(mode)
            target = pd.read_csv(save_dir + mode + "/y_" + mode + ".txt", header=None, delim_whitespace=True).values

            raw_datasets.append((data, target))

        (x_train, y_train), (x_test, y_test) = raw_datasets[0], raw_datasets[1]

    else:
        x_train = pd.read_csv(save_dir + "train/X_train.txt", header=None, delim_whitespace=True).values
        y_train = pd.read_csv(save_dir + "train/y_train.txt", header=None, delim_whitespace=True).values

        x_test = pd.read_csv(save_dir + "test/X_test.txt", header=None, delim_whitespace=True).values
        y_test = pd.read_csv(save_dir + "test/y_test.txt", header=None, delim_whitespace=True).values

    return (x_train, y_train), (x_test, y_test)
