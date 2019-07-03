import os
import zipfile

import pandas as pd

from ..datasets import core


BASE_URL = "https://ti.arc.nasa.gov/"


def load_turbofan_engine():
    """
    Load Turbofan Engine Degradation Simulation Dataset
    from https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    :return:
    """

    save_dir = core.SAVE_DIR + "/NASA Turbofan/"

    if not os.path.isdir(save_dir):
        url = BASE_URL + "m/project/prognostic-repository/CMAPSSData.zip"

        print("Downloading NASA Turbofan Engine Degradation Simulation Dataset ...")
        core.fetch_dataset(url, extract=False)

        with zipfile.ZipFile("CMAPSSData.zip") as f:
            f.extractall("NASA Turbofan")
        os.remove("CMAPSSData.zip")

    modes = ["train", "test"]

    def load_per_type(mode):
        files = ["_FD001", "_FD002", "_FD003", "_FD004"]

        loaded = []
        for file in files:
            tmp = pd.read_csv("NASA Turbofan/" + mode + file + ".txt", header=None, delim_whitespace=True).values
            loaded.append(tmp)

        return loaded

    train_data = load_per_type(modes[0])
    test_data = load_per_type(modes[1])

    return train_data, test_data


def load_phm08(load_all=False):
    """
    Load PHM08 Challenge Data Set
    from https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    :return:
    """

    save_dir = core.SAVE_DIR + "/PHM08/"

    if not os.path.isdir(save_dir):
        url = BASE_URL + "m/project/prognostic-repository/Challenge_Data.zip"

        print("Downloading NASA PHM08 Dataset ...")
        core.fetch_dataset(url, extract=False)

        with zipfile.ZipFile("Challenge_Data.zip") as f:
            f.extractall("PHM08")
        os.remove("Challenge_Data.zip")

    if load_all:
        train = pd.read_csv("PHM08/train.txt", header=None, delim_whitespace=True).values
        test = pd.read_csv("PHM08/test.txt", header=None, delim_whitespace=True).values
        final_test = pd.read_csv("PHM08/final_test.txt", header=None, delim_whitespace=True).values

        return train, test, final_test

    else:
        train = pd.read_csv("PHM08/train.txt", header=None, delim_whitespace=True).values
        test = pd.read_csv("PHM08/test.txt", header=None, delim_whitespace=True).values

        return train, test
