import os
import sys
import subprocess
import webbrowser
from typing import Tuple

import pandas as pd

from . import core

save_dir = core.SAVE_DIR + "/UCRArchive_2018/"


def __fetch_dataset() -> None:
    """
    Downloading Dataset.

    Returns:
        None
    """
    if not os.path.isdir(save_dir):
        sys.stdout.write("Required Password: ")
        password = input()

        print("Downloading UCR Time Series Classification Archive & Extracting files ...")

        core.fetch_dataset(
            "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip",
            extract=True,
            password=password.encode("utf-8")
        )


def load_data(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load dataset

    Args:
        name: Dataset name. Please select from DatasetInformation.show_list() .

    Returns:
        pd.DataFrame
    """
    __fetch_dataset()

    train_data = pd.read_csv(save_dir + name + "/" + name + "_TRAIN.tsv", header=None, delim_whitespace=True)
    test_data = pd.read_csv(save_dir + name + "/" + name + "_TEST.tsv", header=None, delim_whitespace=True)

    return train_data, test_data


class DatasetInformation(object):
    @classmethod
    def show_list(cls):
        return os.listdir(save_dir)

    @classmethod
    def detail(cls, name: str, browser: bool = False) -> str:
        """
        Return detail of dataset.

        Args:
            name: Dataset name. Please select from DatasetInformation.show_list() .
            browser: if True, open your web browser. default False

        Returns: str - URL

        """

        name = save_dir + name + "/README.md"
        cmd = "cat {} | tail -n 1 | cut -c 5-".format(name)
        res = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode("utf-8").\
            replace("\n", "")

        if browser:
            webbrowser.open(res)

        return res
