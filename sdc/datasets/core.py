import os
import zipfile
from typing import List

import tqdm
import requests
import pandas as pd
import numpy as np


SAVE_DIR = os.getcwd()


def fetch_dataset(url: str, extract: bool = True, password: bytes = None) -> None:
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = url.split("/")[-1]

    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR, pwd=password)
        os.remove(filename)


def load_files(filenames: List[str]) -> np.ndarray:
    """

    Args:
        filenames:

    Returns:

    """

    done = []
    for name in filenames:
        data = pd.read_csv(name, header=None, delim_whitespace=True).values
        done.append(data)

    done = np.dstack(done)
    return done
