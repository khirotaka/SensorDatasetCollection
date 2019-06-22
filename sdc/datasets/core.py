import os
import requests
import zipfile
import tqdm

SAVE_DIR = os.environ["HOME"] + "/.SensorSignalDatasets/"


def fetch_dataset(url, extract=True):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = url.split("/")[-1]

    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    print("Downloading {} ...".format(filename))
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)
