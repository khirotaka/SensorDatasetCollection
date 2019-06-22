import os
import zipfile
import requests
import tqdm
import numpy as np
import pandas as pd
from scipy import stats


BASE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
