# Sensor Dataset Collection

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/68a2ad7a458945e4b8a3423e5bcd5b1d)](https://app.codacy.com/app/KawashimaHirotaka/SensorDatasetCollection?utm_source=github.com&utm_medium=referral&utm_content=KawashimaHirotaka/SensorDatasetCollection&utm_campaign=Badge_Grade_Dashboard)

Sensor Dataset Collection for Machine Learning.

---
Image, NLP etc. major domain of machine learning has many famous dataset. 
For example, [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR series](https://www.cs.toronto.edu/~kriz/cifar.html),
[PTB dataset](http://www.fit.vutbr.cz/%7Eimikolov/rnnlm/).
However, Machine learning using sensor data does not have such a famous dataset.

## Purpose
Machine Learning using sensor-data will become an important field in the future, 
However, there is no easy way to access the data sets, nor is there a well-known data set.

Purpose of this project is to provide 3 applications.

 - Sensor Dataset Loader 
 - Preprocessing function for sensor data.
 - Examples using sensor data

## Installation
Before install `SensorDatasetCollection`, please create virtual environment such as `Conda` or `pipenv`

```sh
git clone https://github.com/KawashimaHirotaka/SensorDatasetCollection.git
cd SensorDatasetCollection/
pip install .
```

or

```sh
git clone https://github.com/KawashimaHirotaka/SensorDatasetCollection.git
cd SensorDatasetCollection/
python setup.py install
```

## Usage

```python
import sdc

(x_train, y_train), (x_test, y_test) = sdc.datasets.uci.load_har()

```
