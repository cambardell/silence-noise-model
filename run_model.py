import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.io import wavfile
import h5py
import os

datalist = []
model = keras.models.load_model('model.h5')
path = '/Users/cameronbardell/documents/tensorflow/silence_noise_model/noise-1-1.wav'
fs, data = wavfile.read(path)
datalist.append(data[:fs*5])
datalist = np.asarray(datalist)



predictions_single = model.predict(datalist)

print(predictions_single)
