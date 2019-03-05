import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.io import wavfile
import h5py
import os

## import and format trainging data
path = '/Users/cameronbardell/documents/tensorflow/silence_noise_model/train_data'
data_list = []
label_list = []

for filename in os.listdir(path):
    if ".wav" in filename:
        pathToFile = path+'/'+filename
        fs, data = wavfile.read(pathToFile)
        # Append first five seconds to keep all files same length
        data_list.append(data[:fs*5])
    if "silence" in filename:
        label_list.append(0)
    if "noise" in filename:
        label_list.append(1)
label_list = np.asarray(label_list)

# Test Data
path = '/Users/cameronbardell/documents/tensorflow/silence_noise_model/test_data'
test_data_list = []
test_label_list = []

for filename in os.listdir(path):
    if ".wav" in filename:
        pathToFile = path+'/'+filename
        fs, data = wavfile.read(pathToFile)
        # Append first five seconds to keep all files same length
        test_data_list.append(data[:fs*5])
    if "silence" in filename:
        test_label_list.append(0)
    if "noise" in filename:
        test_label_list.append(1)
test_label_list = np.asarray(test_label_list)

# Noise data list is now a numpy array of numpy arrays, each wav file has two tracks.
data_list = np.asarray(data_list)
test_data_list = np.asarray(test_data_list)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(44100*5, 2)),
    # Returns an array of two probability scores that sum to 1
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data_list, label_list, epochs=5)

test_loss, test_acc = model.evaluate(test_data_list, test_label_list)

print('Test accuracy:', test_acc)

keras.models.save_model(
    model,
    '/Users/cameronbardell/documents/tensorflow/silence_noise_model/model.h5',
    overwrite = True,
    include_optimizer = True
)
