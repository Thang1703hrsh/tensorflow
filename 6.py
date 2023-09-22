
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras import regularizers

(x_train , y_train) , (x_test , y_test) = cifar10.load_data()
print(x_train.shape)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

model = keras.Sequential()
model.add(keras.Input(shape = (None , 28)))
model.add(
    layers.SimpleRNN(512, return_sequences= True , activation = 'relu')
)

model.add(layers.SimpleRNN(512 , activation = 'relu'))
model.add(layers.Dense(10))

print(model.summary())