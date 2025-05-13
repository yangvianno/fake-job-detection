# src/models.py

from tensorflow import keras

def build_mlp(input_shape, num_classes=1):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers3

    