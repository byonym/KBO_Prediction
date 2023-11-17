import pandas as pd
import numpy as np
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import RootMeanSquaredError


# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=1))

def get_model(types):
    es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=30)
    model = models.Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.25))
    model.add(LSTM(16, return_sequences=False, activation='relu'))
    model.add(Dropout(0.25))
    if types=='era':
        model.add(Dense(1, activation='relu'))
    else:
        model.add(Dense(1, activation='sigmoid'))# ERA는 relu, AVG는 sigmoid.

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.8)

    opt = keras.optimizers.RMSprop(learning_rate=lr_schedule)


    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model, es



