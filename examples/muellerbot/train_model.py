
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import re


def train_model(wordvectors, maxlen):
    word_count, dimensionality = wordvectors.shape
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, dimensionality)))
    #model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(dimensionality))
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Finished compiling LSTM')

    # cut the text in semi-redundant sequences of maxlen characters
    step = 1
    sentences = []
    next_words = []
    for i in range(0, len(wordvectors) - maxlen, step):
        sentences.append(wordvectors[i: i + maxlen])
        next_words.append(wordvectors[i + maxlen])
    print('Training on {} sequences'.format(len(sentences)))

    X = np.asarray(sentences)
    y = np.asarray(next_words)

    for iteration in range(1, 1000):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)
        yield model
