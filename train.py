import numpy as np 
import pandas as pd 
import os
import itertools

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Conv2DTranspose
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.preprocessing.image import array_to_img

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import preprocess
from model_00 import get_model


X = np.load("{}.npy".format(os.path.join(preprocess.d_data, preprocess.f_x_train)))
Y = np.load("{}.npy".format(os.path.join(preprocess.d_data, preprocess.f_y_train)))

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)
del X
del Y

print("X_train", X_train.shape)
print("X_valid", X_valid.shape)
print("Y_train", Y_train.shape)
print("Y_valid", Y_valid.shape)

def augment_imgarr(x, i):
    if i % 4 == 0:
        return np.rot90(x, k=i)
    if i == 4:
        return np.flip(x, 0)
    if i == 5:
        return np.flip(x, 1)
    else:
        return np.transpose(x, axes=(1, 0, 2))


def trainDataGenerator(X, Y, batches=1):
    XYS = itertools.cycle(zip(X, Y))
    while True:
        XX = []
        YY = []
        for i in range(batches):
            x, y = next(XYS)
            x = x / 255
            idx = np.random.randint(0, 7)
            XX.append(augment_imgarr(x, idx))
            YY.append(augment_imgarr(y, idx))
        yield np.array(XX), np.array(YY)

def calc_weights(y):
    n_ones = np.sum(y)
    n_total = y.shape[1]**2
    w_ones = 1 - n_ones / n_total
    w_zeros = 1 - w_ones
    return w_ones, w_zeros

def validDataGenerator(X, Y, batches=1):
    XYS = itertools.cycle(zip(X, Y))
    while True:
        XX = []
        YY = []
        WW = []
        for i in range(batches):
            x, y = next(XYS)
            x = x / 255
            idx = np.random.randint(0, 7)
            XX.append(augment_imgarr(x, idx))
            y_aug = augment_imgarr(y, idx)
            w_ones, w_zeros = calc_weights(y_aug)
            w = y_aug * (w_ones - w_zeros) + w_zeros
            YY.append(y_aug)
            WW.append(w)
        yield np.array(XX), np.array(YY), np.array(WW)

model = get_model(n_kernels=32, img_height=X_train.shape[1], img_width=X_train.shape[1])

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.98, nesterov=True),
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1),
             ModelCheckpoint("model00.h5", monitor='val_loss', save_best_only=True, verbose=1)]


train_generator = trainDataGenerator(X_train, Y_train, 1)
valid_generator = validDataGenerator(X_valid, Y_valid, 1)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=50,
                    epochs=10,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator,
                    validation_steps=10,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False)