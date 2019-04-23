import utils
import augment
import numpy as np
from model_03 import get_model
import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ProgbarLogger
from keras import optimizers
from datetime import datetime
import matplotlib.pyplot as plt
import os

img_size = 256
n_kernels = 64
d_train = "./data/{}/train/".format(img_size)
d_test = "./data/{}/test".format(img_size)
files = utils.get_files(d_train, verbose=False)
n_valid = int(len(files) * 0.15)
files_valid = files[:n_valid]
files_train = files[n_valid:]
d = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
f_model = "model_{}.h5".format(d)

X_train, Y_train = utils.load_data(files_train)
X_valid, Y_valid = utils.load_data(files_valid)


n_ones = np.sum(Y_train) + np.sum(Y_valid)
n_total = np.prod(Y_train.shape) + np.prod(Y_valid.shape)
n_zeros = n_total - n_ones
weights_zero = n_ones / n_total
weights_ones = n_zeros / n_total
print("Ones:  ", n_ones)
print("zeros: ", n_zeros)
print("Total: ", n_total)
print("1/T:   ", weights_zero)
print("0/T:   ", weights_ones)
print("n0 * w0 : n1 * w1 = {} : {}".format(n_ones *
                                           weights_ones, n_zeros * weights_zero))


model = get_model(img_size, n_kernels)
print(model.summary())


model.compile(loss=losses.make_bce_loss(weights_ones, weights_zero),
              optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True),
              metrics=['accuracy'])
callbacks = [
             EarlyStopping(monitor='val_loss',
                           patience=4,
                           verbose=1),
             ModelCheckpoint(f_model,
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=1)]

#f_model = "model_04_23_2019_17_00_16.h5"
if os.path.isfile(f_model) and True:
    model.load_weights(f_model)
hist = model.fit(x=X_train,
                 y=Y_train,
                 batch_size=len(files_train),
                 epochs=100,
                 verbose=2, callbacks=callbacks,
                 validation_data=(X_valid, Y_valid))
