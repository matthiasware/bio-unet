import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Conv2DTranspose, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras import backend as K


import preprocess
from model_02 import get_model

import losses

files = preprocess.get_split_files("data/split")
print(len(files))

np.random.shuffle(files)
files = files[0:]
n_valid = int(np.ceil(len(files) * 0.15))
n_test = int(np.ceil((len(files) - n_valid) * 0.15))
n_train = len(files) - n_valid - n_test
f_train = files[0:n_train]
f_valid = files[n_train: n_train + n_valid]
f_test = files[n_train + n_valid:]

X_train, Y_train = preprocess.load_split_data(f_train)
X_valid, Y_valid = preprocess.load_split_data(f_valid)

print(X_train.shape)
print(Y_train.shape)

model = get_model(img_size=512, n_kernels=32)
print(model.summary())

model.compile(loss=losses.weighted_binary_crossentropy,
              optimizer=optimizers.SGD(
                  lr=0.01, decay=1e-6, momentum=0.98, nesterov=True),
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=4,
                  verbose=1),
    ModelCheckpoint("model06.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1)]


hist = model.fit(x=X_train,
                 y=Y_train,
                 batch_size=1,
                 steps_per_epoch=40,
                 validation_steps=10,
                 epochs=20,
                 verbose=2, callbacks=callbacks,
                 validation_data=(X_valid, Y_valid))


training_loss = hist.history['loss']
test_loss = hist.history['val_loss']
training_acc = hist.history['acc']
test_acc = hist.history['val_acc']


print("Train Loss: ", training_loss)
print("Test Loss:  ", test_loss)
print("Train Acc:  ", training_acc)
print("Test Acc:   ", test_acc)
