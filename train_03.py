import numpy as np
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers


import preprocess
from model_01 import get_model
import pickle
import splitdata


import losses

with open(os.path.join(splitdata.d_data, splitdata.f_train), "rb") as file:
    files = pickle.load(file)
if not files:
    raise Exception("Could not load training files!")
print("Files: ", len(files))

np.random.shuffle(files)
n_valid = int(np.ceil(len(files) * 0.15))
files_train = files[n_valid:]
files_valid = files[:n_valid]
print("Files train: ", len(files_train))
print("Files valid: ", len(files_valid))


model = get_model(n_kernels=32, img_height=2084, img_width=2084)
print(model.summary())

model.compile(loss=losses.weighted_binary_crossentropy,
              optimizer=optimizers.SGD(
                  lr=0.01, decay=1e-6, momentum=0.98, nesterov=True),
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=4,
                  verbose=1),
    ModelCheckpoint("model03.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1)]

train_generator = preprocess.dataGenerator(files_train, batches=1)
valid_generator = preprocess.dataGenerator(files_valid, batches=1)
p_weights = "model02.h5"

if os.path.isfile(p_weights) and True:
    model.load_weights(p_weights)

hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=30,
                           epochs=20,
                           verbose=2,
                           callbacks=callbacks,
                           validation_data=valid_generator,
                           validation_steps=16,
                           max_queue_size=2)
