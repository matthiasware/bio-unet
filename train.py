import utils
import numpy as np
from model_03 import get_model
import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from datetime import datetime
import metrics


img_size = 256
n_kernels = 16
p_valid = 0.15
d_train = "./data/{}/train/".format(img_size)
files = utils.get_files(d_train, verbose=False)
files = files
n_valid = int(len(files) * p_valid)
files_valid = files[:n_valid]
files_train = files[n_valid:]
d = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
f_model = "model_k{}_s{}_{}.h5".format(str(n_kernels), str(img_size), d)
X_train, Y_train = utils.load_data(files_train)
X_valid, Y_valid = utils.load_data(files_valid)


n_ones = np.sum(Y_train) + np.sum(Y_valid)
n_total = np.prod(Y_train.shape) + np.prod(Y_valid.shape)
n_zeros = n_total - n_ones
weights_zero = n_ones / n_total
weights_ones = n_zeros / n_total
weights_zero = 0.04
weights_ones = 0.96

model = get_model(img_size, n_kernels)
print(model.summary())

mymetrics = ['accuracy', metrics.acc_zeros, metrics.acc_ones,
             metrics.mae_zeros, metrics.mae_ones, metrics.true_positives,
             metrics.true_negatives, metrics.false_positives,
             metrics.false_negatives]

myloss = losses.make_bce_loss(weights_ones, weights_zero)
cb_es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
cb_cp = ModelCheckpoint(f_model, monitor='val_loss',
                        save_best_only=True, verbose=1),
cb_tb = TensorBoard(log_dir="logs/{}".format(d))

# first loop
callbacks = [cb_cp, cb_tb]
myoptimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.99, nesterov=True)
model.compile(loss=myloss, optimizer=myoptimizer, metrics=mymetrics)


hist = model.fit(x=X_train,
                 y=Y_train,
                 batch_size=1,
                 epochs=X_train.shape[0],
                 verbose=2,
                 callbacks=callbacks,
                 validation_data=(X_valid, Y_valid))

utils.plot(hist.history['loss'], hist.history['val_loss'], "Loss")
utils.plot(hist.history['acc'], hist.history['val_acc'], "ACC")
utils.plot(hist.history['acc_zeros'],
           hist.history['val_acc_zeros'], "ACC_zeros")
utils.plot(hist.history['acc_ones'],
           hist.history['val_acc_ones'], "ACC_ones")
utils.plot(hist.history['mae_zeros'],
           hist.history['val_mae_zeros'], "MAE_zeros")
utils.plot(hist.history['mae_ones'],
           hist.history['val_mae_ones'], "MAE_ones")
utils.plot(hist.history['true_negatives'],
           hist.history['val_true_negatives'], "TN")
utils.plot(hist.history['true_positives'],
           hist.history['val_true_positives'], "TP")
utils.plot(hist.history['false_negatives'],
           hist.history['val_false_negatives'], "FN")
utils.plot(hist.history['false_positives'],
           hist.history['val_false_positives'], "FP")
