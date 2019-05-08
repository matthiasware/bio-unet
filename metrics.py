from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


# KERAS VERSIONS
def mae_zeros(y_true, y_pred):
    return K.sum((1 - y_true) * y_pred) / K.sum(1 - y_true)


def mae_zeros_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(mae_zeros(y_true, y_pred))


def mae_ones(y_true, y_pred):
    return K.sum(y_true - (y_true * y_pred)) / K.sum(y_true)


def mae_ones_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(mae_ones(y_true, y_pred))


def acc_ones(y_true, y_pred):
    return K.sum(K.round(y_true * y_pred)) / K.sum(y_true)


def true_positives(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_tp = tf.logical_and(y_true > 0, y_pred > 0)
    y_tp = tf.reduce_sum(tf.cast(y_tp, tf.float32))
    return y_tp / tf.cast(tf.size(y_true), tf.float32)


def true_negatives(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_tn = tf.logical_and(y_true < 1, y_pred < 1)
    y_tn = tf.reduce_sum(tf.cast(y_tn, tf.float32))
    return y_tn / tf.cast(tf.size(y_true), tf.float32)


def false_positives(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_fp = tf.logical_and(y_true < 1, y_pred > 0)
    y_fp = tf.reduce_sum(tf.cast(y_fp, tf.float32))
    return y_fp / tf.cast(tf.size(y_true), tf.float32)


def false_negatives(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_fn = tf.logical_and(y_true > 0, y_pred < 1)
    y_fn = tf.reduce_sum(tf.cast(y_fn, tf.float32))
    return y_fn / tf.cast(tf.size(y_true), tf.float32)


# NUMPY VERSIONS
def acc_ones_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(acc_ones(y_true, y_pred))


def acc_zeros(y_true, y_pred):
    den = 1 - y_true
    nom = (1 - y_pred) * den
    nom = K.round(nom)
    den = K.sum(den)
    nom = K.sum(nom)
    return nom / den


def acc_zeros_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(acc_zeros(y_true, y_pred))


def acc_custom(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def acc_custom_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(acc_custom(y_true, y_pred))


def false_negatives_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(false_negatives(y_true, y_pred))


def false_positives_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(false_positives(y_true, y_pred))


def true_positives_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(true_positives(y_true, y_pred))


def true_negatives_np(y_true, y_pred):
    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    return K.eval(true_negatives(y_true, y_pred))


def calculate_metrics(y_true, y_pred):
    print("ACC:       ", acc_custom_np(y_true, y_pred))
    print("ACC Zeros: ", acc_zeros_np(y_true, y_pred))
    print("ACC Ones:  ", acc_ones_np(y_true, y_pred))
    print("MAE Zeros: ", mae_zeros_np(y_true, y_pred))
    print("MAE Ones:  ", mae_ones_np(y_true, y_pred))
    print("TP:        ", true_positives_np(y_true, y_pred))
    print("TN:        ", true_negatives_np(y_true, y_pred))
    print("FP:        ", false_positives_np(y_true, y_pred))
    print("FN:        ", false_negatives_np(y_true, y_pred))


if __name__ == "__main__":
    y_true_np = np.array([[1, 0, 0, 0],
                          [1, 0, 0, 1],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0]])
    y_pred_np = np.array([[0.6, 0.1, 0.2, 0.5],
                          [0.4, 0.2, 0.3, 0.7],
                          [0.1, 0.2, 0.3, 0.4],
                          [0.1, 0.2, 0.3, 0.8]])
    y_true = K.variable(value=y_true_np)
    y_pred = K.variable(value=y_pred_np)

    mea_zeros_np = np.sum((0.1, 0.2, 0.5, 0.2, 0.3, 0.1,
                           0.2, 0.3, 0.1, 0.2, 0.3, 0.8)) / 12
    mae_zeros_tf = K.eval(mae_zeros(y_true, y_pred))
    assert abs(mea_zeros_np - mae_zeros_tf) < 1e-7

    mae_ones_np = np.sum((1 - 0.6, 1 - 0.4, 1 - 0.7, 1 - 0.4)) / 4
    mae_ones_tf = K.eval(mae_ones(y_true, y_pred))
    assert abs(mae_ones_np - mae_ones_tf) < 1e-7

    acc_ones_np = np.sum((1, 0, 0, 1)) / 4
    acc_ones_tf = K.eval(acc_ones(y_true, y_pred))
    assert abs(acc_ones_np - acc_ones_tf) < 1e-7

    acc_zeros_np = np.sum((1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0)) / 12
    acc_zeros_tf = K.eval(acc_zeros(y_true, y_pred))
    assert abs(acc_zeros_np - acc_zeros_tf) < 1e-7

    acc_custom_np = np.sum((1, 1, 1, 1,
                            0, 1, 1, 1,
                            1, 1, 1, 0,
                            1, 1, 1, 0)) / 16
    acc_custom_tf = K.eval(acc_custom(y_true, y_pred))
    assert abs(acc_custom_np - acc_custom_tf) < 1e-7

    y_true_np = np.around(np.random.random((100, 100)))
    y_pred_np = np.around(np.random.random((100, 100)))

    tn, fp, fn, tp = confusion_matrix(y_true_np.flatten(), y_pred_np.flatten()).ravel()
    s = tn + fp + fn + tp
    tn = tn / s
    fp = fp / s
    fn = fn / s
    tp = tp / s

    tn_c = true_negatives_np(y_true_np, y_pred_np)
    tp_c = true_positives_np(y_true_np, y_pred_np)
    fn_c = false_negatives_np(y_true_np, y_pred_np)
    fp_c = false_positives_np(y_true_np, y_pred_np)

    assert abs(tn - tn_c) < 1e-5, "Expected: {} Actual {}".format(tn, tn_c)
    assert abs(tp - tp_c) < 1e-5, "Expected: {} Actual {}".format(tp, tp_c)
    assert abs(fn - fn_c) < 1e-5, "Expected: {} Actual {}".format(fn, fn_c)
    assert abs(fp - fp_c) < 1e-5, "Expected: {} Actual {}".format(fp, fp_c)
