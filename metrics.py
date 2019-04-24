from keras import backend as K
import numpy as np


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


def fn(y_true, y_pred):
    pass


def fp(y_true, y_pred):
    pass


def tp(y_true, y_pred):
    pass


def tn(y_true, y_pred):
    pass


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
    acc_ones_tf = acc_ones(y_true, y_pred)
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
