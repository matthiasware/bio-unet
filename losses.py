import keras.backend as K


def weighted_binary_crossentropy(y_true, y_pred):
        # 0.9991
        # 0.0009
    weights_one = 0.9991
    weights_zero = 0.0009

    bce = K.binary_crossentropy(y_true, y_pred)

    weight_vector = y_true * weights_one + (1. - y_true) * weights_zero
    ce_weighted = weight_vector * bce

    return K.mean(ce_weighted)


def make_bce_loss(weights_one, weights_zero):
    def inner_weighted_binary_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        weight_vector = y_true * weights_one + (1. - y_true) * weights_zero
        ce_weighted = weight_vector * bce

        return K.mean(ce_weighted)
    return inner_weighted_binary_crossentropy


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
