import numpy as np
import itertools
import imageio
np.random.seed(0)


def augmentation(x, k):
    if k < 4:
        return np.rot90(x, k)
    if k == 4:
        return np.flip(x, 0)
    if k == 5:
        return np.flip(x, 1)
    else:
        return np.transpose(x, axes=(1, 0, 2))


def dataGenerator(files, batches, normalize=True, augment=False):
    files = itertools.cycle(files)
    f_x, f_y = next(files)
    shape_x = imageio.imread(f_x).shape
    shape_y = imageio.imread(f_y).shape
    while True:
        X = np.zeros((batches, *shape_x), dtype=np.float32)
        Y = np.zeros((batches, *shape_y, 1), dtype=np.float32)
        for i in range(batches):
            f_x, f_y = next(files)
            x = imageio.imread(f_x)
            y = imageio.imread(f_y).reshape((*shape_y, 1))
            if augment:
                ai = np.random.randint(0, 7)
                X[i] = augmentation(x, ai)
                Y[i] = augmentation(y, ai)
            else:
                X[i] = x
                Y[i] = y
        if normalize:
            X = X / 255
            Y = Y / 255
        yield X, Y
