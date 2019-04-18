import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, array_to_img
from sklearn.model_selection import train_test_split
import itertools
import imageio

d_data = "data"
f_x_train = "x_train"
f_x_test = "x_test"
f_y_train = "y_train"
f_y_test = "y_test"
test_size = 0.1
verbose = True


def get_ndarray_from_csv(path, width, height):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        img = np.zeros((width, height, 1))
        for row in csv_reader:
            xx = np.array(row[0::2], dtype=np.int)
            yy = np.array(row[1::2], dtype=np.int)
            img[yy, xx] = 1
    return img


def get_ndarray(path, width, height):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        img = np.zeros((width, height))
        for row in csv_reader:
            xx = np.array(row[0::2], dtype=np.int)
            yy = np.array(row[1::2], dtype=np.int)
            img[yy, xx] = 1
    return img


def augment_imgarr(x, i):
    if i % 4 == 0:
        return np.rot90(x, k=i)
    if i == 4:
        return np.flip(x, 0)
    if i == 5:
        return np.flip(x, 1)
    else:
        return np.transpose(x, axes=(1, 0, 2))


def calc_weights(y):
    n_ones = np.sum(y)
    n_total = y.shape[1]**2
    w_ones = 1 - n_ones / n_total
    w_zeros = 1 - w_ones
    w = y * (w_ones - w_zeros) + w_zeros
    return w


def dataGenerator(files, batches=1):
    files = itertools.cycle(files)
    while True:
        X = np.zeros((batches, 2084, 2084, 3), dtype=np.float32)
        Y = np.zeros((batches, 2084, 2084, 1), dtype=np.float32)
        for i in range(batches):
            f_png, f_csv, _ = next(files)
            x = np.array(load_img(f_png))
            y = get_ndarray_from_csv(f_csv, X[i].shape[0], X[i].shape[1])
            x = x / 255
            idx = np.random.randint(0, 7)
            X[i] = augment_imgarr(x, idx)
            Y[i] = augment_imgarr(y, idx)
        yield X, Y


def plot_triple(f_csv, f_png, f_jpg):
    img_png = load_img(f_png)
    img_csv = array_to_img(get_ndarray_from_csv(
        f_csv, img_png.width, img_png.height))
    img_jpg = load_img(f_jpg)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15 * 3))
    ax[0].imshow(img_png)
    ax[1].imshow(np.asarray(img_csv))
    ax[2].imshow(img_jpg)
    plt.show()


def show_ndimg(img1, img2=None):
    nrows = 1
    ncols = 1 if img2 is None else 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    if img2 is None:
        ax.imshow(np.array(array_to_img(img1)))
    else:
        ax[0].imshow(np.array(array_to_img(img1)))
        ax[1].imshow(np.array(array_to_img(img2)))
    plt.show()


def show_img(img1, img2=None):
    nrows = 1
    ncols = 1 if img2 is None else 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    if img2 is None:
        ax.imshow(img1)
    else:
        ax[0].imshow(img1)
        ax[1].imshow(img2)
    plt.show()


def get_files(path, verbose=True):
    if verbose:
        print("Finding files in '{}'".format(path))
    data = []
    for path, _, files in os.walk(path):
        for file in files:
            if file[-3:] == "png":
                f_png = os.path.join(path, file)
                f_csv = os.path.join(path, file[:-3] + "csv")
                f_jpg = os.path.join(path, file[:-3] + "jpg")
                assert os.path.isfile(f_png), f_png
                assert os.path.isfile(f_csv), f_csv
                assert os.path.isfile(f_jpg), f_jpg
                if verbose:
                    print(f_png, "\n", f_csv, "\n", f_jpg)
                data.append((f_png, f_csv, f_jpg))
    return data


def get_split_files(path, verbose=True):
    if verbose:
        print("Finding files in '{}'".format(path))
    data = []
    for path, _, files in os.walk(path):
        for file in files:
            if file[-5:] == "x.png":
                f_x = os.path.join(path, file)
                f_y = os.path.join(path, file[:-5] + "y.png")
                assert os.path.isfile(f_x), f_x
                assert os.path.isfile(f_y), f_y
                if verbose:
                    print(f_x, "\n", f_y)
                data.append((f_x, f_y))
    return data


def load_data(files, verbose=True):
    if verbose:
        print("Loading data...")
    f_png, _, _ = files[0]
    img = load_img(f_png)

    X = np.zeros((len(files), img.width, img.height, 3), dtype=np.uint8)
    Y = np.zeros((len(files), img.width, img.height, 1), dtype=np.uint8)

    dsize = (X.size * X.itemsize + Y.size * Y.itemsize) * 1e-9

    if verbose:
        print("{:<10}{:4d}".format("samples", len(files)))
        print("{:<10}({:4d},{:4d})".format("Img size", img.width, img.height))
        print("{:<10}{:2.2f} GB".format("size", dsize))

    for i, (f_png, f_csv, _) in enumerate(files):
        X[i] = load_img(f_png)
        Y[i] = get_ndarray_from_csv(f_csv, X[i].shape[0], X[i].shape[1])

    if verbose:
        print("X.shape: ", X.shape)
        print("Y.shape: ", Y.shape)

    return X, Y


def load_imgs(files, verbose=True):
    if verbose:
        print("Loading data...")
    f_png, _, _ = files[0]
    img = load_img(f_png)

    X = np.zeros((len(files), img.width, img.height, 3), dtype=np.uint8)
    Y = np.zeros((len(files), img.width, img.height), dtype=np.uint8)

    dsize = (X.size * X.itemsize + Y.size * Y.itemsize) * 1e-9

    if verbose:
        print("{:<10}{:4d}".format("samples", len(files)))
        print("{:<10}({:4d},{:4d})".format("Img size", img.width, img.height))
        print("{:<10}{:2.2f} GB".format("size", dsize))

    for i, (f_png, f_csv, _) in enumerate(files):
        X[i] = load_img(f_png)
        Y[i] = get_ndarray(f_csv, X[i].shape[0], X[i].shape[1]) * 255

    if verbose:
        print("X.shape: ", X.shape)
        print("Y.shape: ", Y.shape)

    return X, Y


def split_data(X, Y, test_size, verbose=True):
    if verbose:
        print("Splitting data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=0)
    if verbose:
        print("X_train: ", X_train.shape)
        print("X_test:  ", Y_test.shape)
        print("Y_train: ", Y_train.shape)
        print("Y_test:  ", Y_test.shape)
    return X_train, X_test, Y_train, Y_test


def square_img_split_idcs(img_size=2084, partition_size=512):
    assert img_size >= partition_size, "Cannot upsample img!"
    # number of partions on each axis
    partitions = int(np.ceil(img_size / partition_size))
    overhang = partitions * partition_size - img_size
    overlap = int(np.floor(overhang / partitions))
    stride = partition_size - overlap

    start = 0
    end = partition_size

    idcs = []
    for ix in range(partitions - 1):
        idcs.append(slice(start, end, 1))
        start += stride
        end += stride
    end = img_size
    start = end - partition_size
    idcs.append(slice(start, end, 1))

    idcs = list(itertools.product(idcs, repeat=2))

    assert len(idcs) == partitions**2
    return idcs


def split_and_store_img(X, Y, split_idcs, path, file_form, verbose=True):
    file_form = file_form + "{:d}_{}.png"
    i = 0
    X_assembled = (np.random.random(X.shape) * 255).astype(X.dtype)
    Y_assembled = (np.random.random(Y.shape) * 255).astype(Y.dtype)
    for idx in split_idcs:
        X_part = X[idx]
        Y_part = Y[idx]
        X_assembled[idx] = X_part
        Y_assembled[idx] = Y_part

        if np.sum(Y_part) > 0:
            path_x = os.path.join(path, file_form.format(i, "x"))
            path_y = os.path.join(path, file_form.format(i, "y"))

            imageio.imwrite(path_x, X_part)
            imageio.imwrite(path_y, Y_part)

            X_r = imageio.imread(path_x)
            Y_r = imageio.imread(path_y)

            assert np.sum(
                X_part - X_r) == 0, "X does not match written version!"
            assert np.sum(
                Y_part - Y_r) == 0, "Y does not match written version!"
            i += 1
    assert np.sum(X - X_assembled) == 0, "X does not match assembled version!"
    assert np.sum(Y - Y_assembled) == 0, "Y does not match assembled version!"


def load_split_data(files, verbose=True, scale=True):
    if verbose:
        print("Loading data...")
        file, _ = files[0]
    img = imageio.imread(file)
    width = img.shape[0]
    height = img.shape[1]
    X = np.zeros((len(files), width, height, 3), dtype=np.float32)
    Y = np.zeros((len(files), width, height), dtype=np.float32)

    dsize = (X.size * X.itemsize + Y.size * Y.itemsize) * 1e-9

    if verbose:
        print("{:<10}{:4d}".format("samples", len(files)))
        print("{:<10}({:4d},{:4d})".format("Img size", width, height))
        print("{:<10}{:2.2f} GB".format("size", dsize))

    for i, (f_x, f_y) in enumerate(files):
        X[i] = imageio.imread(f_x)
        Y[i] = imageio.imread(f_y)
    Y = np.expand_dims(Y, axis=3)
    if scale:
        X = X / 255
        Y = Y / 255
    return X, Y


if __name__ == "__main__":
    files = get_files("data", verbose=verbose)
    X, Y = load_data(files, verbose=verbose)
    X_train, X_test, Y_train, Y_test = split_data(
        X, Y, test_size, verbose=verbose)

    del X
    del Y

    if verbose:
        print("Saving data ...")

    np.save(os.path.join(d_data, f_x_train), X_train)
    np.save(os.path.join(d_data, f_x_test), X_test)
    np.save(os.path.join(d_data, f_y_train), Y_train)
    np.save(os.path.join(d_data, f_y_test), Y_test)
