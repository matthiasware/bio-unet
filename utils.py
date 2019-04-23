import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import itertools
import imageio


def show_img(img1, img2=None):
    nrows = 1
    ncols = 1 if img2 is None else 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    if img2 is None:
        ax.imshow(img1)
    else:
        ax[0].imshow(img1)
        print(img2.shape)
        if len(img2.shape) == 3:
            img2 = img2.reshape(img2.shape[:-1])
        ax[1].imshow(img2)
    plt.show()


def get_ndarray_from_csv(path, img_width, img_height):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        img = np.zeros((img_width, img_height))
        for row in csv_reader:
            xx = np.array(row[0::2], dtype=np.int)
            yy = np.array(row[1::2], dtype=np.int)
            img[yy, xx] = 1
    return img


def get_raw_files(path, verbose=True):
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


def get_files(path, verbose=True):
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


def load_imgs_from_raw_files(files, verbose=True):
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
        Y[i] = get_ndarray_from_csv(f_csv, X[i].shape[0],
                                    X[i].shape[1]) * 255
    if verbose:
        print("X.shape: ", X.shape)
        print("Y.shape: ", Y.shape)

    return X, Y


def get_square_img_split_idcs(img_size, partition_size):
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


def split_img(X, Y, split_idcs):
    X_assembled = (np.random.random(X.shape) * 255).astype(X.dtype)
    Y_assembled = (np.random.random(Y.shape) * 255).astype(Y.dtype)

    X_part = X[split_idcs[0]]
    Y_part = Y[split_idcs[0]]

    X_new = np.zeros((len(split_idcs), *X_part.shape)).astype(X.dtype)
    Y_new = np.zeros((len(split_idcs), *Y_part.shape)).astype(Y.dtype)

    for i, idx in enumerate(split_idcs):
        X_new[i] = X[idx]
        Y_new[i] = Y[idx]
        X_assembled[idx] = X_new[i]
        Y_assembled[idx] = Y_new[i]

    assert np.sum(X - X_assembled) == 0, "X does not match assembled version!"
    assert np.sum(Y - Y_assembled) == 0, "Y does not match assembled version!"

    return X_new, Y_new


def store_img(X, path):
    imageio.imwrite(path, X)
    X_read = imageio.imread(path)
    assert np.sum(X_read - X) == 0, "X does not match written version!"


def split_and_store_img(X, Y, split_idcs, path, file_form):
    file_form = file_form + "{:d}_{}.png"
    X_assembled = (np.random.random(X.shape) * 255).astype(X.dtype)
    Y_assembled = (np.random.random(Y.shape) * 255).astype(Y.dtype)
    i = 0
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


def load_data(files, verbose=True, normalize=True):
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
    if normalize:
        X = X / 255
        Y = Y / 255
    return X, Y
