import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, array_to_img
from sklearn.model_selection import train_test_split

d_data = "data"
f_x_train = "x_train"
f_x_test = "x_test"
f_y_train = "y_train"
f_y_test = "y_test"
test_size = 0.1
verbose = True


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


def get_ndarray_from_csv(path, width, height):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        img = np.zeros((width, height, 1))
        for row in csv_reader:
            xx = np.array(row[0::2], dtype=np.int)
            yy = np.array(row[1::2], dtype=np.int)
            img[yy, xx] = 1
    return img


def load_data(files, verbose=True):
    if verbose:
        print("Loading data...")
    f_png, _, _ = files[0]
    img = load_img(f_png)

    X = np.zeros((len(files), img.width, img.height, 3), dtype=np.float64)
    Y = np.zeros((len(files), img.width, img.height, 1), dtype=np.float64)

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


if __name__ == "__main__":
    files = get_files("data", verbose=verbose)
    X, Y = load_data(files, verbose=verbose)
    X_train, X_test, Y_train, Y_test = split_data(
        X, Y, test_size, verbose=verbose)

    del X
    del Y

    if verbose:
        print("Saving data ...")

    np.save(f_x_train, X_train)
    np.save(f_x_test, X_test)
    np.save(f_y_train, Y_train)
    np.save(f_y_test, Y_test)
