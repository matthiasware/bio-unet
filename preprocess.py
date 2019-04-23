import utils
import os
import numpy as np
import argparse

np.random.seed(0)

p_test = 0.15

d_data = "data"
d_raw = os.path.join(d_data, "raw")
d_preprocessed = "data"
partition_size = 512
img_size = 2084

d_preprocessed = os.path.join(d_data, str(partition_size))
d_train = os.path.join(d_preprocessed, "train")
d_test = os.path.join(d_preprocessed, "test")

file_form = "{}_{}.png"

if not os.path.exists(d_train):
    os.makedirs(d_train)
else:
    for file in os.listdir(d_train):
        os.unlink(os.path.join(d_train, file))

if not os.path.exists(d_test):
    os.makedirs(d_test)
else:
    for file in os.listdir(d_train):
            os.unlink(os.path.join(d_test, file))

files = utils.get_raw_files(d_raw, verbose=True)
X, Y = utils.load_imgs_from_raw_files(files, verbose=True)

split_idcs = utils.get_square_img_split_idcs(img_size, partition_size)


data = []
for i in range(X.shape[0]):
    xx, yy = utils.split_img(X[i], Y[i], split_idcs)
    for j in range(xx.shape[0]):
        if np.sum(yy[j]) > 0:
            data.append((xx[j], yy[j]))

del X
del Y

np.random.shuffle(data)

n_data = len(data)
n_test = int(np.ceil(n_data * 0.15))
n_train = n_data - n_test

for i in range(n_test):
    X, Y = data[i]
    f_x = os.path.join(d_test, file_form.format(i, "x"))
    f_y = os.path.join(d_test, file_form.format(i, "y"))
    utils.store_img(X, f_x)
    utils.store_img(Y, f_y)

for i in range(n_test, n_data):
    X, Y = data[i]
    f_x = os.path.join(d_train, file_form.format(i, "x"))
    f_y = os.path.join(d_train, file_form.format(i, "y"))
    utils.store_img(X, f_x)
    utils.store_img(Y, f_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesses raw mitosis image data set!")