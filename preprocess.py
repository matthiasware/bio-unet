import argparse
import os
import utils
import numpy as np
np.random.seed(0)


def check_out(val):
    val = int(val)
    if not (val != 0 and ((val & (val - 1)) == 0)):
        raise argparse.ArgumentTypeError("Must be power of 2!")
    return val


def check_split(val):
    val = float(val)
    if not 0 < val < 1:
        raise argparse.ArgumentTypeError("Must be in (0, 1)")
    return val


if __name__ == "__main__":
    default_in = 2084
    default_out = 512
    default_raw = os.path.join("data", "raw")
    default_target = os.path.join("data", str(default_out))
    default_split = 0.15
    parser = argparse.ArgumentParser(description="Preprocesses raw images",
                                     prefix_chars="-")
    parser.add_argument('-r', '--raw',
                        nargs='?',
                        type=str,
                        default=default_raw,
                        help='path to raw files!')
    parser.add_argument('-t', '--target',
                        nargs='?',
                        help="path to target folder",
                        default=default_target,
                        type=str)
    parser.add_argument('-i', '--insize',
                        nargs='?',
                        help="input image size",
                        default=default_in,
                        type=int)
    parser.add_argument('-o', '--outsize',
                        nargs='?',
                        help="output partition size (must be power of 2)",
                        default=default_out,
                        type=check_out)
    parser.add_argument('-s', '--split',
                        nargs='?',
                        default=default_split,
                        help="test split e (0, 1)",
                        type=check_split)

    args = parser.parse_args()
    d_train = os.path.join(args.target, "train")
    d_test = os.path.join(args.target, "test")
    d_raw = args.raw
    img_size = args.insize
    partition_size = args.outsize

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

    print("Searching files in: ", d_raw)
    files = utils.get_raw_files(d_raw, verbose=False)
    print("Found files: ", len(files))
    print("Loading data ...")

    X, Y = utils.load_imgs_from_raw_files(files, verbose=False)

    split_idcs = utils.get_square_img_split_idcs(
        img_size,
        partition_size)

    print("Splitting images ...")
    print("Input size:     ", img_size)
    print("Partition size: ", partition_size)

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
    n_test = int(np.ceil(n_data * args.split))
    n_train = n_data - n_test

    print("Created {} new image mask pairs".format(n_data))
    print("#Train: ", n_train)
    print("#Test:  ", n_test)

    print("split: ", args.split)
    print("train: ", d_train)
    print("test:  ", d_test)

    print("Saving test images to disk ...")
    for i in range(n_test):
        X, Y = data[i]
        f_x = os.path.join(d_test, file_form.format(i, "x"))
        f_y = os.path.join(d_test, file_form.format(i, "y"))
        utils.store_img(X, f_x)
        utils.store_img(Y, f_y)

    print("Saving train images to disk ...")
    for i in range(n_test, n_data):
        X, Y = data[i]
        f_x = os.path.join(d_train, file_form.format(i, "x"))
        f_y = os.path.join(d_train, file_form.format(i, "y"))
        utils.store_img(X, f_x)
        utils.store_img(Y, f_y)
    print("Done")
