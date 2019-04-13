import os
import numpy as np
import pickle
from preprocess import get_files


d_data = "data"
f_train = "train.pkl"
f_test = "test.pkl"
test_size = 0.5

if __name__ == "__main__":
    files = get_files(d_data, verbose=False)
    np.random.shuffle(files)
    np.random.shuffle(files)
    n_test = int(np.ceil(len(files) * 0.15))

    files_test = files[: n_test]
    files_train = files[n_test:]

    with open(os.path.join(d_data, f_train), "wb") as file:
        pickle.dump(files_train, file)

    with open(os.path.join(d_data, f_test), "wb") as file:
        pickle.dump(files_test, file)
