import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, array_to_img


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
