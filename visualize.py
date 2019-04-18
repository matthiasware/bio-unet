import matplotlib.pyplot as plt


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
