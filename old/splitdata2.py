import preprocess

files = preprocess.get_files("data", verbose=False)
X, Y = preprocess.load_imgs(files[0:1])
X = X[0]
Y = Y[0]
print(X.shape, Y.shape)

idcs = preprocess.square_img_split_idcs(2084, 512)
preprocess.split_and_store_img(X, Y, idcs, "data/split/")
