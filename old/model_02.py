from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose


def get_model(img_size, n_kernels):
    assert bin(img_size).count('1') == 1, "Image size must be power of 2!"

    inputs = Input((img_size, img_size, 3))

    # In: 512
    conv1 = Conv2D(n_kernels, (3, 3), padding="same", name="conv_1_1",
                   activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(n_kernels, (3, 3), padding="same", name="conv_2_2",
                   activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1",
                         strides=2, data_format="channels_last")(conv1)

    # In: 512
    conv2 = Conv2D(n_kernels * 2, (3, 3), padding="same", name="conv2_1",
                   activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(n_kernels * 2, (3, 3), padding="same", name="conv2_2",
                   activation="relu", data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2",
                         strides=2, data_format="channels_last")(conv2)

    # In: 512
    conv3 = Conv2D(n_kernels * 4, (3, 3), padding="same", name="conv_3_1",
                   activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(n_kernels * 4, (3, 3), padding="same", name="conv_3_2",
                   activation="relu", data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3",
                         strides=2, data_format="channels_last")(conv3)

    # In: 256
    conv4 = Conv2D(n_kernels * 8, (3, 3), padding="same", name="conv_4_1",
                   activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(n_kernels * 8, (3, 3), padding="same", name="conv_4_2",
                   activation="relu", data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4",
                         strides=2, data_format="channels_last")(conv4)

    # bottom
    # In: 128
    conv5 = Conv2D(n_kernels * 16, (3, 3), padding="same", name="conv_5_1",
                   activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(n_kernels * 16, (3, 3), padding="same", name="conv_5_2",
                   activation="relu", data_format="channels_last")(conv5)
    up_conv5 = Conv2DTranspose(
        n_kernels * 16, (3, 3), strides=(2, 2), padding='same')(conv5)
    # Out: 260 x 260 => no cropping

    up6 = concatenate([up_conv5, conv4], name="conc_5_4", axis=3)

    conv6 = Conv2D(n_kernels * 8, (3, 3), name="conv_6_1", padding="same",
                   activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(n_kernels * 8, (3, 3), name="conv_6_2", padding="same",
                   activation="relu", data_format="channels_last")(conv6)
    up_conv6 = Conv2DTranspose(
        n_kernels * 8, (3, 3), strides=(2, 2), padding='same')(conv6)

    up7 = concatenate([up_conv6, conv3], name="conc_6_3", axis=3)

    conv7 = Conv2D(n_kernels * 4, (3, 3), name="conv_7_1", padding="same",
                   activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(n_kernels * 4, (3, 3), name="conv_7_2", padding="same",
                   activation="relu", data_format="channels_last")(conv7)
    up_conv7 = Conv2DTranspose(
        n_kernels * 4, (3, 3), strides=(2, 2), padding='same')(conv7)

    up8 = concatenate([up_conv7, conv2], name="conc_7_2", axis=3)
    conv8 = Conv2D(n_kernels * 2, (3, 3), name="conv_8_1", padding="same",
                   activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(n_kernels * 2, (3, 3), name="conv_8_2", padding="same",
                   activation="relu", data_format="channels_last")(conv8)
    up_conv8 = Conv2DTranspose(
        n_kernels * 2, (3, 3), strides=(2, 2), padding='same')(conv8)

    up9 = concatenate([up_conv8, conv1], name="conc_8_1", axis=3)
    conv9 = Conv2D(n_kernels * 1, (3, 3), name="conv_9_1", padding="same",
                   activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(n_kernels * 1, (3, 3), name="conv_9_2", padding="same",
                   activation="relu", data_format="channels_last")(conv9)

    conv10 = Conv2D(1, (1, 1), name="conv_10",
                    data_format="channels_last", activation="sigmoid")(conv9)
    model = Model(input=inputs, output=conv10)
    return model


if __name__ == "__main__":
    model = get_model(512, 16)
    print(model.summary())
