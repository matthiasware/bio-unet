from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, \
    Input, ZeroPadding2D


def get_model(n_kernels, img_height, img_width):
    # contraction path
    inputs = Input((img_height, img_width, 3))

    # In: 2084 x 2084
    conv1 = Conv2D(n_kernels, (3, 3), padding="same", name="conv_1_1",
                   activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(n_kernels, (3, 3), padding="same", name="conv_2_2",
                   activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1",
                         strides=2, data_format="channels_last")(conv1)

    # In: 1042 x 1042
    conv2 = Conv2D(n_kernels * 2, (3, 3), padding="same", name="conv2_1",
                   activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(n_kernels * 2, (3, 3), padding="same", name="conv2_2",
                   activation="relu", data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2",
                         strides=2, data_format="channels_last")(conv2)

    # In: 521 x 521
    conv3 = Conv2D(n_kernels * 4, (3, 3), padding="same", name="conv_3_1",
                   activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(n_kernels * 4, (3, 3), padding="same", name="conv_3_2",
                   activation="relu", data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3",
                         strides=2, data_format="channels_last")(conv3)

    # In: 260 x 260
    conv4 = Conv2D(n_kernels * 8, (3, 3), padding="same", name="conv_4_1",
                   activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(n_kernels * 8, (3, 3), padding="same", name="conv_4_2",
                   activation="relu", data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4",
                         strides=2, data_format="channels_last")(conv4)

    # bottom
    # In: 130 x 130
    conv5 = Conv2D(n_kernels * 16, (3, 3), padding="same", name="conv_5_1",
                   activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(n_kernels * 16, (3, 3), padding="same", name="conv_5_2",
                   activation="relu", data_format="channels_last")(conv5)
    up_conv5 = UpSampling2D(size=(2, 2), name="ups_5",
                            data_format="channels_last")(conv5)
    # Out: 260 x 260 => no cropping

    up6 = concatenate([up_conv5, conv4], name="conc_5_4", axis=3)
    conv6 = Conv2D(n_kernels * 8, (3, 3), name="conv_6_1", padding="same",
                   activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(n_kernels * 8, (3, 3), name="conv_6_2", padding="same",
                   activation="relu", data_format="channels_last")(conv6)
    up_conv6 = UpSampling2D(size=(2, 2), name="ups_6",
                            data_format="channels_last")(conv6)
    # Out 520 x 520 => crop

    crop_conv3 = Cropping2D(cropping=(
        (0, 1), (0, 1)),
        name="crop_conv_3", data_format="channels_last")(conv3)
    up7 = concatenate([up_conv6, crop_conv3], name="conc_6_3", axis=3)
    conv7 = Conv2D(n_kernels * 4, (3, 3), name="conv_7_1", padding="same",
                   activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(n_kernels * 4, (3, 3), name="conv_7_2", padding="same",
                   activation="relu", data_format="channels_last")(conv7)
    up_conv7 = UpSampling2D(size=(2, 2), name="ups_7",
                            data_format="channels_last")(conv7)
    # 1040 x 1040 => crop

    crop_conv2 = Cropping2D(cropping=(
        (1, 1), (1, 1)),
        name="crop_conv_2", data_format="channels_last")(conv2)
    up8 = concatenate([up_conv7, crop_conv2], name="conc_7_2", axis=3)
    conv8 = Conv2D(n_kernels * 2, (3, 3), name="conv_8_1", padding="same",
                   activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(n_kernels * 2, (3, 3), name="conv_8_2", padding="same",
                   activation="relu", data_format="channels_last")(conv8)
    up_conv8 = UpSampling2D(size=(2, 2), name="ups_8",
                            data_format="channels_last")(conv8)
    # 2080 x 2080 => crop

    crop_conv1 = Cropping2D(cropping=(
        (2, 2), (2, 2)),
        name="crop_conv_1", data_format="channels_last")(conv1)
    up9 = concatenate([up_conv8, crop_conv1], name="conc_8_1", axis=3)
    conv9 = Conv2D(n_kernels * 1, (3, 3), name="conv_9_1", padding="same",
                   activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(n_kernels * 1, (3, 3), name="conv_9_2", padding="same",
                   activation="relu", data_format="channels_last")(conv9)
    # 2080 x 2080 => padding

    conv9 = ZeroPadding2D(padding=(2, 2), name="0pad",
                          data_format="channels_last")(conv9)
    conv10 = Conv2D(1, (1, 1), name="conv_10",
                    data_format="channels_last", activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)
    return model
