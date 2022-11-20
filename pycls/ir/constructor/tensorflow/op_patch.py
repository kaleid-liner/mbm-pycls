import tensorflow as tf
import keras.layers as layers


def group_conv(x, filters, kernel_size, strides, padding, groups, name):
    if groups == 1:
        return layers.Conv2D(filters, 3, strides=strides, groups=groups, padding="same", name=name)(x)

    slices = tf.split(x, groups, -1, name = name + "_split")
    slices = [
        layers.Conv2D(filters // groups, kernel_size, 
            strides=strides, padding=padding, name=name + "_{}".format(i))(slice)
        for i, slice in enumerate(slices)
    ]
    return layers.Concatenate(name=name + "_concat")(slices)
