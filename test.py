import tensorflow as tf
import torchvision.models.shufflenetv2


def channel_shuffle(inputs, num_groups):
    
    # n, h, w, c = inputs.shape
    # x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    # x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    # output = tf.reshape(x_transposed, [-1, h, w, c])
    n, h, w, c = inputs.shape
    x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    output = tf.reshape(x_reshaped, [-1, h, w, c])
    
    return output

def conv(inputs, filters, kernel_size, strides=1):
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
        
    return x

def depthwise_conv_bn(inputs, kernel_size, strides=1):
    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
        
    return x

def ShuffleNetUnitA(inputs, out_channels):
    
    shortcut, x = tf.split(inputs, 2, axis=-1)
    
    x = conv(x, out_channels // 2, kernel_size=1, strides=1)
    x = depthwise_conv_bn(x, kernel_size=3, strides=1)
    x = conv(x, out_channels // 2, kernel_size=1, strides=1)
    
    x = tf.concat([shortcut, x], axis=-1)
    x = channel_shuffle(x, 2)
    
    return x

def ShuffleNetUnitB(inputs, out_channels):
    
    shortcut = inputs
    
    in_channels = inputs.shape[-1]
    
    x = conv(inputs, out_channels // 2, kernel_size=1, strides=1)
    x = depthwise_conv_bn(x, kernel_size=3, strides=2)
    x = conv(x, out_channels-in_channels, kernel_size=1, strides=1)
    
    shortcut = depthwise_conv_bn(shortcut, kernel_size=3, strides=2)
    shortcut = conv(shortcut, in_channels, kernel_size=1, strides=1)
    
    output = tf.concat([shortcut, x], axis=-1)
    output = channel_shuffle(output, 2)
    
    return output

def stage(inputs, out_channels, n):
    
    x = ShuffleNetUnitB(inputs, out_channels)
    
    for _ in range(n):
        x = ShuffleNetUnitA(x, out_channels)
        
    return x

def ShuffleNet(input_shape=(224, 224, 3), first_stage_channels=116):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=24, 
                               kernel_size=3, 
                               strides=2, 
                               padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = stage(x, first_stage_channels, n=3)
    x = stage(x, first_stage_channels*2, n=7)
    x = stage(x, first_stage_channels*4, n=3)
    
    x = tf.keras.layers.Conv2D(filters=1024, 
                               kernel_size=1, 
                               strides=1, 
                               padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000)(x)
    
    return tf.keras.Model(inputs, x)


net = ShuffleNet(first_stage_channels=48)
converter = tf.lite.TFLiteConverter.from_keras_model(net)
tflite_model = converter.convert()

with open('shufflenetv2_origin.tflite', 'wb') as f:
    f.write(tflite_model)
