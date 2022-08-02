import keras.layers as layers
import keras
import tensorflow as tf


def test_model():
    img_input = layers.Input(shape=(56, 56, 3))

    x = layers.Conv2D(64, 3, 1, 'same', name='cpu_conv1')(img_input)
    x2 = layers.Conv2D(64, 3, 1, 'same', name='cpu_conv2')(x)
    x = layers.Add(name='cpu_mp_add')([x, x2])
    x = layers.ReLU(name='cpu_mp_relu')(x)

    branch1 = layers.Conv2D(64, 3, 1, 'same', name='gpu_mb1_conv')(x)
    branch2 = layers.Conv2D(64, 3, 1, 'same', name='cpu_mb2_conv')(x)

    x = layers.Concatenate(name='gpu_mp_mb2_concat')([branch1, branch2])
    x = layers.GlobalAveragePooling2D(name="gpu_mp_mb2_gap")(x)
    x = layers.Dense(1000, name="gpu_mp_mb2_dense")(x)
    return tf.keras.Model(img_input, x)


net = test_model()
converter = tf.lite.TFLiteConverter.from_keras_model(net)
tflite_model = converter.convert()

with open('test_model.tflite', 'wb') as f:
    f.write(tflite_model)
