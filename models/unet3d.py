import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block_3d(inputs, num_filters):
    x = layers.Conv3D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block_3d(inputs, num_filters):
    x = conv_block_3d(inputs, num_filters)
    x_reg = layers.Dropout(0.3)(x)
    p = layers.MaxPool3D((2, 2, 2), dtype='float32')(x_reg)
    return x, p

def decoder_block_3d(inputs, skip_features, num_filters):
    x = layers.Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block_3d(x, num_filters)
    return x

class UNET3D:
    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config['volume_shape'])
        self.num_classes = config['num_classes']

    def build_model(self):
        """Builds a 3D UNET model for segmentation."""
        inputs = layers.Input(self.input_shape)

        # Encoder
        s1, p1 = encoder_block_3d(inputs, 32)
        s2, p2 = encoder_block_3d(p1, 64)
        s3, p3 = encoder_block_3d(p2, 128)

        # Bridge
        b1 = conv_block_3d(p3, 256)

        # Decoder
        d1 = decoder_block_3d(b1, s3, 128)
        d2 = decoder_block_3d(d1, s2, 64)
        d3 = decoder_block_3d(d2, s1, 32)

        # Output
        outputs = layers.Conv3D(self.num_classes, 1, padding="same", activation="softmax")(d3)

        model = Model(inputs, outputs, name="UNET3D_Segmentation")
        return model
