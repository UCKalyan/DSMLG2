import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same", kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    p = layers.Dropout(0.5)(p) # Add Dropout after the pooling layer
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    #x = layers.Dropout(0.3)(x)
    return x

class UNET2D:
    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config['input_shape'])
        self.num_classes = config['num_classes']

    def build_model(self):
        """Builds a 2D UNET model."""
        inputs = layers.Input(self.input_shape)

        # Encoder
        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
        s4, p4 = encoder_block(p3, 512)

        # Bridge
        b1 = conv_block(p4, 1024)
        b1 = layers.Dropout(0.5)(b1)

        # Decoder
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        # Output
        outputs = layers.Conv2D(self.num_classes, 1, padding="same", activation="softmax")(d4)

        model = Model(inputs, outputs, name="UNET2D")
        return model
