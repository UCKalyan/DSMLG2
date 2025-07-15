import tensorflow as tf
from tensorflow.keras import layers, Model
from models.unet3d import encoder_block_3d, conv_block_3d

class Classifier3D:
    """
    3D CNN for benign vs. malignant classification.
    Can use a UNET-like encoder as a feature extractor.
    """
    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config['volume_shape'])

    def build_model(self):
        """Builds the 3D CNN classifier."""
        inputs = layers.Input(self.input_shape)

        # Encoder Feature Extractor
        _, p1 = encoder_block_3d(inputs, 32)
        _, p2 = encoder_block_3d(p1, 64)
        _, p3 = encoder_block_3d(p2, 128)

        # Bridge
        b1 = conv_block_3d(p3, 256)

        # Classifier Head
        x = layers.GlobalAveragePooling3D()(b1)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x) # Binary classification

        model = Model(inputs, outputs, name="Classifier3D")
        return model
