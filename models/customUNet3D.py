import tensorflow as tf
from tensorflow.keras import layers, Model

class ConvBlock(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = layers.Conv3D(out_channels, 3, padding='same', use_bias=False)
        self.norm1 = layers.LayerNormalization(axis=[1,2,3,4])
        self.act1 = layers.LeakyReLU()
        self.conv2 = layers.Conv3D(out_channels, 3, padding='same', use_bias=False)
        self.norm2 = layers.LayerNormalization(axis=[1,2,3,4])
        self.act2 = layers.LeakyReLU()
    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class CustomUNet3D(Model):
    def __init__(self, input_channels, num_classes, base_features=32):
        super().__init__()
        self.encoder1 = ConvBlock(base_features)
        self.down1 = layers.Conv3D(base_features*2, 2, strides=2)
        self.encoder2 = ConvBlock(base_features*2)
        self.down2 = layers.Conv3D(base_features*4, 2, strides=2)
        self.encoder3 = ConvBlock(base_features*4)
        self.down3 = layers.Conv3D(base_features*8, 2, strides=2)
        self.encoder4 = ConvBlock(base_features*8)
        self.down4 = layers.Conv3D(base_features*16, 2, strides=2)
        self.bottleneck = ConvBlock(base_features*16)
        self.up4 = layers.Conv3DTranspose(base_features*8, 2, strides=2)
        self.decoder4 = ConvBlock(base_features*16)
        self.up3 = layers.Conv3DTranspose(base_features*4, 2, strides=2)
        self.decoder3 = ConvBlock(base_features*8)
        self.up2 = layers.Conv3DTranspose(base_features*2, 2, strides=2)
        self.decoder2 = ConvBlock(base_features*4)
        self.up1 = layers.Conv3DTranspose(base_features, 2, strides=2)
        self.decoder1 = ConvBlock(base_features*2)
        self.final_conv = layers.Conv3D(num_classes, 1)
    def call(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))
        e4 = self.encoder4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))
        d4 = self.decoder4(tf.concat([self.up4(b), e4], axis=-1))
        d3 = self.decoder3(tf.concat([self.up3(d4), e3], axis=-1))
        d2 = self.decoder2(tf.concat([self.up2(d3), e2], axis=-1))
        d1 = self.decoder1(tf.concat([self.up1(d2), e1], axis=-1))
        return self.final_conv(d1)

# Example instantiation for BraTS (4 MR channels, 3 tumor regions):
model = UNet3D(input_channels=4, num_classes=3)
model.build(input_shape=(None, 128, 128, 128, 4))
model.summary()
