import tensorflow as tf
from tensorflow.keras import layers, models


class UNet(tf.keras.Model):
    """
    Implementation of the UNet model in keras, as introduced in:
    https://arxiv.org/abs/1505.04597
    """

    def __init__(self, color_mode="grayscale"):
        """
        Initializes building blocks of the model.

        Args:
            color_mode (str): either "grayscale" or "rgb".

        Returns:
            UNet: Instance of the created model.
        """
        super(UNet, self).__init__()

        self.color_mode = color_mode

        # Encoder Blocks
        self.enc1 = models.Sequential(
            [
                layers.Conv2D(
                    16,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(None, None, 1),
                ),
                layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            ]
        )
        self.pool1 = layers.MaxPooling2D((2, 2))

        self.enc2 = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            ]
        )
        self.pool2 = layers.MaxPooling2D((2, 2))

        self.enc3 = models.Sequential(
            [
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            ]
        )
        self.pool3 = layers.MaxPooling2D((2, 2))

        # Bottleneck
        self.bottleneck = models.Sequential(
            [
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            ]
        )

        # Decoder Blocks
        self.upconv3 = layers.Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding="same"
        )
        self.dec3 = models.Sequential(
            [
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            ]
        )

        self.upconv2 = layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding="same"
        )
        self.dec2 = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            ]
        )

        self.upconv1 = layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding="same"
        )
        self.dec1 = models.Sequential(
            [
                layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
                layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            ]
        )

        # Output Layer
        num_output_channels = 1 if self.color_mode == "grayscale" else 3
        self.final = layers.Conv2D(num_output_channels, (1, 1), activation="sigmoid")

    def call(self, inputs):
        """
        Forwards the inputs through the pipeline.
        """
        # Encoder Path
        x1 = self.enc1(inputs)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        # Bottleneck
        x4 = self.bottleneck(p3)

        # Decoder Path
        up3 = self.upconv3(x4)
        cat3 = layers.concatenate([up3, x3], axis=-1)
        x5 = self.dec3(cat3)

        up2 = self.upconv2(x5)
        cat2 = layers.concatenate([up2, x2], axis=-1)
        x6 = self.dec2(cat2)

        up1 = self.upconv1(x6)
        cat1 = layers.concatenate([up1, x1], axis=-1)
        x7 = self.dec1(cat1)

        # Final Output
        output = self.final(x7)

        return output
