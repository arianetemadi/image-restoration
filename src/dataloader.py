import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt


class Dataloader:
    """
    Class for loading the tensorflow datasets for training and validation.
    Overlays noise and dirt texture on images.
    Performs image augmentation.
    """

    def __init__(
        self,
        image_dir,
        texture_dir,
        image_size,
        batch_size,
        color_mode="grayscale",
        validation_split=0.2,
        seed=123,
    ):
        self.image_dir = image_dir
        self.texture_dir = texture_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.validation_split = validation_split
        self.seed = seed

    def load_datasets(
        self,
        noisy=True,
        textured=True,
        texture_alpha=0.3,
        shuffle=False,
    ):
        """
        Main function of the class.
        It loads, normalizes, augments, adds noise, and overlays dirt texture on top of images.

        Args:
            noisy (boolean): Whether to add noise to the image, to simulate an aged look.
            textured (boolean): Whether to overlay dirt textures on top of images, to simulate an aged look.
            texture_alpha (float): Determines the transparency of texture overlay.
            shuffle (boolean): Whether to shuffle datasets.
        Returns:
            Pair of training dataset and validation dataset.
        """

        # load grayscale images
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            self.image_dir,
            labels=None,
            validation_split=self.validation_split,
            subset="both",
            seed=self.seed,
            color_mode=self.color_mode,
            image_size=self.image_size,
            interpolation="bilinear",
            crop_to_aspect_ratio=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

        # normalize pixel values to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x: normalization_layer(x))
        val_ds = val_ds.map(lambda x: normalization_layer(x))

        # spatial augmentations
        spatial_augmentations = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(
                    0.02, 0.02, fill_mode="reflect", interpolation="bilinear"
                ),
                layers.RandomRotation(
                    0.03, fill_mode="reflect", interpolation="bilinear"
                ),
                layers.RandomZoom(
                    -0.05, -0.05, fill_mode="reflect", interpolation="bilinear"
                ),
            ]
        )
        train_ds = train_ds.map(lambda x: spatial_augmentations(x, training=True))

        # pixel augmentations
        pixel_augmentations = tf.keras.Sequential(
            [
                layers.RandomBrightness(0.05, [0.0, 1.0]),
                layers.RandomContrast(0.05),
            ]
        )
        train_ds = train_ds.map(lambda x: pixel_augmentations(x, training=True))

        # create pairs of (input, target)
        train_ds = train_ds.map(lambda x: (x, x))
        val_ds = val_ds.map(lambda x: (x, x))

        # make x grayscale
        if self.color_mode == "rgb":
            train_ds = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
            val_ds = val_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))

        # simulate old image grain by mixing Gaussian noise
        if noisy:
            noise_layer = tf.keras.layers.GaussianNoise(0.10)
            clip_layer = tf.keras.layers.ReLU(max_value=1)
            train_ds = train_ds.map(
                lambda x, y: (clip_layer(noise_layer(x, training=True)), y)
            )
            val_ds = val_ds.map(
                lambda x, y: (clip_layer(noise_layer(x, training=True)), y)
            )

        # add textures on top to simulate the appearance of old worn-out photos
        if textured:
            texture_ds = tf.keras.utils.image_dataset_from_directory(
                self.texture_dir,
                labels=None,
                seed=self.seed,
                color_mode="grayscale",
                image_size=self.image_size,
                interpolation="bilinear",
                crop_to_aspect_ratio=True,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

            normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
            texture_ds = texture_ds.map(lambda x: normalization_layer(x))
            clip_layer = tf.keras.layers.ReLU(max_value=1)
            train_ds = train_ds.map(
                lambda x, y: (clip_layer(x + next(iter(texture_ds))) * texture_alpha, y)
            )
            val_ds = val_ds.map(
                lambda x, y: (clip_layer(x + next(iter(texture_ds))) * texture_alpha, y)
            )

        self.train_ds = train_ds
        self.val_ds = val_ds

        return self.train_ds, self.val_ds

    def show_samples(self, num_samples, fig_size=(7, 7)):
        """
        Function for visualizing a few samples of the loaded dataset.

        Args:
            num_samples (int): number of samples to show.

        Returns:
            None
        """
        counter = 0
        while True:
            x, y = next(iter(self.train_ds))
            for i in range(len(x)):
                self.show_sample(y[i], x[i], fig_size)
                counter += 1
                if counter == num_samples:
                    return

    def show_sample(self, original, aged, figsize):
        """
        Function that visualizes one sample in one row.
        Original (raw) image is on the left, aged (processed) image is on the right.

        Args:
            original: the raw version of the image.
            aged: the processed version of the image.

        Returns:
            None
        """
        f, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(original, cmap="gray")
        ax[0].set_title("Original (raw)")
        ax[1].imshow(aged, cmap="gray")
        ax[1].set_title("Aged (processed)")
        for j in range(2):
            ax[j].axis("off")
        plt.tight_layout()
        plt.show()
