import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)


class Trainer:
    """
    Class handling the training of our UNet.
    """

    def __init__(
        self,
        model,
        train_ds,
        val_ds,
        checkpoint_filepath,
        epochs=10,
        learning_rate=1e-3,
        patience=20,
    ):
        """
        Prepares the trainer and compiles the model.
        """
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.callbacks = []

        # learning rate scheduler callback
        self.callbacks.append(
            LearningRateScheduler(lambda epoch: learning_rate * (0.95 ** epoch))
        )

        # early stopping callback
        self.callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=patience,
                verbose=1,
                mode="min",
            )
        )

        # model checkpoint saver callback
        self.callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor="val_loss",
                mode="max",
                save_best_only=True,
            )
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
        )

    def train(self, num_samples_shown=5):
        """
        Runs the training loop.
        Args:
            show_samples (int): number of training results to be shown
                for the first 10 epochs and every 5 epochs afterwards
        """
        self.callbacks.append(
            VisualizePredictionsCallback(self.val_ds, num_samples_shown)
        )

        self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            callbacks=self.callbacks,
            verbose=1,
        )


class VisualizePredictionsCallback(keras.callbacks.Callback):
    """
    Custom callback for plotting validation results during training.
    """

    def __init__(self, val_ds, num_samples):
        super().__init__()
        self.val_ds = val_ds
        self.num_samples = num_samples

    def show_prediction(self, original, aged, restored, figsize=(7, 7)):
        """
        Plot one row, one triplet, containing the original image, the aged image,
        and the restored image for comparison.
        """
        f, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(original, cmap="gray")
        ax[0].set_title("Original")
        ax[1].imshow(aged, cmap="gray")
        ax[1].set_title("Aged")
        ax[2].imshow(restored, cmap="gray")
        ax[2].set_title("Restored")
        for j in range(3):
            ax[j].axis("off")
        plt.tight_layout()
        plt.show()

    def on_epoch_begin(self, epoch, logs=None):
        """
        Plot validation results at the beginning of every few epochs.
        """
        if epoch < 10 or epoch % 5 == 0:
            for x, y in self.val_ds.take(self.num_samples):
                pred = self.model.predict(x)
                i = np.random.randint(0, len(x), 1)[0]
                self.show_prediction(y[i], x[i], pred[i], figsize=(20, 20))
