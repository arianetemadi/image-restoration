import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


class Trainer():
    def __init__(
            self,
            model,
            train_ds,
            val_ds,
            epochs=10,
            learning_rate=1e-3,
            patience=20,
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=patience,
            verbose=1,
            mode='min'
        )

        self.lr_schedule = LearningRateScheduler(lambda epoch: learning_rate * (0.95 ** epoch))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

    def train(self, show_samples=5):
        self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            callbacks=[self.lr_schedule, VisualizePredictionsCallback(self.val_ds, show_samples)],
            verbose=1,
        )


class VisualizePredictionsCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, num_samples):
        super().__init__()
        self.val_ds = val_ds
        self.num_samples = num_samples

    def show_prediction(self, original, aged, restored, figsize=(7, 7)):
        f, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title("Original")
        ax[1].imshow(aged, cmap='gray')
        ax[1].set_title("Aged")
        ax[2].imshow(restored, cmap='gray')
        ax[2].set_title("Restored")
        for j in range(3):
            ax[j].axis('off')
        plt.tight_layout()
        plt.show()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 10 or epoch % 5 == 0:
            for x, y in self.val_ds.take(self.num_samples):
                pred = self.model.predict(x)
                i = np.random.randint(0, len(x), 1)[0]
                self.show_prediction(y[i], x[i], pred[i], figsize=(20, 20))
