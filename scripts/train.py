from src.dataloader import Dataloader
from src.train import Trainer
from src.model import UNet


if __name__ == "__main__":
    """
    Script for running the default training.
    For experimenting, refer to the training notebook instead.
    """

    image_dir = "../data/div2k-hr-train/"
    texture_dir = "../data/textures/"

    dataloader = Dataloader(
        image_dir,
        texture_dir,
        image_size=(32 * 21, 32 * 32),  # = (672, 1024)
        batch_size=8,
        validation_split=0.2,
    )

    # load the datasets for training and validation
    train_ds, val_ds = dataloader.load_datasets(
        noisy=True,
        textured=True,
        texture_alpha=0.1,
        shuffle=True,
    )

    # instantiate the model
    model = UNet()

    # instantiate the trainer
    trainer = Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        checkpoint_filepath="../checkpoints/checkpoint.weights.h5",
        epochs=10,
        learning_rate=1e-3,
    )

    # run the training loop
    trainer.train(num_samples_shown=4)
