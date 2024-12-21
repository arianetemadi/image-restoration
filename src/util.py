import matplotlib.pyplot as plt


def show_prediction(original, aged, restored, figsize=(7, 7)):
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
