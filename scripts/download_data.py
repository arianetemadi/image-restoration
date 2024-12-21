from data.download_data import download_and_unzip_from_drive
from data.download_data import download_checkpoint_from_drive


if __name__ == "__main__":
    """
    Downloads the two datasets needed for this project:
        1. The high resolution images of the DIV2K dataset.
        2. Dirt texture for simulating an aged grainy look for images.
    """

    ## Load the DIV2K dataset,
    # for training
    url = "https://drive.google.com/file/d/15M1sLNX7uX16fo6UmszTlfmrKAmLizdB/view?usp=sharing"
    download_and_unzip_from_drive(url, "div2k-hr-train")

    # for testing
    url = "https://drive.google.com/file/d/15JuDY0_nkwnbV9SMmmyTtii6xrG3Gg4F/view?usp=sharing"
    download_and_unzip_from_drive(url, "div2k-hr-test")

    ## Load overlay textures
    url = "https://drive.google.com/file/d/1-35t5gG8JXVJ9n0S_nyvBEWheh2s5Bax/view?usp=sharing"
    download_and_unzip_from_drive(url, "textures")

    ### Load pretrained models
    url = "https://drive.google.com/file/d/15R_p0fHxM8qwHBQtnE7pWeXbqsmdrheP/view?usp=sharing"
    download_checkpoint_from_drive(url, "pretrained-grayscale")
    url = "https://drive.google.com/file/d/15RcEP-rc1fvKla4AHgGfh5MIRqMqnmvH/view?usp=sharing"
    download_checkpoint_from_drive(url, "pretrained-rgb")
