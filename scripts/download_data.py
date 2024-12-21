from data.download_data import load_data_google_drive


if __name__ == "__main__":
    """
    Downloads the two datasets needed for this project:
        1. The high resolution images of the DIV2K dataset.
        2. Dirt texture for simulating an aged grainy look for images.
    """

    ## Load the DIV2K dataset,
    # for training
    url = "https://drive.google.com/file/d/15M1sLNX7uX16fo6UmszTlfmrKAmLizdB/view?usp=sharing"
    load_data_google_drive(url, "div2k-hr-train")

    # for testing
    url = "https://drive.google.com/file/d/15JuDY0_nkwnbV9SMmmyTtii6xrG3Gg4F/view?usp=sharing"
    load_data_google_drive(url, "div2k-hr-test")

    ## Load overlay textures
    url = "https://drive.google.com/file/d/1-35t5gG8JXVJ9n0S_nyvBEWheh2s5Bax/view?usp=sharing"
    load_data_google_drive(url, "textures")
