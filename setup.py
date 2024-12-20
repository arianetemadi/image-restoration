from setuptools import find_packages, setup

setup(
    name="image-restoration",
    version="0.0.1",
    description="Project for Applied Deep Learning course, 2024WS",
    author="Arian Etemadi",
    license="MIT",
    install_requires=[
        "ipykernel==6.29.5",
        "gdown==5.2.0",
        "pysocks==1.7.1",
        "matplotlib==3.10.0",
        "tensorflow==2.18.0",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
