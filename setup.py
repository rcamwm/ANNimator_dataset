from setuptools import setup, find_packages

setup(
    name="ANNimator_dataset",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==1.7.1",
        "torchvision==0.8.2",
        "Pillow==7.2.0",
    ],
)
