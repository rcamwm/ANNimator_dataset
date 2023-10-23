from setuptools import setup, find_packages

setup(
    name="ANNimator_dataset",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
    ],
)
