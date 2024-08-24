from setuptools import setup, find_packages

setup(
    name='object_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'torch',
        'torchvision',
        'Pillow',
        'requests',
        'pycocotools'
    ],
)