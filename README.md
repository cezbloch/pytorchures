# Object Detection project
Welcome to the Object Detection (OO) documentation. Below are instructions how to setup and run this repo.

# Setup

This repo was developed under WSL 2 running Ubuntu 20.04 LTS. The editor of choice is VS Code. 

## Install python 

The code was tested for Python 3.11, if you want to run Python 3.10 or other future version please subsitute the python version in the command below.

In case of running new WSL below are required packages and commands.
```sudo apt-get update```
```sudo apt-get install python3.11```
```sudo apt-get install python3.11-venv```

Install for PIL image.show() to work on WSL
```sudo apt install imagemagick```

## Install relevant VS Code extentions.

If you choose to use the recommended VS Code as editor please install the extensions from  ```extensions.json```.

## Create virtual environment

Create venv 
```python3.11 -m venv .venv```
Install requirements.txt
```pip install -r requirements.txt```

To activate venv type - VS Code should automatically detect your new venv, so select it as your default interpreter.
```source venv/bin/activate```

## Install package in editable mode

In order to be able to develop and run the code install this repo in editable mode.
```pip install -e .```

# Running

The entry point to profiling the selected model 'retinanet_resnet50_fpn' is 
```run_profiling.py``` file.

Example:
```python object_detection/run_profiling.py --device 'cpu' --nr_images 3```

# Testing

All tests are located in 'tests' folder. Please follow Arange-Act-Assert pattern for all tests.
The tests should load in the test explorer.