# Pytorch profiler
Pytorchures is a simple model profiler intended for any pytorch model. 
It measures execution time of model layers individually. Every layer of a model is wrapped with timing class which measures latency when called.

TLDR;

Install
```
pip install pytorchures
```

Run
```
from pytorchures import wrap_model_layers

model = wrap_model_layers(model)

otuput = model(inputs)

profiling_data = model.get_timings()

with open(profiling_filename, "w") as f:
    json.dump(profiling_data, f, indent=4)
```

In the code above the model and all it's sublayers are wrapped with ```TimedLayer``` class using ```wrap_model_layers``` function. ```TimedLayer``` measures execution time when a layer is called and stores it for every time the model is called.
Execution times of every wrapped layer are retrieved as hierarchical dictionary using ```model.get_timings()```.
This dictionary can be saved to json file.

# Setup

This repo was developed under WSL 2 running Ubuntu 20.04 LTS, and Ubuntu 22.04 LTS. The editor of choice is VS Code. 

## Install python 

The code was tested for Python 3.10, if you want to run Python 3.11 or other version please subsitute the python version in the command below.

In case of running new WSL below are required packages and commands.

```sudo apt-get update```

```sudo apt-get install python3.10```

```sudo apt-get install python3.10-venv```

Install for PIL image.show() to work on WSL
```sudo apt install imagemagick```

## Install relevant VS Code extentions.

If you choose to use the recommended VS Code as editor please install the extensions from  ```extensions.json```.

## Create virtual environment

Create venv 

```python3.10 -m venv .venv```

To activate venv type - VS Code should automatically detect your new venv, so select it as your default interpreter.

```source venv/bin/activate```

## Install package in editable mode

In order to be able to develop and run the code install this repo in editable mode.

```pip install -e .```

To install in editable mode with additional dependencies for development use the command below.

```pip install -e .[dev]```

# Running

The entry point to profiling the sample object detection models is 
```run_profiling.py``` file.

## Examples

Running on CPU
```python pytorchures/run_profiling.py --device 'cpu' --nr_images 3```

Running on GPU
```python pytorchures/run_profiling.py --device 'cuda' --nr_images 3```

The script will print CPU wall time of every layer encountered in the model.
Values are printed in a nested manner showing deeper layers at further indentation.

# Testing

All tests are located in 'tests' folder. Please follow Arange-Act-Assert pattern for all tests.
The tests should load in the test explorer.

# Formatting

This repo uses 'Black' code formatter.
