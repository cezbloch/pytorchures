# Pytorch profiler
Pytorchures is a simple model profiler intended for any pytorch model. 
It measures execution time of model layers individually. Every layer of a model is wrapped with timing class which measures latency when called.

## TLDR;

Install
```
pip install pytorchures
```

Run
```
from pytorchures import TimedModule

model = TimedModule(model)

_output_ = model(inputs)

profiling_data = model.get_timings()

with open(profiling_filename, "w") as f:
    json.dump(profiling_data, f, indent=4)
```

One layer extract of sample output ```.json```

```
    {
        "module_name": "InvertedResidual",
        "device_type": "cuda",
        "execution_times_ms": [
            5.021333694458008,
            2.427816390991211,
            2.4025440216064453
        ],
        "mean_time_ms": 3.283898035685221,
        "median_time_ms": 2.427816390991211,
        "sub_modules": [
            {
                "module_name": "Sequential",
                "device_type": "cuda",
                "execution_times_ms": [
                    4.198789596557617,
                    1.9135475158691406,
                    1.9412040710449219
                ],
                "mean_time_ms": 2.684513727823893,
                "median_time_ms": 1.9412040710449219,
                "sub_modules": [
                    {
                        "module_name": "Conv2dNormActivation",
                        "device_type": "cuda",
                        "execution_times_ms": [
                            2.0263195037841797,
                            0.7545948028564453,
                            0.9317398071289062
                        ],
                        "mean_time_ms": 1.2375513712565105,
                        "median_time_ms": 0.9317398071289062,
                        "sub_modules": [
                            ...
                                                    
```

# Setup

This repo was developed under WSL 2 running Ubuntu 20.04 LTS, and Ubuntu 22.04 LTS. The editor of choice is VS Code. 

## Install python 

The code was tested for Python 3.11, if you want to run other release please subsitute the python version in commands below which install python and virtual environment.

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
Values are printed in a nested manner.

## TimedModule wrapper

```
from pytorchures import TimedModule

model = TimedModule(model)

_output = model(inputs)

profiling_data = model.get_timings()

with open(profiling_filename, "w") as f:
    json.dump(profiling_data, f, indent=4)
```

In the code above the model and all it's sublayers are recursively wrapped with ```TimedModule``` class which measures execution times when a layers are called and stores them for every time the model is called.
Execution times of every wrapped layer are retrieved as hierarchical dictionary using ```model.get_timings()```.
This dictionary can be saved to json file.

If for some reason there is a need to clear recorded timings call ```model.clear_timings()```. This may useful in only some of the measurements should be included in the final results. It is often the case that first inference run takes much more time due to resource initialization, so clearing the measurements is a way to exclude this first run.

# Testing

All tests are located in 'tests' folder. Please follow Arange-Act-Assert pattern for all tests.
The tests should load in the test explorer.

# Formatting

This repo uses 'Black' code formatter.

# Publishing to Pypi

Build the package. This command will create a ```dist``` folder with ```pytorchures``` package as ```.whl```  and ```tar.gz```.

```python -m build```

Check if the build pacakge were build correctly.

```twine check dist/*```

Optionally upload the new package to ```testpypi``` server.

```twine upload -r testpypi dist/*```

To test the package from use ```testpypi``` the command:

```pip install --index-url https://test.pypi.org/simple/ pytorchures```

Upload the new package to production ```pypi``` server.

```twine upload dist/*```