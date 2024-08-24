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