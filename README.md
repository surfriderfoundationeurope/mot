[![Build Status](https://travis-ci.com/surfriderfoundationeurope/mot.svg?branch=master)](https://travis-ci.com/surfriderfoundationeurope/mot)
[![codecov.io](https://codecov.io/gh/surfriderfoundationeurope/mot/coverage.svg?branch=master)](https://codecov.io/gh/surfriderfoundationeurope/mot/?branch=master)

# MOT

## Project

### Installation

Make sure you have Python 3.3+ and  1.6 <= tensorflow < 2.0

```bash
apt install libsm6 libxrender-dev libxext6 libcap-dev
```


* System Python3

```bash
pip3 install --user .
pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

```

* If you use pyenv

```bash
pyenv activate my_amazing_surfrider_project
pip install .
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### Object detection

This package is the example FasterRCNN coming from [tensorpack](https://github.com/tensorpack/tensorpack).


### Make a prediction with pre trained weights

- save this [file](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2x.npz) correspoding to the weights of a ResNet-50 with FPN
- python3 mot/src/mot/object_detection/predict.py --load /path/to/weights --predict /path/to/image.jpg

### Using a custom dataset

The main development to perform in this folder is described in [this file](src/mot/object_detection/dataset/dataset.py), in order to read a dataset.

For more details, see the official [README.md](src/mot/object_detection/README.md)

## Developpers

Please read the [CONTRIBUTING.md](./CONTRIBUTING.md)

### Developper installation

You need to install the repository in dev:

```bash
pip install -e ./
```

The following libraries are needed to run the tests: `pytest`, `pytest-cov`


### Run the tests

* To run all the tests:

```bash
make tests
```

* To run a specific test

```bash
pytest my_file.py::my_function
```




