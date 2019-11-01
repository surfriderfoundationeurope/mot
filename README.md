# MOT
_________________

[![Build Status](https://travis-ci.com/surfriderfoundationeurope/mot.svg?branch=master)](https://travis-ci.com/surfriderfoundationeurope/mot)
[![codecov.io](https://codecov.io/gh/surfriderfoundationeurope/mot/coverage.svg?branch=master)](https://codecov.io/gh/surfriderfoundationeurope/mot/?branch=master)
_________________

[Read Latest Documentation](https://surfriderfoundationeurope.github.io/mot/) - [Browse GitHub Code Repository](https://github.com/surfriderfoundationeurope/mot)
_________________

## Project

### Installation

Make sure you have Python 3.3+ and  1.6 <= tensorflow < 2.0

```bash
apt install libsm6 libxrender-dev libxext6 libcap-dev ffmpeg
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

This package is based on the example FasterRCNN coming from [tensorpack](https://github.com/tensorpack/tensorpack). See this [notebook](https://colab.research.google.com/github/surfriderfoundationeurope/mot/blob/master/notebooks/object_detection_training_and_inference.ipynb) for more details on training and inference.

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
