# MOT
_________________

[![Build Status](https://travis-ci.com/surfriderfoundationeurope/mot.svg?branch=master)](https://travis-ci.com/surfriderfoundationeurope/mot)
[![codecov.io](https://codecov.io/gh/surfriderfoundationeurope/mot/coverage.svg?branch=master)](https://codecov.io/gh/surfriderfoundationeurope/mot/?branch=master)
_________________

[Read Latest Documentation](https://surfriderfoundationeurope.github.io/mot/) - [Browse GitHub Code Repository](https://github.com/surfriderfoundationeurope/mot)
_________________

## Project

Welcome to MOT, the garbage detection on river banks github. It is part of a project led by [Surfrider Europe](https://surfrider.eu/), which aims at quantifying plastic pollution in rivers through space and time.

MOT stands for Multi-Object Tracking, as we detect, then track the different plastic trash instances.

The object detection part is based on [tensorpack](https://github.com/tensorpack/tensorpack).

### Installation

You may run directly the [notebook in colab](https://colab.research.google.com/github/surfriderfoundationeurope/mot/blob/master/notebooks/object_detection_training_and_inference.ipynb)

To install locally, make sure you have Python 3.3+ and  1.6 <= tensorflow < 2.0

```bash
apt install libsm6 libxrender-dev libxext6 libcap-dev ffmpeg
pip3 install --user .
```


## Developpers

Please read the [CONTRIBUTING.md](./CONTRIBUTING.md)

### Developper installation

You need to install the repository in dev:

```bash
pip install -e ./
```

The following libraries are needed to run the tests: `pytest`, `pytest-cov`

### Use with pyenv

```bash
pyenv activate my_amazing_surfrider_project
pip install .
```

### Run the tests

* To run all the tests:

```bash
make tests
```

* To run a specific test

```bash
pytest my_file.py::my_function
```
### Status

Model & training
- [x] Object detection training
- [ ] Improving train, validation and test dataset
- [ ] Model improvements
- [ ] Connection with dataset to query dataset
- [ ] Tracking model (WIP)
- [ ] test dataset for tracking

Inference and deployment
- [x] Object detection inference notebook
- [ ] Inference on video (WIP)
- [ ] Connection with input data and inference
- [ ] Small webserver and API
- [ ] Docker build and deployment
