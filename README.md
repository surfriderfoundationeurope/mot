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

### Dataset

You can download a training dataset on this [link](http://files.heuritech.com/raw_files/dataset_surfrider_cleaned.zip).

### Installation

You may run directly the [notebook in colab](https://colab.research.google.com/github/surfriderfoundationeurope/mot/blob/master/notebooks/object_detection_training_and_inference.ipynb).

For more details on training and inference of the object detection please see the following [file](src/mot/object_detection/README.md) which is based on the README of tensorpack.

#### Classic

To install locally, make sure you have Python 3.3+ and  1.6 <= tensorflow < 2.0

```bash
apt install libsm6 libxrender-dev libxext6 libcap-dev ffmpeg
pip3 install --user .
```

#### Docker

The following command will build a docker for development and run interactively.

```bash
PORT_JUPYTER=22222 PORT_TENSORBOARD=22223 make docker-training
```

You don't have to specify the ports at the beginning of the command, but do so if you want to assign a specific port to access jupyter notebook and / or tensorboard.
You can launch a jupyter notebook or a tensorboard server by running the command.

```bash
./scripts/run_jupyter.sh
```
or
```bash
./scripts/run_tensorboard.sh /path/to/the/model/folders/to/track
```
Then, access those servers through the ports you used in the Make command.

Do the following command to exec an already running container:

```bash
make docker-exec-training
```

### Export

First, you need to train an object detection model following the instructions in [this file](src/mot/object_detection/REAME.md).
Then, you need to export this model in SavedModel format

```bash
python3 -m mot.object_detection.predict --load /path/to/your/trained/model --config DATA.BASEDIR=/path/to/the/dataset --serving /path/to/serving
```

The dataset should be the one downloaded following the instructions above. You can also use a folder with only [this file](http://files.heuritech.com/raw_files/surfrider/classes.json) inside if you don't want to download the whole dataset.
Also remember to use the same config as the one used for training (using FPN.CASCADE=True for instance).

### Serving

Then, you can launch the serving with:

```bash
NVIDIA_VISIBLE_DEVICES=2 MODEL_FOLDER=/path/to/serving PORT=the_port_you_want_to_expose make docker-serving
```

`NVIDIA_VISIBLE_DEVICES`allows you to specify the GPU you want to use for inferences.
For `MODEL_FOLDER`, you specify the path to the folder where the `saved_model.pb` file and `variables` folder stored.
The `PORT` is the one you'll use to make inference requests.
Then, you can request the server either by sending HTTP requests or by using the web interface at `host:port`.
For more details on this see the documentation related to [serving](src/mot/serving/utils.py).

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


- Within your local environement:

* To run all the tests:

```bash
make tests
```

* To run a specific test:

```bash
pytest my_file.py::my_function
```

- Within a docker environement:

* To run all the tests:

```bash
make docker-tests
```

* To run a specific test:

```bash
make up-tests
pytest my_file.py::my_function
```

### Status

Model & training:

- [x] Object detection training

- [ ] Improving train, validation and test dataset

- [ ] Model improvements

- [ ] Connection with dataset to query dataset

- [ ] Tracking model (WIP)

- [ ] test dataset for tracking

Inference and deployment:

- [x] Object detection inference notebook

- [ ] Inference on video (WIP)

- [ ] Connection with input data and inference

- [x] Small webserver and API (in local)

- [ ] Docker build and deployment
