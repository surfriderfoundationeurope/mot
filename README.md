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

**The next subsections are useful to read if you want to train models or perform advanced tasks. However, if you just want to launch a serving container or perform inferences on one of those, directly jump to [this file](src/mot/serving/README.md).**

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

You can add arguments to the docker run command by specifying RUN_ARGS, for example:

```bash
RUN_ARGS="-v /srv/data:/srv/data" make docker-training
```

Do the following command to exec an already running container:

```bash
make docker-exec-training
```

### Internal tools

You can launch a jupyter notebook or a tensorboard server by running the command.

```bash
./scripts/run_jupyter.sh
```
or
```bash
./scripts/run_tensorboard.sh /path/to/the/model/folders/to/track
```
Then, access those servers through the ports you used in the Make command.

### Train

See the [original tensorpack README](src/mot/object_detection/README.md) for more details about the configurations and weights.
```bash
python3 -m mot.object_detection.train --load /path/to/pretrained/weights --config DATA.BASEDIR=/path/to/the/dataset --config TODO=SEE_TENSORPACK_README
```

The next files are pretrained weights on the dataset introduced previously:
- https://files.heuritech.com/raw_files/surfrider/resnet50_fpn/model-6000.index
- https://files.heuritech.com/raw_files/surfrider/resnet50_fpn/model-6000.data-00000-of-00001

The command used to train this model was:

```bash
python3 -m mot.object_detection.train --load /path/to/pretrained_weights/COCO-MaskRCNN-R50FPN2x.npz --logdir /path/to/logdir --config DATA.BASEDIR=/path/to/dataset MODE_MASK=False TRAIN.LR_SCHEDULE=250,500,750
```

Put those files in a folder, which will be `/path/to/your/trained/model` in the export section.

### Export

First, you need to train an object detection model. Then, you can export this model in SavedModel format:

```bash
python3 -m mot.object_detection.predict --load /path/to/your/trained/model --serving /path/to/serving --config DATA.BASEDIR=/path/to/the/dataset SAME_CONFIG=AS_TRAINING
```

The dataset should be the one downloaded following the instructions above. You can also use a folder with only [this file](http://files.heuritech.com/raw_files/surfrider/classes.json) inside if you don't want to download the whole dataset.
Also remember to use the same config as the one used for training (using FPN.CASCADE=True for instance).

### Serving

Refer to [this file](src/mot/serving/README.md).

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
