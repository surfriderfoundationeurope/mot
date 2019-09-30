[![Build Status](https://travis-ci.com/surfriderfoundationeurope/mot.svg?branch=master)](https://travis-ci.com/surfriderfoundationeurope/mot)
[![codecov.io](https://codecov.io/gh/surfriderfoundationeurope/mot/coverage.svg?branch=master)](https://codecov.io/gh/surfriderfoundationeurope/mot/?branch=master)

# MOT

## Project

### Installation


* System Python3

```bash
pip3 install --user .
```

* If you use pyenv

```bash
pyenv activate my_amazing_surfrider_project
pip install .
```


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




