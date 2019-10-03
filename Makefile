install:
	pip install ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

install-dev:
	pip install -e ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

tests:
	pytest --cov mot tests/

.PHONY: tests
