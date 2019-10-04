install:
	pip install ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
	python scripts/download_models_and_files.py

install-dev:
	pip install -e ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
	python scripts/download_models_and_files.py

tests:
	pytest --cov mot tests/

.PHONY: tests
