install:
	pip install ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

install-dev:
	pip install -e ./
	pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
	python scripts/download_models_and_files.py

tests:
	python scripts/download_models_and_files.py
	pytest --cov mot tests/

docker:
	docker build -t mot_training -f docker/Dockerfile.training .

docker-tests: docker
	docker build -t mot_tests -f docker/Dockerfile.tests .


.PHONY: tests docker
