install:
	pip install ./

install-dev:
	pip install -e ./
	python scripts/download_models_and_files.py

tests:
	./tests/run_tests.sh

docker-training:
	docker build -t mot_training -f docker/Dockerfile.training .
	docker run -it mot_training

docker-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker run mot_tests

.PHONY: tests docker
