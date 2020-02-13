docker_training_name = mot_training_$(USER)
docker_tests_name = mot_tests_$(USER)
docker_serving_name = mot_serving_$(USER)

install:
	pip install ./

install-dev:
	pip install -e ./
	python scripts/download_models_and_files.py

tests:
	./tests/prepare_tests.sh
	./tests/run_tests.sh

docker-training:
	docker build -t mot_training -f docker/Dockerfile.training .
	docker stop $(docker_training_name) || true
	docker rm $(docker_training_name) || true
	docker run --runtime=nvidia --name $(docker_training_name) -p $(PORT_JUPYTER):8888 -p $(PORT_TENSORBOARD):6006 -v $(shell pwd):/workspace/mot $(RUN_ARGS) -it mot_training

docker-exec-training:
	docker exec -it $(docker_training_name) bash

docker-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker stop $(docker_tests_name) || true
	docker rm $(docker_tests_name) || true
	docker run --name $(docker_tests_name) mot_tests

up-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker stop $(docker_tests_name) || true
	docker rm $(docker_tests_name) || true
	docker run -it --entrypoint bash --name $(docker_tests_name) -v $(HOME):/workspace mot_tests

docker-serving:
	docker stop $(docker_serving_name) || true
	docker rm $(docker_serving_name) || true
	./scripts/prepare_serving.sh
	docker build -f docker/Dockerfile.serving -t mot_serving .
	rm -r serving
	docker run -t --rm --name $(docker_serving_name) -p $(PORT):5000 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES -e RATIO_GPU \
        -e MODEL_NAME=serving \
        mot_serving

.PHONY: tests docker
