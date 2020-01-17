
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
	docker stop mot_training_$(USER) || true
	docker rm mot_training_$(USER) || true
	docker run --name mot_training_$(USER) -p $(PORT_JUPYTER):8888 -p $(PORT_TENSORBOARD):6006 -v /srv/data:/srv/data -v $(HOME):/workspace -it mot_training

docker-exec-training: 
	docker exec -it mot_training_$(USER) bash

docker-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker stop mot_tests_$(USER) || true
	docker rm mot_tests_$(USER) || true
	docker run --name mot_tests_$(USER) mot_tests

up-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker stop mot_tests_$(USER) || true
	docker rm mot_tests_$(USER) || true
	docker run -it --entrypoint bash --name mot_tests_$(USER) -v $(HOME):/workspace mot_tests
	
docker-serving:
	docker stop mot_serving_$(USER) || true
	docker rm mot_serving_$(USER) || true
	rm -r serving || true
	cp -r $(MODEL_FOLDER) serving
	docker build -f docker/Dockerfile.serving -t mot_serving .
	rm -r serving
	docker run -t --rm --name mot_serving_$(USER)  -p $(PORT):5000 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES \
        -e MODEL_NAME=serving \
        mot_serving

.PHONY: tests docker
