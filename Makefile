install:
	pip install ./

install-dev:
	pip install -e ./
	python scripts/download_models_and_files.py

tests:
	./tests/run_tests.sh

docker-training:
	docker build -t mot_training -f docker/Dockerfile.training .
	docker run --name mot_training_$(USER) -it mot_training

docker-tests:
	docker build -t mot_tests -f docker/Dockerfile.tests .
	docker run --name mot_tests_$(USER) mot_tests

docker-serving:
	docker stop mot_serving_$(USER) || true
	docker rm mot_serving_$(USER) || true
	rm -r serving || true
	cp -r $(model_folder) serving
	docker build -f docker/Dockerfile.serving -t mot_serving .
	docker run -t --rm --name mot_serving_$(USER)  -p $(port):5000 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
        -e MODEL_NAME=serving \
        mot_serving

.PHONY: tests docker
