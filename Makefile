install:
	pip install ./

install-dev:
	pip install -e ./

tests:
	pytest --cov mot tests/

.PHONY: tests
