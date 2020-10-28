PACKAGE_NAME=lynks
DOCKER=docker
PYTHON = python

.PHONY: setup unsetup install uninstall clean dist build help

default: help

setup: ## Install poetry, requires python3 installed with pip
	python3 --version
	pip3 --version
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
	poetry --version

unsetup: ## Uninstalls pipenv
	make uninstall
	python3 get-poetry.py --uninstall

install: ## Installs local module and locks dependencies
	poetry check
	poetry install
	make lock_dependencies

lock_dependencies: ## Copy dependencies defined in pyproject.toml to requirements.txt
	poetry export -f requirements.txt > requirements.txt
	make test

resolve_dependencies:
	poetry update

uninstall: ## delete dependencies
	poetry env remove python
	rm -rf poetry.lock requirements.txt

clean: ## Cleans build
	-rm -rf build
	-rm -rf dist
	-find . -depth -name "__pycache__" -type d -exec rm -rf {} \;
	-rm -rf cover
	-rm -rf .coverage
	-rm -rf $(PACKAGE_NAME).egg-info

lint: ## run lint for code quality
	poetry run flake8 $(PACKAGE_NAME)

test: ## run test suite
	make lint
	poetry run pytest --cov ./tests

format: ## run formatting.
	poetry run autopep8 -i -r ./$(PACKAGE_NAME)

dist: ## Build distribution package.
	make clean
	poetry build

install_kernel: ## Install interactive jupyter kernel
	poetry run ipython kernel install --user --name=$(PACKAGE_NAME)

help:
	@grep -E '^[[:alnum:][:punct:]]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
