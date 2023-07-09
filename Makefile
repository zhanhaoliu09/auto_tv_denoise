.PHONY: build
build:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	clean-build clean-pyc ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: lint
lint: ## check style with flake8
	black --check --line-length=120  auto_tv_denoise
	flake8 auto_tv_denoise
	pylint auto_tv_denoise

.PHONY: test
test:
	pytest tests


.PHONY: uninstall
uninstall:
	pip uninstall -y auto_tv_denoise


.PHONY: install
install:
	python setup.py install


.PHONY: format
format:
	isort  --profile black auto_tv_denoise
	black  --line-length=120 auto_tv_denoise