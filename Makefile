.DEFAULT_GOAL := all
autoflake = ruff check --select F401,F841 neuralplexer_benchmarks
black = ruff format neuralplexer_benchmarks
isort = ruff check --select I neuralplexer_benchmarks
mypy = dmypy run --timeout 7200 -- neuralplexer_benchmarks
pylint = ruff check --select PL neuralplexer_benchmarks
pyupgrade = ruff check --select UP neuralplexer_benchmarks


.PHONY: install
install:
	conda env create -f environment.yaml --yes && \
	conda run -n np-bench-env pip install -e . --config-settings editable_mode=compat

.PHONY: install-dev
install-dev:
	conda env create -f environment-dev.yaml --yes && \
	conda run -n np-bench-env pip install -e . --config-settings editable_mode=compat

.PHONY: test
test:
	pytest -vvs --durations=10 neuralplexer_benchmarks/

.PHONY: test-failed
test-failed:
	pytest -v --lf neuralplexer_benchmarks/

.PHONY: all-quality
all-quality: format-check lint test

.PHONY: checks
checks: format-check mypy

.PHONY: format
format:
	$(autoflake) --fix
	$(isort) --fix
	$(black)

.PHONY: format-check
format-check:
	$(autoflake)
	$(isort)
	$(black) --check

.PHONY: format-diff
format-diff:
	$(autoflake) --diff
	$(isort) --diff
	$(black) --diff

.PHONY: lint
lint:
	$(pylint) --fix
	$(mypy)

.PHONY: mypy
mypy:
	$(mypy)

.PHONY: pyupgrade
pyupgrade:
	$(pyupgrade) --fix
	$(MAKE) format

.PHONY: pyupgrade-check
pyupgrade-check:
	$(pyupgrade)
	
.PHONY: check-dist
check-dist:
	python setup.py check -ms
	python setup.py sdist
	twine check dist/*
