.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests install dev serve setup

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

######################
# SETUP AND DEVELOPMENT
######################

install:
	pip install -e .

dev:
	pip install -e .[dev]

setup: dev
	@echo "Project setup complete! You can now run:"
	@echo "  make serve    - Start LangGraph development server"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linting"

serve:
	langgraph dev

test:
	PYTHONPATH=src python -m pytest $(TEST_FILE)

integration_tests:
	PYTHONPATH=src python -m pytest tests/integration_tests 

test_watch:
	PYTHONPATH=src python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	PYTHONPATH=src python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	PYTHONPATH=src python -m pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'Setup:'
	@echo 'setup                        - install dependencies and set up project'
	@echo 'install                      - install main dependencies only'
	@echo 'dev                          - install with dev dependencies'
	@echo 'serve                        - start LangGraph development server'
	@echo ''
	@echo 'Development:'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'

