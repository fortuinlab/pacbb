.DEFAULT_GOAL := help

env_path = ./conda_env
python_path = $(env_path)/bin/python3.11
black_path = $(env_path)/bin/black
isort_path = $(env_path)/bin/isort
lint_path = $(env_path)/bin/pylint
pytest_path = $(env_path)/bin/pytest
core_path = app
test_path = test

.PHONY: format
format:
	@(\
	$(black_path) $(app_path); \
	$(isort_path) $(app_path) --skip __init__.py; \
	$(black_path) $(test_path); \
	$(isort_path) $(test_path) --skip __init__.py; \
	)

.PHONY: lint
lint:
	@(\
	$(lint_path) $(app_path); \
	$(lint_path) $(test_path); \
	)

.PHONY: test
test:
	@(\
	$(pytest_path) $(test_path); \
	)