.DEFAULT_GOAL := help

env_path = ./conda_env
python_path = $(env_path)/bin/python3.11
black_path = $(env_path)/bin/black
isort_path = $(env_path)/bin/isort
lint_path = $(env_path)/bin/pylint
core_path = core
scripts_path = scripts

.PHONY: format
format:
	@(\
	$(black_path) $(core_path); \
	$(isort_path) $(core_path) --skip __init__.py; \
	$(black_path) $(scripts_path); \
	$(isort_path) $(scripts_path) --skip __init__.py; \
	)

.PHONY: lint
lint:
	@(\
	$(lint_path) $(core_path); \
	$(lint_path) $(scripts_path); \
	)
