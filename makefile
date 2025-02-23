.DEFAULT_GOAL := help

core_path = core

.PHONY: docs
docs:
	@(\
	pdoc --docformat google --output-dir docs $(core_path); \
	)
