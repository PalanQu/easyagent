.PHONY: test

test:
	uv run python -m unittest discover -s tests -p "test*.py" -v
