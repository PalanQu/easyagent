.PHONY: test test-model-injection test-mock-hello

test:
	uv run python -m unittest discover -s tests -p "test*.py" -v

test-model-injection:
	uv run python -m unittest tests/test_model_injection.py -v

test-mock-hello:
	uv run python -m unittest tests/test_mock_hello_agent.py -v
