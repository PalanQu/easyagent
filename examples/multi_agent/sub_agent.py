from langchain_core.tools import tool

from easyagent.sdk import CompiledSubAgent, EasyagentSDK, Settings


@tool
def add(a: float, b: float) -> float:
	"""Add two numbers."""
	return a + b


@tool
def subtract(a: float, b: float) -> float:
	"""Subtract b from a."""
	return a - b


@tool
def multiply(a: float, b: float) -> float:
	"""Multiply two numbers."""
	return a * b


@tool
def divide(a: float, b: float) -> float:
	"""Divide a by b."""
	if b == 0:
		raise ValueError("division by zero is not allowed")
	return a / b


def build_math_subagent(settings: Settings) -> CompiledSubAgent:
	sub_sdk = EasyagentSDK(
		settings=settings,
		system_prompt=(
			"You are the math subagent and only handle arithmetic calculations. "
			"Use these tools for calculation: add, subtract, multiply, divide. "
			"If a request is outside arithmetic, clearly state your limitations."
		),
		tools=[add, subtract, multiply, divide],
	)

	return {
		"name": "math_agent",
		"description": "Performs addition, subtraction, multiplication, and division",
		"runnable": sub_sdk.agent_runner._agent,
	}
