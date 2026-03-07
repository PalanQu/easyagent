import unittest

from langchain_core.messages import HumanMessage

from tests.tools.mock_chat_model import MockRule, ScriptedMockChatModel


class TestScriptedMockChatModel(unittest.TestCase):
    def test_returns_plain_text_without_tool_call(self) -> None:
        model = ScriptedMockChatModel(
            rules=[MockRule(when_contains="hello", answer="world", call_tool=False)]
        )
        result = model.invoke([HumanMessage(content="hello there")])
        self.assertEqual(result.content, "world")
        self.assertEqual(result.tool_calls, [])

    def test_returns_tool_call_when_rule_requests_it(self) -> None:
        model = ScriptedMockChatModel(
            rules=[
                MockRule(
                    when_contains="weather",
                    answer="I will call weather tool",
                    call_tool=True,
                    tool_name="get_weather",
                    tool_args={"city": "Shanghai"},
                )
            ]
        )
        bound = model.bind_tools([{"name": "get_weather"}])
        result = bound.invoke([HumanMessage(content="check weather")])
        self.assertEqual(result.content, "I will call weather tool")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "get_weather")
        self.assertEqual(result.tool_calls[0]["args"], {"city": "Shanghai"})


if __name__ == "__main__":
    unittest.main()
