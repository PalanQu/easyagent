import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from easyagent.models.schema.agent import AgentRunRequest
from easyagent.sdk import EasyagentSDK, Settings
from tests.tools import MockRule, ScriptedMockChatModel


class _SkillToolMockModel(BaseChatModel):
    def __init__(self):
        super().__init__()
        self._bound_tools: set[str] = set()

    @property
    def _llm_type(self) -> str:
        return "skill-tool-mock-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        bound = _SkillToolMockModel()
        names: set[str] = set()
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    name = tool.get("name")
                else:
                    name = getattr(tool, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
        bound._bound_tools = names
        return bound

    def _generate(  # noqa: PLR0913
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,  # noqa: ANN003
    ) -> ChatResult:
        user_text = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_text = str(message.content)
                break

        tool_content = None
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                tool_content = str(getattr(message, "content", ""))
                break

        if "weather skill" in user_text.lower():
            if tool_content is None:
                if self._bound_tools and "skill" not in self._bound_tools:
                    raise ValueError("tool `skill` is not bound on model")
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="I will load the weather skill.",
                                tool_calls=[
                                    {
                                        "id": f"call_{uuid4().hex}",
                                        "name": "skill",
                                        "args": {"name": "weather"},
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )
            if '<skill name="weather">' in tool_content and "WEATHER_SKILL_MARKER" in tool_content:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="Loaded weather skill successfully."))]
                )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Weather skill not found."))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mocked: fallback"))])


class TestWeatherSkillAgent(unittest.TestCase):
    @staticmethod
    def _extract_tool_names(sdk: EasyagentSDK) -> set[str]:
        tools_node = sdk.agent_runner._agent.nodes["tools"]
        tool_node = tools_node.bound
        return set(tool_node._tools_by_name.keys())

    def test_local_skill_adds_skill_tool_and_can_load_weather_skill(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            skills_path = base_path / "skills"
            weather_dir = skills_path / "weather"
            weather_dir.mkdir(parents=True, exist_ok=True)
            (weather_dir / "SKILL.md").write_text(
                """---
name: weather
description: Weather helper skill for tests.
---
# Weather Skill

WEATHER_SKILL_MARKER
Use this skill when user asks weather-related questions.
""",
                encoding="utf-8",
            )

            settings = Settings(
                model_key="dummy-key",
                model_base_url="https://example.com/v1",
                model_name="real-model-not-used",
                base_path=base_path,
                skills_path=skills_path,
                local_mode=True,
            )

            sdk_without_skills = EasyagentSDK(
                settings=settings,
                model=ScriptedMockChatModel(rules=[MockRule(when_contains="x", answer="y")]),
            )
            base_tools = self._extract_tool_names(sdk_without_skills)
            sdk_without_skills.agent_runner.close()

            sdk_with_skills = EasyagentSDK(
                settings=settings,
                system_prompt="You can load skills.",
                skills=["/skills/"],
                model=_SkillToolMockModel(),
            )
            skill_tools = self._extract_tool_names(sdk_with_skills)

            self.assertIn("skill", skill_tools)
            self.assertEqual(len(skill_tools), len(base_tools) + 1)

            response = sdk_with_skills.agent_runner.run(
                AgentRunRequest(
                    input="Please load weather skill now.",
                    thread_id="thread_test_weather_skill_001",
                    user_id="user_test_weather_001",
                )
            )
            self.assertIn("loaded weather skill", (response.final_output or "").lower())
            sdk_with_skills.agent_runner.close()


if __name__ == "__main__":
    unittest.main()
