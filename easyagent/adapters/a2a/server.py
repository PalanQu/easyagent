import asyncio
import logging
from dataclasses import dataclass

from fastapi import FastAPI

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from easyagent.agent.agent import DeepAgentRunner
from easyagent.models.schema.agent import AgentRunRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class A2AServerConfig:
    public_base_url: str
    rpc_path: str = "/a2a"
    agent_name: str = "Easyagent"
    agent_description: str = "Easyagent compatible A2A agent."
    version: str = "0.1.0"
    skill_name: str = "General Assistant"
    skill_description: str = "General purpose assistant conversation."
    skill_tags: tuple[str, ...] = ("assistant", "chat")


class EasyagentA2AExecutor(AgentExecutor):
    def __init__(self, runner: DeepAgentRunner) -> None:
        self._runner = runner

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or "task_missing"
        context_id = context.context_id or "context_missing"
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)
        user_input = context.get_user_input().strip()
        if not user_input:
            await updater.reject(
                new_agent_text_message(
                    "A2A request requires text input in message.parts.",
                    context_id=context_id,
                    task_id=task_id,
                )
            )
            return

        try:
            await updater.submit()
            await updater.start_work()
            payload = AgentRunRequest(
                input=user_input,
                thread_id=context_id,
                user_id=self._resolve_user_id(context),
            )
            response = await asyncio.to_thread(self._runner.run, payload)
            final_output = response.final_output or ""
            await updater.complete(
                new_agent_text_message(
                    final_output if final_output else "(empty response)",
                    context_id=context_id,
                    task_id=task_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("A2A request execution failed")
            await updater.failed(
                new_agent_text_message(
                    f"Agent execution failed: {exc}",
                    context_id=context_id,
                    task_id=task_id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or "task_missing"
        context_id = context.context_id or "context_missing"
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)
        await updater.cancel(
            new_agent_text_message(
                "Task canceled.",
                context_id=context_id,
                task_id=task_id,
            )
        )

    def _resolve_user_id(self, context: RequestContext) -> str | None:
        call_context = context.call_context
        if not call_context:
            return None
        user = call_context.user
        if not user.is_authenticated:
            return None
        return user.user_name or None


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def mount_a2a_routes(
    app: FastAPI,
    *,
    runner: DeepAgentRunner,
    config: A2AServerConfig,
) -> None:
    rpc_path = "/" + config.rpc_path.strip("/")
    agent_card = AgentCard(
        name=config.agent_name,
        description=config.agent_description,
        version=config.version,
        url=_join_url(config.public_base_url, rpc_path),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            state_transition_history=True,
        ),
        skills=[
            AgentSkill(
                id="general_assistant",
                name=config.skill_name,
                description=config.skill_description,
                tags=list(config.skill_tags),
            )
        ],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=EasyagentA2AExecutor(runner),
        task_store=InMemoryTaskStore(),
    )
    A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).add_routes_to_app(
        app=app,
        rpc_url=rpc_path,
    )
