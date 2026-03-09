from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph


def mount_copilotkit_routes(
    app: FastAPI,
    *,
    graph: CompiledStateGraph,
    path: str,
    name: str,
    description: str,
) -> None:
    try:
        from ag_ui_langgraph import add_langgraph_fastapi_endpoint
        from copilotkit import LangGraphAGUIAgent
    except ImportError as exc:
        raise RuntimeError(
            "CopilotKit AG-UI integration requires both `copilotkit` and `ag-ui-langgraph` to be installed."
        ) from exc

    agent = LangGraphAGUIAgent(
        name=name,
        description=description,
        graph=graph,
    )
    add_langgraph_fastapi_endpoint(app, agent, path)
