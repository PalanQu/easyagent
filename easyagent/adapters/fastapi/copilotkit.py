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
    except ImportError as exc:
        raise RuntimeError(
            "CopilotKit-compatible AG-UI integration requires `ag-ui-langgraph` to be installed."
        ) from exc

    # `ag-ui-langgraph` can expose a plain compiled LangGraph directly.
    # `name` and `description` are kept in our SDK API for future compatibility,
    # even though the current transport adapter does not consume them.
    _ = (name, description)
    add_langgraph_fastapi_endpoint(app, graph, path)
