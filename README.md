# EasyAgent

## Motivation

In enterprise environments, every department, team, and even individual contributors may have their own specialized agents.
EasyAgent is built to operate at a higher level: to provide a consistent way to build, connect, and manage these agents as a coordinated system rather than isolated tools.

EasyAgent is a Python SDK and service scaffold for building production-style AI agents with FastAPI, DeepAgents, and optional A2A interoperability.

It gives you a clean way to:
- run an LLM-powered agent behind HTTP APIs,
- add tools and subagents (local or remote),
- isolate users/sessions with pluggable auth,
- persist user/session metadata with SQLite or Postgres,
- expose and consume A2A agent endpoints.

## What EasyAgent Can Do

- Single-agent chat service via `POST /agent/run`.
- Multi-agent orchestration with local compiled subagents.
- Cross-service delegation through A2A gateway discovery.
- Gateway-based subagent discovery today, with an extension path for skill/tool discovery.
- User/session management APIs (`/users`, `/sessions`, etc.).
- Pluggable authentication (`Noop`, header-based, or custom callable provider).
- Local runtime sandbox for skills/memories/tmp file routes.
- Optional Langfuse tracing when `LANGFUSE_BASE_URL` is configured.

## Architecture

```text
+---------------------------+                         +---------------------------+
|     Agent A (Provider)    |                         |   Agent B (Orchestrator)  |
|---------------------------|                         |---------------------------|
| - Exposes A2A endpoint    |<========= A2A ==========| - Uses EasyagentSDK       |
| - Publishes AgentCard     |                         | - Runs DeepAgentRunner    |
| - Registers to Gateway    |                         | - Calls /agent/run        |
+-------------+-------------+                         +-------------+-------------+
              ^                                                     |
              |                                                     |
              |                                                     v
              |                                                     ^
              | Register endpoint + agent identity                  |
              v                                                     |
      +-------+-----------------------------------------------------+-------+
      |                      Gateway (Discovery Hub)                        |
      |--------------------------------------------------------------------|
      | - Agent registry (who is available)                                |
      | - Subagent discovery data                                           |
      | - Skills discovery data                                             |
      | - Tools discovery data                                              |
      +-------+-----------------------------------------------------+-------+
              |                                                     ^
              | Discover subagents / skills / tools                 |
              +-----------------------------------------------------+
```

```text
Agent A (provider)
  -> exposes A2A endpoint + AgentCard
  -> registers itself to Gateway

Gateway (registry/discovery hub)
  -> stores registered agent endpoints
  -> serves discovery data for:
     - subagents
     - skills
     - tools

Agent B (orchestrator)
  -> queries Gateway
  -> loads remote agents as subagents
  -> fetches skills/tools discovery metadata
  -> runs orchestration through EasyagentSDK + DeepAgentRunner

Client
  -> calls Agent B APIs (`/agent/run`, etc.)
```

### Layer Breakdown

- `easyagent/sdk.py`
  - Composition root of the system.
  - Creates DB, auth wiring, API router, agent runner, and optional A2A routes.
- `easyagent/agent/agent.py`
  - Builds and invokes the DeepAgent runtime.
  - Handles invoke payload shaping, thread/user propagation, callback wiring, and final output extraction.
- `easyagent/adapters/fastapi/`
  - HTTP API adapter (`/health`, `/agent/run`, `/users`, `/sessions`, ...).
  - Middleware injects request-level auth and logging context.
- `easyagent/adapters/a2a/`
  - Exposes A2A JSON-RPC endpoint and AgentCard.
  - Bridges A2A messages to `DeepAgentRunner`.
- `easyagent/agent/discovery.py`
  - Discovers remote A2A agents from a gateway and wraps them as callable subagents.
- `gateway/`
  - Lightweight agent registry service used by EasyAgent instances to discover remote subagents.
- `easyagent/services/` + `easyagent/repos/`
  - Service/repository split for user and session domain logic.
  - Backend factory supports SQLite and Postgres repo implementations.
- `easyagent/utils/settings.py`
  - Centralized env-based settings for model/runtime/database paths and toggles.

## API Surface

- `GET /health` -> service health.
- `POST /agent/run` -> run the agent with:
  - `input` or raw `invoke_input`,
  - optional `thread_id`, `user_id`, `files`, `invoke_config`.
- `POST /users`, `GET /users/by-external-id`
- `POST /sessions`, `GET /sessions/{session_id}`, `GET /users/{user_id}/sessions`
- A2A (enabled by default): `/.well-known/agent-card.json` and `/a2a`

## Gateway

The `gateway/` service is a lightweight discovery registry for distributed agent deployments.

Current capability:
- Register remote agent endpoints via `POST /agents`.
- List registered endpoints via `GET /agents`.
- Enable runtime subagent discovery in EasyAgent (`a2a_gateway_url`), where discovered A2A agents are converted into callable subagents.

Planned extension direction:
- Add discovery metadata and APIs for reusable skills.
- Add discovery metadata and APIs for tools.
- Evolve from "agent endpoint registry" to a broader "agent capability registry" (agents + skills + tools).

## Quick Start

### 1) Install

```bash
uv sync
```

### 2) Configure environment

Required:
- `EASYAGENT_MODEL_KEY`
- `EASYAGENT_MODEL_BASE_URL`
- `EASYAGENT_MODEL_NAME`

Optional examples:
- `EASYAGENT_BASE_PATH`
- `EASYAGENT_DB_BACKEND` (`sqlite` or `postgres`)
- `EASYAGENT_DB_URL` (required if Postgres)
- `LANGFUSE_BASE_URL` (+ Langfuse keys)

### 3) Run an example

```bash
uv run python examples/hello_world/hello_agent.py
```

Then call:

```bash
curl -X POST "http://127.0.0.1:8000/agent/run" \
  -H "Content-Type: application/json" \
  -d '{"input":"hi","thread_id":"thread_001"}'
```

## Example Scenarios

- Basic chat agent: `examples/hello_world/`
- Auth-integrated agent: `examples/auth/`
- Local multi-agent (master + math subagent): `examples/multi_agent/`
- A2A hello world + client: `examples/a2a/hello_world/`
- A2A gateway multi-agent topology: `examples/a2a/multi_agent/` + `gateway/`

## Project Layout

```text
easyagent/
  sdk.py
  agent/
  adapters/
  auth/
  models/
  repos/
  services/
  utils/
examples/
gateway/
```
