---
title: FlowGraph
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# FlowGraph

A lightweight, async-first workflow orchestration engine for building agent pipelines in Python.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A minimal but powerful graph-based workflow engine similar to [LangGraph](https://github.com/langchain-ai/langgraph). Define sequences of steps (nodes), connect them with edges, maintain shared state, and run workflows via REST APIs.

**Live Demo:** https://kbsss-flowgraph.hf.space/docs

---

## Features

| Feature | Description |
|---------|-------------|
| Nodes | Python functions that read and modify shared state |
| Edges | Define which node runs after which |
| Branching | Conditional routing based on state values |
| Looping | Run nodes repeatedly until conditions are met |
| Async | Full async/await support for scalability |
| WebSocket | Real-time execution streaming |
| Visualization | Auto-generated Mermaid diagrams |

---

## Quick Start

### With Docker (Recommended)

```bash
git clone https://github.com/kbss0000/flowgraph.git
cd flowgraph
docker compose up -d
curl http://localhost:8000/health
```

### Without Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

**Access Points:**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

---

## API Reference

### Graph Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/graph/create` | Create a new workflow graph |
| `GET` | `/graph/` | List all graphs |
| `GET` | `/graph/{graph_id}` | Get graph details + Mermaid diagram |
| `POST` | `/graph/run` | Execute a graph |
| `GET` | `/graph/state/{run_id}` | Get execution state |

### Tool Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tools/` | List all registered tools |
| `POST` | `/tools/register` | Register a new tool dynamically |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/run/{graph_id}` | Execute with real-time streaming |

---

## Sample Workflow: Code Review

The included demo workflow analyzes Python code quality:

```
Extract Functions -> Check Complexity -> Detect Issues --+--> END (pass)
                                                         |
                                                         +--> Improve -> (loop back)
```

### Try It

```bash
curl -X POST "https://kbsss-flowgraph.hf.space/graph/run" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_id": "code-review-demo",
    "initial_state": {
      "code": "def hello():\n    print(\"world\")",
      "quality_threshold": 6.0
    }
  }'
```

---

## Architecture

```
flowgraph/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   ├── api/
│   │   ├── schemas.py       # Pydantic models
│   │   └── routes/
│   │       ├── graph.py     # Graph CRUD + execution
│   │       ├── tools.py     # Tool management
│   │       └── websocket.py # Real-time streaming
│   ├── engine/
│   │   ├── state.py         # Immutable state management
│   │   ├── node.py          # Node definitions + decorators
│   │   ├── graph.py         # Graph structure + validation
│   │   └── executor.py      # Async workflow executor
│   ├── tools/
│   │   ├── registry.py      # Tool registry
│   │   └── builtin.py       # Built-in tools
│   ├── workflows/
│   │   └── code_review.py   # Demo workflow
│   └── storage/
│       └── memory.py        # In-memory storage
├── tests/                   # Test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Immutable state | Predictable flow, easier debugging, clear state transitions |
| Async-first | Scalability for long-running or I/O-bound workflows |
| Tool registry | Decouples node logic from handlers, enables dynamic registration |
| Named conditions | Clean serialization, human-readable graph definitions |
| In-memory storage | Simplicity first; easily swappable for Redis/PostgreSQL |
| Max iterations | Safety guard against infinite loops |

---

## Testing

```bash
# Run tests in Docker
docker compose exec workflow-engine pytest tests/ -v

# Run locally
pytest tests/ -v
```

---

## What I Would Improve

With more time, I would add:

1. Persistent Storage - PostgreSQL/Redis for production
2. Parallel Execution - Run independent nodes concurrently
3. Checkpointing - Save/restore execution state
4. Retry Logic - Automatic retry on node failures
5. Metrics - Prometheus/Grafana integration
6. Authentication - API key / JWT support
7. Visual Editor - Web UI for building workflows

---

## Creating Custom Workflows

### 1. Define a Node Handler

```python
from app.tools.registry import register_tool

@register_tool("my_processor")
def my_processor(data: str) -> dict:
    return {"result": data.upper()}
```

### 2. Create via API

```json
POST /graph/create
{
  "name": "my_workflow",
  "nodes": [
    {"name": "step1", "handler": "my_processor"},
    {"name": "step2", "handler": "another_tool"}
  ],
  "edges": {"step1": "step2"},
  "entry_point": "step1"
}
```

### 3. Run It

```json
POST /graph/run
{
  "graph_id": "returned_graph_id",
  "initial_state": {"data": "hello"}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
