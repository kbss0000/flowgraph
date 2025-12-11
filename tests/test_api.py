"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app


# ============================================================
# Sync Test Client (for simple tests)
# ============================================================

client = TestClient(app)


class TestRootEndpoints:
    """Tests for root endpoints."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


class TestToolsEndpoints:
    """Tests for tools endpoints."""
    
    def test_list_tools(self):
        """Test listing tools."""
        response = client.get("/tools/")
        assert response.status_code == 200
        
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert data["total"] > 0
        
        # Check that built-in tools are present
        tool_names = [t["name"] for t in data["tools"]]
        assert "extract_functions" in tool_names
        assert "calculate_complexity" in tool_names
    
    def test_get_tool(self):
        """Test getting a specific tool."""
        response = client.get("/tools/extract_functions")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "extract_functions"
        assert "description" in data
    
    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        response = client.get("/tools/nonexistent_tool")
        assert response.status_code == 404


class TestGraphEndpoints:
    """Tests for graph endpoints."""
    
    def test_list_graphs(self):
        """Test listing graphs."""
        response = client.get("/graph/")
        assert response.status_code == 200
        
        data = response.json()
        assert "graphs" in data
        assert "total" in data
    
    def test_get_demo_workflow(self):
        """Test getting the demo workflow."""
        response = client.get("/graph/code-review-demo")
        assert response.status_code == 200
        
        data = response.json()
        assert data["graph_id"] == "code-review-demo"
        assert data["name"] == "Code Review Demo"
        assert "mermaid_diagram" in data
    
    def test_create_graph(self):
        """Test creating a new graph."""
        graph_data = {
            "name": "test_workflow",
            "description": "A test workflow",
            "nodes": [
                {"name": "start", "handler": "extract_functions"},
                {"name": "end", "handler": "calculate_complexity"}
            ],
            "edges": {
                "start": "end"
            },
            "entry_point": "start"
        }
        
        response = client.post("/graph/create", json=graph_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "graph_id" in data
        assert data["name"] == "test_workflow"
        assert data["node_count"] == 2
    
    def test_create_graph_invalid_handler(self):
        """Test creating a graph with invalid handler."""
        graph_data = {
            "name": "invalid_workflow",
            "nodes": [
                {"name": "bad", "handler": "nonexistent_handler"}
            ],
            "edges": {}
        }
        
        response = client.post("/graph/create", json=graph_data)
        assert response.status_code == 404


# ============================================================
# Async Tests (for async endpoints)
# ============================================================

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_run_demo_workflow():
    """Test running the demo workflow."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        run_data = {
            "graph_id": "code-review-demo",
            "initial_state": {
                "code": "def hello():\n    print('world')",
                "quality_threshold": 5.0
            },
            "async_execution": False
        }
        
        response = await ac.post("/graph/run", json=run_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "run_id" in data
        assert data["status"] in ["completed", "failed"]
        assert "execution_log" in data


@pytest.mark.asyncio
async def test_async_execution():
    """Test async execution mode."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        run_data = {
            "graph_id": "code-review-demo",
            "initial_state": {
                "code": "def test(): pass",
                "quality_threshold": 5.0
            },
            "async_execution": True
        }
        
        response = await ac.post("/graph/run", json=run_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"
        
        # Check run state
        run_id = data["run_id"]
        state_response = await ac.get(f"/graph/state/{run_id}")
        assert state_response.status_code == 200


@pytest.mark.asyncio
async def test_run_nonexistent_graph():
    """Test running a graph that doesn't exist."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        run_data = {
            "graph_id": "nonexistent-graph",
            "initial_state": {}
        }
        
        response = await ac.post("/graph/run", json=run_data)
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
