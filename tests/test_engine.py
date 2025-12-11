"""
Tests for the Workflow Engine core components.
"""

import pytest
import asyncio
from typing import Dict, Any

from app.engine.state import WorkflowState, StateManager
from app.engine.node import Node, NodeType, node, create_node_from_function
from app.engine.graph import Graph, END
from app.engine.executor import Executor, ExecutionStatus, execute_graph


# ============================================================
# State Tests
# ============================================================

class TestWorkflowState:
    """Tests for WorkflowState."""
    
    def test_create_empty_state(self):
        """Test creating an empty state."""
        state = WorkflowState()
        assert state.data == {}
        assert state.iteration == 0
        assert state.visited_nodes == []
    
    def test_create_state_with_data(self):
        """Test creating state with initial data."""
        state = WorkflowState(data={"key": "value"})
        assert state.get("key") == "value"
        assert state.get("missing") is None
        assert state.get("missing", "default") == "default"
    
    def test_state_immutability(self):
        """Test that state updates return new instances."""
        state1 = WorkflowState(data={"a": 1})
        state2 = state1.set("b", 2)
        
        assert state1.get("b") is None
        assert state2.get("b") == 2
        assert state1 is not state2
    
    def test_state_update_multiple(self):
        """Test updating multiple values at once."""
        state = WorkflowState(data={"a": 1})
        new_state = state.update({"b": 2, "c": 3})
        
        assert new_state.get("a") == 1
        assert new_state.get("b") == 2
        assert new_state.get("c") == 3
    
    def test_state_mark_visited(self):
        """Test marking nodes as visited."""
        state = WorkflowState()
        state = state.mark_visited("node1")
        state = state.mark_visited("node2")
        
        assert "node1" in state.visited_nodes
        assert "node2" in state.visited_nodes
        assert state.current_node == "node2"
    
    def test_state_to_from_dict(self):
        """Test serialization and deserialization."""
        state = WorkflowState(data={"test": 123})
        state_dict = state.to_dict()
        
        assert "data" in state_dict
        assert state_dict["data"]["test"] == 123
        
        restored = WorkflowState.from_dict(state_dict)
        assert restored.get("test") == 123


class TestStateManager:
    """Tests for StateManager."""
    
    def test_initialize(self):
        """Test state manager initialization."""
        manager = StateManager()
        state = manager.initialize({"input": "test"})
        
        assert manager.current_state is not None
        assert manager.current_state.get("input") == "test"
        assert manager.current_state.started_at is not None
    
    def test_update_and_history(self):
        """Test state updates create history."""
        manager = StateManager()
        state = manager.initialize({"count": 0})
        
        new_state = state.set("count", 1)
        manager.update(new_state, "node1")
        
        assert len(manager.history) == 1
        assert manager.history[0].node_name == "node1"
        assert manager.current_state.get("count") == 1


# ============================================================
# Node Tests
# ============================================================

class TestNode:
    """Tests for Node class."""
    
    def test_create_node(self):
        """Test creating a node."""
        def handler(state):
            return state
        
        n = Node(name="test_node", handler=handler)
        
        assert n.name == "test_node"
        assert n.handler == handler
        assert n.node_type == NodeType.STANDARD
    
    def test_node_validation(self):
        """Test node validation."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Node(name="", handler=lambda x: x)
        
        with pytest.raises(ValueError, match="must be callable"):
            Node(name="test", handler="not a function")
    
    @pytest.mark.asyncio
    async def test_sync_node_execution(self):
        """Test executing a sync node."""
        def handler(state):
            state["processed"] = True
            return state
        
        n = Node(name="test", handler=handler)
        result = await n.execute({"input": "data"})
        
        assert result["processed"] is True
        assert result["input"] == "data"
    
    @pytest.mark.asyncio
    async def test_async_node_execution(self):
        """Test executing an async node."""
        async def async_handler(state):
            await asyncio.sleep(0.01)
            state["async_processed"] = True
            return state
        
        n = Node(name="async_test", handler=async_handler)
        assert n.is_async is True
        
        result = await n.execute({"input": "data"})
        assert result["async_processed"] is True
    
    def test_node_decorator(self):
        """Test the @node decorator."""
        @node(name="decorated_node", description="A test node")
        def my_handler(state):
            return state
        
        assert hasattr(my_handler, "_node_metadata")
        assert my_handler._node_metadata["name"] == "decorated_node"


# ============================================================
# Graph Tests
# ============================================================

class TestGraph:
    """Tests for Graph class."""
    
    def test_create_graph(self):
        """Test creating a graph."""
        graph = Graph(name="Test Graph")
        assert graph.name == "Test Graph"
        assert len(graph.nodes) == 0
    
    def test_add_nodes(self):
        """Test adding nodes to a graph."""
        graph = Graph()
        graph.add_node("node1", handler=lambda s: s)
        graph.add_node("node2", handler=lambda s: s)
        
        assert "node1" in graph.nodes
        assert "node2" in graph.nodes
        assert graph.entry_point == "node1"  # First node is entry
    
    def test_add_edges(self):
        """Test adding edges."""
        graph = Graph()
        graph.add_node("a", handler=lambda s: s)
        graph.add_node("b", handler=lambda s: s)
        graph.add_edge("a", "b")
        
        assert graph.edges["a"] == "b"
    
    def test_add_edge_to_end(self):
        """Test adding edge to END."""
        graph = Graph()
        graph.add_node("a", handler=lambda s: s)
        graph.add_edge("a", END)
        
        assert graph.edges["a"] == END
    
    def test_invalid_edge(self):
        """Test adding invalid edges raises error."""
        graph = Graph()
        graph.add_node("a", handler=lambda s: s)
        
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("a", "nonexistent")
    
    def test_conditional_edge(self):
        """Test conditional edges."""
        graph = Graph()
        graph.add_node("check", handler=lambda s: s)
        graph.add_node("yes", handler=lambda s: s)
        graph.add_node("no", handler=lambda s: s)
        
        def condition(state):
            return "yes" if state.get("value") else "no"
        
        graph.add_conditional_edge("check", condition, {"yes": "yes", "no": "no"})
        
        # Test routing
        assert graph.get_next_node("check", {"value": True}) == "yes"
        assert graph.get_next_node("check", {"value": False}) == "no"
    
    def test_graph_validation(self):
        """Test graph validation."""
        graph = Graph()
        
        # Empty graph should fail
        errors = graph.validate()
        assert len(errors) > 0
        
        # Valid graph
        graph.add_node("start", handler=lambda s: s)
        graph.add_edge("start", END)
        
        errors = graph.validate()
        assert len(errors) == 0
    
    def test_mermaid_generation(self):
        """Test Mermaid diagram generation."""
        graph = Graph()
        graph.add_node("a", handler=lambda s: s)
        graph.add_node("b", handler=lambda s: s)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        
        mermaid = graph.to_mermaid()
        
        assert "graph TD" in mermaid
        assert "a" in mermaid
        assert "b" in mermaid


# ============================================================
# Executor Tests
# ============================================================

class TestExecutor:
    """Tests for the Executor."""
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test executing a simple graph."""
        graph = Graph()
        graph.add_node("double", handler=lambda s: {**s, "value": s["value"] * 2})
        graph.add_edge("double", END)
        
        result = await execute_graph(graph, {"value": 5})
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["value"] == 10
    
    @pytest.mark.asyncio
    async def test_multi_node_execution(self):
        """Test executing multiple nodes."""
        graph = Graph()
        graph.add_node("add1", handler=lambda s: {**s, "value": s["value"] + 1})
        graph.add_node("add2", handler=lambda s: {**s, "value": s["value"] + 2})
        graph.add_edge("add1", "add2")
        graph.add_edge("add2", END)
        
        result = await execute_graph(graph, {"value": 0})
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["value"] == 3
        assert len(result.execution_log) == 2
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self):
        """Test conditional branching."""
        graph = Graph()
        graph.add_node("start", handler=lambda s: s)
        graph.add_node("high", handler=lambda s: {**s, "path": "high"})
        graph.add_node("low", handler=lambda s: {**s, "path": "low"})
        
        def route(state):
            return "high" if state["value"] > 5 else "low"
        
        graph.add_conditional_edge("start", route, {"high": "high", "low": "low"})
        graph.add_edge("high", END)
        graph.add_edge("low", END)
        
        # Test high path
        result = await execute_graph(graph, {"value": 10})
        assert result.final_state["path"] == "high"
        
        # Test low path
        result = await execute_graph(graph, {"value": 2})
        assert result.final_state["path"] == "low"
    
    @pytest.mark.asyncio
    async def test_loop_execution(self):
        """Test looping execution."""
        graph = Graph(max_iterations=10)
        
        def increment(state):
            return {**state, "count": state["count"] + 1}
        
        def check_count(state):
            return "done" if state["count"] >= 3 else "continue"
        
        graph.add_node("increment", handler=increment)
        graph.add_conditional_edge("increment", check_count, {"done": END, "continue": "increment"})
        
        result = await execute_graph(graph, {"count": 0})
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["count"] == 3
    
    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Test max iterations limit."""
        graph = Graph(max_iterations=3)
        
        # Infinite loop
        graph.add_node("loop", handler=lambda s: s)
        graph.add_conditional_edge("loop", lambda s: "continue", {"continue": "loop"})
        
        result = await execute_graph(graph, {})
        
        assert result.status == ExecutionStatus.FAILED
        assert "Max iterations" in result.error
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during execution."""
        def failing_handler(state):
            raise ValueError("Intentional error")
        
        graph = Graph()
        graph.add_node("fail", handler=failing_handler)
        
        result = await execute_graph(graph, {})
        
        assert result.status == ExecutionStatus.FAILED
        assert "Intentional error" in result.error
    
    @pytest.mark.asyncio
    async def test_execution_log(self):
        """Test that execution log is properly generated."""
        graph = Graph()
        graph.add_node("step1", handler=lambda s: s)
        graph.add_node("step2", handler=lambda s: s)
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)
        
        result = await execute_graph(graph, {})
        
        assert len(result.execution_log) == 2
        assert result.execution_log[0].node == "step1"
        assert result.execution_log[1].node == "step2"
        assert all(s.duration_ms > 0 for s in result.execution_log)


# ============================================================
# Integration Tests
# ============================================================

class TestCodeReviewWorkflow:
    """Integration tests for the Code Review workflow."""
    
    @pytest.mark.asyncio
    async def test_code_review_workflow(self):
        """Test the full code review workflow."""
        from app.workflows.code_review import create_code_review_workflow
        
        sample_code = '''
def hello():
    """Says hello."""
    print("Hello, World!")

def add(a, b):
    return a + b
'''
        
        workflow = create_code_review_workflow(max_iterations=3, quality_threshold=5.0)
        result = await execute_graph(workflow, {
            "code": sample_code,
            "quality_threshold": 5.0,
        })
        
        assert result.status == ExecutionStatus.COMPLETED
        assert "functions" in result.final_state
        assert "quality_score" in result.final_state
        assert len(result.execution_log) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
