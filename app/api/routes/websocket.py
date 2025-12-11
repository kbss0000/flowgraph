"""
WebSocket Routes for Real-time Execution Streaming.

Provides live updates during workflow execution.
"""

from typing import Any, Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4
import asyncio
import json
import logging

from app.engine.graph import Graph
from app.engine.executor import Executor, ExecutionStep
from app.storage.memory import graph_storage, run_storage


logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, run_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        self.active_connections[run_id].add(websocket)
        logger.info(f"WebSocket connected for run: {run_id}")
    
    def disconnect(self, websocket: WebSocket, run_id: str):
        """Remove a WebSocket connection."""
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
        logger.info(f"WebSocket disconnected for run: {run_id}")
    
    async def broadcast(self, run_id: str, message: Dict[str, Any]):
        """Broadcast a message to all connections for a run."""
        if run_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[run_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.add(websocket)
            
            # Clean up disconnected clients
            for ws in disconnected:
                self.active_connections[run_id].discard(ws)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/run/{graph_id}")
async def websocket_run(websocket: WebSocket, graph_id: str):
    """
    WebSocket endpoint for real-time workflow execution.
    
    Connect to this endpoint and send the initial state as JSON.
    You'll receive step-by-step updates as the workflow executes.
    
    Message format (client -> server):
    ```json
    {"action": "start", "initial_state": {"code": "..."}}
    ```
    
    Message format (server -> client):
    ```json
    {
        "type": "step",
        "step": 1,
        "node": "extract",
        "status": "completed",
        "duration_ms": 15.5,
        "state": {...}
    }
    ```
    """
    # Check if graph exists
    stored = await graph_storage.get(graph_id)
    if not stored:
        await websocket.close(code=4004, reason=f"Graph '{graph_id}' not found")
        return
    
    run_id = str(uuid4())
    await manager.connect(websocket, run_id)
    
    try:
        # Wait for start message
        data = await websocket.receive_json()
        
        if data.get("action") != "start":
            await websocket.send_json({
                "type": "error",
                "error": "Expected 'start' action"
            })
            return
        
        initial_state = data.get("initial_state", {})
        
        # Send acknowledgment
        await websocket.send_json({
            "type": "started",
            "run_id": run_id,
            "graph_id": graph_id,
        })
        
        # Rebuild graph
        graph = await _rebuild_graph(stored.definition)
        
        # Create run record
        await run_storage.create(run_id, graph_id, initial_state)
        
        # Execute with streaming updates
        async def on_step(step: ExecutionStep, state: Dict[str, Any]):
            await manager.broadcast(run_id, {
                "type": "step",
                "step": step.step,
                "node": step.node,
                "status": step.result,
                "duration_ms": step.duration_ms,
                "iteration": step.iteration,
                "route_taken": step.route_taken,
                "error": step.error,
                "state": state,
            })
        
        executor = Executor(graph, run_id=run_id)
        
        # Run with step notifications
        result = await _run_with_streaming(executor, initial_state, on_step)
        
        # Send completion
        await websocket.send_json({
            "type": "completed",
            "run_id": run_id,
            "status": result.status.value,
            "final_state": result.final_state,
            "total_duration_ms": result.total_duration_ms,
            "iterations": result.iterations,
            "error": result.error,
        })
        
        # Update storage
        if result.status.value == "completed":
            await run_storage.complete(
                run_id,
                result.final_state,
                [s.to_dict() for s in result.execution_log],
            )
        else:
            await run_storage.fail(run_id, result.error or "Unknown error")
        
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from run {run_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except Exception:
            pass
    finally:
        manager.disconnect(websocket, run_id)


async def _rebuild_graph(definition: Dict[str, Any]) -> Graph:
    """Rebuild graph from definition (copied from graph.py to avoid circular import)."""
    from app.api.routes.graph import _rebuild_graph_from_definition
    return await _rebuild_graph_from_definition(definition)


async def _run_with_streaming(
    executor: Executor,
    initial_state: Dict[str, Any],
    on_step
):
    """Run executor with async step callbacks."""
    from app.engine.graph import END
    from app.engine.state import StateManager
    import time
    from datetime import datetime
    
    # Execute the workflow
    result = await executor.run(initial_state)
    
    # Stream each step (already executed, but we notify)
    for step in result.execution_log:
        await on_step(step, result.final_state)
        await asyncio.sleep(0.01)  # Small delay for streaming effect
    
    return result


@router.websocket("/ws/subscribe/{run_id}")
async def websocket_subscribe(websocket: WebSocket, run_id: str):
    """
    Subscribe to updates for an existing run.
    
    Use this to watch an async execution started via POST /graph/run.
    """
    # Check if run exists
    stored = await run_storage.get(run_id)
    if not stored:
        await websocket.close(code=4004, reason=f"Run '{run_id}' not found")
        return
    
    await manager.connect(websocket, run_id)
    
    try:
        # Send current state
        await websocket.send_json({
            "type": "current_state",
            "run_id": run_id,
            "status": stored.status,
            "current_node": stored.current_node,
            "iteration": stored.iteration,
            "state": stored.current_state,
        })
        
        # Keep connection open and poll for updates
        last_log_count = len(stored.execution_log)
        
        while True:
            await asyncio.sleep(0.5)  # Poll interval
            
            stored = await run_storage.get(run_id)
            if not stored:
                break
            
            # Send new log entries
            if len(stored.execution_log) > last_log_count:
                for entry in stored.execution_log[last_log_count:]:
                    await websocket.send_json({
                        "type": "step",
                        **entry,
                    })
                last_log_count = len(stored.execution_log)
            
            # Check if completed
            if stored.status in ("completed", "failed", "cancelled"):
                await websocket.send_json({
                    "type": "completed",
                    "run_id": run_id,
                    "status": stored.status,
                    "final_state": stored.final_state,
                    "error": stored.error,
                })
                break
                
    except WebSocketDisconnect:
        logger.info(f"Subscriber disconnected from run {run_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, run_id)
