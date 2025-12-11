"""
Graph API Routes.

Endpoints for creating, managing, and executing workflow graphs.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from uuid import uuid4
import logging

from app.api.schemas import (
    GraphCreateRequest,
    GraphCreateResponse,
    GraphRunRequest,
    GraphRunResponse,
    GraphInfoResponse,
    GraphListResponse,
    RunStateResponse,
    RunListResponse,
    ExecutionLogEntry,
    ExecutionStatus,
    ErrorResponse,
)
from app.engine.graph import Graph, END
from app.engine.node import Node, get_registered_node
from app.engine.executor import Executor, ExecutionResult
from app.storage.memory import graph_storage, run_storage
from app.tools.registry import tool_registry


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph"])


# ============================================================
# Condition Functions Registry
# ============================================================

# Built-in condition functions for routing
_condition_registry: Dict[str, Any] = {}


def register_condition(name: str):
    """Decorator to register a condition function."""
    def decorator(func):
        _condition_registry[name] = func
        return func
    return decorator


@register_condition("quality_check")
def quality_check_condition(state: Dict[str, Any]) -> str:
    """Route based on quality score vs threshold."""
    quality_score = state.get("quality_score", 0)
    threshold = state.get("quality_threshold", 7.0)
    return "pass" if quality_score >= threshold else "fail"


# Also register as quality_meets_threshold (used by code review workflow)
@register_condition("quality_meets_threshold")
def quality_meets_threshold(state: Dict[str, Any]) -> str:
    """Route based on quality score vs threshold."""
    quality_score = state.get("quality_score", 0)
    threshold = state.get("quality_threshold", 7.0)
    return "pass" if quality_score >= threshold else "fail"


@register_condition("always_continue")
def always_continue(state: Dict[str, Any]) -> str:
    """Always returns 'continue' - for unconditional looping."""
    return "continue"


# Also register as always_loop (used by code review workflow)
@register_condition("always_loop")
def always_loop(state: Dict[str, Any]) -> str:
    """Always returns 'continue' - for looping back."""
    return "continue"


@register_condition("always_end")
def always_end(state: Dict[str, Any]) -> str:
    """Always returns 'end' - for explicit termination."""
    return "end"


@register_condition("max_iterations_check")
def max_iterations_check(state: Dict[str, Any]) -> str:
    """Check if max iterations reached."""
    iteration = state.get("_iteration", 0)
    max_iter = state.get("_max_iterations", 3)
    return "stop" if iteration >= max_iter else "continue"


def get_condition(name: str):
    """Get a condition function by name."""
    return _condition_registry.get(name)


# ============================================================
# Graph CRUD Endpoints
# ============================================================

@router.post(
    "/create",
    response_model=GraphCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid graph definition"},
        404: {"model": ErrorResponse, "description": "Handler not found"},
    }
)
async def create_graph(request: GraphCreateRequest) -> GraphCreateResponse:
    """
    Create a new workflow graph.
    
    Define nodes with their handlers, edges for flow control,
    and conditional edges for branching logic.
    """
    graph_id = str(uuid4())
    
    # Build the graph
    graph = Graph(
        graph_id=graph_id,
        name=request.name,
        description=request.description or "",
        max_iterations=request.max_iterations,
    )
    
    # Add nodes
    for node_def in request.nodes:
        # Find the handler function
        handler = get_registered_node(node_def.handler)
        if handler is None:
            # Check tool registry as fallback
            tool = tool_registry.get(node_def.handler)
            if tool:
                handler = _create_node_handler_from_tool(node_def.handler)
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Handler '{node_def.handler}' not found. "
                           f"Available handlers: {list(tool_registry.list_tools())}"
                )
        
        graph.add_node(
            name=node_def.name,
            handler=handler,
            description=node_def.description or "",
        )
    
    # Add direct edges
    for source, target in request.edges.items():
        if source not in graph.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Edge source '{source}' is not a valid node"
            )
        if target != END and target != "__END__" and target not in graph.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Edge target '{target}' is not a valid node"
            )
        # Normalize END
        target = END if target == "__END__" else target
        graph.add_edge(source, target)
    
    # Add conditional edges
    for source, cond_routes in request.conditional_edges.items():
        if source not in graph.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Conditional edge source '{source}' is not a valid node"
            )
        
        # Get condition function
        condition_func = get_condition(cond_routes.condition)
        if condition_func is None:
            raise HTTPException(
                status_code=404,
                detail=f"Condition '{cond_routes.condition}' not found. "
                       f"Available: {list(_condition_registry.keys())}"
            )
        
        # Normalize routes (handle __END__)
        routes = {}
        for key, target in cond_routes.routes.items():
            if target == "__END__":
                routes[key] = END
            else:
                if target not in graph.nodes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Conditional route target '{target}' is not a valid node"
                    )
                routes[key] = target
        
        graph.add_conditional_edge(source, condition_func, routes)
    
    # Set entry point
    if request.entry_point:
        if request.entry_point not in graph.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Entry point '{request.entry_point}' is not a valid node"
            )
        graph.set_entry_point(request.entry_point)
    
    # Validate graph
    errors = graph.validate()
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"Graph validation failed: {errors}"
        )
    
    # Store the graph
    await graph_storage.save(
        graph_id=graph_id,
        name=request.name,
        definition=graph.to_dict(),
    )
    
    logger.info(f"Created graph: {graph_id} ({request.name})")
    
    return GraphCreateResponse(
        graph_id=graph_id,
        name=request.name,
        message="Graph created successfully",
        node_count=len(graph.nodes),
    )


def _create_node_handler_from_tool(tool_name: str):
    """Create a node handler that calls a tool and updates state."""
    def handler(state: Dict[str, Any]) -> Dict[str, Any]:
        tool = tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Check if the tool function expects a 'state' parameter (node handler style)
        # or individual parameters (regular tool style)
        import inspect
        sig = inspect.signature(tool.func)
        param_names = list(sig.parameters.keys())
        
        if len(param_names) == 1 and param_names[0] == 'state':
            # This is a node handler - pass state directly
            result = tool.func(state)
        else:
            # This is a regular tool - extract arguments from state
            result = tool.func(**_extract_tool_args(tool, state))
        
        # Handle the result
        if isinstance(result, dict):
            # If the tool returns a full state, use it directly
            # Check if it looks like a state update (has same keys or adds new ones)
            if result is state:
                return result
            # Merge result into state
            state.update(result)
        
        return state
    
    handler.__name__ = f"{tool_name}_handler"
    return handler


def _extract_tool_args(tool, state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract arguments for a tool from state."""
    import inspect
    sig = inspect.signature(tool.func)
    args = {}
    
    for param_name, param in sig.parameters.items():
        if param_name in state:
            args[param_name] = state[param_name]
        elif param.default != inspect.Parameter.empty:
            pass  # Use default
        # Skip missing optional params
    
    return args


@router.get(
    "/{graph_id}",
    response_model=GraphInfoResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_graph(graph_id: str) -> GraphInfoResponse:
    """Get information about a specific graph."""
    stored = await graph_storage.get(graph_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    
    definition = stored.definition
    
    # Generate mermaid diagram
    mermaid = _generate_mermaid(definition)
    
    return GraphInfoResponse(
        graph_id=stored.graph_id,
        name=stored.name,
        description=definition.get("description"),
        node_count=len(definition.get("nodes", {})),
        nodes=list(definition.get("nodes", {}).keys()),
        entry_point=definition.get("entry_point"),
        max_iterations=definition.get("max_iterations", 100),
        created_at=stored.created_at.isoformat(),
        mermaid_diagram=mermaid,
    )


def _generate_mermaid(definition: Dict[str, Any]) -> str:
    """Generate a Mermaid diagram from graph definition."""
    lines = ["graph TD"]
    
    nodes = definition.get("nodes", {})
    edges = definition.get("edges", {})
    cond_edges = definition.get("conditional_edges", {})
    
    # Add nodes
    for name in nodes:
        label = name.replace("_", " ").title()
        lines.append(f'    {name}["{label}"]')
    
    # Check if END is used
    has_end = END in edges.values()
    for cond in cond_edges.values():
        if END in cond.get("routes", {}).values():
            has_end = True
    
    if has_end:
        lines.append(f'    {END}(("END"))')
    
    # Add direct edges
    for source, target in edges.items():
        lines.append(f"    {source} --> {target}")
    
    # Add conditional edges
    for source, cond in cond_edges.items():
        for route_key, target in cond.get("routes", {}).items():
            lines.append(f"    {source} -->|{route_key}| {target}")
    
    return "\n".join(lines)


@router.get(
    "/",
    response_model=GraphListResponse,
)
async def list_graphs() -> GraphListResponse:
    """List all available graphs."""
    graphs = await graph_storage.list_all()
    
    graph_infos = []
    for stored in graphs:
        definition = stored.definition
        graph_infos.append(GraphInfoResponse(
            graph_id=stored.graph_id,
            name=stored.name,
            description=definition.get("description"),
            node_count=len(definition.get("nodes", {})),
            nodes=list(definition.get("nodes", {}).keys()),
            entry_point=definition.get("entry_point"),
            max_iterations=definition.get("max_iterations", 100),
            created_at=stored.created_at.isoformat(),
            mermaid_diagram=None,  # Skip for list view
        ))
    
    return GraphListResponse(graphs=graph_infos, total=len(graph_infos))


@router.delete(
    "/{graph_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_graph(graph_id: str):
    """Delete a graph."""
    deleted = await graph_storage.delete(graph_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    logger.info(f"Deleted graph: {graph_id}")


# ============================================================
# Execution Endpoints
# ============================================================

@router.post(
    "/run",
    response_model=GraphRunResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse, "description": "Execution failed"},
    }
)
async def run_graph(
    request: GraphRunRequest,
    background_tasks: BackgroundTasks,
) -> GraphRunResponse:
    """
    Execute a workflow graph with the given initial state.
    
    If `async_execution` is True, the workflow runs in the background
    and you can poll the status using GET /graph/state/{run_id}.
    """
    # Get the graph
    stored = await graph_storage.get(request.graph_id)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{request.graph_id}' not found"
        )
    
    # Rebuild the graph from definition
    graph = await _rebuild_graph_from_definition(stored.definition)
    
    # Create run
    run_id = str(uuid4())
    await run_storage.create(run_id, request.graph_id, request.initial_state)
    
    if request.async_execution:
        # Run in background
        background_tasks.add_task(
            _execute_in_background,
            graph,
            run_id,
            request.initial_state,
        )
        
        return GraphRunResponse(
            run_id=run_id,
            graph_id=request.graph_id,
            status=ExecutionStatus.PENDING,
            final_state={},
            execution_log=[],
            started_at=None,
            completed_at=None,
            total_duration_ms=None,
            iterations=0,
        )
    
    # Execute synchronously
    try:
        executor = Executor(
            graph,
            run_id=run_id,
            on_step=lambda step, state: _update_run_state(run_id, step, state),
        )
        result = await executor.run(request.initial_state)
        
        # Update storage
        if result.status.value == "completed":
            await run_storage.complete(
                run_id,
                result.final_state,
                [s.to_dict() for s in result.execution_log],
            )
        else:
            await run_storage.fail(run_id, result.error or "Unknown error", result.final_state)
        
        return _result_to_response(result)
        
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        await run_storage.fail(run_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _rebuild_graph_from_definition(definition: Dict[str, Any]) -> Graph:
    """Rebuild a Graph object from its stored definition."""
    graph = Graph(
        graph_id=definition.get("graph_id", str(uuid4())),
        name=definition.get("name", "Unnamed"),
        description=definition.get("description", ""),
        max_iterations=definition.get("max_iterations", 100),
    )
    
    # Add nodes
    nodes_def = definition.get("nodes", {})
    for node_name, node_info in nodes_def.items():
        handler_name = node_info.get("handler", node_name)
        handler = _create_node_handler_from_tool(handler_name)
        graph.add_node(
            name=node_name,
            handler=handler,
            description=node_info.get("description", ""),
        )
    
    # Add direct edges
    for source, target in definition.get("edges", {}).items():
        graph.add_edge(source, target)
    
    # Add conditional edges
    for source, cond_info in definition.get("conditional_edges", {}).items():
        condition_name = cond_info.get("condition", "always_continue")
        condition_func = get_condition(condition_name)
        if condition_func is None:
            condition_func = always_continue
        
        routes = cond_info.get("routes", {})
        graph.add_conditional_edge(source, condition_func, routes)
    
    # Set entry point
    if definition.get("entry_point"):
        graph.set_entry_point(definition["entry_point"])
    
    return graph


async def _execute_in_background(graph: Graph, run_id: str, initial_state: Dict[str, Any]):
    """Execute a workflow in the background."""
    try:
        executor = Executor(
            graph,
            run_id=run_id,
            on_step=lambda step, state: _update_run_state(run_id, step, state),
        )
        result = await executor.run(initial_state)
        
        if result.status.value == "completed":
            await run_storage.complete(
                run_id,
                result.final_state,
                [s.to_dict() for s in result.execution_log],
            )
        else:
            await run_storage.fail(run_id, result.error or "Unknown error", result.final_state)
            
    except Exception as e:
        logger.exception(f"Background execution failed: {e}")
        await run_storage.fail(run_id, str(e))


def _update_run_state(run_id: str, step, state: Dict[str, Any]):
    """Update run state during execution (sync callback)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(
                run_storage.update_state(run_id, state, step.node, step.iteration)
            )
    except Exception:
        pass  # Ignore errors in callback


def _result_to_response(result: ExecutionResult) -> GraphRunResponse:
    """Convert ExecutionResult to API response."""
    return GraphRunResponse(
        run_id=result.run_id,
        graph_id=result.graph_id,
        status=ExecutionStatus(result.status.value),
        final_state=result.final_state,
        execution_log=[
            ExecutionLogEntry(
                step=s.step,
                node=s.node,
                started_at=s.started_at.isoformat(),
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
                duration_ms=s.duration_ms,
                iteration=s.iteration,
                result=s.result,
                error=s.error,
                route_taken=s.route_taken,
            )
            for s in result.execution_log
        ],
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        total_duration_ms=result.total_duration_ms,
        iterations=result.iterations,
        error=result.error,
    )


# ============================================================
# Run State Endpoints
# ============================================================

@router.get(
    "/state/{run_id}",
    response_model=RunStateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_run_state(run_id: str) -> RunStateResponse:
    """
    Get the current state of a workflow run.
    
    Use this to poll the status of async executions.
    """
    stored = await run_storage.get(run_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    return RunStateResponse(
        run_id=stored.run_id,
        graph_id=stored.graph_id,
        status=ExecutionStatus(stored.status),
        current_node=stored.current_node,
        current_state=stored.current_state,
        iteration=stored.iteration,
        execution_log=[
            ExecutionLogEntry(**entry) for entry in stored.execution_log
        ],
        started_at=stored.started_at.isoformat(),
        completed_at=stored.completed_at.isoformat() if stored.completed_at else None,
        error=stored.error,
    )


@router.get(
    "/runs",
    response_model=RunListResponse,
)
async def list_runs(graph_id: Optional[str] = None) -> RunListResponse:
    """List all runs, optionally filtered by graph_id."""
    if graph_id:
        runs = await run_storage.list_by_graph(graph_id)
    else:
        runs = await run_storage.list_all()
    
    run_states = []
    for stored in runs:
        run_states.append(RunStateResponse(
            run_id=stored.run_id,
            graph_id=stored.graph_id,
            status=ExecutionStatus(stored.status),
            current_node=stored.current_node,
            current_state=stored.current_state,
            iteration=stored.iteration,
            execution_log=[
                ExecutionLogEntry(**entry) for entry in stored.execution_log
            ],
            started_at=stored.started_at.isoformat(),
            completed_at=stored.completed_at.isoformat() if stored.completed_at else None,
            error=stored.error,
        ))
    
    return RunListResponse(runs=run_states, total=len(run_states))
