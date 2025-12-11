"""
Async Workflow Executor.

The executor runs a workflow graph, managing state transitions,
handling loops, and generating execution logs.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import time
import logging

from app.engine.graph import Graph, END
from app.engine.state import WorkflowState, StateManager


# Configure logging
logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionStep:
    """A single step in the execution log."""
    step: int
    node: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    iteration: int = 0
    result: str = "success"
    error: Optional[str] = None
    route_taken: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "node": self.node,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "iteration": self.iteration,
            "result": self.result,
            "error": self.error,
            "route_taken": self.route_taken,
        }


@dataclass
class ExecutionResult:
    """Result of a workflow execution."""
    run_id: str
    graph_id: str
    status: ExecutionStatus
    final_state: Dict[str, Any]
    execution_log: List[ExecutionStep] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    error: Optional[str] = None
    iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "final_state": self.final_state,
            "execution_log": [step.to_dict() for step in self.execution_log],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "error": self.error,
            "iterations": self.iterations,
        }


class Executor:
    """
    Async workflow executor.
    
    Executes a graph with given initial state, handling:
    - Sequential node execution
    - Conditional branching
    - Loop iterations with max limit
    - Detailed execution logging
    - Error handling
    
    Usage:
        executor = Executor(graph)
        result = await executor.run({"input": "data"})
    """
    
    def __init__(
        self,
        graph: Graph,
        run_id: Optional[str] = None,
        on_step: Optional[Callable[[ExecutionStep, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the executor.
        
        Args:
            graph: The workflow graph to execute
            run_id: Optional run ID (generated if not provided)
            on_step: Optional callback for each step (for WebSocket streaming)
        """
        self.graph = graph
        self.run_id = run_id or str(uuid.uuid4())
        self.on_step = on_step
        
        # Execution state
        self._state_manager: Optional[StateManager] = None
        self._execution_log: List[ExecutionStep] = []
        self._step_counter = 0
        self._status = ExecutionStatus.PENDING
        self._cancelled = False
    
    @property
    def status(self) -> ExecutionStatus:
        """Get the current execution status."""
        return self._status
    
    @property
    def current_state(self) -> Optional[Dict[str, Any]]:
        """Get the current state data."""
        if self._state_manager and self._state_manager.current_state:
            return self._state_manager.current_state.data
        return None
    
    @property
    def current_node(self) -> Optional[str]:
        """Get the current node being executed."""
        if self._state_manager and self._state_manager.current_state:
            return self._state_manager.current_state.current_node
        return None
    
    def cancel(self) -> None:
        """Cancel the execution."""
        self._cancelled = True
        self._status = ExecutionStatus.CANCELLED
    
    async def run(self, initial_state: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the workflow with the given initial state.
        
        Args:
            initial_state: Initial state data
            
        Returns:
            ExecutionResult with final state and logs
        """
        start_time = time.time()
        self._status = ExecutionStatus.RUNNING
        self._state_manager = StateManager(self.run_id)
        
        # Initialize state
        state = self._state_manager.initialize(initial_state)
        
        # Validate graph
        errors = self.graph.validate()
        if errors:
            return self._create_error_result(
                f"Graph validation failed: {errors}",
                start_time
            )
        
        current_node = self.graph.entry_point
        iteration = 0
        visited_in_iteration: set = set()
        
        try:
            while current_node and current_node != END:
                # Check cancellation
                if self._cancelled:
                    logger.info(f"Execution cancelled at node '{current_node}'")
                    break
                
                # Check max iterations
                if iteration >= self.graph.max_iterations:
                    return self._create_error_result(
                        f"Max iterations ({self.graph.max_iterations}) exceeded",
                        start_time
                    )
                
                # Get the node
                node = self.graph.nodes.get(current_node)
                if not node:
                    return self._create_error_result(
                        f"Node '{current_node}' not found in graph",
                        start_time
                    )
                
                # Execute the node
                step = await self._execute_node(node, state, iteration)
                
                # Handle error
                if step.result == "error":
                    return self._create_error_result(
                        step.error or "Unknown error",
                        start_time
                    )
                
                # Update state from state manager
                state = self._state_manager.current_state
                
                # Get next node
                next_node = self.graph.get_next_node(current_node, state.data)
                
                # Track route for conditional edges
                if current_node in self.graph.conditional_edges:
                    cond_edge = self.graph.conditional_edges[current_node]
                    route_key = cond_edge.condition(state.data)
                    step.route_taken = route_key
                    logger.debug(f"Conditional route: {route_key} -> {next_node}")
                
                # Detect loops and increment iteration
                if next_node in visited_in_iteration:
                    iteration += 1
                    visited_in_iteration.clear()
                    state = state.increment_iteration()
                    logger.debug(f"Loop detected, iteration: {iteration}")
                
                visited_in_iteration.add(current_node)
                current_node = next_node
            
            # Finalize
            self._status = ExecutionStatus.COMPLETED
            final_state = self._state_manager.finalize()
            
            return ExecutionResult(
                run_id=self.run_id,
                graph_id=self.graph.graph_id,
                status=self._status,
                final_state=final_state.data,
                execution_log=self._execution_log,
                started_at=final_state.started_at,
                completed_at=final_state.completed_at,
                total_duration_ms=(time.time() - start_time) * 1000,
                iterations=iteration + 1,
            )
            
        except Exception as e:
            logger.exception(f"Execution failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _execute_node(
        self,
        node,
        state: WorkflowState,
        iteration: int
    ) -> ExecutionStep:
        """Execute a single node and update state."""
        self._step_counter += 1
        step_start = datetime.now()
        node_start_time = time.time()
        
        step = ExecutionStep(
            step=self._step_counter,
            node=node.name,
            started_at=step_start,
            iteration=iteration,
        )
        
        logger.info(f"Executing node: {node.name} (step {self._step_counter})")
        
        try:
            # Execute node handler
            result_data = await node.execute(state.data)
            
            # Update state
            new_state = state.update(result_data).mark_visited(node.name)
            self._state_manager.update(new_state, node.name)
            
            # Complete step
            step.completed_at = datetime.now()
            step.duration_ms = (time.time() - node_start_time) * 1000
            step.result = "success"
            
        except Exception as e:
            logger.error(f"Node {node.name} failed: {e}")
            step.completed_at = datetime.now()
            step.duration_ms = (time.time() - node_start_time) * 1000
            step.result = "error"
            step.error = str(e)
        
        # Add to log
        self._execution_log.append(step)
        
        # Notify callback
        if self.on_step:
            try:
                self.on_step(step, self._state_manager.current_state.data)
            except Exception as e:
                logger.warning(f"Step callback failed: {e}")
        
        return step
    
    def _create_error_result(
        self,
        error: str,
        start_time: float
    ) -> ExecutionResult:
        """Create an error result."""
        self._status = ExecutionStatus.FAILED
        return ExecutionResult(
            run_id=self.run_id,
            graph_id=self.graph.graph_id,
            status=ExecutionStatus.FAILED,
            final_state=self.current_state or {},
            execution_log=self._execution_log,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_ms=(time.time() - start_time) * 1000,
            error=error,
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution."""
        return {
            "run_id": self.run_id,
            "graph_id": self.graph.graph_id,
            "status": self._status.value,
            "current_node": self.current_node,
            "current_state": self.current_state,
            "step_count": self._step_counter,
            "iteration": self._state_manager.current_state.iteration if self._state_manager and self._state_manager.current_state else 0,
        }


async def execute_graph(
    graph: Graph,
    initial_state: Dict[str, Any],
    run_id: Optional[str] = None,
    on_step: Optional[Callable] = None
) -> ExecutionResult:
    """
    Convenience function to execute a graph.
    
    Args:
        graph: The workflow graph
        initial_state: Initial state data
        run_id: Optional run ID
        on_step: Optional step callback
        
    Returns:
        ExecutionResult
    """
    executor = Executor(graph, run_id, on_step)
    return await executor.run(initial_state)
