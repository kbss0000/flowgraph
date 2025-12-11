"""
State Management for Workflow Engine.

This module provides the state management system that flows through the workflow.
State is immutable - each node receives state and returns a new modified state.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from copy import deepcopy
import uuid


class StateSnapshot(BaseModel):
    """A snapshot of state at a specific point in execution."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    node_name: str
    state_data: Dict[str, Any]
    iteration: int = 0


class WorkflowState(BaseModel):
    """
    The shared state that flows through the workflow.
    
    This is a flexible container that holds all data being processed
    by the workflow nodes. Each node can read from and write to this state.
    
    Attributes:
        data: The actual workflow data (flexible dictionary)
        metadata: Execution metadata (iteration count, visited nodes, etc.)
    """
    
    # The actual data being processed
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution metadata
    current_node: Optional[str] = None
    iteration: int = 0
    visited_nodes: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state data."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> "WorkflowState":
        """Set a value in state data and return a new state (immutable pattern)."""
        new_data = deepcopy(self.data)
        new_data[key] = value
        return self.model_copy(update={"data": new_data})
    
    def update(self, updates: Dict[str, Any]) -> "WorkflowState":
        """Update multiple values and return a new state."""
        new_data = deepcopy(self.data)
        new_data.update(updates)
        return self.model_copy(update={"data": new_data})
    
    def mark_visited(self, node_name: str) -> "WorkflowState":
        """Mark a node as visited."""
        new_visited = self.visited_nodes + [node_name]
        return self.model_copy(update={
            "visited_nodes": new_visited,
            "current_node": node_name
        })
    
    def increment_iteration(self) -> "WorkflowState":
        """Increment the iteration counter."""
        return self.model_copy(update={"iteration": self.iteration + 1})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a plain dictionary."""
        return {
            "data": self.data,
            "current_node": self.current_node,
            "iteration": self.iteration,
            "visited_nodes": self.visited_nodes,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create a WorkflowState from a dictionary."""
        if "data" in data:
            return cls(**data)
        # If it's just raw data, wrap it
        return cls(data=data)


class StateManager:
    """
    Manages state history and snapshots for a workflow run.
    
    This provides debugging capabilities by tracking state changes
    throughout the workflow execution.
    """
    
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.history: List[StateSnapshot] = []
        self._current_state: Optional[WorkflowState] = None
    
    @property
    def current_state(self) -> Optional[WorkflowState]:
        """Get the current state."""
        return self._current_state
    
    def initialize(self, initial_data: Dict[str, Any]) -> WorkflowState:
        """Initialize the state manager with initial data."""
        self._current_state = WorkflowState(
            data=initial_data,
            started_at=datetime.now()
        )
        return self._current_state
    
    def update(self, new_state: WorkflowState, node_name: str) -> None:
        """Update the current state and record a snapshot."""
        # Record snapshot
        snapshot = StateSnapshot(
            node_name=node_name,
            state_data=deepcopy(new_state.data),
            iteration=new_state.iteration
        )
        self.history.append(snapshot)
        
        # Update current state
        self._current_state = new_state
    
    def finalize(self) -> WorkflowState:
        """Mark the workflow as complete."""
        if self._current_state:
            self._current_state = self._current_state.model_copy(
                update={"completed_at": datetime.now()}
            )
        return self._current_state
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the state history as a list of dictionaries."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "node": s.node_name,
                "iteration": s.iteration,
                "state": s.state_data
            }
            for s in self.history
        ]
