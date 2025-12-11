"""
In-Memory Storage for Workflow Engine.

Provides thread-safe storage for graphs and execution runs.
Can be easily replaced with a database implementation.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass, field


@dataclass
class StoredGraph:
    """A stored graph definition."""
    graph_id: str
    name: str
    definition: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "definition": self.definition,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class StoredRun:
    """A stored execution run."""
    run_id: str
    graph_id: str
    status: str
    initial_state: Dict[str, Any]
    current_state: Dict[str, Any] = field(default_factory=dict)
    final_state: Optional[Dict[str, Any]] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    current_node: Optional[str] = None
    iteration: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "status": self.status,
            "initial_state": self.initial_state,
            "current_state": self.current_state,
            "final_state": self.final_state,
            "execution_log": self.execution_log,
            "current_node": self.current_node,
            "iteration": self.iteration,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class GraphStorage:
    """
    Thread-safe in-memory storage for workflow graphs.
    
    Stores graph definitions by their ID, allowing creation,
    retrieval, update, and deletion operations.
    """
    
    def __init__(self):
        self._graphs: Dict[str, StoredGraph] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, graph_id: str, name: str, definition: Dict[str, Any]) -> StoredGraph:
        """
        Save a graph definition.
        
        Args:
            graph_id: Unique graph identifier
            name: Graph name
            definition: Graph definition dict
            
        Returns:
            The stored graph
        """
        async with self._lock:
            stored = StoredGraph(
                graph_id=graph_id,
                name=name,
                definition=definition,
            )
            self._graphs[graph_id] = stored
            return stored
    
    async def get(self, graph_id: str) -> Optional[StoredGraph]:
        """Get a graph by ID."""
        async with self._lock:
            return self._graphs.get(graph_id)
    
    async def update(self, graph_id: str, definition: Dict[str, Any]) -> Optional[StoredGraph]:
        """Update a graph definition."""
        async with self._lock:
            if graph_id not in self._graphs:
                return None
            stored = self._graphs[graph_id]
            stored.definition = definition
            stored.updated_at = datetime.now()
            return stored
    
    async def delete(self, graph_id: str) -> bool:
        """Delete a graph."""
        async with self._lock:
            if graph_id in self._graphs:
                del self._graphs[graph_id]
                return True
            return False
    
    async def list_all(self) -> List[StoredGraph]:
        """List all stored graphs."""
        async with self._lock:
            return list(self._graphs.values())
    
    async def exists(self, graph_id: str) -> bool:
        """Check if a graph exists."""
        async with self._lock:
            return graph_id in self._graphs
    
    def __len__(self) -> int:
        return len(self._graphs)


class RunStorage:
    """
    Thread-safe in-memory storage for execution runs.
    
    Stores run state, allowing real-time updates and queries
    for ongoing and completed runs.
    """
    
    def __init__(self):
        self._runs: Dict[str, StoredRun] = {}
        self._lock = asyncio.Lock()
    
    async def create(
        self,
        run_id: str,
        graph_id: str,
        initial_state: Dict[str, Any]
    ) -> StoredRun:
        """
        Create a new run.
        
        Args:
            run_id: Unique run identifier
            graph_id: Associated graph ID
            initial_state: Initial state data
            
        Returns:
            The stored run
        """
        async with self._lock:
            stored = StoredRun(
                run_id=run_id,
                graph_id=graph_id,
                status="pending",
                initial_state=initial_state,
                current_state=initial_state.copy(),
            )
            self._runs[run_id] = stored
            return stored
    
    async def get(self, run_id: str) -> Optional[StoredRun]:
        """Get a run by ID."""
        async with self._lock:
            return self._runs.get(run_id)
    
    async def update_state(
        self,
        run_id: str,
        current_state: Dict[str, Any],
        current_node: Optional[str] = None,
        iteration: Optional[int] = None
    ) -> Optional[StoredRun]:
        """Update the current state of a run."""
        async with self._lock:
            if run_id not in self._runs:
                return None
            stored = self._runs[run_id]
            stored.current_state = current_state
            stored.status = "running"
            if current_node is not None:
                stored.current_node = current_node
            if iteration is not None:
                stored.iteration = iteration
            return stored
    
    async def add_log_entry(
        self,
        run_id: str,
        entry: Dict[str, Any]
    ) -> Optional[StoredRun]:
        """Add an entry to the execution log."""
        async with self._lock:
            if run_id not in self._runs:
                return None
            self._runs[run_id].execution_log.append(entry)
            return self._runs[run_id]
    
    async def complete(
        self,
        run_id: str,
        final_state: Dict[str, Any],
        execution_log: List[Dict[str, Any]]
    ) -> Optional[StoredRun]:
        """Mark a run as completed."""
        async with self._lock:
            if run_id not in self._runs:
                return None
            stored = self._runs[run_id]
            stored.status = "completed"
            stored.final_state = final_state
            stored.execution_log = execution_log
            stored.completed_at = datetime.now()
            return stored
    
    async def fail(
        self,
        run_id: str,
        error: str,
        final_state: Optional[Dict[str, Any]] = None
    ) -> Optional[StoredRun]:
        """Mark a run as failed."""
        async with self._lock:
            if run_id not in self._runs:
                return None
            stored = self._runs[run_id]
            stored.status = "failed"
            stored.error = error
            stored.final_state = final_state
            stored.completed_at = datetime.now()
            return stored
    
    async def list_all(self) -> List[StoredRun]:
        """List all runs."""
        async with self._lock:
            return list(self._runs.values())
    
    async def list_by_graph(self, graph_id: str) -> List[StoredRun]:
        """List all runs for a specific graph."""
        async with self._lock:
            return [r for r in self._runs.values() if r.graph_id == graph_id]
    
    async def delete(self, run_id: str) -> bool:
        """Delete a run."""
        async with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
                return True
            return False
    
    def __len__(self) -> int:
        return len(self._runs)


# Global storage instances
graph_storage = GraphStorage()
run_storage = RunStorage()
