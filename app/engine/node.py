"""
Node Definition for Workflow Engine.

Nodes are the building blocks of a workflow. Each node is a function
that receives state, performs some operation, and returns modified state.
"""

from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import inspect
import functools


class NodeType(str, Enum):
    """Types of nodes in the workflow."""
    STANDARD = "standard"      # Regular processing node
    CONDITIONAL = "conditional"  # Branching decision node
    ENTRY = "entry"            # Entry point
    EXIT = "exit"              # Exit point


@dataclass
class Node:
    """
    A node in the workflow graph.
    
    Each node has a name and a handler function. The handler receives
    the current state data (as a dict) and returns modified state data.
    
    Attributes:
        name: Unique identifier for the node
        handler: Function that processes state (sync or async)
        node_type: Type of node (standard, conditional, etc.)
        description: Human-readable description
        metadata: Additional node metadata
    """
    
    name: str
    handler: Callable[[Dict[str, Any]], Union[Dict[str, Any], Any]]
    node_type: NodeType = NodeType.STANDARD
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the node after initialization."""
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if not callable(self.handler):
            raise ValueError(f"Handler for node '{self.name}' must be callable")
    
    @property
    def is_async(self) -> bool:
        """Check if the handler is an async function."""
        return asyncio.iscoroutinefunction(self.handler)
    
    async def execute(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node handler with the given state data.
        
        Handles both sync and async handlers transparently.
        
        Args:
            state_data: The current state data dictionary
            
        Returns:
            Modified state data dictionary
        """
        try:
            if self.is_async:
                result = await self.handler(state_data)
            else:
                # Run sync handler in executor to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    functools.partial(self.handler, state_data)
                )
            
            # If handler returns None, return original state
            if result is None:
                return state_data
            
            # If handler returns a dict, use it as the new state
            if isinstance(result, dict):
                return result
            
            # Otherwise, something unexpected happened
            raise ValueError(
                f"Node '{self.name}' handler must return a dict or None, "
                f"got {type(result).__name__}"
            )
            
        except Exception as e:
            # Add context to the error
            raise RuntimeError(f"Error in node '{self.name}': {str(e)}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the node to a dictionary."""
        return {
            "name": self.name,
            "type": self.node_type.value,
            "description": self.description,
            "handler": self.handler.__name__ if hasattr(self.handler, '__name__') else str(self.handler),
            "metadata": self.metadata,
        }


# Registry to hold decorated node functions
_node_registry: Dict[str, Callable] = {}


def node(
    name: Optional[str] = None,
    node_type: NodeType = NodeType.STANDARD,
    description: str = ""
) -> Callable:
    """
    Decorator to register a function as a workflow node.
    
    Usage:
        @node(name="extract_functions", description="Extract functions from code")
        def extract_functions(state: dict) -> dict:
            # ... process state
            return state
    
    Args:
        name: Node name (defaults to function name)
        node_type: Type of node
        description: Human-readable description
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        node_name = name or func.__name__
        
        # Store metadata on the function
        func._node_metadata = {
            "name": node_name,
            "type": node_type,
            "description": description or func.__doc__ or "",
        }
        
        # Register in global registry
        _node_registry[node_name] = func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._node_metadata = func._node_metadata
        return wrapper
    
    return decorator


def get_registered_node(name: str) -> Optional[Callable]:
    """Get a registered node function by name."""
    return _node_registry.get(name)


def list_registered_nodes() -> Dict[str, Dict[str, Any]]:
    """List all registered nodes and their metadata."""
    return {
        name: func._node_metadata 
        for name, func in _node_registry.items()
        if hasattr(func, '_node_metadata')
    }


def create_node_from_function(
    func: Callable,
    name: Optional[str] = None,
    node_type: NodeType = NodeType.STANDARD,
    description: str = ""
) -> Node:
    """
    Create a Node instance from a function.
    
    Args:
        func: The handler function
        name: Node name (defaults to function name)
        node_type: Type of node
        description: Human-readable description
        
    Returns:
        A Node instance
    """
    return Node(
        name=name or func.__name__,
        handler=func,
        node_type=node_type,
        description=description or func.__doc__ or "",
    )
