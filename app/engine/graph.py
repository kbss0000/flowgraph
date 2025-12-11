"""
Graph Definition for Workflow Engine.

The Graph is the core structure that defines the workflow - nodes, edges,
conditional routing, and execution flow.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from app.engine.node import Node, NodeType, get_registered_node, create_node_from_function


# Special node names
END = "__END__"
START = "__START__"


class EdgeType(str, Enum):
    """Types of edges between nodes."""
    DIRECT = "direct"           # Always follow this edge
    CONDITIONAL = "conditional"  # Choose based on condition


@dataclass
class Edge:
    """An edge connecting two nodes."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECT
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value
        }


@dataclass
class ConditionalEdge:
    """
    A conditional edge that routes to different nodes based on a condition.
    
    The condition function receives the current state and returns a route key.
    The routes dict maps route keys to target node names.
    """
    source: str
    condition: Callable[[Dict[str, Any]], str]
    routes: Dict[str, str]  # route_key -> target_node_name
    
    def evaluate(self, state_data: Dict[str, Any]) -> str:
        """Evaluate the condition and return the target node name."""
        route_key = self.condition(state_data)
        if route_key not in self.routes:
            raise ValueError(
                f"Condition returned unknown route '{route_key}'. "
                f"Available routes: {list(self.routes.keys())}"
            )
        return self.routes[route_key]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "condition": self.condition.__name__ if hasattr(self.condition, '__name__') else str(self.condition),
            "routes": self.routes
        }


@dataclass
class Graph:
    """
    A workflow graph consisting of nodes and edges.
    
    The graph defines the structure of a workflow:
    - Nodes: Processing units that transform state
    - Edges: Connections between nodes
    - Conditional Edges: Branching logic based on state
    
    Attributes:
        graph_id: Unique identifier for this graph
        name: Human-readable name
        nodes: Dict of node_name -> Node
        edges: List of direct edges
        conditional_edges: Dict of source_node -> ConditionalEdge
        entry_point: Name of the first node to execute
        max_iterations: Maximum loop iterations allowed
    """
    
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Workflow"
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)  # source -> target for direct edges
    conditional_edges: Dict[str, ConditionalEdge] = field(default_factory=dict)
    entry_point: Optional[str] = None
    max_iterations: int = 100
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(
        self,
        name: str,
        handler: Optional[Callable] = None,
        node_type: NodeType = NodeType.STANDARD,
        description: str = ""
    ) -> "Graph":
        """
        Add a node to the graph.
        
        If handler is not provided, attempts to find a registered node
        with the given name.
        
        Args:
            name: Unique name for the node
            handler: Function to execute (optional if registered)
            node_type: Type of node
            description: Human-readable description
            
        Returns:
            Self for chaining
        """
        if handler is None:
            # Try to find a registered handler
            handler = get_registered_node(name)
            if handler is None:
                raise ValueError(
                    f"No handler provided for node '{name}' and no registered "
                    f"node found with that name"
                )
        
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in the graph")
        
        node = create_node_from_function(handler, name, node_type, description)
        self.nodes[name] = node
        
        # Set as entry point if it's the first node or marked as entry
        if self.entry_point is None or node_type == NodeType.ENTRY:
            self.entry_point = name
        
        return self
    
    def add_edge(self, source: str, target: str) -> "Graph":
        """
        Add a direct edge from source to target.
        
        Args:
            source: Source node name
            target: Target node name (or END)
            
        Returns:
            Self for chaining
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target != END and target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found in graph")
        
        # Check for conflicts with conditional edges
        if source in self.conditional_edges:
            raise ValueError(
                f"Node '{source}' already has a conditional edge. "
                f"Cannot add a direct edge."
            )
        
        self.edges[source] = target
        return self
    
    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[Dict[str, Any]], str],
        routes: Dict[str, str]
    ) -> "Graph":
        """
        Add a conditional edge from source node.
        
        The condition function receives state and returns a route key.
        
        Args:
            source: Source node name
            condition: Function that returns route key
            routes: Dict mapping route keys to target nodes
            
        Returns:
            Self for chaining
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        
        # Validate all targets
        for route_key, target in routes.items():
            if target != END and target not in self.nodes:
                raise ValueError(
                    f"Target node '{target}' for route '{route_key}' not found in graph"
                )
        
        # Check for conflicts with direct edges
        if source in self.edges:
            raise ValueError(
                f"Node '{source}' already has a direct edge. "
                f"Cannot add a conditional edge."
            )
        
        self.conditional_edges[source] = ConditionalEdge(
            source=source,
            condition=condition,
            routes=routes
        )
        return self
    
    def set_entry_point(self, node_name: str) -> "Graph":
        """Set the entry point of the graph."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.entry_point = node_name
        return self
    
    def get_next_node(self, current_node: str, state_data: Dict[str, Any]) -> Optional[str]:
        """
        Get the next node to execute based on edges and state.
        
        Args:
            current_node: Current node name
            state_data: Current state data
            
        Returns:
            Next node name, END, or None if no edge defined
        """
        # Check for conditional edge first
        if current_node in self.conditional_edges:
            conditional = self.conditional_edges[current_node]
            return conditional.evaluate(state_data)
        
        # Check for direct edge
        if current_node in self.edges:
            return self.edges[current_node]
        
        # No edge defined - implicit end
        return None
    
    def validate(self) -> List[str]:
        """
        Validate the graph structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Must have at least one node
        if not self.nodes:
            errors.append("Graph must have at least one node")
            return errors
        
        # Must have an entry point
        if not self.entry_point:
            errors.append("Graph must have an entry point")
        elif self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' not found in nodes")
        
        # Check for orphan nodes (not reachable from entry point)
        reachable = self._get_reachable_nodes()
        orphans = set(self.nodes.keys()) - reachable
        if orphans:
            errors.append(f"Orphan nodes (not reachable): {orphans}")
        
        # Check that nodes without outgoing edges make sense
        for node_name in self.nodes:
            if node_name not in self.edges and node_name not in self.conditional_edges:
                # This is an implicit end node - that's okay
                pass
        
        return errors
    
    def _get_reachable_nodes(self) -> Set[str]:
        """Get all nodes reachable from the entry point."""
        if not self.entry_point:
            return set()
        
        reachable = set()
        to_visit = [self.entry_point]
        
        while to_visit:
            node = to_visit.pop()
            if node in reachable or node == END:
                continue
            
            reachable.add(node)
            
            # Add direct edge target
            if node in self.edges:
                to_visit.append(self.edges[node])
            
            # Add conditional edge targets
            if node in self.conditional_edges:
                for target in self.conditional_edges[node].routes.values():
                    to_visit.append(target)
        
        return reachable
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": self.edges,
            "conditional_edges": {
                name: edge.to_dict() 
                for name, edge in self.conditional_edges.items()
            },
            "entry_point": self.entry_point,
            "max_iterations": self.max_iterations,
            "metadata": self.metadata,
        }
    
    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram of the graph."""
        lines = ["graph TD"]
        
        # Add nodes
        for name, node in self.nodes.items():
            label = name.replace("_", " ").title()
            if node.node_type == NodeType.ENTRY:
                lines.append(f'    {name}["{label} 🚀"]')
            elif node.node_type == NodeType.EXIT:
                lines.append(f'    {name}["{label} 🏁"]')
            else:
                lines.append(f'    {name}["{label}"]')
        
        # Add END node if used
        has_end = END in self.edges.values()
        for cond in self.conditional_edges.values():
            if END in cond.routes.values():
                has_end = True
                break
        
        if has_end:
            lines.append(f'    {END}(("END"))')
        
        # Add direct edges
        for source, target in self.edges.items():
            lines.append(f"    {source} --> {target}")
        
        # Add conditional edges
        for source, cond in self.conditional_edges.items():
            for route_key, target in cond.routes.items():
                lines.append(f"    {source} -->|{route_key}| {target}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"Graph(name='{self.name}', nodes={list(self.nodes.keys())}, "
            f"entry='{self.entry_point}')"
        )
