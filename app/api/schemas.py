"""
Pydantic Schemas for API Request/Response Models.

These schemas define the structure of data flowing through the API,
providing automatic validation and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================
# Enums
# ============================================================

class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# Node Schemas
# ============================================================

class NodeDefinition(BaseModel):
    """Definition of a node in the graph."""
    name: str = Field(..., description="Unique name for the node")
    handler: str = Field(..., description="Name of the handler function (must be registered)")
    description: Optional[str] = Field(None, description="Human-readable description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "extract",
                "handler": "extract_functions",
                "description": "Extract function definitions from code"
            }
        }


# ============================================================
# Edge Schemas
# ============================================================

class ConditionalRoutes(BaseModel):
    """Routes for a conditional edge."""
    condition: str = Field(..., description="Name of the condition function")
    routes: Dict[str, str] = Field(
        ..., 
        description="Mapping of condition results to target nodes"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "condition": "quality_check",
                "routes": {
                    "pass": "__END__",
                    "fail": "improve"
                }
            }
        }


# ============================================================
# Graph Schemas
# ============================================================

class GraphCreateRequest(BaseModel):
    """Request to create a new workflow graph."""
    name: str = Field(..., description="Name of the workflow")
    description: Optional[str] = Field(None, description="Description of what this workflow does")
    nodes: List[NodeDefinition] = Field(..., description="List of nodes in the graph")
    edges: Dict[str, str] = Field(
        default_factory=dict, 
        description="Direct edges: source -> target"
    )
    conditional_edges: Dict[str, ConditionalRoutes] = Field(
        default_factory=dict,
        description="Conditional edges with routing logic"
    )
    entry_point: Optional[str] = Field(None, description="Entry node (defaults to first node)")
    max_iterations: int = Field(100, description="Maximum loop iterations", ge=1, le=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "code_review_workflow",
                "description": "Automated code review with quality checks",
                "nodes": [
                    {"name": "extract", "handler": "extract_functions"},
                    {"name": "complexity", "handler": "calculate_complexity"},
                    {"name": "issues", "handler": "detect_issues"},
                    {"name": "improve", "handler": "suggest_improvements"}
                ],
                "edges": {
                    "extract": "complexity",
                    "complexity": "issues"
                },
                "conditional_edges": {
                    "issues": {
                        "condition": "quality_check",
                        "routes": {"pass": "__END__", "fail": "improve"}
                    },
                    "improve": {
                        "condition": "always_continue",
                        "routes": {"continue": "issues"}
                    }
                },
                "entry_point": "extract",
                "max_iterations": 10
            }
        }


class GraphCreateResponse(BaseModel):
    """Response after creating a graph."""
    graph_id: str = Field(..., description="Unique identifier for the created graph")
    name: str = Field(..., description="Name of the workflow")
    message: str = Field(default="Graph created successfully")
    node_count: int = Field(..., description="Number of nodes in the graph")
    
    class Config:
        json_schema_extra = {
            "example": {
                "graph_id": "abc123-def456",
                "name": "code_review_workflow",
                "message": "Graph created successfully",
                "node_count": 4
            }
        }


class GraphInfoResponse(BaseModel):
    """Response with graph information."""
    graph_id: str
    name: str
    description: Optional[str]
    node_count: int
    nodes: List[str]
    entry_point: Optional[str]
    max_iterations: int
    created_at: str
    mermaid_diagram: Optional[str] = Field(None, description="Mermaid diagram of the graph")


class GraphListResponse(BaseModel):
    """Response listing all graphs."""
    graphs: List[GraphInfoResponse]
    total: int


# ============================================================
# Run Schemas
# ============================================================

class GraphRunRequest(BaseModel):
    """Request to run a workflow graph."""
    graph_id: str = Field(..., description="ID of the graph to run")
    initial_state: Dict[str, Any] = Field(
        ..., 
        description="Initial state data for the workflow"
    )
    async_execution: bool = Field(
        False, 
        description="If true, run in background and return immediately"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "graph_id": "abc123-def456",
                "initial_state": {
                    "code": "def hello():\n    print('world')",
                    "quality_threshold": 7.0
                },
                "async_execution": False
            }
        }


class ExecutionLogEntry(BaseModel):
    """A single entry in the execution log."""
    step: int
    node: str
    started_at: str
    completed_at: Optional[str]
    duration_ms: Optional[float]
    iteration: int
    result: str
    error: Optional[str]
    route_taken: Optional[str]


class GraphRunResponse(BaseModel):
    """Response after running a graph."""
    run_id: str = Field(..., description="Unique identifier for this run")
    graph_id: str
    status: ExecutionStatus
    final_state: Dict[str, Any]
    execution_log: List[ExecutionLogEntry]
    started_at: Optional[str]
    completed_at: Optional[str]
    total_duration_ms: Optional[float]
    iterations: int
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "run-xyz789",
                "graph_id": "abc123-def456",
                "status": "completed",
                "final_state": {
                    "code": "def hello():\n    print('world')",
                    "functions": [{"name": "hello"}],
                    "quality_score": 8.5
                },
                "execution_log": [
                    {
                        "step": 1,
                        "node": "extract",
                        "started_at": "2024-01-01T12:00:00",
                        "completed_at": "2024-01-01T12:00:01",
                        "duration_ms": 15.5,
                        "iteration": 0,
                        "result": "success",
                        "error": None,
                        "route_taken": None
                    }
                ],
                "started_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:00:05",
                "total_duration_ms": 5000.0,
                "iterations": 1,
                "error": None
            }
        }


class RunStateResponse(BaseModel):
    """Response with current run state."""
    run_id: str
    graph_id: str
    status: ExecutionStatus
    current_node: Optional[str]
    current_state: Dict[str, Any]
    iteration: int
    execution_log: List[ExecutionLogEntry]
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]


class RunListResponse(BaseModel):
    """Response listing runs."""
    runs: List[RunStateResponse]
    total: int


# ============================================================
# Tool Schemas
# ============================================================

class ToolInfo(BaseModel):
    """Information about a registered tool."""
    name: str
    description: str
    parameters: Dict[str, str]


class ToolListResponse(BaseModel):
    """Response listing all registered tools."""
    tools: List[ToolInfo]
    total: int


class ToolRegisterRequest(BaseModel):
    """Request to register a new tool (for dynamic registration)."""
    name: str = Field(..., description="Unique name for the tool")
    description: str = Field("", description="Description of what the tool does")
    code: str = Field(..., description="Python code for the tool function")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "custom_validator",
                "description": "Custom validation logic",
                "code": "def custom_validator(data):\n    return {'valid': True}"
            }
        }


class ToolRegisterResponse(BaseModel):
    """Response after registering a tool."""
    name: str
    message: str
    warning: Optional[str] = None


# ============================================================
# Error Schemas
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "Validation Error"
    detail: List[Dict[str, Any]]
    status_code: int = 422
