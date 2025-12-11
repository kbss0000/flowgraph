"""
Tools API Routes.

Endpoints for listing and managing registered tools.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
import logging

from app.api.schemas import (
    ToolInfo,
    ToolListResponse,
    ToolRegisterRequest,
    ToolRegisterResponse,
    ErrorResponse,
)
from app.tools.registry import tool_registry


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tools"])


@router.get(
    "/",
    response_model=ToolListResponse,
)
async def list_tools() -> ToolListResponse:
    """
    List all registered tools.
    
    Tools are functions that workflow nodes can use during execution.
    """
    tools = tool_registry.list_tools()
    
    tool_infos = [
        ToolInfo(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in tools
    ]
    
    return ToolListResponse(tools=tool_infos, total=len(tool_infos))


@router.get(
    "/{tool_name}",
    response_model=ToolInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_tool(tool_name: str) -> ToolInfo:
    """Get information about a specific tool."""
    tool = tool_registry.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found"
        )
    
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


@router.post(
    "/register",
    response_model=ToolRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid tool code"},
        409: {"model": ErrorResponse, "description": "Tool already exists"},
    }
)
async def register_tool(request: ToolRegisterRequest) -> ToolRegisterResponse:
    """
    Register a new tool dynamically.
    
    **Warning**: This endpoint executes Python code. Use with caution
    and only in trusted environments.
    
    The code should define a function that:
    - Takes parameters as needed
    - Returns a dictionary with results
    """
    # Check if tool already exists
    if tool_registry.has(request.name):
        raise HTTPException(
            status_code=409,
            detail=f"Tool '{request.name}' already exists"
        )
    
    # Try to compile and execute the code
    try:
        # Create a restricted namespace
        namespace: Dict[str, Any] = {}
        
        # Execute the code to define the function
        exec(request.code, namespace)
        
        # Find the function in the namespace
        func = None
        for name, value in namespace.items():
            if callable(value) and not name.startswith("_"):
                func = value
                break
        
        if func is None:
            raise HTTPException(
                status_code=400,
                detail="No callable function found in the provided code"
            )
        
        # Register the tool
        tool_registry.add(
            func=func,
            name=request.name,
            description=request.description,
        )
        
        logger.info(f"Registered dynamic tool: {request.name}")
        
        return ToolRegisterResponse(
            name=request.name,
            message=f"Tool '{request.name}' registered successfully",
            warning="Dynamic tool registration executes code. Use responsibly.",
        )
        
    except SyntaxError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Syntax error in tool code: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error registering tool: {e}"
        )


@router.delete(
    "/{tool_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_tool(tool_name: str):
    """Delete a registered tool."""
    # Protect built-in tools
    builtin_tools = {
        "extract_functions",
        "calculate_complexity", 
        "detect_issues",
        "suggest_improvements",
        "quality_check",
    }
    
    if tool_name in builtin_tools:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete built-in tool '{tool_name}'"
        )
    
    deleted = tool_registry.remove(tool_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found"
        )
    
    logger.info(f"Deleted tool: {tool_name}")
