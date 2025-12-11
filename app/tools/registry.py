"""
Tool Registry for Workflow Engine.

The tool registry maintains a collection of callable tools that
workflow nodes can use. Tools are simple Python functions that
perform specific operations.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import functools
import inspect
import logging


logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """
    A registered tool.
    
    Attributes:
        name: Unique identifier for the tool
        func: The callable function
        description: Human-readable description
        parameters: Parameter descriptions
    """
    name: str
    func: Callable
    description: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the tool function."""
        return self.func(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    Registry for workflow tools.
    
    Tools are simple Python functions that nodes can call to perform
    specific operations. The registry allows dynamic registration
    and lookup of tools.
    
    Usage:
        registry = ToolRegistry()
        
        @registry.register("my_tool")
        def my_tool(data: str) -> dict:
            return {"result": data.upper()}
        
        # Later
        tool = registry.get("my_tool")
        result = tool("hello")
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, str]] = None
    ) -> Callable:
        """
        Decorator to register a function as a tool.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            parameters: Parameter descriptions
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            
            # Extract parameters from signature if not provided
            params = parameters or {}
            if not params:
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    if param_name not in ("self", "cls"):
                        params[param_name] = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            
            # Create and store tool
            tool = Tool(
                name=tool_name,
                func=func,
                description=tool_desc.strip(),
                parameters=params,
            )
            self._tools[tool_name] = tool
            
            logger.debug(f"Registered tool: {tool_name}")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def add(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Directly add a function as a tool (non-decorator version).
        
        Args:
            func: The function to register
            name: Tool name (defaults to function name)
            description: Tool description
            parameters: Parameter descriptions
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        tool = Tool(
            name=tool_name,
            func=func,
            description=tool_desc.strip(),
            parameters=parameters or {},
        )
        self._tools[tool_name] = tool
        logger.debug(f"Added tool: {tool_name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def call(self, name: str, *args, **kwargs) -> Any:
        """
        Call a tool by name.
        
        Args:
            name: Tool name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool result
            
        Raises:
            KeyError: If tool not found
        """
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool(*args, **kwargs)
    
    def remove(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata."""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __contains__(self, name: str) -> bool:
        return self.has(name)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __iter__(self):
        return iter(self._tools.values())


# Global tool registry instance
tool_registry = ToolRegistry()


def register_tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Convenience decorator to register a tool in the global registry.
    
    Usage:
        @register_tool("my_tool", description="Does something cool")
        def my_tool(data: str) -> dict:
            return {"result": data}
    """
    return tool_registry.register(name, description, parameters)


def get_tool(name: str) -> Optional[Tool]:
    """Get a tool from the global registry."""
    return tool_registry.get(name)
