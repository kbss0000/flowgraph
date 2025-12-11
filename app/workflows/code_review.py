"""
Code Review Workflow Implementation.

This is the sample workflow demonstrating the workflow engine capabilities:
1. Extract functions from code
2. Check complexity
3. Detect issues
4. Suggest improvements
5. Loop until quality_score >= threshold
"""

from typing import Any, Dict
import logging

from app.engine.graph import Graph, END
from app.engine.node import node, NodeType
from app.tools.builtin import (
    extract_functions,
    calculate_complexity,
    detect_issues,
    suggest_improvements,
    quality_check,
)
from app.tools.registry import tool_registry


logger = logging.getLogger(__name__)


# ============================================================
# Node Handlers (using the @node decorator)
# ============================================================

@node(name="extract_node", description="Extract functions from the input code")
def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract function definitions from the code.
    
    Input state requires:
    - code: str - The Python source code to analyze
    
    Updates state with:
    - functions: List[dict] - Extracted function information
    - function_count: int - Number of functions found
    """
    code = state.get("code", "")
    result = extract_functions(code)
    state.update(result)
    logger.info(f"Extracted {result.get('function_count', 0)} functions")
    return state


@node(name="complexity_node", description="Calculate code complexity metrics")
def complexity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate complexity metrics for the code.
    
    Uses state:
    - code: str - Source code
    - functions: List[dict] - Previously extracted functions
    
    Updates state with:
    - lines_of_code: int
    - cyclomatic_complexity: int
    - complexity_score: int (1-10)
    """
    code = state.get("code", "")
    functions = state.get("functions", [])
    result = calculate_complexity(code, functions)
    state.update(result)
    logger.info(f"Complexity score: {result.get('complexity_score', 0)}")
    return state


@node(name="issues_node", description="Detect code quality issues")
def issues_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect code quality issues and calculate quality score.
    
    Uses state:
    - code: str - Source code
    - functions: List[dict] - Extracted functions
    - complexity_score: int - From complexity check
    
    Updates state with:
    - issues: List[dict] - Detected issues
    - issue_count: int
    - quality_score: float (1-10)
    """
    code = state.get("code", "")
    functions = state.get("functions", [])
    complexity_score = state.get("complexity_score")
    
    result = detect_issues(code, functions, complexity_score)
    state.update(result)
    
    logger.info(
        f"Found {result.get('issue_count', 0)} issues, "
        f"quality score: {result.get('quality_score', 0)}"
    )
    return state


@node(name="improve_node", description="Generate improvement suggestions")
def improve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate improvement suggestions based on detected issues.
    
    Uses state:
    - issues: List[dict] - Detected issues
    - functions: List[dict] - Extracted functions
    - quality_score: float - Current quality score
    
    Updates state with:
    - suggestions: List[dict] - Improvement suggestions
    - suggestion_count: int
    - potential_quality_score: float - Score after improvements
    """
    issues = state.get("issues", [])
    functions = state.get("functions", [])
    quality_score = state.get("quality_score", 5.0)
    
    result = suggest_improvements(issues, functions, quality_score)
    state.update(result)
    
    # Simulate improvement by slightly increasing quality score
    # In a real scenario, this would involve actual code modifications
    improvement = min(0.5, result.get("suggestion_count", 0) * 0.2)
    state["quality_score"] = min(10, quality_score + improvement)
    
    logger.info(
        f"Generated {result.get('suggestion_count', 0)} suggestions, "
        f"quality improved to {state['quality_score']}"
    )
    return state


# Register node handlers as tools so they can be retrieved when rebuilding from storage
def _wrapper_handler(handler_func):
    """Create a wrapper that works with tool registry."""
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return handler_func(state)
    wrapper.__name__ = handler_func.__name__
    wrapper.__doc__ = handler_func.__doc__
    return wrapper

tool_registry.add(_wrapper_handler(extract_node), name="extract_node", description="Extract functions from code")
tool_registry.add(_wrapper_handler(complexity_node), name="complexity_node", description="Calculate complexity")
tool_registry.add(_wrapper_handler(issues_node), name="issues_node", description="Detect quality issues")
tool_registry.add(_wrapper_handler(improve_node), name="improve_node", description="Suggest improvements")


# ============================================================
# Condition Functions
# ============================================================

def quality_meets_threshold(state: Dict[str, Any]) -> str:
    """
    Routing condition: check if quality meets threshold.
    
    Returns:
    - "pass" if quality_score >= quality_threshold
    - "fail" if more improvement needed
    """
    quality_score = state.get("quality_score", 0)
    threshold = state.get("quality_threshold", 7.0)
    
    if quality_score >= threshold:
        logger.info(f"Quality {quality_score} meets threshold {threshold}")
        return "pass"
    else:
        logger.info(f"Quality {quality_score} below threshold {threshold}")
        return "fail"


def always_loop(state: Dict[str, Any]) -> str:
    """Always return to issues check after improvement."""
    return "continue"


# ============================================================
# Workflow Factory
# ============================================================

def create_code_review_workflow(
    max_iterations: int = 5,
    quality_threshold: float = 7.0
) -> Graph:
    """
    Create a Code Review workflow graph.
    
    Workflow flow:
    ```
    extract → complexity → issues ─┬─→ END (if pass)
                                   │
                                   └─→ improve → issues (loop if fail)
    ```
    
    Args:
        max_iterations: Maximum improvement loops
        quality_threshold: Minimum quality score to pass
        
    Returns:
        Configured Graph instance
    """
    graph = Graph(
        name="Code Review Workflow",
        description=(
            "Analyzes Python code for quality issues and suggests improvements. "
            f"Loops until quality score >= {quality_threshold} or max {max_iterations} iterations."
        ),
        max_iterations=max_iterations,
    )
    
    # Add nodes
    graph.add_node("extract", handler=extract_node, description="Extract functions from code")
    graph.add_node("complexity", handler=complexity_node, description="Calculate complexity")
    graph.add_node("issues", handler=issues_node, description="Detect quality issues")
    graph.add_node("improve", handler=improve_node, description="Suggest improvements")
    
    # Add edges
    graph.add_edge("extract", "complexity")
    graph.add_edge("complexity", "issues")
    
    # Conditional edge: issues → END or improve
    graph.add_conditional_edge(
        "issues",
        quality_meets_threshold,
        {"pass": END, "fail": "improve"}
    )
    
    # Loop back from improve to issues
    graph.add_conditional_edge(
        "improve",
        always_loop,
        {"continue": "issues"}
    )
    
    # Set entry point
    graph.set_entry_point("extract")
    
    return graph


async def register_code_review_workflow():
    """
    Register a pre-built Code Review workflow in storage.
    
    This makes the workflow available immediately via the API
    without needing to create it first.
    """
    from app.storage.memory import graph_storage
    
    workflow = create_code_review_workflow()
    
    await graph_storage.save(
        graph_id="code-review-demo",
        name="Code Review Demo",
        definition=workflow.to_dict(),
    )
    
    logger.info("Registered Code Review workflow with ID: code-review-demo")
    return workflow


# ============================================================
# Example Usage
# ============================================================

async def run_code_review_demo():
    """
    Demo function showing how to run the code review workflow.
    
    Usage:
        import asyncio
        from app.workflows.code_review import run_code_review_demo
        asyncio.run(run_code_review_demo())
    """
    from app.engine.executor import execute_graph
    
    # Sample code to review
    sample_code = '''
def calculate_total(items):
    total = 0
    for item in items:
        if item.price > 0:
            if item.quantity > 0:
                if item.discount:
                    total += item.price * item.quantity * (1 - item.discount)
                else:
                    total += item.price * item.quantity
    return total

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 100:
            result.append(data[i] * 2)
        else:
            result.append(data[i])
    print(result)
    return result


def helper():
    x = 42
    return x * 1000
'''
    
    # Create workflow
    workflow = create_code_review_workflow(max_iterations=3, quality_threshold=6.0)
    
    # Initial state
    initial_state = {
        "code": sample_code,
        "quality_threshold": 6.0,
    }
    
    # Execute
    print("Starting Code Review...")
    result = await execute_graph(workflow, initial_state)
    
    # Print results
    print(f"\nExecution Status: {result.status.value}")
    print(f"Total Duration: {result.total_duration_ms:.2f}ms")
    print(f"Iterations: {result.iterations}")
    print(f"\nFinal Quality Score: {result.final_state.get('quality_score', 'N/A')}")
    print(f"Issues Found: {result.final_state.get('issue_count', 'N/A')}")
    print(f"\nSuggestions:")
    for suggestion in result.final_state.get("suggestions", []):
        print(f"  - [{suggestion['priority']}] {suggestion['suggestion']}")
    
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_code_review_demo())
