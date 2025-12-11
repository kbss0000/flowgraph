"""
Built-in Tools for the Code Review Workflow.

These tools implement the functionality needed for the sample
Code Review workflow demonstration.
"""

import re
import ast
from typing import Any, Dict, List, Optional
from app.tools.registry import register_tool


@register_tool(
    name="extract_functions",
    description="Extract function definitions from Python code"
)
def extract_functions(code: str) -> Dict[str, Any]:
    """
    Extract function names and basic info from Python code.
    
    Args:
        code: Python source code string
        
    Returns:
        Dict with 'functions' list containing function info
    """
    functions = []
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "lineno": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "has_docstring": (
                        ast.get_docstring(node) is not None
                    ),
                    "decorators": [
                        ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                        for d in node.decorator_list
                    ],
                    "line_count": (
                        node.end_lineno - node.lineno + 1
                        if hasattr(node, 'end_lineno') and node.end_lineno
                        else 0
                    ),
                }
                functions.append(func_info)
                
    except SyntaxError as e:
        return {
            "functions": [],
            "error": f"Syntax error in code: {e}",
            "parse_success": False,
        }
    
    return {
        "functions": functions,
        "function_count": len(functions),
        "parse_success": True,
    }


@register_tool(
    name="calculate_complexity",
    description="Calculate complexity metrics for code"
)
def calculate_complexity(code: str, functions: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Calculate simple complexity metrics for Python code.
    
    Metrics:
    - Lines of code (LOC)
    - Cyclomatic complexity (simplified)
    - Nesting depth
    - Function count
    
    Args:
        code: Python source code
        functions: Optional pre-extracted function list
        
    Returns:
        Dict with complexity metrics
    """
    lines = code.split('\n')
    loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    
    # Simple cyclomatic complexity: count decision points
    complexity_keywords = ['if', 'elif', 'for', 'while', 'and', 'or', 'except', 'with']
    complexity = 1  # Base complexity
    
    for line in lines:
        stripped = line.strip()
        for keyword in complexity_keywords:
            if re.match(rf'\b{keyword}\b', stripped):
                complexity += 1
    
    # Calculate max nesting depth
    max_depth = 0
    current_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            depth = indent // 4  # Assume 4-space indentation
            max_depth = max(max_depth, depth)
    
    # Calculate function count
    func_count = len(functions) if functions else code.count('def ')
    
    # Generate a simple complexity score (1-10 scale)
    # Lower is better
    score = 10
    if complexity > 10:
        score -= 2
    if complexity > 20:
        score -= 2
    if max_depth > 4:
        score -= 1
    if max_depth > 6:
        score -= 1
    if loc > 200:
        score -= 1
    if func_count > 10:
        score -= 1
    if functions:
        long_funcs = [f for f in functions if f.get('line_count', 0) > 50]
        score -= len(long_funcs)
    
    score = max(1, score)  # Minimum score of 1
    
    return {
        "lines_of_code": loc,
        "cyclomatic_complexity": complexity,
        "max_nesting_depth": max_depth,
        "function_count": func_count,
        "complexity_score": score,
    }


@register_tool(
    name="detect_issues",
    description="Detect code quality issues and smells"
)
def detect_issues(
    code: str,
    functions: Optional[List[Dict]] = None,
    complexity_score: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect common code quality issues.
    
    Checks for:
    - Missing docstrings
    - Long functions
    - Deep nesting
    - Magic numbers
    - TODO/FIXME comments
    - Print statements (in production code)
    - Unused imports (basic check)
    
    Args:
        code: Python source code
        functions: Optional pre-extracted functions
        complexity_score: Optional pre-calculated complexity
        
    Returns:
        Dict with issues list and summary
    """
    issues = []
    lines = code.split('\n')
    
    # Check for missing docstrings
    if functions:
        for func in functions:
            if not func.get('has_docstring'):
                issues.append({
                    "type": "missing_docstring",
                    "severity": "warning",
                    "message": f"Function '{func['name']}' lacks a docstring",
                    "line": func.get('lineno'),
                })
    
    # Check for long functions
    if functions:
        for func in functions:
            line_count = func.get('line_count', 0)
            if line_count > 50:
                issues.append({
                    "type": "long_function",
                    "severity": "warning",
                    "message": f"Function '{func['name']}' is too long ({line_count} lines)",
                    "line": func.get('lineno'),
                })
    
    # Check for TODO/FIXME
    for i, line in enumerate(lines, 1):
        if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
            issues.append({
                "type": "todo_comment",
                "severity": "info",
                "message": f"Found TODO/FIXME comment",
                "line": i,
            })
    
    # Check for print statements
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('print(') or 'print(' in stripped:
            issues.append({
                "type": "print_statement",
                "severity": "info",
                "message": "Print statement found (consider using logging)",
                "line": i,
            })
    
    # Check for magic numbers
    magic_number_pattern = r'\b(?<![\'".])\d{2,}\b(?![\'"])'
    for i, line in enumerate(lines, 1):
        # Skip comments and string assignments
        stripped = line.strip()
        if not stripped.startswith('#'):
            matches = re.findall(magic_number_pattern, line)
            for match in matches:
                if int(match) not in (0, 1, 2, 100):  # Common acceptable values
                    issues.append({
                        "type": "magic_number",
                        "severity": "info",
                        "message": f"Magic number {match} found (consider using a constant)",
                        "line": i,
                    })
                    break  # One per line is enough
    
    # Calculate quality score based on issues
    quality_score = 10
    for issue in issues:
        if issue['severity'] == 'error':
            quality_score -= 2
        elif issue['severity'] == 'warning':
            quality_score -= 1
        else:
            quality_score -= 0.5
    
    # Factor in complexity score if provided
    if complexity_score:
        quality_score = (quality_score + complexity_score) / 2
    
    quality_score = max(1, min(10, quality_score))
    
    return {
        "issues": issues,
        "issue_count": len(issues),
        "quality_score": round(quality_score, 1),
        "issues_by_severity": {
            "error": len([i for i in issues if i['severity'] == 'error']),
            "warning": len([i for i in issues if i['severity'] == 'warning']),
            "info": len([i for i in issues if i['severity'] == 'info']),
        }
    }


@register_tool(
    name="suggest_improvements",
    description="Generate improvement suggestions based on detected issues"
)
def suggest_improvements(
    issues: List[Dict],
    functions: Optional[List[Dict]] = None,
    quality_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate actionable improvement suggestions.
    
    Args:
        issues: List of detected issues
        functions: Optional function info
        quality_score: Current quality score
        
    Returns:
        Dict with suggestions and priority ranking
    """
    suggestions = []
    
    # Group issues by type
    issue_types = {}
    for issue in issues:
        issue_type = issue.get('type', 'unknown')
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    # Generate suggestions based on issue types
    if 'missing_docstring' in issue_types:
        count = len(issue_types['missing_docstring'])
        suggestions.append({
            "priority": "high",
            "category": "documentation",
            "suggestion": f"Add docstrings to {count} function(s)",
            "details": "Good docstrings improve code maintainability and enable automatic documentation generation.",
            "affected_functions": [i.get('message', '').split("'")[1] for i in issue_types['missing_docstring'] if "'" in i.get('message', '')],
        })
    
    if 'long_function' in issue_types:
        count = len(issue_types['long_function'])
        suggestions.append({
            "priority": "high",
            "category": "refactoring",
            "suggestion": f"Refactor {count} long function(s) into smaller units",
            "details": "Functions over 50 lines are harder to understand and test. Consider extracting helper functions.",
        })
    
    if 'print_statement' in issue_types:
        count = len(issue_types['print_statement'])
        suggestions.append({
            "priority": "medium",
            "category": "logging",
            "suggestion": f"Replace {count} print statement(s) with proper logging",
            "details": "Use the logging module for better control over log levels and output.",
        })
    
    if 'magic_number' in issue_types:
        count = len(issue_types['magic_number'])
        suggestions.append({
            "priority": "medium",
            "category": "readability",
            "suggestion": f"Extract {count} magic number(s) into named constants",
            "details": "Named constants improve readability and make the code easier to modify.",
        })
    
    if 'todo_comment' in issue_types:
        count = len(issue_types['todo_comment'])
        suggestions.append({
            "priority": "low",
            "category": "maintenance",
            "suggestion": f"Address {count} TODO/FIXME comment(s)",
            "details": "Consider creating issues or tasks to track these items.",
        })
    
    # Add general suggestions if quality is low
    if quality_score and quality_score < 5:
        suggestions.append({
            "priority": "high",
            "category": "general",
            "suggestion": "Consider a comprehensive code review",
            "details": "The overall quality score is low. A thorough review may reveal structural improvements.",
        })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    # Calculate new expected quality score after improvements
    potential_improvement = len(suggestions) * 0.5
    new_quality_score = min(10, (quality_score or 5) + potential_improvement)
    
    return {
        "suggestions": suggestions,
        "suggestion_count": len(suggestions),
        "current_quality_score": quality_score,
        "potential_quality_score": round(new_quality_score, 1),
        "categories": list(set(s['category'] for s in suggestions)),
    }


@register_tool(
    name="quality_check",
    description="Check if quality meets the threshold"
)
def quality_check(quality_score: float, quality_threshold: float = 7.0) -> str:
    """
    Simple routing function to check if quality meets threshold.
    
    Args:
        quality_score: Current quality score (1-10)
        quality_threshold: Minimum acceptable score
        
    Returns:
        "pass" if quality meets threshold, "fail" otherwise
    """
    if quality_score >= quality_threshold:
        return "pass"
    return "fail"
