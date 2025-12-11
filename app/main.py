"""
FlowGraph - FastAPI Application Entry Point.

A lightweight, async-first workflow orchestration engine for building agent pipelines.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.api.routes import graph, tools, websocket
from app.workflows.code_review import register_code_review_workflow

# Import builtin tools to register them
import app.tools.builtin  # noqa: F401


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Register the demo workflow
    await register_code_review_workflow()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Workflow Engine API

A minimal but powerful workflow/graph engine for building agent workflows.

### Features
- **Nodes**: Python functions that read and modify shared state
- **Edges**: Define execution flow between nodes
- **Branching**: Conditional routing based on state values
- **Looping**: Support for iterative workflows
- **Real-time Updates**: WebSocket support for live execution streaming

### Quick Start
1. List available tools: `GET /tools`
2. Create a graph: `POST /graph/create`
3. Run the graph: `POST /graph/run`
4. Check execution state: `GET /graph/state/{run_id}`

### Demo Workflow
A pre-registered Code Review workflow is available with ID: `code-review-demo`
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(graph.router)
app.include_router(tools.router)
app.include_router(websocket.router)


# ============================================================
# Root Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic info and links."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "A minimal workflow/graph engine for agent workflows",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "graphs": "/graph",
            "tools": "/tools",
            "websocket_run": "/ws/run/{graph_id}",
            "websocket_subscribe": "/ws/subscribe/{run_id}",
        },
        "demo_workflow": "code-review-demo",
    }


@app.get("/health", tags=["Root"])
async def health():
    """Health check endpoint."""
    from app.storage.memory import graph_storage, run_storage
    
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "graphs_count": len(graph_storage),
        "runs_count": len(run_storage),
    }


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )
