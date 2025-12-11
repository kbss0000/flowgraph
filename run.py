#!/usr/bin/env python3
"""
Simple run script for the Workflow Engine.

Usage:
    python run.py
    
Or with custom settings:
    HOST=127.0.0.1 PORT=8080 python run.py
"""

import uvicorn
import os


def main():
    """Run the FastAPI application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                      FlowGraph 🔄                             ║
║                                                               ║
║  A lightweight workflow orchestration engine                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Server:    http://{host}:{port}                                 ║
║  API Docs:  http://{host}:{port}/docs                            ║
║  ReDoc:     http://{host}:{port}/redoc                           ║
╠═══════════════════════════════════════════════════════════════╣
║  Demo workflow ID: code-review-demo                           ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
