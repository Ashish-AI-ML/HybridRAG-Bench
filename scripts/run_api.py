"""
scripts/run_api.py — Launch the FastAPI service.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --reload
"""

import argparse
import sys
import os
import uvicorn

# Inject project root into path so uvicorn can find 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch HybridRAG Bench API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    print(f"\n🚀 Starting HybridRAG Bench API on http://{args.host}:{args.port}")
    print(f"   Docs: http://localhost:{args.port}/docs")
    print(f"   Health: http://localhost:{args.port}/health\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )
