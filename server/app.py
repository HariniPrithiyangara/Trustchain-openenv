
"""
FastAPI application for the Openenv Environment.

This module creates an HTTP server that exposes the OpenenvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
"""

import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required for the web interface. Install with 'pip install openenv-core[core]'"
    ) from e

from models import CloudOpsAction, CloudOpsObservation
from server.openenv_environment import CloudOpsEnvironment


# Create the app with web interface and README integration
app = create_app(
    CloudOpsEnvironment,
    CloudOpsAction,
    CloudOpsObservation,
    env_name="cloudops",
    max_concurrent_envs=1,
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
