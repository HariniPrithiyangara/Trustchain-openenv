
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

try:
    from models import TrustchainAction, TrustchainObservation
    from server.openenv_environment import TrustchainEnvironment
except ImportError:
    # Fallback for different import paths
    from openenv.models import TrustchainAction, TrustchainObservation
    from openenv.server.openenv_environment import TrustchainEnvironment


# Create the app with web interface and README integration
app = create_app(
    TrustchainEnvironment,
    TrustchainAction,
    TrustchainObservation,
    env_name="trustchain",
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
