
"""
FastAPI application for the Openenv Environment.

This module creates an HTTP server that exposes the OpenenvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
