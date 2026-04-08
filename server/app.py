
"""
FastAPI application for the Openenv Environment.
"""

import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required for the web interface. Install with 'pip install openenv-core[core]'"
    ) from e

from my_env_v4.env import MyEnvV4Env
from my_env_v4.models import MyEnvV4Action, MyEnvV4Observation


# Create the app with web interface and README integration
app = create_app(
    MyEnvV4Env,
    MyEnvV4Action,
    MyEnvV4Observation,
    env_name="backend_workflow",
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
