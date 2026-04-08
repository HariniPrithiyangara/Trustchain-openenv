"""
FastAPI application for the TrustChain Verification Environment.
"""

import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from server.environment import TrustChainEnvironment
from server.models import TrustChainAction, TrustChainObservation

# Create the app with web interface
app = create_app(
    TrustChainEnvironment,
    TrustChainAction,
    TrustChainObservation,
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
