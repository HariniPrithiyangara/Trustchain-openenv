
"""Data models for the Backend API Automation environment."""

from typing import Literal, Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MyEnvV4Action(Action):
    """Action for the Backend Workflow agent."""

    action_type: Literal["fetch_data", "validate_input", "fix_error", "return_response", "log_audit", "query_db"] = Field(
        ..., description="The type of backend operation to perform."
    )
    resource: Optional[str] = Field(
        None, description="The target resource (e.g., 'user_db', 'payload_v1', 'auth_token')."
    )
    params: Optional[Dict[str, Any]] = Field(
        None, description="Optional parameters for the operation."
    )


class MyEnvV4Observation(Observation):
    """Observation from the Backend Workflow environment."""

    status: str = Field(default="idle", description="Current workflow status.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Data returned from the previous action.")
    error_message: Optional[str] = Field(default=None, description="Error message if the previous action failed.")
    feedback: str = Field(default="", description="Continuous learning signal from the environment.")
    difficulty: str = Field(default="easy", description="Current task difficulty.")
