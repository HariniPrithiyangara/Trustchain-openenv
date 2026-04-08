
"""Data models for the CloudOps Incident Responder environment."""

from typing import Literal, Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class CloudOpsAction(Action):
    """Action for the CloudOps environment."""

    action_type: Literal["list_resources", "read_logs", "restart_service", "scale_up", "rollback", "delete_resource"] = Field(
        ..., description="The type of SRE action to perform."
    )
    resource_id: Optional[str] = Field(
        None, description="The ID of the resource (VM, DB, or Service) to act upon."
    )


class CloudOpsObservation(Observation):
    """Observation from the CloudOps environment."""

    alert: str = Field(default="", description="The current system alert or incident description.")
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="List of infrastructure resources and their metrics.")
    logs: Optional[str] = Field(default=None, description="Logs retrieved if the last action was 'read_logs'.")
    feedback: str = Field(default="", description="Feedback on the effectiveness of the previous action.")
    difficulty: str = Field(default="easy", description="The difficulty tier of the current incident.")
