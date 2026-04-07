

"""Data models for the TrustChain OpenEnv environment."""

from typing import Literal, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TrustchainAction(Action):
    """Action for the Trustchain environment - accept, reject, or verify."""

    decision: Literal["accept", "reject", "verify"] = Field(
        ..., description="The decision to make regarding the claim: 'accept', 'reject', or 'verify'."
    )


class TrustchainObservation(Observation):
    """Observation from the Trustchain environment - claim to verify."""

    claim: str = Field(default="", description="The claim made by the reporter agent")
    context: Optional[str] = Field(default=None, description="Optional context or source document")
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, or hard")
    feedback: str = Field(default="", description="Feedback on the previous action")
