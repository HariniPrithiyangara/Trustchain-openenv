"""Data models for TrustChain Verification Environment."""

from typing import Literal, Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TrustChainAction(Action):
    """Action for the TrustChain verification agent."""

    action_type: Literal["verify_claim", "cross_reference", "flag_hallucination", "pass_forward", "request_source"] = Field(
        ..., description="The type of verification action to perform."
    )
    claim_id: str = Field(
        ..., description="The ID of the claim being acted upon."
    )


class TrustChainObservation(Observation):
    """Observation from the TrustChain environment."""

    claim_id: str = Field(default="", description="The unique identifier for the current claim.")
    claim: str = Field(default="", description="The text of the claim to verify.")
    sender: str = Field(default="", description="The theoretical agent or reporter that made the claim.")
    context: Optional[str] = Field(default=None, description="Additional context, if requested.")
    difficulty: str = Field(default="easy", description="The difficulty tier of the current task.")
    feedback: str = Field(default="", description="Feedback from the previous action.")
    claims_remaining: int = Field(default=0, description="How many claims remain in this task session.")
