
"""Openenv Environment Client for CloudOps."""

from typing import Dict, Any, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import CloudOpsAction, CloudOpsObservation
except ImportError:
    from openenv.models import CloudOpsAction, CloudOpsObservation


class CloudOpsEnv(
    EnvClient[CloudOpsAction, CloudOpsObservation, State]
):
    """
    Client for the CloudOps Environment.
    """

    async def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> StepResult[CloudOpsObservation]:
        """
        Reset the environment and return the initial observation.
        """
        payload = {"task": task, "seed": seed}
        return await super().reset(**payload)

    def _step_payload(self, action: CloudOpsAction) -> Dict[str, Any]:
        """
        Convert CloudOpsAction to JSON payload for step message.
        """
        return {
            "action_type": action.action_type,
            "resource_id": action.resource_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CloudOpsObservation]:
        """
        Parse server response into StepResult[CloudOpsObservation].
        """
        obs_data = payload.get("observation", {})
        observation = CloudOpsObservation(
            alert=obs_data.get("alert", ""),
            resources=obs_data.get("resources", []),
            logs=obs_data.get("logs"),
            difficulty=obs_data.get("difficulty", "easy"),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
