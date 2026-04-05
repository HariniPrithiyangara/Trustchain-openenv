# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Openenv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TrustchainAction, TrustchainObservation
except ImportError:
    from models import TrustchainAction, TrustchainObservation


class TrustchainEnv(
    EnvClient[TrustchainAction, TrustchainObservation, State]
):
    """
    Client for the Trustchain Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: TrustchainAction) -> Dict:
        """
        Convert TrustchainAction to JSON payload for step message.
        """
        return {
            "decision": action.decision,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrustchainObservation]:
        """
        Parse server response into StepResult[TrustchainObservation].
        """
        obs_data = payload.get("observation", {})
        observation = TrustchainObservation(
            claim=obs_data.get("claim", ""),
            context=obs_data.get("context"),
            difficulty=obs_data.get("difficulty", ""),
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

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
