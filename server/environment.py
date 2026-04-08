"""TrustChain Environment Logic."""

import uuid
from typing import Optional, List, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from server.models import TrustChainAction, TrustChainObservation


TASKS_DATA = {
    "easy": [
        {"id": "claim_e1", "claim": "The Eiffel Tower is located in Paris.", "sender": "agent_alpha", "true": True, "type": "direct"},
        {"id": "claim_e2", "claim": "Water boils at 200 degrees Celsius at sea level.", "sender": "agent_beta", "true": False, "type": "direct"},
        {"id": "claim_e3", "claim": "The sun rises in the east.", "sender": "agent_alpha", "true": True, "type": "direct"}
    ],
    "medium": [
        {"id": "claim_m1", "claim": "Company X reported a 20% increase in revenue.", "sender": "news_agent", "true": False, "type": "needs_source", "context": "Q3 earnings report: Company X revenue grew by 15%."},
        {"id": "claim_m2", "claim": "Elon Musk founded Tesla in 2003.", "sender": "history_agent", "true": False, "type": "direct"},
        {"id": "claim_m3", "claim": "Python was released in 1991.", "sender": "tech_agent", "true": True, "type": "direct"}
    ],
    "hard": [
        {"id": "claim_h1", "claim": "The new study claims X cures cancer entirely.", "sender": "science_agent", "true": False, "type": "needs_cross", "context": "Study says X shows promise in mice for one specific cancer type."},
        {"id": "claim_h2", "claim": "According to the UN, global population is expected to peak at 10.4 billion.", "sender": "stat_agent", "true": True, "type": "needs_source", "context": "UN DESA expects human population to peak at 10.4 billion in the 2080s."},
        {"id": "claim_h3", "claim": "The speed of light is exactly 300,000 km/s.", "sender": "physics_agent", "true": False, "type": "direct"}
    ]
}


class TrustChainEnvironment(Environment):
    """
    TrustChain Verification Environment.
    Agents act as truth-validators verifying claims from upstream agents.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._difficulty = "easy"
        self._claims = []
        self._current_index = 0
        self._score = 0.0

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> TrustChainObservation:
        """Reset the environment with a specific task (easy, medium, hard)."""
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._difficulty = task if task in TASKS_DATA else "easy"
        
        # Deep copy to allow fresh start without state pollution
        self._claims = [c.copy() for c in TASKS_DATA[self._difficulty]]
        self._current_index = 0
        self._score = 0.0
        
        return self._get_observation("Started new episode. Verify the first claim.")

    def step(self, action: TrustChainAction) -> TrustChainObservation:
        if self._current_index >= len(self._claims):
            return self._get_observation("Session is already complete.", done=True, reward=0.0)

        self._state.step_count += 1
        current_claim = self._claims[self._current_index]
        reward = 0.0
        feedback = ""
        done = False
        
        # Action validation
        if action.claim_id != current_claim["id"]:
            feedback = f"Error: You acted on claim_id '{action.claim_id}' but the current claim is '{current_claim['id']}'"
            reward = -0.1
            return self._get_observation(feedback, done=False, reward=reward)

        if action.action_type in ["request_source", "cross_reference"]:
            if "context" in current_claim:
                current_claim["revealed_context"] = current_claim["context"]
                reward = 0.1  # small reward for taking the correct investigative step
                feedback = "Context retrieved successfully. Please make a final decision."
            else:
                feedback = "No additional context available for this claim. Must decide based on common knowledge."
                reward = 0.0
        elif action.action_type == "verify_claim":
            feedback = "Verification process started. Please either flag_hallucination or pass_forward now."
            reward = 0.0
        elif action.action_type in ["flag_hallucination", "pass_forward"]:
            is_hallucination = not current_claim["true"]
            if (action.action_type == "flag_hallucination" and is_hallucination) or \
               (action.action_type == "pass_forward" and not is_hallucination):
                reward = 1.0  # Full point for correct final decision
                feedback = "Correct! Claim successfully evaluated."
            else:
                reward = -0.5 # Penalty for incorrect validation (passing false, or flagging true)
                feedback = "Incorrect! You made the wrong judgment on this claim."
            
            # Progress to the next claim
            self._current_index += 1
            if self._current_index >= len(self._claims):
                done = True
                feedback += " Episode complete."
            else:
                feedback += " Moving to the next claim."
        else:
            feedback = "Unknown action."
            reward = -0.1

        # We normalize rewards on the inference side, but here we just return the raw step reward
        # To strictly stay in [0, 1] for OpenEnv reward parameter:
        return self._get_observation(feedback, done=done, reward=reward)

    def _get_observation(self, feedback: str, done: bool = False, reward: float = 0.0) -> TrustChainObservation:
        if self._current_index >= len(self._claims):
            return TrustChainObservation(
                claim_id="None",
                claim="All claims verified.",
                sender="System",
                context=None,
                difficulty=self._difficulty,
                feedback=feedback,
                claims_remaining=0,
                done=True,
                reward=reward
            )
            
        current = self._claims[self._current_index]
        return TrustChainObservation(
            claim_id=current["id"],
            claim=current["claim"],
            sender=current["sender"],
            context=current.get("revealed_context"),
            difficulty=self._difficulty,
            feedback=feedback,
            claims_remaining=len(self._claims) - self._current_index,
            done=done,
            reward=reward
        )

    @property
    def state(self) -> State:
        return self._state
