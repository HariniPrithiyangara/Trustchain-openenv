"""TrustChain Environment Logic."""

import uuid
from typing import Optional, List, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from server.models import TrustChainAction, TrustChainObservation


TASKS = {
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


def compute_reward(claim: dict, action_type: str) -> float:
    """Core reward function for all difficulty levels."""
    is_hallucination = not claim["true"]
    if (action_type == "flag_hallucination" and is_hallucination) or \
       (action_type == "pass_forward" and not is_hallucination):
        return 1.0
    elif action_type in ["cross_reference", "request_source"]:
        return 0.1
    elif action_type == "verify_claim":
        return 0.0
    else:
        return -0.5


def grade_easy(claim: dict, action_type: str) -> float:
    """Grader for easy tasks: direct truth/false checking."""
    return compute_reward(claim, action_type)


def grade_medium(claim: dict, action_type: str) -> float:
    """Grader for medium tasks: nuanced reporting with source requests rewarded."""
    return compute_reward(claim, action_type)


def grade_hard(claim: dict, action_type: str) -> float:
    """Grader for hard tasks: adversarial hallucinations with cross-reference signals."""
    return compute_reward(claim, action_type)



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
        self._difficulty = task if task in TASKS else "easy"
        
        # Deep copy to allow fresh start without state pollution
        self._claims = [c.copy() for c in TASKS[self._difficulty]]
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

        # Select grader based on task difficulty
        grader = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}.get(
            self._difficulty, grade_easy
        )

        if action.action_type in ["request_source", "cross_reference"]:
            if "context" in current_claim:
                current_claim["revealed_context"] = current_claim["context"]
                reward = grader(current_claim, action.action_type)
                feedback = "Context retrieved successfully. Please make a final decision."
            else:
                feedback = "No additional context available for this claim. Must decide based on common knowledge."
                reward = 0.0
        elif action.action_type == "verify_claim":
            feedback = "Verification process started. Please either flag_hallucination or pass_forward now."
            reward = grader(current_claim, action.action_type)
        elif action.action_type in ["flag_hallucination", "pass_forward"]:
            reward = grader(current_claim, action.action_type)
            if reward >= 1.0:
                feedback = "Correct! Claim successfully evaluated."
            else:
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
