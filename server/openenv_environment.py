

"""TrustChain OpenEnv environment implementation."""

import random
from typing import Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from openenv.models import TrustchainAction, TrustchainObservation


EASY_TASKS = [
    {"claim": "The Earth has 2 moons.", "context": None, "truth": "reject"},
    {"claim": "Water boils at 100 degrees Celsius under standard atmospheric pressure.", "context": None, "truth": "accept"},
    {"claim": "Python is a compiled language that does not require an interpreter.", "context": None, "truth": "reject"},
    {"claim": "The area of a circle with radius r is pi times r squared.", "context": None, "truth": "accept"},
    {"claim": "Penicillin is an antibiotic used to treat bacterial infections.", "context": None, "truth": "accept"},
    {"claim": "The Great Wall of China is visible from space with the naked eye.", "context": None, "truth": "reject"},
]

MEDIUM_TASKS = [
    {"claim": "Revenue increased by 20% in Q3.", "context": "In Q2, revenue was $10M. In Q3, it reached $12M.", "truth": "accept"},
    {"claim": "The company acquired a new startup using only cash.", "context": "The acquisition involved $5M in cash and $2M in company stock.", "truth": "reject"},
    {"claim": "The API rate limit is 100 requests per minute.", "context": "Users can make up to 6000 requests per hour. (Note: 6000 per hour = 100 per minute)", "truth": "accept"},
    {"claim": "The bug was caused by a memory leak in the core module.", "context": "The investigation revealed a race condition in the auth component, leading to crashes.", "truth": "reject"},
    {"claim": "The patient's blood pressure is within normal range.", "context": "Patient readings: systolic 118 mmHg, diastolic 76 mmHg. Normal range is 90-120 / 60-80 mmHg.", "truth": "accept"},
    {"claim": "The contract was signed by both parties before the deadline.", "context": "Party A signed on March 12. Party B signed on March 16. The deadline was March 15.", "truth": "reject"},
]

HARD_TASKS = [
    {"claim": "The system architecture is fully compliant with SOC2.", "context": "The recent audit showed passes on 90% of controls, but the access logging control is still pending review.", "truth": "verify", "difficulty": "hard"},
    {"claim": "The CEO explicitly promised a hybrid work policy starting next month.", "context": "In the all-hands, the CEO said 'we are looking into hybrid options and will have a decision soon.'", "truth": "reject", "difficulty": "hard"},
    {"claim": "The new encryption algorithm is unbreakable.", "context": "We implemented AES-256 for all at-rest data. Research implies it's highly secure against known non-quantum attacks.", "truth": "verify", "difficulty": "hard"},
    {"claim": "The legacy endpoint returns user passwords.", "context": "The v1/users endpoint returns user records containing id, email, and password_hash.", "truth": "reject", "difficulty": "hard"},
    {"claim": "The drug is safe for all patients.", "context": "Clinical trials showed efficacy in 94% of participants. Contraindicated for patients with renal impairment or known sulfa allergies.", "truth": "verify", "difficulty": "hard"},
    {"claim": "The defendant's liability is fully established by the evidence.", "context": "CCTV footage shows the defendant at the location. No forensic evidence links them directly to the crime.", "truth": "verify", "difficulty": "hard"},
    {"claim": "The investment fund outperformed the S&P 500 this year.", "context": "Fund returned 14.2% YTD. The S&P 500 returned 13.8% over the same period.", "truth": "accept", "difficulty": "hard"},
    {"claim": "The medical procedure is experimental.", "context": "The procedure has FDA approval for use in adults over 65, but is still in Phase 2 trials for patients under 40.", "truth": "verify", "difficulty": "hard"},
]


class TrustchainEnvironment(Environment):
    """
    TrustChain Environment for evaluating AI truth-checking agents.
    Agents observe claims with optional context and decide whether to accept, reject, or verify them.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the trustchain environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._seed = 42
        self._rng = random.Random(self._seed)
        self._tasks = self._build_tasks()
        self._current_task_idx = 0

    def _build_tasks(self) -> List[Dict[str, str]]:
        """Build a deterministic but non-fixed episode task order."""
        easy = [dict(t, difficulty="easy") for t in EASY_TASKS]
        medium = [dict(t, difficulty="medium") for t in MEDIUM_TASKS]
        hard = [dict(t, difficulty="hard") for t in HARD_TASKS]
        self._rng.shuffle(easy)
        self._rng.shuffle(medium)
        self._rng.shuffle(hard)
        return easy + medium + hard

    def reset(self) -> TrustchainObservation:
        """Reset the environment to start a new episode."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Keep episodes reproducible while preventing simple memorization.
        self._rng = random.Random(self._seed + self._reset_count)
        self._tasks = self._build_tasks()
        self._current_task_idx = 0

        task = self._tasks[self._current_task_idx]
        difficulty = task["difficulty"]

        return TrustchainObservation(
            claim=task["claim"],
            context=task["context"],
            difficulty=difficulty,
            feedback="Environment reset. Please evaluate the first claim.",
            done=False,
            reward=0.0,
        )

    def step(self, action: TrustchainAction) -> TrustchainObservation:  # type: ignore[override]
        """Execute a step in the environment by evaluating the agent's decision."""
        self._state.step_count += 1

        task = self._tasks[self._current_task_idx]
        expected_action = task["truth"]
        
        # Reward shaping: dense and bounded in [0.0, 1.0].
        if action.decision == expected_action:
            reward = 1.0
            feedback = f"Your decision '{action.decision}' was correct!"
        elif action.decision == "verify" and task["difficulty"] in {"easy", "medium"}:
            reward = 0.3
            feedback = f"Your decision '{action.decision}' shows appropriate caution, but the claim was actually clear enough to '{expected_action}' directly."
        elif task["difficulty"] == "hard" and expected_action == "verify":
            reward = 0.5
            feedback = f"Your decision '{action.decision}' was close — this claim is fundamentally ambiguous and required 'verify'."
        elif task["difficulty"] == "hard" and action.decision != "verify":
            reward = 0.2
            feedback = f"Hard claim judged as '{action.decision}'. Use caution for uncertain evidence; expected '{expected_action}'."
        else:
            reward = 0.0
            feedback = f"Your decision '{action.decision}' was incorrect. The correct answer was '{expected_action}'."

        self._current_task_idx += 1
        done = self._current_task_idx >= len(self._tasks)

        next_task = self._tasks[self._current_task_idx] if not done else None
        next_difficulty = next_task["difficulty"] if next_task else ""

        return TrustchainObservation(
            claim=next_task["claim"] if next_task else "",
            context=next_task["context"] if next_task else None,
            difficulty=next_difficulty,
            feedback=feedback,
            done=done,
            reward=reward,
            metadata={
                "expected": expected_action,
                "step": self._state.step_count
            }
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
