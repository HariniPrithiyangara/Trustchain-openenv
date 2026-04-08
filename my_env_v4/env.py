
"""Backend API Automation Environment Implementation."""

import random
from typing import Dict, List, Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from my_env_v4.models import MyEnvV4Action, MyEnvV4Observation
from my_env_v4.tasks import TASKS


class MyEnvV4Env(Environment):
    """
    Backend API Automation Environment.
    Agents act as backend logic processors to fetch, validate, and handle errors.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = "easy"
        self._history = []
        self._resolved = False
        self._current_task = None

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> MyEnvV4Observation:
        """Reset the environment for a specific backend task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Standardize task naming (handle 'trustchain_easy' or just 'easy')
        if task:
            clean_task = task.split("_")[-1] if "_" in task else task
            self._task_id = clean_task if clean_task in TASKS else "easy"
        else:
            self._task_id = "easy"

        self._current_task = TASKS[self._task_id].copy()
        self._history = []
        self._resolved = False
        
        return MyEnvV4Observation(
            status=self._current_task["initial_status"],
            feedback=f"Task assigned: {self._task_id}. {self._current_task['description']}",
            difficulty=self._task_id,
            done=False,
            reward=0.0
        )

    def step(self, action: MyEnvV4Action) -> MyEnvV4Observation:  # type: ignore[override]
        """Execute a backend operation and transition the workflow state."""
        if self._resolved:
            return MyEnvV4Observation(done=True, reward=0.0, feedback="Workflow already completed.")

        self._state.step_count += 1
        reward = 0.0
        feedback = ""
        error_message = None
        
        target_seq = self._current_task["expected_sequence"]
        
        # 🏆 Shaped Reward Logic
        step_idx = len(self._history)
        if step_idx < len(target_seq):
            expected_action = target_seq[step_idx]
            
            if action.action_type == expected_action:
                # Correct action in sequence
                reward = 0.3 if step_idx < len(target_seq) - 1 else 1.0
                feedback = f"Action '{action.action_type}' was successful. Progressing workflow."
                self._history.append(action.action_type)
                
                if len(self._history) == len(target_seq):
                    self._resolved = True
                    feedback = "SUCCESS: Workflow completed accurately."
            else:
                # Incorrect action or out of sequence
                reward = -0.1
                feedback = f"Action '{action.action_type}' was not expected at this stage."
                error_message = "Unexpected workflow state transition."
        else:
            reward = 0.0
            feedback = "No further actions required for this workflow."

        # Handle hard-task specific logic (error detection)
        if self._task_id == "hard" and action.action_type == "validate_input" and "fetch_data" in self._history:
            feedback += " [ALERT] Malformed data detected. Error recovery required."
            error_message = "Payload corruption: missing 'auth_sig'."

        return MyEnvV4Observation(
            status=f"step_{self._state.step_count}",
            feedback=feedback,
            error_message=error_message,
            done=self._resolved or self._state.step_count >= 6,
            reward=max(min(reward, 1.0), -1.0),
            difficulty=self._task_id
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
