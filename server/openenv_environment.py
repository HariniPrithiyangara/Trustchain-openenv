"""CloudOps Incident Responder OpenEnv environment implementation."""

import random
from typing import Dict, List, Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from openenv.models import CloudOpsAction, CloudOpsObservation


INCIDENTS = {
    "cloudops_easy": {
        "alert": "CRITICAL: High CPU usage (98%) on frontend-vm-01. Response times exceeding 5s.",
        "resources": [
            {"id": "frontend-vm-01", "type": "vm", "status": "running", "cpu": 98.2, "memory": 45.0},
            {"id": "db-vm-01", "type": "vm", "status": "running", "cpu": 12.0, "memory": 30.0},
        ],
        "logs": {
            "frontend-vm-01": "2026-04-08 10:00:01 INFO: Incoming request spike detected.\n2026-04-08 10:00:05 WARN: CPU saturation in progress.",
        },
        "solution": ["scale_up"],
        "target_id": "frontend-vm-01"
    },
    "cloudops_medium": {
        "alert": "WARNING: Memory leak suspected in 'api-service' on api-vm-01. Memory usage at 92% and climbing.",
        "resources": [
            {"id": "api-vm-01", "type": "vm", "status": "running", "cpu": 25.0, "memory": 92.5},
            {"id": "db-vm-01", "type": "vm", "status": "running", "cpu": 15.0, "memory": 40.0},
        ],
        "logs": {
            "api-vm-01": "2026-04-08 10:05:00 ERROR: java.lang.OutOfMemoryError: Java heap space\n2026-04-08 10:05:10 INFO: Garbage collection failed to reclaim memory.",
        },
        "solution": ["read_logs", "restart_service"],
        "target_id": "api-vm-01"
    },
    "cloudops_hard": {
        "alert": "CRITICAL: Cascading failure. 'auth-service' returning 500s after deployment v2.1. Database connections maxed out.",
        "resources": [
            {"id": "auth-vm-01", "type": "vm", "status": "running", "cpu": 10.0, "memory": 15.0, "version": "v2.1"},
            {"id": "db-vm-01", "type": "vm", "status": "running", "cpu": 99.5, "memory": 95.0, "max_conns": 100},
        ],
        "logs": {
            "auth-vm-01": "2026-04-08 10:10:00 FATAL: Database connection pool exhausted.\n2026-04-08 10:10:05 INFO: Retrying connection in 1s...",
            "db-vm-01": "2026-04-08 10:10:00 WARN: Max connections reached (100/100).\n2026-04-08 10:10:10 INFO: Vacuuming tables...",
        },
        "solution": ["read_logs", "rollback", "scale_up"],
        "target_id": "auth-vm-01",
        "secondary_target": "db-vm-01"
    }
}


class CloudOpsEnvironment(Environment):
    """
    CloudOps Incident Responder Environment.
    Agents act as SREs to diagnose and resolve production incidents.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the CloudOps environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_incident = None
        self._incident_id = "cloudops_easy"
        self._history = []
        self._resolved = False

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> CloudOpsObservation:
        """Reset the environment for a specific incident."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._incident_id = task if task in INCIDENTS else "cloudops_easy"
        self._current_incident = INCIDENTS[self._incident_id].copy()
        self._history = []
        self._resolved = False
        
        return CloudOpsObservation(
            alert=self._current_incident["alert"],
            resources=self._current_incident["resources"],
            difficulty=self._incident_id.split("_")[-1],
            feedback=f"Incident assigned: {self._incident_id}. Diagnose and remediate.",
            done=False,
            reward=0.0
        )

    def step(self, action: CloudOpsAction) -> CloudOpsObservation:  # type: ignore[override]
        """Execute an SRE action and transition the environment state."""
        if self._resolved:
            return CloudOpsObservation(done=True, reward=0.0, feedback="Incident already resolved.")

        self._state.step_count += 1
        reward = 0.0
        feedback = ""
        logs = None
        
        target_sol = self._current_incident["solution"]
        target_id = self._current_incident["target_id"]
        
        # Step logic
        if action.action_type == "read_logs":
            logs = self._current_incident["logs"].get(action.resource_id, "No logs available.")
            if action.action_type == target_sol[len(self._history)]:
                reward = 0.3 # Informational reward
                feedback = "Logs reviewed. Diagnostic progress made."
                if action.action_type not in [a.action_type for a in self._history]:
                    self._history.append(action)
            else:
                reward = 0.1
                feedback = "Logs reviewed, but may not be the primary bottleneck."

        elif action.action_type in ["scale_up", "restart_service", "rollback"]:
            if len(self._history) < len(target_sol) and action.action_type == target_sol[len(self._history)]:
                # Check if acting on correct resource
                if action.resource_id == target_id or (self._incident_id == "cloudops_hard" and action.resource_id == self._current_incident.get("secondary_target")):
                    reward = 0.5 # Partial remediation
                    feedback = f"Action '{action.action_type}' applied successfully to {action.resource_id}."
                    self._history.append(action)
                else:
                    reward = -0.2
                    feedback = f"Action '{action.action_type}' applied to WRONG resource {action.resource_id}!"
            else:
                reward = 0.0
                feedback = f"Action '{action.action_type}' had no effect at this stage."

        # Resolution check
        if len(self._history) == len(target_sol):
            reward = 1.0
            feedback = "SUCCESS: Incident resolved. All systems operational."
            self._resolved = True

        # Penalize dangerous actions
        if action.action_type == "delete_resource":
            reward = -1.0
            feedback = "CRITICAL ERROR: Production resource deleted! Instant failure."
            self._resolved = True # Terminate episode on catastrophe

        return CloudOpsObservation(
            alert=self._current_incident["alert"],
            resources=self._current_incident["resources"],
            logs=logs,
            feedback=feedback,
            done=self._resolved or self._state.step_count >= 10,
            reward=max(min(reward, 1.0), -1.0),
            difficulty=self._incident_id.split("_")[-1]
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
