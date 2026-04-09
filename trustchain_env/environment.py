"""
trustchain_env/environment.py
==============================
Full TrustChain environment implementing all 3 tasks with deterministic ledger state.
"""

from __future__ import annotations

import random
import hashlib
from typing import Any, Optional

from .models import TrustChainAction, TrustChainObservation


# ---------------------------------------------------------------------------
# Deterministic ledger generator (seed=42 → reproducible)
# ---------------------------------------------------------------------------

def _make_ledger(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    issuers = ["did:issuer:MFG-001", "did:issuer:DIST-007", "did:issuer:RET-042", "did:issuer:UNKNOWN-999"]
    records = []
    base_ts = 1_700_000_000
    for i in range(6):
        cred_id = f"CRED-{i:04d}"
        ts = base_ts + i * 3600
        valid = i != 3  # CRED-0003 is the tampered record
        sig = hashlib.sha256(f"{cred_id}{ts}".encode()).hexdigest()[:16]
        if not valid:
            sig = "BADSIG0000000000"  # tampered signature
        records.append({
            "credential_id": cred_id,
            "product_id": f"PROD-{(i % 2) + 1:04d}",
            "issuer_did": issuers[i % len(issuers)],
            "holder_did": issuers[(i + 1) % len(issuers)],
            "timestamp": ts if i != 3 else base_ts - 100,  # timestamp out of order
            "signature": sig,
            # valid flag hidden from agent; only grader sees it
        })
    return records


LEDGER = _make_ledger(seed=42)
CHAIN_OF_CUSTODY = ["MFG-001", "DIST-007", "RET-042"]  # expected provenance chain
TAMPERED_CREDENTIAL_ID = "CRED-0003"
ANOMALY_REASON = "timestamp_out_of_order"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrustChainEnvironment:
    """Full TrustChain verification environment with 3 tasks."""

    TASK_IDS = ["verify_credential", "trace_provenance", "audit_anomaly"]
    MAX_STEPS = {"verify_credential": 5, "trace_provenance": 10, "audit_anomaly": 15}

    def __init__(self):
        self._task_id: str = "verify_credential"
        self._step: int = 0
        self._done: bool = False
        # Grader-compatible state (populated during episode)
        self._grader_state: dict[str, Any] = {}

    # ------------------------------------------------------------------ reset
    def reset(self, task_id: Optional[str] = None) -> dict:
        self._task_id = task_id if task_id in self.TASK_IDS else "verify_credential"
        self._step = 0
        self._done = False
        self._grader_state = {
            "task_id": self._task_id,
            "max_steps": self.MAX_STEPS[self._task_id],
            # verify_credential
            "expected_verdict": "INVALID",  # CRED-0000 scenario shown to agent
            "agent_verdict": "",
            # trace_provenance
            "expected_chain": CHAIN_OF_CUSTODY,
            "agent_chain": [],
            # audit_anomaly
            "tampered_credential_id": TAMPERED_CREDENTIAL_ID,
            "anomaly_reason": ANOMALY_REASON,
            "agent_credential_id": "",
            "agent_reason": "",
            "agent_remediation": "",
        }
        # Expose ledger (hide validity flag)
        visible_ledger = [{k: v for k, v in r.items()} for r in LEDGER]
        obs = TrustChainObservation(
            ledger_snapshot=visible_ledger,
            last_result=f"Episode started. Task: {self._task_id}",
            done=False,
            task_id=self._task_id,
            step=0,
        )
        return {"observation": obs.model_dump(), "done": False, "reward": 0.01}

    # ------------------------------------------------------------------ step
    def step(self, action: dict) -> dict:
        if self._done:
            obs = TrustChainObservation(done=True, task_id=self._task_id,
                                        step=self._step, last_result="Episode already complete.")
            return {"observation": obs.model_dump(), "done": True, "reward": 0.01,
                    "state": self._grader_state}

        self._step += 1
        self._grader_state["steps_taken"] = self._step
        action_type = str(action.get("action_type", "done"))
        target_id = str(action.get("target_id", ""))
        reward = 0.01
        last_result = ""
        error = None
        done = False

        max_steps = self.MAX_STEPS.get(self._task_id, 10)
        if self._step >= max_steps:
            done = True

        if action_type == "lookup":
            match = next((r for r in LEDGER if r["credential_id"] == target_id), None)
            if match:
                last_result = f"Credential {target_id}: issuer={match['issuer_did']}, ts={match['timestamp']}, sig={match['signature']}"
            else:
                error = f"Credential '{target_id}' not found."

        elif action_type == "verify":
            verdict = str(action.get("verdict", "")).upper()
            self._grader_state["agent_verdict"] = verdict
            match = next((r for r in LEDGER if r["credential_id"] == target_id), None)
            if match:
                # Simple reward signal: +0.5 if verdict matches ground truth
                ground_truth = "INVALID" if match["signature"] == "BADSIG0000000000" or match["timestamp"] < 1_700_000_000 else "VALID"
                reward = 0.5 if verdict == ground_truth else -0.3
                last_result = f"Verdict logged: {verdict}"
                done = True
            else:
                error = f"Credential '{target_id}' not found."

        elif action_type == "trace":
            # Return chain of custody for product
            chain = [r["issuer_did"].split(":")[-1] for r in LEDGER if r["product_id"] == target_id]
            last_result = " → ".join(chain) if chain else "No chain found."
            self._grader_state["agent_chain"] = chain
            reward = 0.31

        elif action_type == "report":
            self._grader_state["agent_credential_id"] = target_id
            self._grader_state["agent_reason"] = str(action.get("reason", ""))
            self._grader_state["agent_remediation"] = str(action.get("remediation", ""))
            reward = 0.51
            last_result = "Report submitted."
            done = True

        elif action_type == "done":
            done = True
            last_result = "Agent signalled done."
            reward = 0.01

        else:
            error = f"Unknown action_type: '{action_type}'"

        if done:
            self._done = True

        obs = TrustChainObservation(
            ledger_snapshot=[{k: v for k, v in r.items()} for r in LEDGER],
            last_result=last_result,
            error=error,
            done=done,
            task_id=self._task_id,
            step=self._step,
        )
        return {
            "observation": obs.model_dump(),
            "done": done,
            "reward": reward,
            "state": self._grader_state,
        }

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "step": self._step,
            "done": self._done,
            **self._grader_state,
        }
