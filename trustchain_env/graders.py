"""
trustchain_env/graders.py
=========================
Deterministic graders for the three TrustChain tasks.
Each grader receives the final environment state and returns a float in [0.0, 1.0].

Grader contract (OpenEnv spec):
  - __call__(self, state: dict) -> float
  - Deterministic — same state always returns same score
  - Score is clamped to [0.0, 1.0]
"""

from __future__ import annotations

from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Task 1 — Easy: Verify a Single Credential
# ---------------------------------------------------------------------------

class VerifyCredentialGrader:
    """
    Score based on whether the agent correctly classified a single credential
    as VALID or INVALID.

    State keys consumed:
      state["expected_verdict"]  : "VALID" | "INVALID"
      state["agent_verdict"]     : the agent's final answer string
      state["steps_taken"]       : int, used to give partial credit for speed
      state["max_steps"]         : int
    """

    def __call__(self, state: dict) -> float:
        expected: str = str(state.get("expected_verdict", "")).upper().strip()
        agent: str = str(state.get("agent_verdict", "")).upper().strip()

        if not expected or not agent:
            return 0.0

        # Full credit for correct verdict
        if expected in agent:
            steps_taken = int(state.get("steps_taken", state.get("max_steps", 5)))
            max_steps = int(state.get("max_steps", 5))
            efficiency_bonus = 0.2 * max(0, (max_steps - steps_taken) / max_steps)
            return _clamp(0.8 + efficiency_bonus)

        # Partial credit: agent expressed uncertainty rather than wrong answer
        uncertain_keywords = {"unknown", "unsure", "unclear", "cannot", "ambiguous"}
        if any(kw in agent.lower() for kw in uncertain_keywords):
            return 0.3

        return 0.0


# ---------------------------------------------------------------------------
# Task 2 — Medium: Trace Full Provenance Chain
# ---------------------------------------------------------------------------

class TraceProvenanceGrader:
    """
    Score based on how many hops of the provenance chain the agent correctly
    identified, in the right order.

    State keys consumed:
      state["expected_chain"]  : list[str]  e.g. ["MFG-001", "DIST-007", "RET-042"]
      state["agent_chain"]     : list[str]  the agent's output chain
    """

    def __call__(self, state: dict) -> float:
        expected = [s.strip() for s in state.get("expected_chain", [])]
        agent = [s.strip() for s in state.get("agent_chain", [])]

        if not expected:
            return 0.0
        if not agent:
            return 0.0

        lcs_len = _lcs_length(expected, agent)
        order_score = lcs_len / len(expected)

        extra = max(0, len(agent) - len(expected))
        hallucination_penalty = min(0.3, extra * 0.1)

        return _clamp(order_score - hallucination_penalty)


def _lcs_length(a: list, b: list) -> int:
    """Standard dynamic-programming LCS length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ---------------------------------------------------------------------------
# Task 3 — Hard: Detect and Explain Supply-Chain Anomaly
# ---------------------------------------------------------------------------

class AuditAnomalyGrader:
    """
    Score across three sub-criteria:
      1. Correct identification of the tampered credential ID  (0.4 weight)
      2. Correct anomaly reason                               (0.4 weight)
      3. Plausible remediation step proposed                  (0.2 weight)

    State keys consumed:
      state["tampered_credential_id"]  : str
      state["anomaly_reason"]          : str
      state["agent_credential_id"]     : str
      state["agent_reason"]            : str
      state["agent_remediation"]       : str
    """

    _REASON_ALIASES: dict = {
        "timestamp_out_of_order": ["timestamp", "order", "sequence", "chronolog"],
        "signature_mismatch":     ["signature", "sig", "mismatch", "invalid sig"],
        "unknown_issuer":         ["issuer", "unknown", "unrecognised", "unrecognized", "untrusted"],
    }

    def __call__(self, state: dict) -> float:
        expected_cred: str = str(state.get("tampered_credential_id", "")).strip()
        expected_reason: str = str(state.get("anomaly_reason", "")).strip().lower()
        agent_cred: str = str(state.get("agent_credential_id", "")).strip()
        agent_reason: str = str(state.get("agent_reason", "")).strip().lower()
        agent_remediation: str = str(state.get("agent_remediation", "")).strip()

        score = 0.0

        if expected_cred and expected_cred.lower() in agent_cred.lower():
            score += 0.4

        if expected_reason:
            aliases = self._REASON_ALIASES.get(expected_reason, [expected_reason])
            if any(alias in agent_reason for alias in aliases):
                score += 0.4

        if len(agent_remediation) >= 10:
            score += 0.2

        return _clamp(score)
