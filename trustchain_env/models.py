"""
trustchain_env/models.py
========================
Typed Pydantic models for TrustChain OpenEnv actions and observations.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class CredentialRecord(BaseModel):
    credential_id: str
    product_id: str
    issuer_did: str
    holder_did: str
    timestamp: int
    signature: str
    valid: bool


class TrustChainAction(BaseModel):
    action_type: Literal["lookup", "verify", "trace", "report", "done"] = Field(...)
    target_id: Optional[str] = None
    verdict: Optional[Literal["VALID", "INVALID"]] = None
    reason: Optional[str] = None
    remediation: Optional[str] = None


class TrustChainObservation(BaseModel):
    ledger_snapshot: list = Field(default_factory=list)
    last_result: str = ""
    error: Optional[str] = None
    done: bool = False
    task_id: str = ""
    step: int = 0
