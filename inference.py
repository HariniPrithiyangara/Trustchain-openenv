"""
inference.py — TrustChain OpenEnv Inference Script
====================================================
Runs all three tasks sequentially and emits exactly:
  [START] task=<task> env=trustchain model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL   — LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — model id      (default: meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       — API key       (no default; must be set)
  LOCAL_IMAGE_NAME — Docker image name (optional)
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — strictly following hackathon spec
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")  # MUST NOT HAVE DEFAULT
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL: str = os.getenv("TRUSTCHAIN_ENV_URL", "http://localhost:7860")
BENCHMARK: str = "trustchain"
MAX_STEPS: int = 12

TASKS = ["verify_credential", "trace_provenance", "audit_anomaly"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent auditing a blockchain-based supply-chain ledger called TrustChain.

Available action_types:
  lookup   — fetch a credential record by ID
  verify   — assert VALID or INVALID for a credential (target_id + verdict)
  trace    — retrieve the provenance chain for a product (target_id)
  report   — submit anomaly finding (target_id + reason + remediation)
  done     — signal end of episode

Reply with a JSON object only. No prose, no markdown fences.
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, task_id: str, obs: dict, step: int, history: List[str]) -> dict:
    hist_block = "\n".join(history[-4:]) if history else "None"
    user_msg = textwrap.dedent(f"""
        Task: {task_id}
        Step: {step}
        Last result: {obs.get('last_result', '')}
        Error: {obs.get('error') or 'none'}
        Recent history:
        {hist_block}

        Decide your next action and reply with a JSON object only.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {"action_type": "done"}

# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env_reset(task_id)
        obs = result.get("observation", {})
        history: List[str] = []

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            action = get_action(client, task_id, obs, step, history)
            action_str = str(action.get("action_type", "done"))

            try:
                result = env_step(action)
            except Exception as exc:
                log_step(step, action_str, 0.0, True, str(exc))
                break

            obs = result.get("observation", {})
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = obs.get("error")

            rewards.append(reward)
            steps_taken = step
            history.append(f"step={step} action={action_str} reward={reward:.2f}")
            log_step(step, action_str, reward, done, error)

            if done:
                score = float(result.get("score", sum(rewards) / max(len(rewards), 1)))
                score = max(0.0, min(1.0, score))
                break

        if not score and rewards:
            score = max(0.0, min(1.0, sum(rewards) / len(rewards)))
        success = score >= 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    single_task = os.getenv("TRUSTCHAIN_TASK", "")
    tasks_to_run = [single_task] if single_task and single_task in TASKS else TASKS

    for task_id in tasks_to_run:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
