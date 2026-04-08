"""
TrustChain OpenEnv — Inference Script
======================================
Hackathon-compliant baseline inference script for the TrustChain environment.
Iterates through easy, medium, and hard tasks.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import TrustchainEnv
from models import TrustchainAction

# ── Environment configuration (per hackathon spec) ────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be provided
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "trustchain"
TASKS = ["trustchain_easy", "trustchain_medium", "trustchain_hard"]

# ── System prompt: chain-of-thought reasoning, single-word output ─────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert fact-checking and reasoning agent.

    You will receive a CLAIM, optional CONTEXT, and a DIFFICULTY level.
    Think through the problem step by step INTERNALLY, then output EXACTLY ONE word.

    OUTPUT RULES (strictly enforced):
      accept  → The claim is factually correct. The context (or general knowledge) clearly confirms it.
      reject  → The claim is factually wrong, or directly contradicts the context.
      verify  → The claim is ambiguous, uncertain, uses absolute language where context is partial,
                or requires external information not available here.

    REASONING CHECKLIST:
      1. NUMERICAL: Compute exact math. (e.g. 6000/hour = 100/minute).
      2. CONTEXTUAL: Does context confirm or contradict?
      3. ABSOLUTE LANGUAGE: "fully", "always", "unbreakable" often mean → verify.
      4. GENERAL KNOWLEDGE: For no-context tasks, use world knowledge.

    Reply with EXACTLY ONE word: accept, reject, or verify.
    No explanation. No punctuation. No other text.
    """
).strip()


# ── STDOUT logging helpers — exact hackathon format ───────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(claim: str, context: Optional[str], difficulty: str) -> str:
    ctx_line = f"Context: {context}" if context else "Context: (none — use general knowledge)"
    return textwrap.dedent(
        f"""
        Difficulty: {difficulty}
        {ctx_line}
        Claim: {claim}

        Think step by step, then reply with exactly one word: accept, reject, or verify.
        """
    ).strip()


# ── Two-pass Chain-of-Thought model call ──────────────────────────────────────
def get_model_decision(
    client: OpenAI,
    claim: str,
    context: Optional[str],
    difficulty: str,
) -> str:
    user_prompt = build_user_prompt(claim, context, difficulty)
    try:
        # Pass 1: generate step-by-step reasoning
        reasoning_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Let me reason through this carefully:\n"},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        thought = reasoning_resp.choices[0].message.content or ""

        # Pass 2: extract final single-word decision
        decision_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise decision extractor. Given reasoning, output EXACTLY ONE word: accept, reject, or verify.",
                },
                {
                    "role": "user",
                    "content": f"Reasoning:\n{thought}\n\nFinal answer (one word only):",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = (decision_resp.choices[0].message.content or "").strip().lower()

        for word in ["accept", "reject", "verify"]:
            if word in text:
                return word
        return "verify"  # safe fallback

    except Exception as exc:
        import sys
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "verify"


# ── Task Execution Loop ───────────────────────────────────────────────────────
# Total possible steps per task bucket for normalization
TASK_LENGTHS = {
    "trustchain_easy": 6,
    "trustchain_medium": 6,
    "trustchain_hard": 8
}

async def run_task(client: OpenAI, env: TrustchainEnv, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False
    max_steps = TASK_LENGTHS.get(task_id, 10)
    
    try:
        result = await env.reset(task_id=task_id)
        done = result.done

        # OpenEnv episodes for TrustChain are finite (6-8 steps per task)
        for step in range(1, 25):
            if done:
                break

            obs = result.observation
            decision = get_model_decision(client, obs.claim, obs.context, obs.difficulty)
            action = TrustchainAction(decision=decision)  # type: ignore[call-arg]

            try:
                result = await env.step(action)
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=decision, reward=reward, done=done, error=error)

            if error:
                break

        # Normalize score: sum(rewards) / max_steps_possible
        score = sum(rewards) / float(max_steps)
        score = max(0.0, min(1.0, score))
        success = score >= 0.7

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main entry point ──────────────────────────────────────────────────────────
async def main() -> None:
    if not HF_TOKEN:
        import sys
        print("[ERROR] HF_TOKEN not set. Mandatory for evaluation.", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await TrustchainEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = TrustchainEnv(base_url="http://localhost:8000")
        await env.connect()

    try:
        for task_id in TASKS:
            await run_task(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
