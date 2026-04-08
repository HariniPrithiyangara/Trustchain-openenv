
"""
TrustChain OpenEnv — Inference Script
======================================
Hackathon-compliant baseline inference script for the TrustChain environment.

Environment Variables (required):
    API_BASE_URL      The API endpoint for the LLM (default: HF router)
    MODEL_NAME        The model identifier (default: Llama-3.3-70B-Instruct)
    HF_TOKEN          Your Hugging Face / API key (NO default — must be set)
    LOCAL_IMAGE_NAME  Optional: local Docker image name for from_docker_image()

Usage:
    export HF_TOKEN="hf_..."
    uv run python inference.py
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import TrustchainEnv
from models import TrustchainAction

# ── Environment configuration (per hackathon spec) ────────────────────────────
# Defaults set ONLY for API_BASE_URL and MODEL_NAME — never for HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be provided
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: for from_docker_image()

TASK_NAME = "trustchain_eval"
BENCHMARK = "trustchain"

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

    REASONING CHECKLIST (do this mentally before answering):
      1. NUMERICAL: If the claim involves numbers, compute the exact math.
         e.g., 6000/hour ÷ 60 = 100/minute → equivalent → accept.
         e.g., ($12M - $10M) / $10M = 20% → matches claim → accept.
      2. CONTEXTUAL: Does the context directly confirm or contradict the claim?
         If context says "partially" or "pending" or "looking into", the claim is NOT fully confirmed.
      3. ABSOLUTE LANGUAGE: Words like "fully", "always", "unbreakable", "explicitly", "all patients"
         almost always mean → verify (unless context is 100% definitive).
      4. CONTRADICTION: If context directly disproves the claim → reject.
      5. GENERAL KNOWLEDGE: For no-context tasks, use world knowledge to accept or reject.

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
    # score uses 2 decimal places to match spec exactly (e.g. 1.00)
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
    """
    Two-pass CoT strategy:
      Pass 1 — Ask model to reason step-by-step through the claim.
      Pass 2 — Extract the final single-word decision from the reasoning.

    Falls back to "verify" (safest option) on any API error.
    """
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
            max_tokens=200,
        )
        thought = reasoning_resp.choices[0].message.content or ""

        # Pass 2: extract final single-word decision
        decision_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise decision extractor. "
                        "Given a reasoning block, output EXACTLY ONE word: accept, reject, or verify."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Reasoning:\n{thought}\n\nFinal answer (one word only):",
                },
            ],
            temperature=0.0,
            max_tokens=5,
        )
        text = (decision_resp.choices[0].message.content or "").strip().lower()

        if "accept" in text:
            return "accept"
        if "reject" in text:
            return "reject"
        if "verify" in text:
            return "verify"
        return "verify"  # safe fallback

    except Exception as exc:
        import sys
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "verify"


# ── Main episode loop ─────────────────────────────────────────────────────────
async def main() -> None:
    # Check for HF_TOKEN as it is mandatory
    if not HF_TOKEN:
        import sys
        print("[ERROR] HF_TOKEN environment variable is not set. Disqualified.", file=sys.stderr)
        return

    # All LLM calls use the OpenAI client with env-var credentials
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Connect to environment (Docker image or localhost server)
    if LOCAL_IMAGE_NAME:
        env = await TrustchainEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = TrustchainEnv(base_url="http://localhost:8000")
        await env.connect()

    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        done = result.done

        # Episode ends when done=True (after 20 tasks)
        for step in range(1, 25):
            if done:
                break

            obs = result.observation
            decision = get_model_decision(
                client,
                obs.claim,
                obs.context,
                obs.difficulty,
            )
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

        # Normalise score to [0, 1]
        total_possible = float(steps_taken) if steps_taken > 0 else 1.0
        score = sum(rewards) / total_possible
        score = max(0.0, min(1.0, score))
        success = score >= 0.7  # Higher success threshold for 100/100

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
