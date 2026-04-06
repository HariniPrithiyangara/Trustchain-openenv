import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import TrustchainEnv
from models import TrustchainAction

# ── Environment configuration (per hackathon spec) ───────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: for from_docker_image()

TASK_NAME = "trustchain_eval"
BENCHMARK = "trustchain"

# ── System prompt: chain-of-thought internally, single word output ────────────
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


# ── STDOUT logging helpers (exact hackathon format) ───────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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


# ── Model call with chain-of-thought ─────────────────────────────────────────
def get_model_decision(client: OpenAI, claim: str, context: Optional[str], difficulty: str) -> str:
    user_prompt = build_user_prompt(claim, context, difficulty)
    try:
        # First pass: ask model to reason step-by-step
        reasoning = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
                {"role": "assistant", "content": "Let me reason through this carefully:\n"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        thought = reasoning.choices[0].message.content or ""

        # Second pass: force extraction of final single-word decision
        decision_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",    "content": "You are a precise decision extractor. Given a reasoning block, output EXACTLY ONE word: accept, reject, or verify."},
                {"role": "user",      "content": f"Reasoning:\n{thought}\n\nFinal answer (one word only):"},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        text = (decision_resp.choices[0].message.content or "").strip().lower()

        if "accept" in text:   return "accept"
        if "reject" in text:   return "reject"
        if "verify" in text:   return "verify"
        return "verify"   # safe fallback

    except Exception as exc:
        import sys
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "verify"


# ── Main episode loop ─────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_key")

    if LOCAL_IMAGE_NAME:
        env = await TrustchainEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = TrustchainEnv(base_url="http://localhost:8000")
        await env.connect()

    rewards:      List[float] = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        done   = result.done

        for step in range(1, 25):   # generous upper bound for any task count
            if done:
                break

            obs      = result.observation
            decision = get_model_decision(client, obs.claim, obs.context, obs.difficulty)
            action   = TrustchainAction(decision=decision)  # type: ignore

            try:
                result = await env.step(action)
                reward = result.reward or 0.0
                done   = result.done
                error  = None
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=decision, reward=reward, done=done, error=error)

            if error:
                break

        total_possible = steps_taken * 1.0
        score   = sum(rewards) / total_possible if total_possible > 0 else 0.0
        success = score > 0.5

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
