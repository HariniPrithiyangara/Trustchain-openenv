"""
TrustChain: Multi-Agent Verification Environment — Inference Script
===================================================================
Baseline inference script for the TrustChain framework.
Evaluates claims over easy, medium, and hard validation tasks.
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from server.environment import TrustChainEnvironment
from server.models import TrustChainAction, TrustChainObservation

# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # MUST NOT HAVE DEFAULT
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "trustchain"
MAX_STEPS_PER_EPISODE = 20

# ── STDOUT formatting helpers ───────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ") if error else "null"
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

# ── Model logic ─────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, obs: TrustChainObservation) -> TrustChainAction:
    sys_prompt = textwrap.dedent(
        """
        You are a Verification Agent in a newsroom.
        Your job is to read claims sent by other agents, cross reference or request sources if needed, and finally pass or flag.
        Actions available: "verify_claim", "cross_reference", "flag_hallucination", "pass_forward", "request_source".
        You MUST reply with ONLY a JSON block like:
        {"action_type": "<action>", "claim_id": "<id>"}
        """
    ).strip()

    user_prompt = f"Claim ID: {obs.claim_id}\nClaim: {obs.claim}\nContext: {obs.context}\nFeedback: {obs.feedback}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Extract json safely
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        
        data = json.loads(text)
        return TrustChainAction(
            action_type=data.get("action_type", "verify_claim"),
            claim_id=data.get("claim_id", obs.claim_id)
        )
    except Exception:
        # Fallback action
        return TrustChainAction(action_type="cross_reference", claim_id=obs.claim_id)

async def run_task(task_id: str, client: OpenAI) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    # Initialize the Env logic directly for inference baseline
    env = TrustChainEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    
    try:
        obs = env.reset(task=task_id)

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            if obs.done:
                break
                
            action_obj = get_model_action(client, obs)
            obs = env.step(action_obj)
            
            reward = float(obs.reward or 0.0)
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            
            # Simple conversion of errors
            error_val = None
            if reward < 0:
                error_val = obs.feedback
                
            log_step(step=step, action=action_obj.action_type, reward=reward, done=done, error=None)
            
            if done:
                break
                
        # In TrustChain, max normalized score calculates positive validations
        # 3 claims = max 3.0 points from direct right decisions
        total_reward = sum(rewards)
        # We clamp score to [0, 1] mapped roughly to our max points
        max_possible = 3.0
        score = max(0.0, min(1.0, total_reward / max_possible))
        success = score >= 0.6  # Threshold to be successful

    except Exception as e:
        print(f"[DEBUG] Error running task {task_id}: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not HF_TOKEN:
        import sys
        print("[ERROR] HF_TOKEN missing.", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    for task_id in TASKS:
        asyncio.run(run_task(task_id, client))

if __name__ == "__main__":
    main()
