
"""
AI Backend Workflow Environment — Inference Script
=================================================
Hackathon-compliant baseline inference script for the Backend Automation environment.
Iterates through easy, medium, and hard workflow tasks.
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from my_env_v4.env import MyEnvV4Env
from my_env_v4.models import MyEnvV4Action

# ✅ REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # ❗ MUST NOT HAVE DEFAULT
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ❗ STRICT: no fallback allowed for client creation in eval mode
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_key")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 6
BENCHMARK = "backend_workflow"

# ── STDOUT logging helpers — exact hackathon format ───────────────────────────
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


# ── Model call ───────────────────────────────────────────────────────────────
def get_model_action(obs) -> MyEnvV4Action:
    prompt = textwrap.dedent(
        f"""
        You are an AI Backend Automation Agent.
        Current Status: {obs.status}
        Feedback: {obs.feedback}
        Error: {obs.error_message or 'None'}
        Difficulty: {obs.difficulty}

        Choose the next optimal action from: fetch_data, validate_input, fix_error, return_response.
        Reply with a JSON object: {{"action_type": "...", "resource": "..."}}
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
            
        data = json.loads(text)
        return MyEnvV4Action(
            action_type=data.get("action_type", "fetch_data"),
            resource=data.get("resource")
        )

    except Exception:
        return MyEnvV4Action(action_type="fetch_data")


# ── Task Execution Loop ───────────────────────────────────────────────────────
async def run_task(task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    # In OpenEnv, we either use local class or from_docker_image
    if LOCAL_IMAGE_NAME:
        # Note: In real OpenEnv, you'd use EnvClient to connect to Docker
        # For baseline, we use the local class if not testing container connectivity
        env = MyEnvV4Env() 
    else:
        env = MyEnvV4Env()

    rewards: List[float] = []
    steps_taken: int = 0
    success = False
    
    try:
        obs = env.reset(task=task_id)

        for step in range(1, MAX_STEPS + 1):
            action_obj = get_model_action(obs)
            
            obs = env.step(action_obj)
            
            reward = float(obs.reward or 0.0)
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_obj.action_type, reward=reward, done=done, error=obs.error_message)

            if done:
                break

        score = sum(r for r in rewards if r > 0) / 1.0 # Simple normalized score
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

    for task_id in TASKS:
        await run_task(task_id)


if __name__ == "__main__":
    asyncio.run(main())
