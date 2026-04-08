"""
CloudOps Incident Responder — Inference Script
==============================================
Hackathon-compliant baseline inference script for the CloudOps environment.
Iterates through easy, medium, and hard SRE incidents.
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from client import CloudOpsEnv
from models import CloudOpsAction

# ── Environment configuration (per hackathon spec) ────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # ❗ Mandatory: no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # ✅ Required name

BENCHMARK = "cloudops"
TASKS = ["cloudops_easy", "cloudops_medium", "cloudops_hard"]

# ── System prompt for SRE Agent ──────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Cloud Operations and SRE agent.
    You will receive an ALERT and a list of current RESOURCES.
    Your goal is to resolve the incident with minimal steps.

    AVAILABLE ACTIONS:
    - list_resources: View all resources.
    - read_logs(resource_id): Read recent logs for a specific VM or service.
    - restart_service(resource_id): Restart the service on a specific VM.
    - scale_up(resource_id): Increase CPU/RAM for a specific VM.
    - rollback(resource_id): Revert to the previous stable version.
    - delete_resource(resource_id): Delete a resource (USE WITH EXTREME CAUTION).

    OUTPUT FORMAT:
    Reply with a JSON object containing 'action_type' and 'resource_id'.
    Example: {"action_type": "read_logs", "resource_id": "vm-01"}
    
    DO NOT provide any explanation. ONLY the JSON object.
    """
).strip()


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


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(obs) -> str:
    resources_str = json.dumps(obs.resources, indent=2)
    logs_str = f"\nRecent Logs:\n{obs.logs}" if obs.logs else ""
    return textwrap.dedent(
        f"""
        Difficulty: {obs.difficulty}
        Alert: {obs.alert}
        Resources:
        {resources_str}
        {logs_str}
        
        Feedback: {obs.feedback}

        What is your next action? (Reply with JSON)
        """
    ).strip()


# ── Model call ───────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, obs) -> CloudOpsAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # JSON extraction
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
            
        data = json.loads(text)
        return CloudOpsAction(
            action_type=data.get("action_type"),
            resource_id=data.get("resource_id")
        )

    except Exception as exc:
        import sys
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return CloudOpsAction(action_type="list_resources")


# ── Task Execution Loop ───────────────────────────────────────────────────────
async def run_task(client: OpenAI, env: CloudOpsEnv, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False
    
    try:
        result = await env.reset(task=task_id)
        done = result.done

        for step in range(1, 15):
            if done:
                break

            obs = result.observation
            action_obj = get_model_action(client, obs)
            action_str = f"{action_obj.action_type}({action_obj.resource_id})"

            try:
                result = await env.step(action_obj)
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if error:
                break

        # Normalize score: sum(rewards) / ideal_steps (approx 3)
        score = sum(r for r in rewards if r > 0) / 2.0 
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
        env = await CloudOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = CloudOpsEnv(base_url="http://localhost:8000")
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
