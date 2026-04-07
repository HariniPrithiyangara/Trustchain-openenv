---
title: TrustChain Environment Server
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TrustChain: Multi-Agent Verification Environment

TrustChain is a real-world OpenEnv benchmark where an agent acts as a newsroom verification layer for upstream AI-generated claims. The agent must decide whether to `accept`, `reject`, or `verify` each claim before it propagates through a multi-agent pipeline.

## Why This Environment Matters

Multi-agent systems fail when one hallucinated claim is trusted and passed downstream. TrustChain directly targets that failure mode and evaluates skeptical reasoning under uncertainty, ambiguity, and partial context.

Real-world mappings:
- content moderation pipelines
- automated fact-checking assistants
- AI safety verification layers
- LLM output auditing in enterprise workflows

## OpenEnv Spec Compliance

This environment implements the full OpenEnv API contract:
- typed models: `TrustchainAction`, `TrustchainObservation`
- `reset()` returns initial observation
- `step(action)` returns observation + reward + done
- `state` returns current `State` object
- `openenv.yaml` included at repo root

## Action and Observation Spaces

Action (`TrustchainAction`):
- `decision`: one of `accept`, `reject`, `verify`

Observation (`TrustchainObservation`):
- `claim`: claim from upstream reporter agent
- `context`: optional supporting context
- `difficulty`: `easy` / `medium` / `hard`
- `feedback`: grader feedback from previous action
- `done`, `reward`, `metadata`: standard OpenEnv observation fields

## Task Set and Difficulty Progression

TrustChain contains **20 tasks** total:
- 6 easy (grounded factual checks)
- 6 medium (context interpretation and contradiction detection)
- 8 hard (ambiguous/partial evidence, calibration under uncertainty)

Each reset keeps the same curriculum order (easy -> medium -> hard) but shuffles tasks deterministically within each bucket to reduce pure memorization while preserving reproducibility.

## Grader and Reward Design

Deterministic grader emits rewards in `[0.0, 1.0]`:
- `1.0`: exact correct decision
- `0.5`: near miss on hard ambiguous task when expected `verify`
- `0.3`: cautious `verify` on clear easy/medium task
- `0.2`: non-verify decision on hard uncertain task
- `0.0`: incorrect with no useful partial signal

This creates dense learning signals rather than only sparse binary reward.

## Setup

### 1) Install dependencies

```bash
uv sync
```

### 2) Run server locally

```bash
uv run --project . server
```

### 3) Run in Docker

```bash
docker build -t trustchain-env:latest .
docker run -p 8000:8000 trustchain-env:latest
```

## Baseline Inference (`inference.py`)

The root `inference.py` follows the required evaluator format and uses the OpenAI client for all LLM calls.

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optional `LOCAL_IMAGE_NAME` (only when using `from_docker_image()`)

Run:

```bash
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" \
HF_TOKEN="<your-token>" \
uv run python inference.py
```

Expected logs:
- `[START] ...`
- `[STEP] ...` for each step
- `[END] ...` with score and rewards

Sample logs are in `outputs/sample_output.txt`.

## Hugging Face Space Deployment

Deploy:

```bash
openenv push --repo-id <your-org>/trustchain-env
```

Set your live URL here before submission:
- Space URL: `https://<your-org>-trustchain-env.hf.space`

Validator requires:
- `POST <space_url>/reset` returns HTTP 200
- Docker build succeeds
- `openenv validate` passes

## Pre-Submission Checklist

- [ ] `openenv validate` passes
- [ ] `docker build .` passes on clean machine
- [ ] HF Space deployed and `/reset` reachable
- [ ] `inference.py` runs with required env vars
- [ ] logs match exact `[START]/[STEP]/[END]` format
- [ ] rewards/scores stay within `[0.0, 1.0]`
