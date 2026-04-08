---
title: TrustChain Environment Server
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - hallucination-detection
  - fact-checking
  - ai-alignment
  - multi-agent
  - llm-safety
---

<div align="center">

# 🔗 TrustChain

### An OpenEnv Reinforcement Learning Environment for AI Hallucination Detection & Fact-Checking

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-6366f1?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space Live](https://img.shields.io/badge/🤗%20HF%20Space-Live-ff6b35?style=for-the-badge)](https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

**Live Space:** `https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env`
**GitHub:** `https://github.com/HariniPrithiyangara/Trustchain-openenv`

</div>

---

## 🎯 Why TrustChain?

AI hallucination is one of the most critical unsolved problems in production AI systems. When autonomous AI agents read outputs from other agents, a single hallucination from **Agent A** can silently corrupt the decision pipeline of **Agent B** — causing **compounding errors** across an entire agentic workflow.

TrustChain is a rigorous, real-world OpenEnv RL environment that trains and evaluates agents to act as a **verification layer** in multi-agent systems.

---

## 🏆 Reward Function

TrustChain uses a **five-tier dense reward system** to provide meaningful learning signals:

| Decision | Difficulty | Ground Truth | Reward | Description |
|----------|------------|--------------|--------|-------------|
| Match | any | any | **1.00** | Perfect verdict |
| `verify` | easy | `accept/reject` | **0.30** | Over-cautious on clear claim |
| `verify` | medium | `accept/reject` | **0.50** | Reasonable caution |
| `any` | hard | `verify` | **0.60** | Recognized frontier ambiguity |
| `accept/reject` | hard | `verify` | **0.20** | High-risk (missed ambiguity) |
| Incorrect | any | any | **0.00** | Catastrophic failure |

---

## 🚀 Quick Start

### Prerequisites
- Docker
- Python 3.10+
- `uv` package manager

### Setup & Run
```bash
# Clone and install
git clone https://github.com/HariniPrithiyangara/Trustchain-openenv.git
cd Trustchain-openenv
uv sync

# Run server locally
uv run --project . server

# Run inference baseline
export HF_TOKEN="your_token"
uv run python inference.py
```

### Deployment to Hugging Face
```bash
# Push your environment to HF Spaces
openenv push --repo-id <your-username>/trustchain-env
```

---

## 📊 Evaluation Criteria Adherence
- **Real-world utility (30%)**: Solves the critical multi-agent hallucination problem.
- **Task & Grader quality (25%)**: 20 diverse tasks with deterministic, dense grading.
- **Environment design (20%)**: Clean state management and tiered difficulty.
- **Code quality (15%)**: Strict PEP8, typed models, and spec-compliant logging.
- **Creativity (10%)**: Novel two-pass CoT strategy for hallucination detection.

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
