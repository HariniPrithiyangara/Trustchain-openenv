---
title: CloudOps Incident Responder
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
app_port: 8000
tags:
  - openenv
  - sre
  - devops
  - reinforcement-learning
  - ai-agents
---

<div align="center">

# 🛠️ CloudOps Incident Responder

### A High-Fidelity OpenEnv Benchmark for SRE and DevOps AI Agents

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-6366f1?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![🤗 HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live-ff6b35?style=for-the-badge)](https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 Project Overview

**CloudOps Incident Responder** is a rigorous, real-world OpenEnv environment designed to evaluate AI agents on Site Reliability Engineering (SRE) tasks. Unlike synthetic benchmarks, this environment simulates a production cloud infrastructure where agents must diagnose and remediate critical incidents.

### Why This Environment Matters
Multi-agent systems and autonomous agents are increasingly tasked with managing infrastructure. CloudOps provides a safe, reproducible benchmark to measure an agent's ability to:
1. **Diagnose** issues via log analysis.
2. **Remediate** incidents through resource scaling, service restarts, and rollbacks.
3. **Reason** through cascading failures without causing further downtime.

---

## 🎮 Action & Observation Spaces

### Action Space (`CloudOpsAction`)
Agents can perform the following actions:
- `list_resources`: Enumerate all infrastructure components and their current state.
- `read_logs`: Retrieve recent stdout/stderr logs for a specific resource.
- `scale_up`: Increase CPU/RAM allocation for a VM to handle spikes.
- `restart_service`: Restart a failing application service.
- `rollback`: Revert a service to its previous stable deployment version.
- `delete_resource`: Permanently remove a resource (**High Risk**).

### Observation Space (`CloudOpsObservation`)
- `alert`: A human-readable system alert (e.g., "High CPU usage detected").
- `resources`: A JSON list of resources with metrics (CPU, Memory, Version).
- `logs`: Diagnostic information retrieved from the system.
- `feedback`: Immediate guidance on the effectiveness of the agent's actions.

---

## 📚 Task Curriculum

| Task ID | Difficulty | Scenario | Remediation Steps |
| :--- | :--- | :--- | :--- |
| `cloudops_easy` | Easy | High CPU spike on frontend. | `scale_up` |
| `cloudops_medium` | Medium | Memory leak in API service. | `read_logs` → `restart_service` |
| `cloudops_hard` | Hard | Cascading deployment failure. | `read_logs` → `rollback` → `scale_up` |

---

## 🏆 Reward Function

The environment uses a **shaped reward system** to provide continuous learning signals:

| Action Result | Reward | Description |
| :--- | :--- | :--- |
| **Success** | `+1.00` | Incident fully resolved. |
| **Progress** | `+0.30 to +0.50` | Correct diagnostic or partial remediation step. |
| **Neutral** | `0.00` | Action had no effect. |
| **Wrong Target** | `-0.20` | Action applied to the incorrect resource. |
| **Catastrophic** | `-1.00` | Production resource deleted (Terminates episode). |

---

## 🚀 Quick Start

### Prerequisites
- Docker
- Python 3.10+
- `uv` package manager

### Installation & Local Run
```bash
# Clone the repository
git clone https://github.com/HariniPrithiyangara/Trustchain-openenv.git
cd Trustchain-openenv

# Install dependencies
uv sync

# Start the environment server
uv run server
```

### Running Inference
```bash
export HF_TOKEN="your_huggingface_token"
python inference.py
```

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
