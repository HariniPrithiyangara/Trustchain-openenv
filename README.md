---
title: AI Backend Workflow Automation
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: true
app_port: 8000
tags:
  - openenv
  - backend-automation
  - ai-agents
  - workflow-optimization
---

<div align="center">

# ⚙️ AI Backend Workflow Automation

### A Real-World OpenEnv Benchmark for Backend API Automation Agents

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-6366f1?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![🤗 HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live-ff6b35?style=for-the-badge)](https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 Overview

The **AI Backend Workflow Automation** environment is a high-utility benchmark designed to evaluate AI agents on complex, multi-step backend operations. Unlike synthetic tasks, this environment simulates real-world API workflows including data fetching, input validation, and error recovery.

### Key Capabilities Evaluated
- **Data Retrieval**: Fetching structured data from diverse database resources.
- **Validation**: Enforcing schema compliance and detecting payload corruption.
- **Error Remediation**: Applying logic to fix malformed data and ensure workflow continuity.
- **Response Management**: Returning finalized, validated responses to upstream services.

---

## 🎮 Action & Observation Spaces

### Action Space (`MyEnvV4Action`)
- `fetch_data`: Retrieve raw data from a specified resource.
- `validate_input`: Run validation checks against the current data payload.
- `fix_error`: Apply remediation logic to malformed or corrupted data.
- `return_response`: Finalize the workflow and return the processed output.
- `query_db`: Execute complex queries against internal data stores.

### Observation Space (`MyEnvV4Observation`)
- `status`: Current stage of the backend workflow.
- `data`: The payload returned by the previous successful action.
- `error_message`: Detailed diagnostic info when an action fails or detects issues.
- `feedback`: A continuous learning signal providing progress guidance.

---

## 📚 Task Curriculum

| Task ID | Level | Scenario | Ideal Action Sequence |
| :--- | :--- | :--- | :--- |
| `easy` | 🟢 | Basic Query Handling | `fetch_data` |
| `medium` | 🟡 | Multi-step Workflow | `fetch_data` → `validate_input` → `return_response` |
| `hard` | 🔴 | Error Recovery | `fetch_data` → `validate_input` (fails) → `fix_error` → `return_response` |

---

## 🏆 Reward System

The environment implements a **highly shaped reward function** to facilitate agent learning:

| Result | Reward | Description |
| :--- | :--- | :--- |
| **Full Success** | `+1.00` | Entire workflow sequence completed accurately. |
| **Progress** | `+0.30` | Correct action performed in the expected sequence. |
| **Out of Order** | `-0.10` | Action performed at the wrong stage of the workflow. |
| **Trivial Action** | `0.00` | Redundant or no-op action. |

---

## 🚀 Quick Start

```bash
# Clone and Sync
git clone https://github.com/HariniPrithiyangara/Trustchain-openenv.git
cd Trustchain-openenv
uv sync

# Run Environment Server
uv run server

# Run Inference
export HF_TOKEN="your_token"
python inference.py
```

---

## 📄 License
MIT License.
