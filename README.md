---
title: Trustchain Env
emoji: 🔗
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🔗 TrustChain: Blockchain Supply-Chain Auditing Environment

**GitHub Repository:** [https://github.com/HariniPrithiyangara/Trustchain-openenv](https://github.com/HariniPrithiyangara/Trustchain-openenv)  
**Hugging Face Space:** [https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env](https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env)

## 🌟 Overview
TrustChain is a rigorous, real-world OpenEnv reinforcement learning environment that trains and evaluates agents to act as **Compliance Auditors** for complex supply chains. The environment simulates a cryptographically secured ledger of product credentials, where agents must verify authenticity, trace provenance chains, and detect sophisticated anomalies.

This environment addresses the real-world need for automated auditing in global trade, where tracking a product from manufacturer to consumer involves verifying multiple digital signatures and timestamps to prevent counterfeiting.

---

## 🚀 Environment Features
- **Deterministic Ledger:** Powered by a seeded random engine (seed 42) ensuring reproducible states for agent training.
- **Deniable Credentials:** Simulates "valid" vs "invalid" signatures and "unknown" issuers.
- **Multi-Step Provenance:** Products move through Manufacturer -> Distributor -> Retailer, requiring multi-step state lookup.
- **Structured Explanations:** Far beyond binary rewards, the environment requires agents to explain *why* a record is an anomaly and propose remediation.

## ♟️ Tasks & Difficulty Curriculum

1. **Easy: Single-Credential Verification (`verify_credential`)**
   - **Goal:** Determine if a specific record has a valid signature and authorized issuer.
   - **Difficulty:** Low — single action lookup.

2. **Medium: Multi-Hop Provenance Trace (`trace_provenance`)**
   - **Goal:** Retrieve the complete custody chain for a product ID in chronological order.
   - **Difficulty:** Medium — requires sequential `trace` and `lookup` actions across the ledger state.

3. **Hard: Supply-Chain Anomaly Audit (`audit_anomaly`)**
   - **Goal:** Identify the one tampered record in a complex chain, explain the failure (e.g., "Timestamp out of order"), and provide a remediation plan.
   - **Difficulty:** Hard — requires comprehensive analysis and causal reasoning.

## 🛠️ State and Actions

**Observations (`TrustChainObservation`)**
- `ledger_snapshot (list)`: Known records retrieved from the blockchain.
- `last_result (str)`: Descriptive feedback from the previous action.
- `error (Optional[str])`: Detailed error message if an action fails.
- `done (bool)`: Episode terminal flag.
- `task_id (str)`: Current active task.
- `step (int)`: Cumulative step count.

**Actions (`TrustChainAction`)**
- `action_type`: `lookup`, `verify`, `trace`, `report`, or `done`.
- `target_id`: The product or credential ID to act upon.
- `verdict`: `VALID` or `INVALID` for verification tasks.
- `reason / remediation`: Textual findings for audit reports.

## 📈 Reward Shaping
- **Accuracy (0.0 - 1.0):** Deterministic graders score the precision of the agent's findings.
- **Efficiency Bonus:** Rewards for completing tasks in fewer steps (0.1 boost for early completion).
- **Hard Task weighting:** The Audit task uses weighted scoring for record selection (0.4), reasoning (0.3), and remediation (0.3).

---

## ⚙️ Quickstart

### Prerequisites
- Python 3.10+
- `openenv-core`
- `docker`

### Initializing Locally
```bash
# Provide inference variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-hf-token"

# Install package
pip install -e .

# Run inference script conforming to Meta Hackathon standard
python inference.py
```

### Docker Verification
```bash
docker build -t trustchain-openenv .
docker run -p 7860:7860 trustchain-openenv
```

### OpenEnv Validation
```bash
openenv validate
```
