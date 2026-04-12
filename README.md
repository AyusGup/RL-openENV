# RL-OpenEnv: SRE Incident Response Benchmark

A production-grade evaluation environment for AI agents specialized in **Site Reliability Engineering (SRE)**. This repository provides a suite of complex, multi-service incident scenarios designed to test an agent's ability to triage, debug, and remediate production issues.

## 🌟 Project Overview
This project leverages [OpenEnv](https://github.com/openenv) to create a high-fidelity sandbox for SRE tasks. It goes beyond simple code-fixing by introducing realistic operational constraints:
- **Observability-First**: Alerts and logs are the primary drivers for investigation.
- **Microservice Architecture**: Scenarios involve inter-service communication and cascading failures.
- **RCA Documentation**: Success depends on both fixing the system and accurately documenting the root cause.

## 📁 Repository Structure
- `rl_env/`: Core OpenEnv implementation, Docker configuration, and task fixtures.
- `rl_env/inference.py`: The reference inference runner for agent evaluation.
- `tests/`: End-to-end and unit tests for environment mechanics, including reward shaping and change tracking.
- `validate-submission.sh`: Script to verify package integrity before submission.

## 🚦 Getting Started
For detailed setup, deployment, and task descriptions, please refer to the package documentation:
👉 **[rl_env/README.md](./rl_env/README.md)**

### Running Inference Locally
1. **Initialize the Environment**:
   ```bash
   source .venv/bin/activate
   # Start the server (see rl_env/README for details)
   ```
2. **Execute Agent Policy**:
   ```bash
   export API_BASE_URL="https://router.huggingface.co/v1"
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   python rl_env/inference.py --task 2
   ```

## 📊 Deployment
This environment is designed to be hosted on **Hugging Face Spaces**. You can deploy it using the `openenv push` command or through the automated GitHub Actions workflow.

---
**License**: MIT
