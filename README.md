# rl-openenv

Root workspace for the SRE OpenEnv project.

## Structure
- `sre_env/`: main OpenEnv package, Docker setup, fixtures, and environment documentation.
- `sre_env/inference.py`: baseline runner client.
- `validate-submission.sh`: pre-submission validator script.
- `tests/`: local test suite.

## Start Here
For package/environment setup and deployment details, see:
- [`sre_env/README.md`](./sre_env/README.md)

## Run Inference
Start the environment server first, then from this repository root run:
```bash
source .venv/bin/activate
# Optional override. If omitted, the client reads .openenv_port then falls back to 7860.
# export OPENENV_BASE_URL="http://127.0.0.1:7860"
export SRE_TASK_NAME="task1_wrong_status"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
python sre_env/inference.py
```

`sre_env/inference.py` uses the typed async SDK wrapper (`SREEnv`/`SREAction`) for `reset`, `step`, and `state`.
By default it runs `task1_wrong_status`; set `SRE_TASK_NAME` to target a different task.

You can load values from `.env.example` using your preferred dotenv workflow. Do not commit real secrets.

## Validate Submission
From this repository root:
```bash
./validate-submission.sh https://jha-ayush-rl-openenv.hf.space ./sre_env
```
