# rl-openenv

Root workspace for the SRE OpenEnv project.

## Structure
- `rl_env/`: main OpenEnv package, Docker setup, fixtures, and environment documentation.
- `rl_env/inference.py`: inference runner client.
- `validate-submission.sh`: pre-submission validator script.
- `tests/`: local test suite.

## Start Here
For package/environment setup and deployment details, see:
- [`rl_env/README.md`](./rl_env/README.md)

## Run Inference
Start the environment server first, then from this repository root run:
```bash
source .venv/bin/activate
export EVAL_BASE_URL="http://127.0.0.1:8000"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
python rl_env/inference.py --task 1
```

Task selector:
- `--task 1`
- `--task 2`
- `--task 3`
- omit `--task` to run all three tasks.

Optional grading breakdown logs:
```bash
export ENABLE_GRADE_BREAKDOWN_LOGS=1
python rl_env/inference.py --task 2
```

You can load values from `.env.example` using your preferred dotenv workflow. Do not commit real secrets.

## Validate Submission
From this repository root:
```bash
./validate-submission.sh https://jha-ayush-rl-openenv.hf.space ./rl_env
```
