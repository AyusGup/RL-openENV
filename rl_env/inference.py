"""
Inference Script (SRE Tasks)
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import httpx
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from .models import SREAction
except ImportError:
    from models import SREAction  # type: ignore


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("BENCHMARK", "rl_env")
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")

MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))

TASK_MAP: Dict[int, str] = {
    1: "task1_wrong_status",
    2: "task2_retry_logic",
    3: "task3_cascading_failure",
}

REPLAY_MAP: Dict[str, str] = {
    "task1_wrong_status": "create_item_contract",
    "task2_retry_logic": "retry_health_contract",
    "task3_cascading_failure": "cascading_timeout_budget",
}

ACTION_SYSTEM_PROMPT = """
You are controlling an SRE incident-response environment.
Choose exactly one next action.

Return ONLY JSON with keys:
{"tool":"terminal|editor|replay|submit","command":"","file_path":"","file_content":""}

Rules:
- Do not repeat the same `cat <file>` on unchanged files.
- Prefer progress: inspect -> edit -> replay -> submit.
- Replay command must match the task replay name.
- Submit once replay shows success.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


def _extract_model_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            raw = candidate

    # Python-literal fallback (single quotes/triple quotes from some models)
    pythonish = re.sub(r"\btrue\b", "True", raw, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
    parsed = ast.literal_eval(pythonish)
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output is not an object.")
    return parsed


def _normalize_action(raw: Dict[str, Any]) -> Dict[str, str]:
    tool = str(raw.get("tool") or "").strip().lower()
    if tool not in {"terminal", "editor", "replay", "submit"}:
        raise RuntimeError(f"Unsupported tool: {tool or 'empty'}")

    action = {
        "tool": tool,
        "command": str(raw.get("command") or "").strip(),
        "file_path": str(raw.get("file_path") or "").strip(),
        "file_content": str(raw.get("file_content") or ""),
    }
    if action["tool"] == "terminal" and not action["command"]:
        raise RuntimeError("terminal action missing command")
    if action["tool"] == "replay" and not action["command"]:
        raise RuntimeError("replay action missing command")
    if action["tool"] == "editor" and not action["file_path"]:
        raise RuntimeError("editor action missing file_path")
    return action


def _build_prompt(
    task_id: str,
    observation: Dict[str, Any],
    history: List[str],
    seen_cats: Set[str],
    replay_name: str,
    step: int,
    max_steps: int,
) -> str:
    stdout = str(observation.get("stdout") or "")
    stderr = str(observation.get("stderr") or "")
    file_tree = observation.get("file_tree") or []
    hist_block = "\n".join(history[-8:]) if history else "None"
    seen_cats_block = ", ".join(sorted(seen_cats)) if seen_cats else "None"
    return (
        f"Task: {task_id}\n"
        f"Step: {step}/{max_steps}\n"
        f"Replay command for this task: {replay_name}\n\n"
        f"Latest stdout:\n{stdout[:2000] or 'None'}\n\n"
        f"Latest stderr:\n{stderr[:1200] or 'None'}\n\n"
        f"Workspace files:\n{_safe_json(file_tree)}\n\n"
        f"Recent actions:\n{hist_block}\n\n"
        f"Already-cat files (avoid repeating unless edited): {seen_cats_block}\n"
    )


def _choose_action_from_llm(
    client: OpenAI,
    task_id: str,
    obs: Dict[str, Any],
    history: List[str],
    seen_cats: Set[str],
    step: int,
    max_steps: int,
) -> Dict[str, str]:
    replay_name = REPLAY_MAP.get(task_id, "create_item_contract")
    prompt = _build_prompt(
        task_id=task_id,
        observation=obs,
        history=history,
        seen_cats=seen_cats,
        replay_name=replay_name,
        step=step,
        max_steps=max_steps,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=350,
    )
    content = response.choices[0].message.content or ""
    return _normalize_action(_extract_model_json(content))


FALLBACK_TERMINAL_PLAN: Dict[str, List[str]] = {
    "task1_wrong_status": ["cat logs/error.log", "cat app/main.py"],
    "task2_retry_logic": ["cat logs/app.log", "cat app/retry_handler.py", "cat app/main.py"],
    "task3_cascading_failure": [
        "cat logs/service_a.log",
        "cat logs/service_b.log",
        "cat service_a/config.py",
        "cat service_a/main.py",
        "cat service_b/main.py",
    ],
}


def _fallback_action(task_id: str, replay_done: bool, seen_cats: Set[str]) -> Dict[str, str]:
    replay_name = REPLAY_MAP.get(task_id, "create_item_contract")
    if replay_done:
        return {"tool": "submit", "command": "", "file_path": "", "file_content": ""}
    for cmd in FALLBACK_TERMINAL_PLAN.get(task_id, []):
        if cmd.startswith("cat "):
            target = cmd[4:].strip()
            if target in seen_cats:
                continue
        return {"tool": "terminal", "command": cmd, "file_path": "", "file_content": ""}
    return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}


def _apply_policy(
    action: Dict[str, str],
    task_id: str,
    seen_cats: Set[str],
    edited_files: Set[str],
    replay_done: bool,
    step: int,
    max_steps: int,
) -> Dict[str, str]:
    replay_name = REPLAY_MAP.get(task_id, "create_item_contract")

    if replay_done:
        return {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

    if step >= max_steps - 1:
        return {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

    if action["tool"] == "replay":
        action["command"] = replay_name
        return action

    if action["tool"] == "terminal":
        cmd = action["command"].strip()
        if cmd.startswith("cat "):
            target = cmd[4:].strip()
            if target in seen_cats and target not in edited_files:
                return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}
        return action

    if action["tool"] == "editor":
        # If model forgot content, force a replay instead of empty write.
        if not action["file_content"].strip():
            return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}
        return action

    return action


async def _get_score(base_url: str, task_id: str, fallback_rewards: List[float]) -> float:
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.get(f"{base_url}/grader")
            if resp.status_code == 200:
                payload = resp.json()
                scores = payload.get("task_scores", {})
                if task_id in scores:
                    return float(scores[task_id] or 0.0)
                mapped = task_id.replace("task", "task_")
                if mapped in scores:
                    return float(scores[mapped] or 0.0)
    except Exception:
        pass
    return float(fallback_rewards[-1]) if fallback_rewards else 0.0


def _parse_reward(result_payload: Dict[str, Any]) -> float:
    reward_payload = result_payload.get("reward", 0.0)
    if isinstance(reward_payload, dict):
        return float(reward_payload.get("value", 0.0) or 0.0)
    return float(reward_payload or 0.0)


async def run_inference(task: int) -> Dict[str, Any]:
    task_id = TASK_MAP.get(task, f"task{task}")
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    seen_cats: Set[str] = set()
    edited_files: Set[str] = set()
    history: List[str] = []
    replay_done = False
    llm_disabled = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if IMAGE_NAME:
            print("[WARN] IMAGE_NAME mode is not used in HTTP inference flow; using EVAL_BASE_URL/local server.", flush=True)
        eval_base_url = LOCAL_BASE_URL

        async with httpx.AsyncClient(base_url=eval_base_url, timeout=60.0) as http:
            reset_resp = await http.post("/reset", json={"task_id": task_id})
            reset_resp.raise_for_status()
            result_payload = reset_resp.json()
            obs_dict = result_payload.get("observation", {})

            done = bool(result_payload.get("done", False))

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                if not llm_disabled:
                    try:
                        raw_action = _choose_action_from_llm(
                            client=llm_client,
                            task_id=task_id,
                            obs=obs_dict,
                            history=history,
                            seen_cats=seen_cats,
                            step=step,
                            max_steps=MAX_STEPS,
                        )
                    except Exception as exc:
                        err = str(exc).lower()
                        if "402" in err or "depleted" in err or "credits" in err:
                            llm_disabled = True
                        raw_action = _fallback_action(task_id, replay_done, seen_cats)
                else:
                    raw_action = _fallback_action(task_id, replay_done, seen_cats)
                action_dict = _apply_policy(
                    action=raw_action,
                    task_id=task_id,
                    seen_cats=seen_cats,
                    edited_files=edited_files,
                    replay_done=replay_done,
                    step=step,
                    max_steps=MAX_STEPS,
                )
                # Validate action payload against schema before sending.
                _ = SREAction(**action_dict)

                step_resp = await http.post("/step", json=action_dict)
                step_resp.raise_for_status()
                result_payload = step_resp.json()

                obs_dict = result_payload.get("observation", {})
                reward = _parse_reward(result_payload)
                done = bool(result_payload.get("done", False))
                info = result_payload.get("info") or {}
                error = info.get("last_action_error") or obs_dict.get("stderr") or None

                if action_dict["tool"] == "terminal" and action_dict["command"].startswith("cat "):
                    target = action_dict["command"][4:].strip()
                    seen_cats.add(target)
                if action_dict["tool"] == "editor" and action_dict["file_path"]:
                    edited_files.add(action_dict["file_path"])
                if action_dict["tool"] == "replay":
                    if "contract_ok=true" in str(obs_dict.get("stdout", "")).lower():
                        replay_done = True

                rewards.append(reward)
                steps_taken = step
                action_log = _safe_json(action_dict).replace('"', "'")
                history.append(f"{step}. {action_log} -> reward={reward:.2f} done={str(done).lower()}")
                log_step(step=step, action=action_log, reward=reward, done=done, error=error)

            if not done:
                submit_action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}
                submit_resp = await http.post("/step", json=submit_action)
                submit_resp.raise_for_status()
                submit_payload = submit_resp.json()
                reward = _parse_reward(submit_payload)
                rewards.append(reward)
                steps_taken += 1
                submit_info = submit_payload.get("info") or {}
                submit_obs = submit_payload.get("observation") or {}
                submit_done = bool(submit_payload.get("done", False))
                submit_error = submit_info.get("last_action_error") or submit_obs.get("stderr") or None
                log_step(
                    step=steps_taken,
                    action="{'tool':'submit'}",
                    reward=reward,
                    done=submit_done,
                    error=submit_error,
                )

            score = await _get_score(eval_base_url, task_id, rewards)
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
        "task": task_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference on rl_env tasks")
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Task number (1=easy, 2=medium, 3=hard). Omit to run all 3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks_to_run = [args.task] if args.task is not None else [1, 2, 3]
    all_scores: Dict[str, float] = {}

    for task_num in tasks_to_run:
        outcome = asyncio.run(run_inference(task=task_num))
        all_scores[f"task_{task_num}"] = float(outcome.get("score", 0.0))

    print(f"[SCORES] {json.dumps(all_scores)}", flush=True)
