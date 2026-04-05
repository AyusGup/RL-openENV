"""LLM-driven inference script for the SRE OpenEnv environment."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL") or "http://127.0.0.1:7860"
TASK_NAME = os.getenv("SRE_TASK_NAME") or "task1_wrong_status"
BENCHMARK = os.getenv("SRE_BENCHMARK") or "sre_env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1

ACTION_SYSTEM_PROMPT = """
You are controlling an SRE incident-response environment.

At every turn, choose exactly one next action to execute. Available tools:
- terminal: run one shell command in the task workspace
- editor: write the full replacement contents of one file
- submit: finish and trigger grading

Return exactly one JSON object and nothing else with these keys:
{"tool":"terminal|editor|submit","command":"","file_path":"","file_content":""}

Rules:
- Output valid JSON only. No markdown fences.
- Use terminal commands to inspect logs, inspect source, and run tests.
- Use editor only after you have inspected the target file.
- For editor actions, set file_path and leave file_content empty. The client will request the full file contents separately.
- For terminal actions, set command and leave file_path/file_content empty.
- For submit, leave command/file_path/file_content empty.
- Prefer short deterministic commands like: cat logs/error.log, cat app/main.py, python -m pytest -q.
- Submit once the fix is verified or when more probing is unlikely to help.
""".strip()

EDITOR_SYSTEM_PROMPT = """
You are preparing a full replacement file for an SRE environment edit action.
Use the incident alert, recent logs, prior action history, and the current file contents to infer the fix.
Return only the complete corrected file contents. Do not add explanations or markdown fences.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = sanitize_log_value(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={sanitize_log_value(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def sanitize_log_value(value: str) -> str:
    """Keep log fields single-line and compact."""
    compact = value.replace("\r", "\\r").replace("\n", "\\n")
    return compact.strip() or "null"


def sanitize_action_for_log(action: Dict[str, Any]) -> str:
    """Collapse an action dict into a single log-friendly string."""
    if action["tool"] == "terminal":
        return str(action.get("command") or "")
    if action["tool"] == "editor":
        return f"write({action.get('file_path') or ''})"
    return "submit"


def extract_model_output(content: str) -> str:
    """Strip common markdown fences from model output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def truncate_text(value: str, limit: int = 3000) -> str:
    """Truncate long observation text to keep prompts bounded."""
    if len(value) <= limit:
        return value
    return value[: limit - 15] + "\n...[truncated]"


def extract_json_object(content: str) -> Dict[str, Any]:
    """Parse a JSON object from the model response."""
    cleaned = extract_model_output(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Model did not return a JSON action.")
        return json.loads(match.group(0))


def normalize_action(raw_action: Dict[str, Any]) -> Dict[str, str]:
    """Validate and normalize the model action payload."""
    tool = str(raw_action.get("tool") or "").strip().lower()
    if tool not in {"terminal", "editor", "submit"}:
        raise RuntimeError(f"Unsupported tool from model: {tool or 'empty'}")

    action = {
        "tool": tool,
        "command": str(raw_action.get("command") or "").strip(),
        "file_path": str(raw_action.get("file_path") or "").strip(),
        "file_content": "",
    }

    if tool == "terminal" and not action["command"]:
        raise RuntimeError("Model returned terminal action without command.")
    if tool == "editor" and not action["file_path"]:
        raise RuntimeError("Model returned editor action without file_path.")
    return action


def build_action_prompt(
    alert_message: str,
    file_tree: List[str],
    history: List[Dict[str, str]],
    latest_observation: Dict[str, Any],
) -> str:
    """Build the observation summary for the next-action model call."""
    history_lines = []
    for item in history[-6:]:
        history_lines.append(
            f"- action={item['action']} reward={item['reward']} done={item['done']} "
            f"error={item['error']} stdout={item['stdout']} stderr={item['stderr']}"
        )

    history_block = "\n".join(history_lines) if history_lines else "None"
    stdout = truncate_text(str(latest_observation.get("stdout") or ""))
    stderr = truncate_text(str(latest_observation.get("stderr") or ""))

    return (
        f"Task: {TASK_NAME}\n"
        f"Alert:\n{alert_message or 'None'}\n\n"
        f"Workspace files:\n" + "\n".join(file_tree) + "\n\n"
        f"Latest observation stdout:\n{stdout or 'None'}\n\n"
        f"Latest observation stderr:\n{stderr or 'None'}\n\n"
        f"Recent history:\n{history_block}\n"
    )


def choose_next_action(
    client: OpenAI,
    alert_message: str,
    file_tree: List[str],
    history: List[Dict[str, str]],
    latest_step_result: Dict[str, Any],
) -> Dict[str, str]:
    """Ask the model to choose the next environment action."""
    step_score = latest_step_result["info"].get("score")
    step_score_display = "null" if step_score is None else f"{float(step_score):.3f}"
    prompt = build_action_prompt(
        alert_message=alert_message,
        file_tree=file_tree,
        history=history,
        latest_observation=latest_step_result["observation"],
    )
    prompt += (
        f"\nStep reward: {float(latest_step_result['reward']['value']):.2f}\n"
        f"Step done: {str(bool(latest_step_result['done'])).lower()}\n"
        f"Step info score: {step_score_display}\n"
        f"Step last_action_error: {latest_step_result['info'].get('last_action_error') or 'null'}\n"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    return normalize_action(extract_json_object(content))


def build_editor_content(
    client: OpenAI,
    file_path: str,
    current_source: str,
    alert_message: str,
    known_logs: Dict[str, str],
    history: List[Dict[str, str]],
) -> str:
    """Ask the model for the full replacement contents of a file."""
    history_lines = []
    for item in history[-6:]:
        history_lines.append(
            f"- action={item['action']} reward={item['reward']} error={item['error']}"
        )
    history_block = "\n".join(history_lines) if history_lines else "None"

    log_sections = []
    for log_name, log_content in known_logs.items():
        log_sections.append(f"{log_name}:\n{truncate_text(log_content, limit=2000)}")
    logs_block = "\n\n".join(log_sections) if log_sections else "None"

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {TASK_NAME}\n"
                    f"Target file: {file_path}\n\n"
                    f"Alert:\n{alert_message or 'None'}\n\n"
                    f"Known logs:\n{logs_block}\n\n"
                    f"Recent history:\n{history_block}\n\n"
                    f"Current file contents:\n{current_source}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=1200,
        stream=False,
    )
    candidate = extract_model_output(completion.choices[0].message.content or "")
    if not candidate:
        raise RuntimeError("Model returned empty editor content.")
    return candidate


def post_json(http_client: httpx.Client, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON and return the parsed response payload."""
    response = http_client.post(path, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def maybe_capture_file_contents(
    action: Dict[str, str],
    observation: Dict[str, Any],
    known_files: Dict[str, str],
    known_logs: Dict[str, str],
) -> None:
    """Record file/log contents after successful cat commands."""
    if action["tool"] != "terminal":
        return
    command = action["command"].strip()
    if not command.startswith("cat "):
        return
    target = command[4:].strip()
    stdout = str(observation.get("stdout") or "")
    if not stdout:
        return
    if target.endswith(".log"):
        known_logs[target] = stdout
    else:
        known_files[target] = stdout


def main() -> None:
    rewards: List[float] = []
    score = 0.0
    success = False
    steps_taken = 0
    latest_step_result: Dict[str, Any] = {}
    history: List[Dict[str, str]] = []
    known_files: Dict[str, str] = {}
    known_logs: Dict[str, str] = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for submission inference runs.")
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        with httpx.Client(base_url=ENV_BASE_URL, follow_redirects=True) as env_client:
            initial_observation = post_json(env_client, "/reset", {"task_id": TASK_NAME})
            alert_message = str(initial_observation.get("alert_message") or "")
            file_tree = list(initial_observation.get("file_tree") or [])
            latest_step_result = {
                "observation": initial_observation,
                "reward": {"value": 0.0},
                "done": False,
                "info": {"score": None, "message": "Episode reset", "last_action_error": None},
            }

            for step in range(1, MAX_STEPS + 1):
                if latest_step_result.get("done"):
                    break

                action = choose_next_action(
                    client=llm_client,
                    alert_message=alert_message,
                    file_tree=file_tree,
                    history=history,
                    latest_step_result=latest_step_result,
                )

                if action["tool"] == "editor":
                    current_source = known_files.get(action["file_path"])
                    if current_source is None:
                        action = {
                            "tool": "terminal",
                            "command": f"cat {action['file_path']}",
                            "file_path": "",
                            "file_content": "",
                        }
                    else:
                        action["file_content"] = build_editor_content(
                            client=llm_client,
                            file_path=action["file_path"],
                            current_source=current_source,
                            alert_message=alert_message,
                            known_logs=known_logs,
                            history=history,
                        )

                latest_step_result = post_json(env_client, "/step", action)
                reward = float(latest_step_result["reward"]["value"] or 0.0)
                rewards.append(reward)
                steps_taken = step
                log_step(
                    step=step,
                    action=sanitize_action_for_log(action),
                    reward=reward,
                    done=bool(latest_step_result.get("done")),
                    error=latest_step_result["info"].get("last_action_error") or None,
                )

                maybe_capture_file_contents(
                    action=action,
                    observation=latest_step_result["observation"],
                    known_files=known_files,
                    known_logs=known_logs,
                )
                file_tree = list(latest_step_result["observation"].get("file_tree") or file_tree)
                history.append(
                    {
                        "action": sanitize_log_value(sanitize_action_for_log(action)),
                        "reward": f"{reward:.2f}",
                        "done": str(bool(latest_step_result.get('done'))).lower(),
                        "error": sanitize_log_value(str(latest_step_result["info"].get("last_action_error") or "null")),
                        "stdout": sanitize_log_value(truncate_text(str(latest_step_result["observation"].get("stdout") or ""), 600)),
                        "stderr": sanitize_log_value(truncate_text(str(latest_step_result["observation"].get("stderr") or ""), 600)),
                    }
                )

            if not latest_step_result.get("done"):
                submit_action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}
                latest_step_result = post_json(env_client, "/step", submit_action)
                reward = float(latest_step_result["reward"]["value"] or 0.0)
                rewards.append(reward)
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=sanitize_action_for_log(submit_action),
                    reward=reward,
                    done=bool(latest_step_result.get("done")),
                    error=latest_step_result["info"].get("last_action_error") or None,
                )

            score = float(latest_step_result["info"].get("score") or 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD and not latest_step_result["info"].get("last_action_error")
    except Exception as exc:
        success = False
        score = 0.0
        if steps_taken == 0:
            history.append(
                {
                    "action": "startup",
                    "reward": "0.00",
                    "done": "false",
                    "error": sanitize_log_value(str(exc)),
                    "stdout": "null",
                    "stderr": sanitize_log_value(str(exc)),
                }
            )
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
