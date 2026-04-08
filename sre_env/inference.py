"""LLM-driven inference script for the SRE OpenEnv environment."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from sre_env import SREAction, SREEnv

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL") or "http://127.0.0.1:7861"
TASK_NAME = os.getenv("SRE_TASK_NAME") or "task1_wrong_status"
BENCHMARK = os.getenv("SRE_BENCHMARK") or "sre_env"
DEFAULT_MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1


@lru_cache(maxsize=16)
def task_requires_rca(task_name: str) -> bool:
    """Return whether the task requires an RCA.md before submission."""
    candidate_paths = [
        Path(__file__).resolve().parent / "sre_env" / "fixtures" / task_name / "task_config.json",
        Path(__file__).resolve().parent / "fixtures" / task_name / "task_config.json",
        Path("fixtures") / task_name / "task_config.json",
        Path("sre_env") / "fixtures" / task_name / "task_config.json",
    ]
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        expected_fix_files = config.get("expected_fix_files") or []
        return "RCA.md" in expected_fix_files

    # Conservative fallback for current benchmark tasks when config is unavailable.
    return task_name in {
        "task1_wrong_status",
        "task2_retry_logic",
        "task3_cascading_failure",
    }


ACTION_SYSTEM_PROMPT = """
You are controlling an SRE incident-response environment.

At every turn, choose exactly one next action to execute. Available tools:
- terminal: run one shell command in the task workspace
- editor: write the full replacement contents of one file
- replay: run a deterministic replay probe against the current workspace code
- submit: finish and trigger grading

Return exactly one JSON object and nothing else with these keys:
{"tool":"terminal|editor|replay|submit","command":"","file_path":"","file_content":""}

Rules:
- Output valid JSON only. No markdown fences.
- Use terminal commands to inspect logs and inspect source.
- Use editor only after you have inspected the target file.
- For editor actions, set file_path and leave file_content empty. The client will request the full file contents separately.
- For terminal actions, set command and leave file_path/file_content empty.
- For replay, set command to the replay name:
  - task1_wrong_status: create_item_contract
  - task2_retry_logic: retry_health_contract
  - task3_cascading_failure: cascading_timeout_budget
- For submit, leave command/file_path/file_content empty.
- Prefer short deterministic commands like: cat logs/error.log, cat app/main.py, ls .
- Do not repeat `cat` on the same file unless that file was edited since your last read.
- Treat RCA.md as a final incident document: investigate first, fix and verify the code, then complete the RCA near the end.
- Create RCA.md before submit with headings: Root Cause, Affected Services, Fix Applied, Prevention.
- After replay confirms success and RCA.md exists, submit immediately.
- Keep RCA language concise, factual, and non-speculative.
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
    error_val = error if error else "null"
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
    if action["tool"] == "replay":
        return f"replay({action.get('command') or ''})"
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
    if tool not in {"terminal", "editor", "replay", "submit"}:
        raise RuntimeError(f"Unsupported tool from model: {tool or 'empty'}")

    action = {
        "tool": tool,
        "command": str(raw_action.get("command") or "").strip(),
        "file_path": str(raw_action.get("file_path") or "").strip(),
        "file_content": "",
    }

    if tool == "terminal" and not action["command"]:
        raise RuntimeError("Model returned terminal action without command.")
    if tool == "replay" and not action["command"]:
        raise RuntimeError("Model returned replay action without command.")
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


async def choose_next_action(
    client: OpenAI,
    alert_message: str,
    file_tree: List[str],
    history: List[Dict[str, str]],
    latest_step_result: Dict[str, Any],
) -> Dict[str, str]:
    """Ask the model to choose the next environment action."""
    step_score = latest_step_result["info"].get("score")
    step_score_display = "null" if step_score is None else f"{float(step_score):.3f}"
    has_rca = "RCA.md" in file_tree
    replay_succeeded = history_has_successful_replay(history)
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
        f"Has RCA.md: {str(has_rca).lower()}\n"
        f"Replay already succeeded: {str(replay_succeeded).lower()}\n"
        "Guidance: Avoid duplicate cat reads on unchanged files. "
        "If replay already succeeded and RCA.md exists, choose submit now.\n"
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


async def build_editor_content(
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


def extract_http_error_detail(exc: httpx.HTTPStatusError) -> str:
    """Return a concise error detail for non-2xx HTTP responses."""
    response = exc.response
    if response is None:
        return str(exc)
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
    except Exception:
        pass
    text = (response.text or "").strip()
    return text or str(exc)


def history_has_code_edit(history: List[Dict[str, str]]) -> bool:
    """Return whether the agent already edited a likely source file."""
    return any(
        item.get("action", "").startswith("write(")
        and item.get("action", "").endswith(".py)")
        for item in history
    )


def history_has_successful_replay(history: List[Dict[str, str]]) -> bool:
    """Return whether a replay action already reported a successful contract."""
    for item in history:
        action = item.get("action", "")
        stdout = item.get("stdout", "").lower()
        if action.startswith("replay(") and "contract_ok=true" in stdout:
            return True
    return False


def choose_forced_action(
    task_name: str,
    file_tree: List[str],
    known_files: Dict[str, str],
    history: List[Dict[str, str]],
    latest_step_result: Dict[str, Any],
    alert_message: str,
    step_number: int,
    max_steps: int,
) -> Optional[Dict[str, str]]:
    """Inject a small amount of task-aware structure into the baseline."""
    if not task_requires_rca(task_name):
        return None

    has_rca = "RCA.md" in file_tree or "RCA.md" in known_files
    if has_rca:
        return None

    remaining_actions = max_steps - step_number + 1
    last_error = str(latest_step_result["info"].get("last_action_error") or "")
    has_code_progress = history_has_code_edit(history)
    has_replay_success = history_has_successful_replay(history)
    ready_for_rca = has_code_progress and (has_replay_success or remaining_actions <= 2)

    if (
        ready_for_rca
        or ("RCA.md" in last_error and has_code_progress)
        or (remaining_actions <= 2 and has_code_progress)
    ):
        return {
            "tool": "editor",
            "command": "",
            "file_path": "RCA.md",
            "file_content": "",
        }

    return None


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


async def main() -> None:
    rewards: List[float] = []
    score = 0.0
    success = False
    steps_taken = 0
    latest_step_result: Dict[str, Any] = {}
    history: List[Dict[str, str]] = []
    known_files: Dict[str, str] = {}
    known_logs: Dict[str, str] = {}
    max_steps = DEFAULT_MAX_STEPS
    abort_episode = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("HF_TOKEN is required for submission inference runs.")
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        async with SREEnv(base_url=ENV_BASE_URL) as env_client:
            initial_observation_obj = await env_client.reset(task_id=TASK_NAME)
            state_obj = await env_client.state()
            state = state_obj.model_dump() if state_obj is not None else None
            initial_observation = initial_observation_obj.model_dump()
            if isinstance(state, dict):
                max_steps = int(state.get("max_steps") or DEFAULT_MAX_STEPS)
            alert_message = str(initial_observation.get("alert_message") or "")
            file_tree = list(initial_observation.get("file_tree") or [])
            latest_step_result = {
                "observation": initial_observation,
                "reward": {"value": 0.0},
                "done": False,
                "info": {"score": None, "message": "Episode reset", "last_action_error": None},
            }

            for step in range(1, max_steps + 1):
                if latest_step_result.get("done"):
                    break

                try:
                    action = choose_forced_action(
                        task_name=TASK_NAME,
                        file_tree=file_tree,
                        known_files=known_files,
                        history=history,
                        latest_step_result=latest_step_result,
                        alert_message=alert_message,
                        step_number=step,
                        max_steps=max_steps,
                    )
                    if action is None:
                        action = await choose_next_action(
                            client=llm_client,
                            alert_message=alert_message,
                            file_tree=file_tree,
                            history=history,
                            latest_step_result=latest_step_result,
                        )

                    if (
                        action["tool"] == "submit"
                        and task_requires_rca(TASK_NAME)
                        and "RCA.md" not in file_tree
                        and "RCA.md" not in known_files
                    ):
                        action = {
                            "tool": "editor",
                            "command": "",
                            "file_path": "RCA.md",
                            "file_content": "",
                        }

                    if action["tool"] == "editor":
                        current_source = known_files.get(action["file_path"])
                        if action["file_path"] == "RCA.md":
                            action["file_content"] = await build_editor_content(
                                client=llm_client,
                                file_path=action["file_path"],
                                current_source=current_source or "",
                                alert_message=alert_message,
                                known_logs=known_logs,
                                history=history,
                            )
                        elif current_source is None:
                            action = {
                                "tool": "terminal",
                                "command": f"cat {action['file_path']}",
                                "file_path": "",
                                "file_content": "",
                            }
                        else:
                            action["file_content"] = await build_editor_content(
                                client=llm_client,
                                file_path=action["file_path"],
                                current_source=current_source,
                                alert_message=alert_message,
                                known_logs=known_logs,
                                history=history,
                            )

                    latest_step_obj = await env_client.step(SREAction.model_validate(action))
                    latest_step_result = latest_step_obj.model_dump()
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
                    if action["tool"] == "editor" and action["file_path"] == "RCA.md" and action["file_content"]:
                        known_files["RCA.md"] = action["file_content"]
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
                except Exception as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        detail = extract_http_error_detail(exc)
                        steps_taken = step
                        log_step(
                            step=step,
                            action=sanitize_action_for_log(action),
                            reward=0.0,
                            done=False,
                            error=detail,
                        )
                        latest_step_result = {
                            "observation": {
                                "stdout": "",
                                "stderr": detail,
                                "exit_code": 1,
                                "file_tree": file_tree,
                                "alert_message": "",
                            },
                            "reward": {"value": 0.0},
                            "done": False,
                            "info": {
                                "score": None,
                                "message": "Step rejected by environment; continuing episode.",
                                "last_action_error": detail,
                            },
                        }
                        history.append(
                            {
                                "action": sanitize_log_value(sanitize_action_for_log(action)),
                                "reward": "0.00",
                                "done": "false",
                                "error": sanitize_log_value(detail),
                                "stdout": "null",
                                "stderr": sanitize_log_value(detail),
                            }
                        )
                        continue

                    log_step(
                        step=step,
                        action=sanitize_action_for_log(action),
                        reward=0.0,
                        done=False,
                        error=str(exc),
                    )
                    abort_episode = True
                    break

            if not latest_step_result.get("done") and not abort_episode:
                submit_action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}
                latest_step_obj = await env_client.step(SREAction.model_validate(submit_action))
                latest_step_result = latest_step_obj.model_dump()
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
        log_step(
            step=steps_taken,
            action="startup",
            reward=0.0,
            done=False,
            error=str(exc),
        )
        if steps_taken == 0:
            log_step(
                step=steps_taken,
                action="startup",
                reward=0.0,
                done=False,
                error=str(exc),
            )
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
    asyncio.run(main())
