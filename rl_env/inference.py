"""
Inference Script (SRE Tasks) — refactored
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import difflib
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = os.getenv("BENCHMARK", "rl_env")
LOCAL_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "180"))
ENABLE_GRADE_BREAKDOWN_LOGS = os.getenv("ENABLE_GRADE_BREAKDOWN_LOGS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_STEPS               = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
SCORE_EPSILON           = float(os.getenv("SCORE_EPSILON", "1e-6"))

TASK_MAP: Dict[int, str] = {
    1: "task1_wrong_status",
    2: "task2_retry_logic",
    3: "task3_cascading_failure",
}

REPLAY_MAP: Dict[str, str] = {
    "task1_wrong_status":     "create_item_contract",
    "task2_retry_logic":      "retry_health_contract",
    "task3_cascading_failure":"cascading_timeout_budget",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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
- For replay, set command to the correct replay name for this task (provided in the prompt).
- For submit, leave command/file_path/file_content empty.
- Prefer short deterministic commands like: cat logs/error.log, cat app/main.py, ls .
- Do not repeat `cat` on the same file unless that file was edited since your last read.
- Treat RCA.md as a final incident document: investigate first, fix the code, verify with replay, then write RCA.
- Create RCA.md before submit with headings: Root Cause, Affected Services, Fix Applied, Prevention.
- In "Fix Applied", describe the exact code-level change using literal identifiers/expressions from edited source files.
- Never choose submit before at least one meaningful code edit to a `.py` file.
- Before submit, ensure replay has most recently confirmed success (`contract_ok=true`).
- For RCA-required tasks, ensure RCA.md exists and reflects the latest code edit.
- Replay discipline: after the first successful replay (`contract_ok=true`), stop replaying and move to RCA/submit.
- Never replay repeatedly to farm reward; each extra replay without a new edit is harmful.
- If replay fails after an edit, make another code/config change before replaying again.
- After replay confirms success and RCA.md exists, submit immediately.
- Keep RCA language concise, factual, and non-speculative.
- If uncertain, prefer another fix/verify step instead of submit.
""".strip()

EDITOR_SYSTEM_PROMPT = """
You are preparing a full replacement file for an SRE environment edit action.
Use the incident alert, recent logs, prior action history, and the current file contents to infer the fix.
Return only the complete corrected file contents. Do not add explanations or markdown fences.
""".strip()


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    action: str
    reward: float
    done: bool
    error: Optional[str]
    stdout: str
    stderr: str


@dataclass
class PersistentState:
    # file/log memory
    known_files:   Dict[str, str] = field(default_factory=dict)
    known_logs:    Dict[str, str] = field(default_factory=dict)
    edited_files:  Set[str]       = field(default_factory=set)
    seen_cats:     Set[str]       = field(default_factory=set)
    # change tracking: file_path -> list of {before, after, step, diff}
    edit_diffs:    Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    # replay state
    replay_attempted:    bool           = False
    replay_passed:       bool           = False
    last_replay_stdout:  Optional[str]  = None
    last_replay_step:    Optional[int]  = None
    last_code_edit_step: Optional[int]  = None
    last_rca_edit_step:  Optional[int]  = None
    consecutive_replays_without_edit: int = 0
    # artifact state
    rca_written: bool = False
    submitted:   bool = False
    # history
    history: List[StepRecord] = field(default_factory=list)


@dataclass
class DerivedState:
    # code progress
    has_code_edit:      bool
    last_action_type:   str
    last_action_target: str
    # replay
    has_replay_attempt: bool
    has_replay_pass:    bool
    has_replay_after_latest_code_edit: bool
    # artifacts
    has_rca:                  bool
    unread_candidate_files:   List[str]
    # budget
    remaining_steps: int
    # derived decisions
    needs_rca_now:     bool
    should_force_replay: bool
    should_replay_after_latest_code_edit: bool
    must_submit_now:   bool


# ---------------------------------------------------------------------------
# Derived-state computation
# ---------------------------------------------------------------------------

def compute_derived_state(
    persistent: PersistentState,
    file_tree: List[str],
    step: int,
    max_steps: int,
) -> DerivedState:
    remaining = max_steps - step + 1

    last = persistent.history[-1] if persistent.history else None
    last_action_type   = last.action.split("(")[0] if last else ""
    last_action_target = last.action if last else ""

    has_code_edit = any(
        r.action.startswith("write(") and r.action.endswith(".py)")
        for r in persistent.history
    )
    has_replay_after_latest_code_edit = (
        persistent.last_code_edit_step is None
        or (
            persistent.last_replay_step is not None
            and persistent.last_replay_step > persistent.last_code_edit_step
        )
    )

    has_rca = "RCA.md" in file_tree or "RCA.md" in persistent.known_files

    unread_candidate_files = [
        f for f in file_tree
        if f not in persistent.seen_cats and not f.endswith(".md")
    ]

    # RCA evidence: code edited + (replay passed OR replay attempted with stdout evidence)
    enough_evidence = has_code_edit and (
        persistent.replay_passed
        or (persistent.replay_attempted and bool(persistent.last_replay_stdout))
    )
    needs_rca_now = enough_evidence and not has_rca

    # Force replay when: code edited, replay not attempted, budget closing
    steps_needed = (
        (1 if not persistent.replay_attempted else 0)
        + (1 if not has_rca else 0)
        + 2  # safety buffer for submit + server round-trip
    )
    should_force_replay = (
        has_code_edit
        and not persistent.replay_attempted
        and remaining <= steps_needed
    )
    should_replay_after_latest_code_edit = (
        has_code_edit and not has_replay_after_latest_code_edit
    )

    must_submit_now = remaining <= 1

    return DerivedState(
        has_code_edit=has_code_edit,
        last_action_type=last_action_type,
        last_action_target=last_action_target,
        has_replay_attempt=persistent.replay_attempted,
        has_replay_pass=persistent.replay_passed,
        has_replay_after_latest_code_edit=has_replay_after_latest_code_edit,
        has_rca=has_rca,
        unread_candidate_files=unread_candidate_files,
        remaining_steps=remaining,
        needs_rca_now=needs_rca_now,
        should_force_replay=should_force_replay,
        should_replay_after_latest_code_edit=should_replay_after_latest_code_edit,
        must_submit_now=must_submit_now,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def log_grade_breakdown(breakdown: Dict[str, float]) -> None:
    print(
        "[GRADE] "
        f"file_change={breakdown.get('file_change', 0.0):.3f} "
        f"tests_pass={breakdown.get('tests_pass', 0.0):.3f} "
        f"regex_match={breakdown.get('regex_match', 0.0):.3f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


def _generate_concise_diff_hint(before: str, after: str, context_lines: int = 3) -> str:
    """Return a compact unified-diff string capped at ~30 lines."""
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        before_lines, after_lines,
        fromfile="before", tofile="after",
        n=context_lines,
    )
    diff_lines = list(diff_iter)
    if not diff_lines:
        return "(no diff - content unchanged)"
    MAX_DIFF_LINES = 30
    if len(diff_lines) > MAX_DIFF_LINES:
        diff_lines = diff_lines[:MAX_DIFF_LINES] + ["... (diff truncated)\n"]
    return "".join(diff_lines)


def _normalize_task_score(raw_score: Any, epsilon: float = SCORE_EPSILON) -> float:
    try:
        score = float(raw_score)
    except Exception:
        score = 0.0
    if not math.isfinite(score):
        score = 0.0
    eps = min(max(float(epsilon), 1e-12), 0.49)
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


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
    pythonish = re.sub(r"\btrue\b",  "True",  raw,       flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b",  "None",  pythonish, flags=re.IGNORECASE)
    parsed = ast.literal_eval(pythonish)
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output is not an object.")
    return parsed


def _normalize_action(raw: Dict[str, Any]) -> Dict[str, str]:
    tool = str(raw.get("tool") or "").strip().lower()
    if tool not in {"terminal", "editor", "replay", "submit"}:
        raise RuntimeError(f"Unsupported tool: {tool or 'empty'}")
    action: Dict[str, str] = {
        "tool":         tool,
        "command":      str(raw.get("command")      or "").strip(),
        "file_path":    str(raw.get("file_path")    or "").strip(),
        "file_content": str(raw.get("file_content") or ""),
    }
    if action["tool"] == "terminal" and not action["command"]:
        raise RuntimeError("terminal action missing command")
    if action["tool"] == "replay" and not action["command"]:
        raise RuntimeError("replay action missing command")
    if action["tool"] == "editor" and not action["file_path"]:
        raise RuntimeError("editor action missing file_path")
    return action


def _sanitize_action_for_log(action: Dict[str, str]) -> str:
    if action["tool"] == "terminal":
        return action.get("command") or ""
    if action["tool"] == "editor":
        return f"write({action.get('file_path') or ''})"
    if action["tool"] == "replay":
        return f"replay({action.get('command') or ''})"
    return "submit"


def _parse_reward(result_payload: Dict[str, Any]) -> float:
    reward_payload = result_payload.get("reward", 0.0)
    if isinstance(reward_payload, dict):
        return float(reward_payload.get("value", 0.0) or 0.0)
    return float(reward_payload or 0.0)


@lru_cache(maxsize=16)
def _task_requires_rca(task_id: str) -> bool:
    candidate_paths = [
        Path(__file__).resolve().parent / "fixtures" / task_id / "task_config.json",
        Path("fixtures") / task_id / "task_config.json",
    ]
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected = config.get("expected_fix_files") or []
            return "RCA.md" in expected
        except Exception:
            continue
    return task_id in {"task1_wrong_status", "task2_retry_logic", "task3_cascading_failure"}


@lru_cache(maxsize=16)
def _task_max_steps(task_id: str) -> int:
    candidate_paths = [
        Path(__file__).resolve().parent / "fixtures" / task_id / "task_config.json",
        Path("fixtures") / task_id / "task_config.json",
    ]
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            max_steps = int(config.get("max_steps") or 0)
            if max_steps > 0:
                return max_steps
        except Exception:
            continue
    return MAX_STEPS


# ---------------------------------------------------------------------------
# Action selection helpers
# ---------------------------------------------------------------------------

def _apply_hard_guards(
    action: Dict[str, str],
    derived: DerivedState,
    persistent: PersistentState,
    task_id: str,
    replay_name: str,
) -> Dict[str, str]:
    """
    Normalizes an LLM-proposed action through a series of hard guards.
    Order matters: each guard either rewrites the action or passes through.
    """
    tool = action["tool"]

    # Guard 1: single deterministic submit gate.
    # While budget remains, block submit until code is edited, replay passes,
    # and (for RCA tasks) RCA.md is present and up-to-date.
    if tool == "submit" and derived.remaining_steps > 1:
        requires_rca = _task_requires_rca(task_id)

        if not derived.has_code_edit:
            candidate_files = _candidate_edit_files(persistent, derived)
            if candidate_files:
                return {
                    "tool": "editor",
                    "command": "",
                    "file_path": candidate_files[0],
                    "file_content": "",
                }
            unread_sources = [
                path for path in derived.unread_candidate_files
                if path.endswith(".py")
            ]
            if unread_sources:
                return {
                    "tool": "terminal",
                    "command": f"cat {unread_sources[0]}",
                    "file_path": "",
                    "file_content": "",
                }
            return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}

        if derived.should_replay_after_latest_code_edit or not persistent.replay_passed:
            return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}

        if requires_rca:
            rca_outdated = (
                persistent.last_code_edit_step is not None
                and (
                    persistent.last_rca_edit_step is None
                    or persistent.last_rca_edit_step < persistent.last_code_edit_step
                )
            )
            if (not derived.has_rca) or rca_outdated:
                return {"tool": "editor", "command": "", "file_path": "RCA.md", "file_content": ""}

    # Guard 2: editor on a file not yet read → cat it first
    if tool == "editor":
        fp = action["file_path"]
        if fp != "RCA.md" and fp not in persistent.seen_cats and fp not in persistent.edited_files:
            return {"tool": "terminal", "command": f"cat {fp}", "file_path": "", "file_content": ""}

    # Guard 3: enforce correct replay command for this task
    if tool == "replay":
        requires_rca = _task_requires_rca(task_id)

        # Guard 3a: if replay already passed and RCA is required but missing,
        # immediately redirect to RCA.md — never let extra replays happen here.
        if (
            persistent.replay_passed
            and requires_rca
            and not derived.has_rca
        ):
            return {"tool": "editor", "command": "", "file_path": "RCA.md", "file_content": ""}

        # Guard 3b: hard cap replay spam — at most 2 consecutive replays without a .py edit.
        if (
            persistent.consecutive_replays_without_edit >= 2
            and derived.remaining_steps > 2
        ):
            # If RCA is required and missing, write it before anything else.
            if requires_rca and not derived.has_rca and derived.has_code_edit:
                return {"tool": "editor", "command": "", "file_path": "RCA.md", "file_content": ""}
            candidate_files = _candidate_edit_files(persistent, derived)
            if candidate_files:
                return {
                    "tool": "editor",
                    "command": "",
                    "file_path": candidate_files[0],
                    "file_content": "",
                }
            # If we have no useful edit target, read one unread source file.
            unread_sources = [
                path for path in derived.unread_candidate_files
                if path.endswith(".py")
            ]
            if unread_sources:
                return {
                    "tool": "terminal",
                    "command": f"cat {unread_sources[0]}",
                    "file_path": "",
                    "file_content": "",
                }

        # Guard 3c: avoid replay-only loops before any meaningful code change.
        if (
            not derived.has_code_edit
            and derived.remaining_steps > 2
        ):
            candidate_files = _candidate_edit_files(persistent, derived)
            if candidate_files:
                return {
                    "tool": "editor",
                    "command": "",
                    "file_path": candidate_files[0],
                    "file_content": "",
                }
            unread_sources = [
                path for path in derived.unread_candidate_files
                if path.endswith(".py")
            ]
            if unread_sources:
                return {
                    "tool": "terminal",
                    "command": f"cat {unread_sources[0]}",
                    "file_path": "",
                    "file_content": "",
                }
        action["command"] = replay_name

    return action


def _candidate_edit_files(persistent: PersistentState, derived: DerivedState) -> List[str]:
    """Rank likely code edit targets to break replay/cat loops."""
    seen_sources = sorted(
        path for path in persistent.seen_cats
        if path.endswith(".py") and path != "RCA.md"
    )
    unseen_but_known = sorted(
        path for path in persistent.known_files.keys()
        if path.endswith(".py") and path != "RCA.md" and path not in persistent.seen_cats
    )
    unread_sources = sorted(
        path for path in derived.unread_candidate_files
        if path.endswith(".py") and path != "RCA.md"
    )
    # Prefer files the model has already read, then known files, then unread files.
    return seen_sources + unseen_but_known + unread_sources


def _choose_forced_action(
    derived: DerivedState,
    persistent: PersistentState,
    task_id: str,
    replay_name: str,
) -> Optional[Dict[str, str]]:
    """
    Forced actions (priority 1 = done→break is in the main loop).
    Returns None if no forced action is required.
    """
    requires_rca = _task_requires_rca(task_id)

    # Priority 2: force replay
    if derived.should_force_replay:
        return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}

    # Priority 3: replay after latest code edit before submit (if budget allows).
    if (
        derived.should_replay_after_latest_code_edit
        and derived.remaining_steps > 1
    ):
        return {"tool": "replay", "command": replay_name, "file_path": "", "file_content": ""}

    # Priority 4: RCA evidence sufficient, RCA missing — fire immediately after any replay completes
    if derived.needs_rca_now and requires_rca:
        return {"tool": "editor", "command": "", "file_path": "RCA.md", "file_content": ""}

    # Priority 4b: replay has passed, RCA is required but not written yet — catch any gap that
    # Priority 4 misses (e.g., if needs_rca_now was gated on a broader evidence check).
    if (
        requires_rca
        and persistent.replay_passed
        and derived.has_code_edit
        and not derived.has_rca
        and not derived.should_replay_after_latest_code_edit
    ):
        return {"tool": "editor", "command": "", "file_path": "RCA.md", "file_content": ""}

    # Priority 5: if replay has passed and RCA requirement is met, submit early.
    if (
        derived.has_code_edit
        and derived.has_replay_pass
        and not derived.should_replay_after_latest_code_edit
        and (not requires_rca or derived.has_rca)
    ):
        return {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

    # Priority 6: final step
    if derived.must_submit_now:
        return {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

    return None


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _build_action_prompt(
    task_id: str,
    obs: Dict[str, Any],
    persistent: PersistentState,
    derived: DerivedState,
    replay_name: str,
    step: int,
    max_steps: int,
    alert_message: str,
) -> str:
    stdout = str(obs.get("stdout") or "")
    stderr = str(obs.get("stderr") or "")
    file_tree = obs.get("file_tree") or []

    history_lines = []
    for rec in persistent.history[-8:]:
        history_lines.append(
            f"- step={rec.step} action={rec.action} reward={rec.reward:.2f} "
            f"done={str(rec.done).lower()} error={rec.error or 'null'} "
            f"stdout={rec.stdout[:400]} stderr={rec.stderr[:300]}"
        )
    hist_block      = "\n".join(history_lines) if history_lines else "None"
    seen_cats_block = ", ".join(sorted(persistent.seen_cats)) if persistent.seen_cats else "None"
    unread_block    = ", ".join(derived.unread_candidate_files) if derived.unread_candidate_files else "None"
    edited_block    = ", ".join(sorted(persistent.edited_files)) if persistent.edited_files else "None"

    # Build a one-line-per-file diff summary so the LLM knows what it already changed
    diff_summary_lines = []
    for path, diffs in sorted(persistent.edit_diffs.items()):
        latest = diffs[-1]
        first_diff_line = next(
            (l.rstrip() for l in latest["diff"].splitlines() if l.startswith(("@@", "+", "-"))),
            "edited",
        )
        diff_summary_lines.append(f"  {path} (step {latest['step']}): {first_diff_line}")
    diff_summary_block = "\n".join(diff_summary_lines) if diff_summary_lines else "  (No changes recorded in this session yet.)"

    return (
        f"Task: {task_id}\n"
        f"Step: {step}/{max_steps} | Remaining: {derived.remaining_steps}\n"
        f"Replay command for this task: {replay_name}\n\n"
        f"Alert:\n{alert_message or 'None'}\n\n"
        f"Latest stdout:\n{stdout[:2000] or 'None'}\n\n"
        f"Latest stderr:\n{stderr[:1200] or 'None'}\n\n"
        f"Workspace file tree:\n{_safe_json(file_tree)}\n\n"
        f"Recent actions (last 8):\n{hist_block}\n\n"
        f"--- Derived State ---\n"
        f"has_code_edit:     {derived.has_code_edit}\n"
        f"has_replay_attempt:{derived.has_replay_attempt}\n"
        f"has_replay_pass:   {derived.has_replay_pass}\n"
        f"replay_after_latest_code_edit: {derived.has_replay_after_latest_code_edit}\n"
        f"has_rca:           {derived.has_rca}\n"
        f"Edited files:                           {edited_block}\n"
        f"Files already viewed: {seen_cats_block}\n"
        f"Unread candidate files:                 {unread_block}\n"
        f"Session technical changes (authoritative log):\n{diff_summary_block}\n"
        f"--- Action Guidance (FOLLOW STRICTLY) ---\n"
        f"1. If has_replay_pass=True and has_rca=False: choose editor with file_path=RCA.md NOW. Do NOT replay again.\n"
        f"2. If has_replay_pass=True and has_rca=True: choose submit NOW.\n"
        f"3. If has_code_edit=False: inspect files with terminal then use editor to fix a .py file.\n"
        f"4. Replaying after a successful replay (contract_ok=true) wastes steps. Write RCA.md instead.\n"
    )


def _choose_action_from_llm(
    client: OpenAI,
    task_id: str,
    obs: Dict[str, Any],
    persistent: PersistentState,
    derived: DerivedState,
    step: int,
    max_steps: int,
    alert_message: str,
    replay_name: str,
) -> Dict[str, str]:
    prompt = _build_action_prompt(
        task_id=task_id,
        obs=obs,
        persistent=persistent,
        derived=derived,
        replay_name=replay_name,
        step=step,
        max_steps=max_steps,
        alert_message=alert_message,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=350,
    )
    content = response.choices[0].message.content or ""
    return _normalize_action(_extract_model_json(content))


async def _build_editor_content(
    client: OpenAI,
    file_path: str,
    current_source: str,
    task_id: str,
    alert_message: str,
    known_logs: Dict[str, str],
    known_files: Dict[str, str],
    edited_files: Set[str],
    history: List[StepRecord],
    replay_evidence: str,
    edit_diffs: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> str:
    history_lines = [
        f"- step={r.step} action={r.action} reward={r.reward:.2f} error={r.error or 'null'}"
        for r in history[-6:]
    ]
    history_block = "\n".join(history_lines) if history_lines else "None"

    log_sections = [
        f"{log_name}:\n{log_content[:2000]}"
        for log_name, log_content in known_logs.items()
    ]
    logs_block = "\n\n".join(log_sections) if log_sections else "None"

    # For RCA, pass the current (post-fix) source snapshot of each edited .py file
    edited_source_sections = []
    if file_path == "RCA.md":
        for edited in sorted(edited_files):
            if edited.endswith(".py") and edited in known_files:
                edited_source_sections.append(
                    f"{edited} (current/fixed version):\n{known_files[edited][:2000]}"
                )
    edited_source_block = "\n\n".join(edited_source_sections) if edited_source_sections else "None"
    replay_evidence_block = replay_evidence[:2000] if replay_evidence else "None"

    # Build an authoritative change-history block from recorded diffs
    change_history_block = "None"
    if file_path == "RCA.md" and edit_diffs:
        change_sections = []
        for path, diffs in sorted(edit_diffs.items()):
            if path.endswith(".py"):
                for d in diffs:
                    change_sections.append(
                        f"File edited: {path} (at step {d['step']})\n"
                        f"--- diff (before -> after) ---\n{d['diff']}\n"
                        f"BEFORE snippet:\n{d['before_snippet']}\n"
                        f"AFTER snippet:\n{d['after_snippet']}"
                    )
        if change_sections:
            change_history_block = "\n\n".join(change_sections)

    # Hardened RCA instruction - no misleading examples
    rca_fix_instruction = ""
    if file_path == "RCA.md":
        rca_fix_instruction = (
            "IMPORTANT for 'Fix Applied' section:\n"
            "- Use the 'Change History' block above as the AUTHORITATIVE record of what changed.\n"
            "- Describe the change in the AFTER direction (what the code now says), not the BEFORE.\n"
            "- Quote the exact AFTER expression from the diff (lines starting with '+').\n"
            "- Do NOT invert or guess the fix direction."
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task_id}\n"
                    f"Target file: {file_path}\n\n"
                    f"Alert:\n{alert_message or 'None'}\n\n"
                    f"Known logs:\n{logs_block}\n\n"
                    f"Change History (authoritative before->after diffs):\n"
                    f"{change_history_block}\n\n"
                    f"Edited source snapshots (current/fixed state):\n{edited_source_block}\n\n"
                    f"Replay evidence:\n{replay_evidence_block}\n\n"
                    f"{rca_fix_instruction}\n\n"
                    f"Recent history:\n{history_block}\n\n"
                    f"Current file contents:\n{current_source}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    if not raw:
        raise RuntimeError("Model returned empty editor content.")
    return raw


# ---------------------------------------------------------------------------
# State update after each step
# ---------------------------------------------------------------------------

def _update_persistent_state(
    persistent: PersistentState,
    action: Dict[str, str],
    obs: Dict[str, Any],
    reward: float,
    done: bool,
    last_error: Optional[str],
    step: int,
) -> None:
    stdout = str(obs.get("stdout") or "")
    stderr = str(obs.get("stderr") or "")
    tool   = action["tool"]

    # Terminal: track cat'd files
    if tool == "terminal":
        cmd = action["command"]
        if cmd.startswith("cat "):
            target = cmd[4:].strip()
            persistent.seen_cats.add(target)
            if stdout:
                if target.endswith(".log"):
                    persistent.known_logs[target] = stdout
                else:
                    persistent.known_files[target] = stdout

    # Editor: track written files + record before/after diffs
    if tool == "editor":
        fp      = action["file_path"]
        content = action.get("file_content", "")
        # Capture old content BEFORE updating known_files
        old_content = persistent.known_files.get(fp, "")
        persistent.edited_files.add(fp)
        if content:
            persistent.known_files[fp] = content
        # Record diff when content actually changed
        if content and content != old_content:
            diff_hint = _generate_concise_diff_hint(old_content, content)
            if fp not in persistent.edit_diffs:
                persistent.edit_diffs[fp] = []
            persistent.edit_diffs[fp].append({
                "before_snippet": old_content[:800],
                "after_snippet":  content[:800],
                "step":           str(step),
                "diff": diff_hint,
            })
        if fp.endswith(".py") and content.strip():
            persistent.last_code_edit_step = step
            persistent.consecutive_replays_without_edit = 0
        if fp == "RCA.md":
            persistent.rca_written = True
            persistent.last_rca_edit_step = step
        # Only reset consecutive_replays counter when an actual .py code edit happened
        # (not for RCA.md or other markdown edits — those don't break replay loops)

    # Replay: update replay state precisely
    if tool == "replay":
        persistent.replay_attempted   = True
        persistent.last_replay_stdout = stdout
        persistent.last_replay_step   = step
        persistent.consecutive_replays_without_edit += 1
        stdout_lower = stdout.lower()
        if "contract_ok=true" in stdout_lower:
            persistent.replay_passed = True
        elif "contract_ok=false" in stdout_lower:
            persistent.replay_passed = False
        # if neither keyword is present, leave replay_passed unchanged
    # NOTE: only .py code edits (handled above at line 861) reset the replay spam counter;
    # terminal/submit/RCA actions do NOT reset it, preventing the reset-and-loop cycle

    # Submit
    if tool == "submit":
        persistent.submitted = True

    # Build a stdout hint for history: for editor steps include a diff summary
    history_stdout = stdout
    if tool == "editor":
        fp = action["file_path"]
        diffs = persistent.edit_diffs.get(fp)
        if diffs:
            history_stdout = f"{stdout}\n[DIFF]\n{diffs[-1]['diff'][:400]}"

    # Append to history
    action_log = _sanitize_action_for_log(action)
    persistent.history.append(StepRecord(
        step=step,
        action=action_log,
        reward=reward,
        done=done,
        error=last_error,
        stdout=history_stdout[:600],
        stderr=stderr[:600],
    ))


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

async def run_inference(task: int) -> Dict[str, Any]:
    task_id     = TASK_MAP.get(task, f"task{task}")
    effective_max_steps = max(MAX_STEPS, _task_max_steps(task_id))
    llm_client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    replay_name = REPLAY_MAP.get(task_id, "create_item_contract")

    persistent   = PersistentState()
    rewards: List[float] = []
    steps_taken  = 0
    score        = 0.0
    success      = False
    llm_disabled = False
    alert_message= ""
    current_step = 0
    current_action_log = ""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(base_url=LOCAL_BASE_URL, timeout=HTTP_TIMEOUT_SECONDS) as http:

            # ── Reset ──────────────────────────────────────────────────────
            reset_resp = await http.post("/reset", json={"task_id": task_id})
            reset_resp.raise_for_status()
            result_payload = reset_resp.json()
            final_payload: Dict[str, Any] = result_payload
            obs_dict       = result_payload.get("observation", {})
            alert_message  = str(obs_dict.get("alert_message") or "")
            file_tree: List[str] = list(obs_dict.get("file_tree") or [])
            done = bool(result_payload.get("done", False))

            for step in range(1, effective_max_steps + 1):

                # ── Priority 1: done ──────────────────────────────────────
                if done:
                    break

                # ── Compute derived state ─────────────────────────────────
                derived = compute_derived_state(
                    persistent=persistent,
                    file_tree=file_tree,
                    step=step,
                    max_steps=effective_max_steps,
                )

                # ── Priorities 2-4: forced actions ────────────────────────
                action_dict = _choose_forced_action(
                    derived=derived,
                    persistent=persistent,
                    task_id=task_id,
                    replay_name=replay_name,
                )

                # ── Priority 5: LLM proposes action ───────────────────────
                if action_dict is None:
                    if not llm_disabled:
                        try:
                            action_dict = _choose_action_from_llm(
                                client=llm_client,
                                task_id=task_id,
                                obs=obs_dict,
                                persistent=persistent,
                                derived=derived,
                                step=step,
                                max_steps=effective_max_steps,
                                alert_message=alert_message,
                                replay_name=replay_name,
                            )
                        except Exception as exc:
                            err = str(exc).lower()
                            if "402" in err or "depleted" in err or "credits" in err:
                                llm_disabled = True
                            # Safe fallback: submit when resolution criteria are met,
                            # otherwise attempt replay.
                            if (
                                derived.has_code_edit
                                and derived.has_replay_pass
                                and not derived.should_replay_after_latest_code_edit
                                and (not _task_requires_rca(task_id) or derived.has_rca)
                            ):
                                action_dict = {
                                    "tool": "submit", "command": "",
                                    "file_path": "", "file_content": "",
                                }
                            else:
                                action_dict = {
                                    "tool": "replay", "command": replay_name,
                                    "file_path": "", "file_content": "",
                                }
                    else:
                        if (
                            derived.has_code_edit
                            and derived.has_replay_pass
                            and not derived.should_replay_after_latest_code_edit
                            and (not _task_requires_rca(task_id) or derived.has_rca)
                        ):
                            action_dict = {
                                "tool": "submit", "command": "",
                                "file_path": "", "file_content": "",
                            }
                        else:
                            action_dict = {
                                "tool": "replay", "command": replay_name,
                                "file_path": "", "file_content": "",
                            }

                # ── Priority 6: hard guards on LLM action ────────────────
                action_dict = _apply_hard_guards(
                    action=action_dict,
                    derived=derived,
                    persistent=persistent,
                    task_id=task_id,
                    replay_name=replay_name,
                )

                # ── Priority 7: editor → generate content ─────────────────
                if action_dict["tool"] == "editor":
                    fp = action_dict["file_path"]
                    if not action_dict.get("file_content", "").strip():
                        current_source = persistent.known_files.get(fp, "")
                        try:
                            action_dict["file_content"] = await _build_editor_content(
                                client=llm_client,
                                file_path=fp,
                                current_source=current_source,
                                task_id=task_id,
                                alert_message=alert_message,
                                known_logs=persistent.known_logs,
                                known_files=persistent.known_files,
                                edited_files=persistent.edited_files,
                                history=persistent.history,
                                replay_evidence=persistent.last_replay_stdout or "",
                                edit_diffs=persistent.edit_diffs,
                            )
                        except Exception as exc:
                            err = str(exc).lower()
                            if "402" in err or "depleted" in err or "credits" in err:
                                llm_disabled = True
                                if persistent.last_code_edit_step is not None:
                                    action_dict = {
                                        "tool": "replay",
                                        "command": replay_name,
                                        "file_path": "",
                                        "file_content": "",
                                    }
                                else:
                                    # Avoid silent early submit on editor-generation failure.
                                    fallback_cmd = f"cat {fp}" if fp and fp != "RCA.md" else "ls ."
                                    action_dict = {
                                        "tool": "terminal",
                                        "command": fallback_cmd,
                                        "file_path": "",
                                        "file_content": "",
                                    }
                            else:
                                raise

                # ── Validate with pydantic model ──────────────────────────
                _ = SREAction(**action_dict)

                # ── Priority 8: POST /step ────────────────────────────────
                action_log = _sanitize_action_for_log(action_dict)
                current_step = step
                current_action_log = action_log
                step_resp = await http.post("/step", json=action_dict)
                step_resp.raise_for_status()
                result_payload = step_resp.json()
                final_payload = result_payload

                obs_dict      = result_payload.get("observation", {})
                reward        = _parse_reward(result_payload)
                done          = bool(result_payload.get("done", False))
                info          = result_payload.get("info") or {}
                last_error    = str(info.get("last_action_error") or "") or None
                error_display = last_error or obs_dict.get("stderr") or None

                # ── Priority 9: update persistent state ───────────────────
                _update_persistent_state(
                    persistent=persistent,
                    action=action_dict,
                    obs=obs_dict,
                    reward=reward,
                    done=done,
                    last_error=last_error,
                    step=step,
                )

                file_tree = list(obs_dict.get("file_tree") or file_tree)
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_log, reward=reward, done=done, error=error_display)

            # ── Priority 10: loop exits without done → force submit ────────
            if not done:
                submit_action = {
                    "tool": "submit", "command": "", "file_path": "", "file_content": "",
                }
                current_step = steps_taken + 1
                current_action_log = "submit"
                submit_resp = await http.post("/step", json=submit_action)
                submit_resp.raise_for_status()
                submit_payload = submit_resp.json()
                final_payload  = submit_payload
                reward         = _parse_reward(submit_payload)
                rewards.append(reward)
                steps_taken   += 1
                submit_info    = submit_payload.get("info") or {}
                submit_obs     = submit_payload.get("observation") or {}
                submit_done    = bool(submit_payload.get("done", False))
                done           = submit_done
                submit_error   = submit_info.get("last_action_error") or submit_obs.get("stderr") or None
                log_step(step=steps_taken, action="submit(auto)", reward=reward, done=submit_done, error=submit_error)

            info_payload = final_payload.get("info") or {}
            raw_score = float(info_payload.get("score") or 0.0)
            score     = _normalize_task_score(raw_score)
            success   = raw_score >= SUCCESS_SCORE_THRESHOLD
            breakdown = info_payload.get("grading_breakdown")
            if ENABLE_GRADE_BREAKDOWN_LOGS and isinstance(breakdown, dict):
                numeric_breakdown = {
                    str(key): float(value)
                    for key, value in breakdown.items()
                    if isinstance(value, (int, float))
                }
                if numeric_breakdown:
                    log_grade_breakdown(numeric_breakdown)

    except Exception as exc:
        error_text = str(exc)
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                detail = (exc.response.text or "").strip()
                if detail:
                    error_text = detail
            except Exception:
                pass
        # Always surface the error — silent failures make debugging impossible
        log_step(
            step=max(current_step, 1),
            action=current_action_log or "error",
            reward=0.0,
            done=False,
            error=error_text,
        )
        steps_taken = max(steps_taken, current_step)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "success": success,
        "steps":   steps_taken,
        "score":   score,
        "rewards": rewards,
        "task":    task_id,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
