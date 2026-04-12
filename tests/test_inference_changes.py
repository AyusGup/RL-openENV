"""Tests for change-tracking, diff hint generation, and enriched prompt content
added in the inference.py refactor."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_env.inference import (
    PersistentState,
    DerivedState,
    _apply_hard_guards,
    _generate_concise_diff_hint,
    _update_persistent_state,
    _build_action_prompt,
    _build_editor_content,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(**kw) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "file_tree": [],
        "alert_message": "",
    }
    defaults.update(kw)
    return defaults


def _make_derived(**kw) -> DerivedState:
    defaults = dict(
        has_code_edit=False,
        last_action_type="terminal",
        last_action_target="",
        has_replay_attempt=False,
        has_replay_pass=False,
        has_replay_after_latest_code_edit=True,
        has_rca=False,
        unread_candidate_files=[],
        remaining_steps=10,
        needs_rca_now=False,
        should_force_replay=False,
        should_replay_after_latest_code_edit=False,
        must_submit_now=False,
    )
    defaults.update(kw)
    return DerivedState(**defaults)


def _make_mock_client(return_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = return_text
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# _generate_concise_diff_hint
# ---------------------------------------------------------------------------

class TestGenerateConciseDiffHint:
    def test_identical_content_returns_no_diff_message(self) -> None:
        result = _generate_concise_diff_hint("line1\nline2\n", "line1\nline2\n")
        assert "no diff" in result.lower()

    def test_changed_line_appears_with_plus_and_minus(self) -> None:
        before = "for attempt in range(max_retries):\n    pass\n"
        after  = "for attempt in range(max_retries + 1):\n    pass\n"
        result = _generate_concise_diff_hint(before, after)
        # unified diff uses --- / +++ for headers and -/+ for changed lines
        assert "max_retries + 1" in result
        assert "-for attempt in range(max_retries):" in result
        assert "+for attempt in range(max_retries + 1):" in result

    def test_diff_is_capped_at_30_lines(self) -> None:
        before = "\n".join(f"old line {i}" for i in range(50))
        after  = "\n".join(f"new line {i}" for i in range(50))
        result = _generate_concise_diff_hint(before, after)
        lines = result.splitlines()
        assert len(lines) <= 31  # 30 diff lines + possible truncation notice

    def test_empty_before_produces_additions_only(self) -> None:
        result = _generate_concise_diff_hint("", "new content\n")
        assert "+" in result
        assert "new content" in result

    def test_empty_after_produces_deletions_only(self) -> None:
        result = _generate_concise_diff_hint("old content\n", "")
        assert "-" in result
        assert "old content" in result

    def test_return_value_is_string(self) -> None:
        result = _generate_concise_diff_hint("a\n", "b\n")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _update_persistent_state — diff recording
# ---------------------------------------------------------------------------

class TestUpdatePersistentStateDiffRecording:
    def _editor_action(self, fp: str, content: str) -> Dict[str, str]:
        return {"tool": "editor", "file_path": fp, "file_content": content, "command": ""}

    def test_first_edit_creates_diff_entry(self) -> None:
        p = PersistentState()
        p.known_files["app/retry_handler.py"] = "for attempt in range(max_retries):\n    pass\n"
        obs = _make_obs(stdout="SUCCESS: Wrote content to app/retry_handler.py")
        action = self._editor_action(
            "app/retry_handler.py",
            "for attempt in range(max_retries + 1):\n    pass\n",
        )
        _update_persistent_state(p, action, obs, reward=0.0, done=False, last_error=None, step=3)

        assert "app/retry_handler.py" in p.edit_diffs
        diffs = p.edit_diffs["app/retry_handler.py"]
        assert len(diffs) == 1
        assert diffs[0]["step"] == "3"
        assert "max_retries" in diffs[0]["before_snippet"]
        assert "max_retries + 1" in diffs[0]["after_snippet"]
        assert "max_retries + 1" in diffs[0]["diff"]

    def test_identical_rewrite_does_not_create_diff_entry(self) -> None:
        p = PersistentState()
        same = "for attempt in range(max_retries):\n    pass\n"
        p.known_files["app/retry_handler.py"] = same
        obs = _make_obs(stdout="SUCCESS: Wrote content to app/retry_handler.py")
        action = self._editor_action("app/retry_handler.py", same)
        _update_persistent_state(p, action, obs, reward=0.0, done=False, last_error=None, step=5)

        assert p.edit_diffs.get("app/retry_handler.py") is None

    def test_second_edit_appends_to_diff_list(self) -> None:
        p = PersistentState()
        p.known_files["app/retry_handler.py"] = "v1\n"
        obs = _make_obs(stdout="SUCCESS")

        _update_persistent_state(p, self._editor_action("app/retry_handler.py", "v2\n"), obs, 0.0, False, None, step=2)
        _update_persistent_state(p, self._editor_action("app/retry_handler.py", "v3\n"), obs, 0.0, False, None, step=4)

        assert len(p.edit_diffs["app/retry_handler.py"]) == 2
        assert p.edit_diffs["app/retry_handler.py"][0]["step"] == "2"
        assert p.edit_diffs["app/retry_handler.py"][1]["step"] == "4"

    def test_history_stdout_includes_diff_hint_for_editor_step(self) -> None:
        p = PersistentState()
        p.known_files["app/retry_handler.py"] = "old code\n"
        obs = _make_obs(stdout="SUCCESS: Wrote content to app/retry_handler.py")
        action = self._editor_action("app/retry_handler.py", "new code here\n")
        _update_persistent_state(p, action, obs, 0.0, False, None, step=3)

        last_rec = p.history[-1]
        assert "[DIFF]" in last_rec.stdout

    def test_known_files_updated_with_new_content(self) -> None:
        p = PersistentState()
        p.known_files["app/retry_handler.py"] = "old\n"
        obs = _make_obs(stdout="SUCCESS")
        action = self._editor_action("app/retry_handler.py", "new\n")
        _update_persistent_state(p, action, obs, 0.0, False, None, step=2)

        assert p.known_files["app/retry_handler.py"] == "new\n"

    def test_edit_diffs_initialized_as_empty_dict(self) -> None:
        p = PersistentState()
        assert p.edit_diffs == {}

    def test_last_code_edit_step_is_set_for_py_files(self) -> None:
        p = PersistentState()
        p.known_files["app/retry_handler.py"] = "old\n"
        obs = _make_obs(stdout="SUCCESS")
        _update_persistent_state(
            p, self._editor_action("app/retry_handler.py", "new\n"),
            obs, 0.0, False, None, step=5,
        )
        assert p.last_code_edit_step == 5


# ---------------------------------------------------------------------------
# _build_action_prompt — enriched with diff summary and edited files
# ---------------------------------------------------------------------------

class TestBuildActionPromptEnriched:
    def test_edited_files_block_present_when_files_edited(self) -> None:
        p = PersistentState()
        p.edited_files.add("app/retry_handler.py")
        # Use a diff that starts with a diff hunk line (@@) so the first_diff_line
        # extractor picks up the +/- change line, not just the --- header.
        p.edit_diffs["app/retry_handler.py"] = [{
            "step": "3",
            "before_snippet": "range(max_retries)",
            "after_snippet": "range(max_retries + 1)",
            "diff": "@@ -1 +1 @@\n-range(max_retries)\n+range(max_retries + 1)\n",
        }]
        derived = _make_derived(has_code_edit=True)
        obs = _make_obs(file_tree=["app/retry_handler.py"])

        prompt = _build_action_prompt(
            task_id="task2_retry_logic",
            obs=obs,
            persistent=p,
            derived=derived,
            replay_name="retry_health_contract",
            step=4,
            max_steps=16,
            alert_message="HIGH: retry failing.",
        )

        assert "Edited files:" in prompt
        assert "app/retry_handler.py" in prompt
        assert "Session technical changes (authoritative log):" in prompt
        # The diff summary line picks the first @@/+/- line, which is the @@ hunk header
        assert "app/retry_handler.py (step 3):" in prompt

    def test_no_diff_summary_when_no_edits(self) -> None:
        p = PersistentState()
        derived = _make_derived()
        obs = _make_obs(file_tree=[])

        prompt = _build_action_prompt(
            task_id="task2_retry_logic",
            obs=obs,
            persistent=p,
            derived=derived,
            replay_name="retry_health_contract",
            step=1,
            max_steps=16,
            alert_message="",
        )

        assert "Session technical changes (authoritative log):" in prompt
        assert "(No changes recorded in this session yet.)" in prompt

    def test_edited_files_block_shows_none_when_no_edits(self) -> None:
        p = PersistentState()
        derived = _make_derived()
        obs = _make_obs()

        prompt = _build_action_prompt(
            task_id="task2_retry_logic",
            obs=obs,
            persistent=p,
            derived=derived,
            replay_name="retry_health_contract",
            step=2,
            max_steps=16,
            alert_message="",
        )

        assert "Edited files:                           None" in prompt


# ---------------------------------------------------------------------------
# _build_editor_content — change history block in RCA prompt
# ---------------------------------------------------------------------------

class TestBuildEditorContentChangeHistory:
    """Verify that _build_editor_content passes an authoritative change-history
    block to the model when generating RCA.md."""

    def test_rca_prompt_contains_change_history(self) -> None:
        mock_client = _make_mock_client("## Root Cause\nfoo\n## Fix Applied\nbar\n")
        edit_diffs = {
            "app/retry_handler.py": [{
                "step": "3",
                "before_snippet": "range(max_retries)",
                "after_snippet": "range(max_retries + 1)",
                "diff": "--- before\n+++ after\n-range(max_retries)\n+range(max_retries + 1)\n",
            }]
        }

        asyncio.run(_build_editor_content(
            client=mock_client,
            file_path="RCA.md",
            current_source="",
            task_id="task2_retry_logic",
            alert_message="retry failing",
            known_logs={},
            known_files={"app/retry_handler.py": "range(max_retries + 1)"},
            edited_files={"app/retry_handler.py"},
            history=[],
            replay_evidence="contract_ok=true",
            edit_diffs=edit_diffs,
        ))

        call_kwargs = mock_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        assert "Change History" in user_content
        assert "max_retries + 1" in user_content
        assert "AUTHORITATIVE" in user_content
        assert "Do NOT invert" in user_content

    def test_rca_prompt_shows_none_when_no_edit_diffs(self) -> None:
        mock_client = _make_mock_client("## Root Cause\nfoo\n")

        asyncio.run(_build_editor_content(
            client=mock_client,
            file_path="RCA.md",
            current_source="",
            task_id="task2_retry_logic",
            alert_message="",
            known_logs={},
            known_files={},
            edited_files=set(),
            history=[],
            replay_evidence="",
            edit_diffs=None,
        ))

        call_kwargs = mock_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        assert "Change History" in user_content
        assert "None" in user_content

    def test_non_rca_file_omits_authoritative_instruction(self) -> None:
        mock_client = _make_mock_client("def ok(): pass")

        asyncio.run(_build_editor_content(
            client=mock_client,
            file_path="app/retry_handler.py",
            current_source="def old(): pass",
            task_id="task2_retry_logic",
            alert_message="",
            known_logs={},
            known_files={},
            edited_files=set(),
            history=[],
            replay_evidence="",
            edit_diffs=None,
        ))

        call_kwargs = mock_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        # AUTHORITATIVE instruction only appears for RCA files
        assert "AUTHORITATIVE" not in user_content

    def test_rca_edited_source_labeled_as_current_fixed_version(self) -> None:
        """The edited source snapshot should be labeled 'current/fixed version'."""
        mock_client = _make_mock_client("## Root Cause\nfoo\n")

        asyncio.run(_build_editor_content(
            client=mock_client,
            file_path="RCA.md",
            current_source="",
            task_id="task2_retry_logic",
            alert_message="",
            known_logs={},
            known_files={"app/retry_handler.py": "range(max_retries + 1)"},
            edited_files={"app/retry_handler.py"},
            history=[],
            replay_evidence="",
            edit_diffs={},
        ))

        call_kwargs = mock_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        assert "current/fixed version" in user_content


# ---------------------------------------------------------------------------
# _apply_hard_guards — submit gating after failed replay
# ---------------------------------------------------------------------------

class TestSubmitGuardAfterFailedReplay:
    def test_submit_forced_to_replay_when_code_edited_but_replay_not_passing(self) -> None:
        p = PersistentState()
        p.last_code_edit_step = 8
        p.last_rca_edit_step = 8
        p.replay_passed = False
        derived = _make_derived(
            has_code_edit=True,
            has_rca=True,
            has_replay_attempt=True,
            has_replay_pass=False,
            has_replay_after_latest_code_edit=True,
            should_replay_after_latest_code_edit=False,
            remaining_steps=4,
        )
        action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

        guarded = _apply_hard_guards(
            action=action,
            derived=derived,
            persistent=p,
            task_id="task3_cascading_failure",
            replay_name="cascading_timeout_budget",
        )

        assert guarded["tool"] == "replay"
        assert guarded["command"] == "cascading_timeout_budget"

    def test_submit_allowed_on_last_remaining_step(self) -> None:
        p = PersistentState()
        p.last_code_edit_step = 8
        p.last_rca_edit_step = 8
        p.replay_passed = False
        derived = _make_derived(
            has_code_edit=True,
            has_rca=True,
            has_replay_attempt=True,
            has_replay_pass=False,
            has_replay_after_latest_code_edit=True,
            should_replay_after_latest_code_edit=False,
            remaining_steps=1,
        )
        action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

        guarded = _apply_hard_guards(
            action=action,
            derived=derived,
            persistent=p,
            task_id="task3_cascading_failure",
            replay_name="cascading_timeout_budget",
        )

        assert guarded["tool"] == "submit"


class TestSubmitGuardBeforeFirstCodeEdit:
    def test_submit_forced_to_editor_before_first_code_edit(self) -> None:
        p = PersistentState()
        p.seen_cats.add("app/main.py")
        derived = _make_derived(
            has_code_edit=False,
            has_rca=True,
            has_replay_attempt=False,
            has_replay_pass=False,
            has_replay_after_latest_code_edit=True,
            remaining_steps=6,
            unread_candidate_files=["app/main.py", "logs/error.log"],
        )
        action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

        guarded = _apply_hard_guards(
            action=action,
            derived=derived,
            persistent=p,
            task_id="task1_wrong_status",
            replay_name="create_item_contract",
        )

        assert guarded["tool"] == "editor"
        assert guarded["file_path"].endswith(".py")

    def test_submit_allowed_on_last_step_even_before_first_edit(self) -> None:
        p = PersistentState()
        derived = _make_derived(
            has_code_edit=False,
            has_rca=True,
            has_replay_attempt=False,
            has_replay_pass=False,
            has_replay_after_latest_code_edit=True,
            remaining_steps=1,
            unread_candidate_files=["app/main.py"],
        )
        action = {"tool": "submit", "command": "", "file_path": "", "file_content": ""}

        guarded = _apply_hard_guards(
            action=action,
            derived=derived,
            persistent=p,
            task_id="task1_wrong_status",
            replay_name="create_item_contract",
        )

        assert guarded["tool"] == "submit"
