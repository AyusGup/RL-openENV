"""Deterministic grading for SRE incident response tasks."""

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

try:
    from ..providers.sandbox_executor import SandboxExecutor
    from ..tasks import RegexCheck
    from ..tasks.config import TaskConfig
except ImportError:
    from providers.sandbox_executor import SandboxExecutor
    from tasks import RegexCheck
    from tasks.config import TaskConfig


class SREGrader:
    """Calculates final scores for SRE episodes."""

    def __init__(self, executor: SandboxExecutor = SandboxExecutor()):
        self.executor = executor

    async def grade_episode(
        self,
        task_config: TaskConfig,
        fixture_path: Path,
        workspace_path: Path,
    ) -> float:
        """Evaluate the final workspace after task submission.

        Returns:
            float: Total score (0.0 - 1.0).
        """
        result = await self.grade_episode_with_breakdown(task_config, fixture_path, workspace_path)
        return float(result["total_score"])

    async def grade_episode_with_breakdown(
        self,
        task_config: TaskConfig,
        fixture_path: Path,
        workspace_path: Path,
    ) -> Dict[str, float]:
        """Evaluate workspace and return normalized total + per-component breakdown."""
        weights = task_config.grading_weights

        file_change_score = self._check_file_changes(
            task_config.expected_fix_files, fixture_path, workspace_path
        )
        test_score = await self._check_tests(task_config.id, fixture_path, workspace_path)
        regex_score = self._check_regex(task_config.regex_checks, workspace_path)

        weighted = (
            file_change_score * weights.get("file_change", 0.0)
            + test_score * weights.get("tests_pass", 0.0)
            + regex_score * weights.get("regex_match", 0.0)
        )
        total_score = max(0.0, min(1.0, weighted))
        return {
            "total_score": total_score,
            "file_change": file_change_score,
            "tests_pass": test_score,
            "regex_match": regex_score,
        }

    def _check_file_changes(
        self, expected_files: List[str], fixture_path: Path, workspace_path: Path
    ) -> float:
        """Compare workspace vs original fixture for modified files."""
        if not expected_files:
            return 1.0

        modified_count = 0
        for rel_path in expected_files:
            orig = fixture_path / rel_path
            new = workspace_path / rel_path

            if orig.exists() and new.exists():
                with open(orig, "r", encoding="utf-8") as f_orig, open(
                    new, "r", encoding="utf-8"
                ) as f_new:
                    if f_orig.read() != f_new.read():
                        modified_count += 1
                continue

            if not orig.exists() and new.exists():
                with open(new, "r", encoding="utf-8") as f_new:
                    if f_new.read().strip():
                        modified_count += 1

        return modified_count / len(expected_files)

    async def _check_tests(self, task_id: str, fixture_path: Path, workspace_path: Path) -> float:
        """Run hidden fixture tests against the workspace code."""
        _ = task_id
        tests_root = fixture_path / "tests"
        if not tests_root.exists():
            return 0.0

        python_cmd = ["py", "-3", "-m", "pytest", "-q", str(tests_root)] if os.name == "nt" else [
            "python",
            "-m",
            "pytest",
            "-q",
            str(tests_root),
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(workspace_path)
            if not existing_pythonpath
            else str(workspace_path) + os.pathsep + existing_pythonpath
        )

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                python_cmd,
                cwd=str(workspace_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            return 0.0
        summary_text = f"{completed.stdout}\n{completed.stderr}"
        passed, failed, errors = self._extract_pytest_counts(summary_text)
        denominator = passed + failed + errors
        if denominator > 0:
            return passed / denominator
        return 1.0 if completed.returncode == 0 else 0.0

    def _extract_pytest_counts(self, output: str) -> tuple[int, int, int]:
        """Parse pytest summary counts from output text."""
        passed = 0
        failed = 0
        errors = 0
        for count_str, label in re.findall(
            r"(\d+)\s+(passed|failed|error|errors|xpassed|xfailed|skipped)",
            output,
            flags=re.IGNORECASE,
        ):
            count = int(count_str)
            key = label.lower()
            if key in {"passed", "xpassed"}:
                passed += count
            elif key == "failed":
                failed += count
            elif key in {"error", "errors"}:
                errors += count
        return passed, failed, errors

    def _check_regex(self, checks: List[RegexCheck], workspace_path: Path) -> float:
        """Verify the logic of the fix using regex."""
        if not checks:
            return 1.0
            
        pass_count = 0
        for check in checks:
            target_file = workspace_path / check.file
            if not target_file.exists():
                continue
                
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Strip comment-only lines only for source files where '#' is
                # a code comment. Markdown RCA headings also start with '#'
                # and must remain visible to regex checks.
                if target_file.suffix == ".py":
                    content = self._strip_comment_only_lines(content)
                if re.search(check.pattern, content):
                    pass_count += 1

        return pass_count / len(checks)

    def _strip_comment_only_lines(self, content: str) -> str:
        """Remove full-line comments so regex checks target executable code."""
        filtered_lines = []
        for line in content.splitlines():
            if line.lstrip().startswith("#"):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)
