"""Deterministic grading for SRE incident response tasks."""

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional

import asyncio
from ..providers.sandbox_executor import SandboxExecutor
from ..tasks.config import TaskConfig


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
        weights = task_config.grading_weights
        total_score = 0.0

        # 1. File Change Scoring (Diff original fixture vs workspace)
        file_change_score = self._check_file_changes(
            task_config.expected_fix_files, fixture_path, workspace_path
        )
        total_score += file_change_score * weights.get("file_change", 0.0)

        # 2. Test Success Scoring (Run pytest)
        test_score = await self._check_tests(workspace_path)
        total_score += test_score * weights.get("tests_pass", 0.0)

        # 3. Correctness Scoring (Regex match in source code)
        regex_score = self._check_regex(task_config.regex_checks, workspace_path)
        total_score += regex_score * weights.get("regex_match", 0.0)

        # Normalize to 0.0-1.0
        return max(0.0, min(1.0, total_score))

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
            
            if not orig.exists() or not new.exists():
                continue
                
            # If the files are different, the agent modified it
            with open(orig, "r") as f_orig, open(new, "r") as f_new:
                if f_orig.read() != f_new.read():
                    modified_count += 1
                    
        return modified_count / len(expected_files)

    async def _check_tests(self, workspace_path: Path) -> float:
        """Run pytest in the workspace."""
        # Use our executor with a 30s timeout for tests
        stdout, stderr, exit_code = await self.executor.execute(
            "pytest -v", workspace_path, timeout=30
        )
        
        # 1.0 if exit code 0 (all pass), 0.0 otherwise
        return 1.0 if exit_code == 0 else 0.0

    def _check_regex(self, checks: List[any], workspace_path: Path) -> float:
        """Verify the logic of the fix using regex."""
        if not checks:
            return 1.0
            
        pass_count = 0
        for check in checks:
            target_file = workspace_path / check.file
            if not target_file.exists():
                continue
                
            with open(target_file, "r") as f:
                content = f.read()
                if re.search(check.pattern, content):
                    pass_count += 1
                    
        return pass_count / len(checks)
