"""Outcome-aware reward shaping for SRE incident response."""

from pathlib import PurePosixPath
from typing import Iterable, Set

from ..models import SREAction, SREObservation


class SREStepRewarder:
    """Calculate conservative, less gameable partial rewards."""

    def __init__(self):
        self.initial_files: Set[str] = set()
        self.seen_logs: Set[str] = set()
        self.seen_source: Set[str] = set()
        self.seen_test_runs: Set[str] = set()
        self.rewarded_edits: Set[str] = set()
        self.seen_replays: Set[str] = set()
        self.last_cat_stdout_by_target: dict[str, str] = {}

    def calculate_reward(
        self,
        action: SREAction,
        observation: SREObservation,
        expected_fix_files: Iterable[str],
    ) -> float:
        """Assign conservative rewards for meaningful progress.

        The final task score should remain the dominant signal. Step rewards are
        intentionally small and task-aware to discourage reward hacking.
        """
        reward = -0.01
        expected_files = {path.replace("\\", "/") for path in expected_fix_files}

        if action.tool == "terminal":
            cmd = action.command.lower()
            normalized_cmd = " ".join(cmd.split())

            if any(
                blocked in normalized_cmd
                for blocked in ("rm -rf", "shutdown", "reboot", "mkfs", "dd ", "sudo ", "killall")
            ):
                reward -= 0.25

            if observation.exit_code == 0 and cmd.startswith("cat "):
                target = cmd.split("cat ", maxsplit=1)[-1].strip().replace("\\", "/")
                previous_stdout = self.last_cat_stdout_by_target.get(target)
                if previous_stdout is not None and previous_stdout == observation.stdout:
                    reward -= 0.02
                self.last_cat_stdout_by_target[target] = observation.stdout
                if target.startswith("logs/") and target.endswith(".log") and target not in self.seen_logs:
                    reward += 0.05
                    self.seen_logs.add(target)
                elif self._is_relevant_source_file(target) and target not in self.seen_source:
                    reward += 0.05
                    self.seen_source.add(target)

            if "pytest" in cmd:
                if normalized_cmd not in self.seen_test_runs:
                    reward += 0.01
                    self.seen_test_runs.add(normalized_cmd)
                if observation.exit_code == 0:
                    reward += 0.04

        elif action.tool == "editor":
            normalized_path = action.file_path.replace("\\", "/")
            if not action.file_content.strip():
                reward -= 0.10
            elif normalized_path not in expected_files:
                reward -= 0.03
            elif normalized_path not in self.rewarded_edits:
                if normalized_path == "RCA.md":
                    reward += 0.04 if len(action.file_content.strip()) >= 120 else -0.02
                else:
                    reward += 0.03
                self.rewarded_edits.add(normalized_path)
            # Allow one immediate validation read after edits.
            self.last_cat_stdout_by_target.pop(normalized_path, None)

        elif action.tool == "replay":
            replay_name = " ".join(action.command.lower().split())
            if replay_name:
                if replay_name in self.seen_replays:
                    reward -= 0.01
                else:
                    reward += 0.02
                    self.seen_replays.add(replay_name)
            if "contract_ok=true" in observation.stdout.lower():
                reward += 0.01

        elif action.tool == "submit":
            return 0.0

        return reward

    def _is_relevant_source_file(self, target: str) -> bool:
        """Return whether a read target looks like task source code."""
        path = PurePosixPath(target)
        if path.suffix != ".py":
            return False
        if target not in self.initial_files:
            return False
        return path.parts[:1] in {("app",), ("service_a",), ("service_b",)}

    def seed_initial_files(self, file_tree: Iterable[str]) -> None:
        """Register the files that existed at the start of the episode."""
        self.initial_files = {path.replace("\\", "/") for path in file_tree}
