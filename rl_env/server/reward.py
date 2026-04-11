"""Outcome-aware reward shaping for SRE incident response."""

import ast
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Set

try:
    from ..models import SREAction, SREObservation
except ImportError:
    from models import SREAction, SREObservation


class SREStepRewarder:
    """Calculate conservative, less gameable partial rewards."""

    DEFAULT_BASE_STEP_PENALTY = -0.005
    DEFAULT_W_COMPILE = 0.2
    DEFAULT_W_TEST = 1.0
    DEFAULT_W_COMPLEXITY = 0.05
    DEFAULT_HEURISTICS_SCALE = 0.1
    DEFAULT_HEURISTICS_CAP = 0.05

    def __init__(self):
        self.initial_files: Set[str] = set()
        self.seen_logs: Set[str] = set()
        self.seen_source: Set[str] = set()
        self.rewarded_edits: Set[str] = set()
        self.last_cat_stdout_by_target: dict[str, str] = {}
        self.has_relevant_code_edit: bool = False
        self.edit_since_last_replay: bool = False
        self.replay_success_since_last_edit: bool = False
        self.last_replay_name: str | None = None
        self.base_step_penalty: float = self.DEFAULT_BASE_STEP_PENALTY
        self.w_compile: float = self.DEFAULT_W_COMPILE
        self.w_test: float = self.DEFAULT_W_TEST
        self.w_complexity: float = self.DEFAULT_W_COMPLEXITY
        self.heuristics_scale: float = self.DEFAULT_HEURISTICS_SCALE
        self.heuristics_cap: float = self.DEFAULT_HEURISTICS_CAP
        self.best_replay_ratio: float = 0.0
        self.compile_validity_by_file: dict[str, bool] = {}
        self.baseline_complexity_by_file: dict[str, float] = {}
        self.last_breakdown: dict[str, Any] = {}
        self._logger = logging.getLogger("rl_env.reward")

    def calculate_reward(
        self,
        action: SREAction,
        observation: SREObservation,
        expected_fix_files: Iterable[str],
        reward_policy: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Assign conservative rewards for meaningful progress."""
        self._apply_reward_policy(reward_policy)
        reward = self.base_step_penalty
        expected_files = {path.replace("\\", "/") for path in expected_fix_files}
        compile_component = 0.0
        replay_test_component = 0.0
        complexity_component = 0.0
        heuristic_raw = 0.0
        compile_valid_now: Optional[bool] = None
        compile_prev_valid: Optional[bool] = None
        compile_error_type: Optional[str] = None

        if action.tool == "terminal":
            heuristic_raw += self._terminal_heuristic_reward(action, observation)

        elif action.tool == "editor":
            normalized_path = action.file_path.replace("\\", "/")
            heuristic_raw += self._editor_heuristic_reward(action, normalized_path, expected_files)

            if (
                normalized_path in expected_files
                and normalized_path.endswith(".py")
                and action.file_content.strip()
            ):
                compile_prev_valid = self.compile_validity_by_file.get(normalized_path)
                compile_valid_now, compile_error_type = self._safe_parse_python(action.file_content)
                self.compile_validity_by_file[normalized_path] = compile_valid_now
                if compile_prev_valid is not None:
                    if (not compile_prev_valid) and compile_valid_now:
                        compile_component = self.w_compile * 1.0
                    elif compile_prev_valid and (not compile_valid_now):
                        compile_component = self.w_compile * -0.5
                complexity_signal = self._complexity_growth_signal(
                    normalized_path, action.file_content
                )
                complexity_component = self.w_complexity * complexity_signal

        elif action.tool == "replay":
            heuristic_raw += self._replay_heuristic_reward(action, observation)
            replay_ratio = self._extract_replay_ratio(observation.stdout)
            if replay_ratio is not None:
                delta = max(0.0, replay_ratio - self.best_replay_ratio)
                replay_test_component = self.w_test * delta
                self.best_replay_ratio = max(self.best_replay_ratio, replay_ratio)

        elif action.tool == "submit":
            self.last_breakdown = {
                "total_step_reward": 0.0,
                "base_penalty": 0.0,
                "compile_component": 0.0,
                "replay_test_component": 0.0,
                "complexity_component": 0.0,
                "heuristic_component": 0.0,
                "compile_valid_now": None,
                "compile_prev_valid": None,
                "compile_error_type": None,
            }
            return 0.0

        heuristic_component = self._clamp(
            self.heuristics_scale * heuristic_raw,
            -self.heuristics_cap,
            self.heuristics_cap,
        )
        reward = (
            self.base_step_penalty
            + compile_component
            + replay_test_component
            - complexity_component
            + heuristic_component
        )
        self.last_breakdown = {
            "total_step_reward": reward,
            "base_penalty": self.base_step_penalty,
            "compile_component": compile_component,
            "replay_test_component": replay_test_component,
            "complexity_component": complexity_component,
            "heuristic_component": heuristic_component,
            "compile_valid_now": compile_valid_now,
            "compile_prev_valid": compile_prev_valid,
            "compile_error_type": compile_error_type,
        }
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

    def seed_initial_state(
        self,
        file_tree: Iterable[str],
        workspace_root: Path,
        expected_fix_files: Iterable[str],
    ) -> None:
        """Seed initial files plus compile/complexity baselines for expected fix files."""
        self.seed_initial_files(file_tree)
        self.compile_validity_by_file = {}
        self.baseline_complexity_by_file = {}
        for path in expected_fix_files:
            normalized = path.replace("\\", "/")
            if not normalized.endswith(".py"):
                continue
            target = workspace_root / normalized
            if not target.exists():
                continue
            try:
                source = target.read_text(encoding="utf-8")
            except Exception:
                continue
            is_valid, _ = self._safe_parse_python(source)
            self.compile_validity_by_file[normalized] = is_valid
            complexity = self._source_complexity(source)
            if complexity is not None:
                self.baseline_complexity_by_file[normalized] = complexity

    def _terminal_heuristic_reward(self, action: SREAction, observation: SREObservation) -> float:
        reward = 0.0
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
                reward -= 0.01
            self.last_cat_stdout_by_target[target] = observation.stdout
            if target.startswith("logs/") and target.endswith(".log") and target not in self.seen_logs:
                reward += 0.05
                self.seen_logs.add(target)
            elif self._is_relevant_source_file(target) and target not in self.seen_source:
                reward += 0.05
                self.seen_source.add(target)
        return reward

    def _editor_heuristic_reward(
        self,
        action: SREAction,
        normalized_path: str,
        expected_files: set[str],
    ) -> float:
        reward = 0.0
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
        if (
            normalized_path in expected_files
            and normalized_path != "RCA.md"
            and action.file_content.strip()
        ):
            self.has_relevant_code_edit = True
            self.edit_since_last_replay = True
            self.replay_success_since_last_edit = False
        self.last_cat_stdout_by_target.pop(normalized_path, None)
        return reward

    def _replay_heuristic_reward(self, action: SREAction, observation: SREObservation) -> float:
        reward = 0.0
        replay_name = " ".join(action.command.lower().split())
        if self.has_relevant_code_edit and replay_name:
            if self.edit_since_last_replay and replay_name != self.last_replay_name:
                reward += 0.01
            if replay_name == self.last_replay_name and not self.edit_since_last_replay:
                reward -= 0.01
        if (
            self.has_relevant_code_edit
            and "contract_ok=true" in observation.stdout.lower()
            and not self.replay_success_since_last_edit
        ):
            reward += 0.02
            self.replay_success_since_last_edit = True
        self.last_replay_name = replay_name or None
        self.edit_since_last_replay = False
        return reward

    def _safe_parse_python(self, source: str) -> tuple[bool, Optional[str]]:
        """Safely parse Python code and never raise."""
        try:
            ast.parse(source)
            return True, None
        except SyntaxError:
            return False, "SyntaxError"
        except Exception:
            self._logger.warning("Unexpected parse failure while scoring compile component.", exc_info=True)
            return False, "ParseError"

    def _source_complexity(self, source: str) -> Optional[float]:
        try:
            tree = ast.parse(source)
        except Exception:
            return None
        ast_nodes = sum(1 for _ in ast.walk(tree))
        non_comment_loc = sum(
            1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")
        )
        return float(ast_nodes + non_comment_loc)

    def _complexity_growth_signal(self, path: str, source: str) -> float:
        current = self._source_complexity(source)
        if current is None:
            return 0.0
        baseline = self.baseline_complexity_by_file.get(path)
        if baseline is None:
            self.baseline_complexity_by_file[path] = current
            return 0.0
        growth = max(0.0, (current - baseline) / max(1.0, baseline))
        return self._clamp(growth, 0.0, 1.0)

    def _extract_replay_ratio(self, stdout: str) -> Optional[float]:
        text = stdout.lower()
        if "contract_ok=true" in text:
            return 1.0
        if "contract_ok=false" in text:
            return 0.0
        return None

    def _apply_reward_policy(self, reward_policy: Optional[Mapping[str, float]]) -> None:
        policy = reward_policy or {}
        self.base_step_penalty = float(policy.get("base_penalty", self.DEFAULT_BASE_STEP_PENALTY))
        self.w_compile = float(policy.get("w_compile", self.DEFAULT_W_COMPILE))
        self.w_test = float(policy.get("w_test", self.DEFAULT_W_TEST))
        self.w_complexity = float(policy.get("w_complexity", self.DEFAULT_W_COMPLEXITY))
        self.heuristics_scale = float(policy.get("heuristics_scale", self.DEFAULT_HEURISTICS_SCALE))
        self.heuristics_cap = float(policy.get("heuristics_cap", self.DEFAULT_HEURISTICS_CAP))

    def get_last_breakdown(self) -> dict[str, Any]:
        """Return the latest component-level step breakdown."""
        return dict(self.last_breakdown)

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
