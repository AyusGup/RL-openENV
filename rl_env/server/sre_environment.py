"""Core SRE Environment controller (sync implementation)."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, SREObservation, SREState
    from ..providers.sandbox_executor import SandboxExecutor
    from ..providers.static_alert import StaticAlertProvider
    from ..tasks.registry import TaskRegistry
    from ..utils.file_ops import get_file_tree, setup_workspace
    from .grader import SREGrader
    from .replay import ReplayExecutor
    from .reward import SREStepRewarder
except ImportError:
    from models import SREAction, SREObservation, SREState
    from providers.sandbox_executor import SandboxExecutor
    from providers.static_alert import StaticAlertProvider
    from tasks.registry import TaskRegistry
    from utils.file_ops import get_file_tree, setup_workspace
    from server.grader import SREGrader
    from server.replay import ReplayExecutor
    from server.reward import SREStepRewarder


class SREEnvironment(Environment):
    """SRE incident-response environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, fixtures_dir: Path, workspace_root: Path):
        self.fixtures_dir = fixtures_dir
        self.workspace_root = workspace_root
        self.registry = TaskRegistry(fixtures_dir)
        self.executor = SandboxExecutor()
        self.alert_provider = StaticAlertProvider(fixtures_dir)
        self.grader = SREGrader(self.executor)
        self.replay_executor = ReplayExecutor()
        self.rewarder = SREStepRewarder()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_initialized = False
        self._episode_done = False
        self._task_id = ""
        self._task_name = ""
        self._max_steps = 50
        self._cumulative_reward = 0.0
        self.logger = logging.getLogger("rl_env")

    def reset(self, task_id: str | None = None) -> SREObservation:
        resolved_task_id = task_id or self.registry.default_task_id()
        if not resolved_task_id:
            return SREObservation(stderr="Error: No tasks are configured.")

        task_config = self.registry.get_task(resolved_task_id)
        if not task_config:
            return SREObservation(stderr=f"Error: Task {resolved_task_id} not found.")

        fixture_path = self.fixtures_dir / resolved_task_id
        if not setup_workspace(
            fixture_path,
            self.workspace_root,
            extra_ignore_patterns=("tests",),
        ):
            return SREObservation(stderr="Error: Could not setup workspace.")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_initialized = True
        self._episode_done = False
        self._task_id = resolved_task_id
        self._task_name = task_config.name
        self._max_steps = task_config.max_steps
        self._cumulative_reward = 0.0
        self.rewarder = SREStepRewarder()

        alert_data = asyncio.run(self.alert_provider.get_alert(resolved_task_id))
        file_tree = get_file_tree(self.workspace_root)
        self.rewarder.seed_initial_files(file_tree)

        return SREObservation(
            alert_message=alert_data.get("message", ""),
            file_tree=file_tree,
            reward=0.0,
            done=False,
        )

    def step(self, action: SREAction) -> SREObservation:
        if not self._episode_initialized or self._episode_done:
            return self._error_observation("Error: No active episode. Call reset() first.")

        task_config = self.registry.get_task(self._task_id)
        if not task_config:
            return self._error_observation("Error: Active task configuration lost.")

        self._state.step_count += 1
        reached_step_limit = self._state.step_count >= self._max_steps

        observation = SREObservation()
        info_message = ""
        last_action_error: Optional[str] = None
        final_score: Optional[float] = None

        if action.tool == "terminal":
            observation, last_action_error = self._run_terminal_action(action)
        elif action.tool == "editor":
            observation, info_message, last_action_error = self._run_editor_action(action)
        elif action.tool == "replay":
            observation, info_message, last_action_error = self._run_replay_action(action)
        elif action.tool == "submit":
            self._episode_done = True
            final_score = self._grade_current_workspace(task_config)
            observation.stdout = f"Episode submitted for grading. FINAL SCORE: {final_score:.2f}"
            info_message = observation.stdout
        else:
            observation.stderr = f"Error: Unsupported tool {action.tool}"
            last_action_error = observation.stderr

        if reached_step_limit and action.tool != "submit":
            self._episode_done = True
            final_score = self._grade_current_workspace(task_config)
            info_message = (
                f"Step budget exhausted at {self._state.step_count}/{self._max_steps}. "
                f"Workspace auto-graded with FINAL SCORE: {final_score:.2f}"
            )

        observation.file_tree = get_file_tree(self.workspace_root)
        reward_value = self._compute_reward(
            action,
            observation,
            task_config.expected_fix_files,
            final_score,
        )
        self._cumulative_reward += reward_value

        observation.reward = reward_value
        observation.done = self._episode_done
        observation.metadata = {
            "score": final_score,
            "message": info_message,
            "last_action_error": last_action_error,
        }
        return observation

    @property
    def state(self) -> State:
        return self._state

    def get_api_state(self) -> Optional[SREState]:
        if not self._episode_initialized:
            return None
        return SREState(
            episode_id=self._state.episode_id,
            task_id=self._task_id,
            task_name=self._task_name,
            step_count=self._state.step_count,
            max_steps=self._max_steps,
            cumulative_reward=self._cumulative_reward,
            done=self._episode_done,
            workspace_root=str(self.workspace_root),
        )

    def get_internal_state(self) -> dict:
        if not self._episode_initialized:
            return {"state": None}
        return {
            "episode_id": self._state.episode_id,
            "task_id": self._task_id,
            "task_name": self._task_name,
            "step_count": self._state.step_count,
            "max_steps": self._max_steps,
            "cumulative_reward": self._cumulative_reward,
            "done": self._episode_done,
            "workspace_root": str(self.workspace_root),
        }

    def _run_terminal_action(self, action: SREAction) -> Tuple[SREObservation, Optional[str]]:
        timeout = 30 if "pytest" in action.command.lower() else 10
        stdout, stderr, exit_code = asyncio.run(
            self.executor.execute(action.command, self.workspace_root, timeout=timeout)
        )
        observation = SREObservation(stdout=stdout, stderr=stderr, exit_code=exit_code)
        return observation, (stderr or None)

    def _run_editor_action(self, action: SREAction) -> Tuple[SREObservation, str, Optional[str]]:
        observation = SREObservation()
        info_message = ""
        last_action_error: Optional[str] = None
        try:
            target_path = (self.workspace_root / action.file_path).resolve()
            workspace_root = self.workspace_root.resolve()
            if workspace_root not in target_path.parents and target_path != workspace_root:
                raise ValueError("Editor target must stay inside workspace root.")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(action.file_content)
            observation.stdout = f"SUCCESS: Wrote content to {action.file_path}"
            info_message = observation.stdout
        except Exception as exc:  # noqa: BLE001
            observation.stderr = f"FAIL: Error writing file: {str(exc)}"
            last_action_error = observation.stderr
        return observation, info_message, last_action_error

    def _run_replay_action(self, action: SREAction) -> Tuple[SREObservation, str, Optional[str]]:
        observation = SREObservation()
        info_message = ""
        last_action_error: Optional[str] = None
        replay_name = action.command.strip()
        if not replay_name:
            observation.stderr = "Error: Replay action requires command=<replay_name>."
            return observation, info_message, observation.stderr

        try:
            assert self._episode_initialized
            replay_result = self.replay_executor.run(
                task_id=self._task_id,
                replay_name=replay_name,
                workspace_root=self.workspace_root,
            )
            observation.stdout = replay_result.evidence_log
            observation.exit_code = 0 if replay_result.success else 1
            info_message = (
                f"Replay '{replay_result.replay_name}' completed with "
                f"status={replay_result.status_code} "
                f"success={str(replay_result.success).lower()}."
            )
        except Exception as exc:  # noqa: BLE001
            observation.stderr = f"Error: {str(exc)}"
            last_action_error = observation.stderr
        return observation, info_message, last_action_error

    def _grade_current_workspace(self, task_config) -> float:
        assert self._episode_initialized
        fixture_path = self.fixtures_dir / self._task_id
        return asyncio.run(self.grader.grade_episode(task_config, fixture_path, self.workspace_root))

    def _compute_reward(
        self,
        action: SREAction,
        observation: SREObservation,
        expected_fix_files: list[str],
        final_score: Optional[float],
    ) -> float:
        if final_score is not None:
            return final_score
        if action.tool == "submit":
            return 0.0
        return self.rewarder.calculate_reward(action, observation, expected_fix_files)

    def _error_observation(self, message: str) -> SREObservation:
        return SREObservation(
            stderr=message,
            reward=0.0,
            done=True,
            metadata={"score": None, "message": "", "last_action_error": message},
        )
