"""Core SRE Environment controller."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from sre_env.models import SREAction, SREObservation, SREState
from sre_env.providers.sandbox_executor import SandboxExecutor
from sre_env.providers.static_alert import StaticAlertProvider
from sre_env.server.grader import SREGrader
from sre_env.server.replay import ReplayExecutor
from sre_env.server.reward import SREStepRewarder
from sre_env.tasks.registry import TaskRegistry
from sre_env.utils.file_ops import get_file_tree, setup_workspace


class SREEnvironment(Environment[SREAction, SREObservation, SREState]):
    """Orchestrates incident response episodes."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        fixtures_dir: Path,
        workspace_root: Path,
        default_task_id: str | None = None,
    ):
        self.fixtures_dir = fixtures_dir
        self.workspace_root = workspace_root
        self.registry = TaskRegistry(fixtures_dir)
        self.executor = SandboxExecutor()
        self.alert_provider = StaticAlertProvider(fixtures_dir)
        self.grader = SREGrader(self.executor)
        self.replay_executor = ReplayExecutor()
        self.rewarder = SREStepRewarder()
        self.default_task_id = default_task_id
        self._state: Optional[SREState] = None
        self.logger = logging.getLogger("sre_env")

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose environment metadata for OpenEnv /metadata."""
        task_names = [task.id for task in self.registry.list_tasks()]
        task_text = ", ".join(task_names) if task_names else "none"
        return EnvironmentMetadata(
            name="SRE Incident Response",
            description="Multi-task SRE incident-response benchmark with deterministic grading.",
            version="1.0.0",
            readme_content=f"Available task IDs: {task_text}",
            author="AyusGup",
        )

    @property
    def state(self) -> SREState:
        """Return the current environment state."""
        return self._state or SREState()

    def _run_coroutine_sync(self, coro: Any) -> Any:
        """Run async methods for sync callers outside event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Use reset_async/step_async when inside an event loop.")

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        return self._run_coroutine_sync(
            self.reset_async(seed=seed, episode_id=episode_id, **kwargs)
        )

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        """Initialize a new incident episode."""
        _ = seed
        requested_task_id = kwargs.get("task_id")
        resolved_task_id = (
            requested_task_id
            or self.default_task_id
            or self.registry.default_task_id()
        )
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

        self._state = SREState(
            episode_id=episode_id or str(uuid4()),
            task_id=resolved_task_id,
            task_name=task_config.name,
            step_count=0,
            max_steps=task_config.max_steps,
            cumulative_reward=0.0,
            done=False,
            workspace_root=str(self.workspace_root),
        )
        self.rewarder = SREStepRewarder()

        alert_data = await self.alert_provider.get_alert(resolved_task_id)
        file_tree = get_file_tree(self.workspace_root)
        self.rewarder.seed_initial_files(file_tree)

        return SREObservation(
            alert_message=alert_data.get("message", ""),
            file_tree=file_tree,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: SREAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        return self._run_coroutine_sync(
            self.step_async(action=action, timeout_s=timeout_s, **kwargs)
        )

    async def step_async(
        self,
        action: SREAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SREObservation:
        """Handle one agent action and return an OpenEnv observation."""
        _ = kwargs
        if not self._state or self._state.done:
            return SREObservation(
                stderr="Error: No active episode. Call reset() first.",
                done=True,
                reward=0.0,
                score=None,
                message="",
                last_action_error="Error: No active episode. Call reset() first.",
            )

        task_config = self.registry.get_task(self._state.task_id)
        if not task_config:
            return SREObservation(
                stderr="Error: Active task configuration lost.",
                done=True,
                reward=0.0,
                score=None,
                message="",
                last_action_error="Error: Active task configuration lost.",
            )

        self._state.step_count += 1
        reached_step_limit = self._state.step_count >= self._state.max_steps

        observation = SREObservation()
        last_action_error: Optional[str] = None
        info_message = ""
        final_score: Optional[float] = None

        if action.tool == "terminal":
            timeout = timeout_s if timeout_s is not None else (
                30 if "pytest" in action.command.lower() else 10
            )
            stdout, stderr, exit_code = await self.executor.execute(
                action.command, self.workspace_root, timeout=int(timeout)
            )
            observation.stdout = stdout
            observation.stderr = stderr
            observation.exit_code = exit_code
            last_action_error = stderr or None

        elif action.tool == "editor":
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
            except Exception as exc:
                observation.stderr = f"FAIL: Error writing file: {exc}"
                last_action_error = observation.stderr

        elif action.tool == "replay":
            replay_name = action.command.strip()
            if not replay_name:
                observation.stderr = "Error: Replay action requires command=<replay_name>."
                last_action_error = observation.stderr
            else:
                try:
                    replay_result = self.replay_executor.run(
                        task_id=self._state.task_id,
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
                except Exception as exc:
                    observation.stderr = f"Error: {exc}"
                    last_action_error = observation.stderr

        elif action.tool == "submit":
            self._state.done = True
            fixture_path = self.fixtures_dir / self._state.task_id
            final_score = await self.grader.grade_episode(
                task_config, fixture_path, self.workspace_root
            )
            observation.stdout = f"Episode submitted for grading. FINAL SCORE: {final_score:.2f}"
            info_message = observation.stdout

        else:
            observation.stderr = f"Error: Unsupported tool {action.tool}"
            last_action_error = observation.stderr

        if reached_step_limit and action.tool != "submit":
            self._state.done = True
            fixture_path = self.fixtures_dir / self._state.task_id
            final_score = await self.grader.grade_episode(
                task_config, fixture_path, self.workspace_root
            )
            info_message = (
                f"Step budget exhausted at {self._state.step_count}/{self._state.max_steps}. "
                f"Workspace auto-graded with FINAL SCORE: {final_score:.2f}"
            )

        observation.file_tree = get_file_tree(self.workspace_root)

        if final_score is not None:
            reward_value = final_score
        elif action.tool != "submit":
            reward_value = self.rewarder.calculate_reward(
                action,
                observation,
                task_config.expected_fix_files,
            )
        else:
            reward_value = 0.0

        self._state.cumulative_reward += reward_value
        observation.done = self._state.done
        observation.reward = reward_value
        observation.score = final_score
        observation.message = info_message
        observation.last_action_error = last_action_error
        return observation
