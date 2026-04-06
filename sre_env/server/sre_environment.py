"""Core SRE Environment controller."""

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from ..models import SREAction, SREObservation, SREReward, SREState, SREStepInfo, SREStepResult
from ..providers.sandbox_executor import SandboxExecutor
from ..providers.static_alert import StaticAlertProvider
from ..tasks.registry import TaskRegistry
from ..utils.file_ops import get_file_tree, setup_workspace
from .grader import SREGrader
from .reward import SREStepRewarder


class SREEnvironment:
    """Orchestrates incident response episodes.

    Handles:
        - Workspace lifecycle (reset/step).
        - Tool execution via SandboxExecutor.
        - State management and recording observations.
    """

    def __init__(
        self,
        fixtures_dir: Path,
        workspace_root: Path,
    ):
        self.fixtures_dir = fixtures_dir
        self.workspace_root = workspace_root
        self.registry = TaskRegistry(fixtures_dir)
        self.executor = SandboxExecutor()
        self.alert_provider = StaticAlertProvider(fixtures_dir)
        self.grader = SREGrader(self.executor)
        self.rewarder = SREStepRewarder()
        
        # Track active state
        self.state: Optional[SREState] = None
        
        # Simple logging
        self.logger = logging.getLogger("sre_env")

    async def reset(self, task_id: str | None = None) -> SREObservation:
        """Initialize a new incident episode.

        1. Setup a fresh workspace from the task fixture.
        2. Create the initial SREState.
        3. Fetch the first observation (alert + file tree).
        """
        resolved_task_id = task_id or self.registry.default_task_id()
        if not resolved_task_id:
            return SREObservation(stderr="Error: No tasks are configured.")

        task_config = self.registry.get_task(resolved_task_id)
        if not task_config:
            return SREObservation(stderr=f"Error: Task {resolved_task_id} not found.")

        # Prepare clear workspace
        fixture_path = self.fixtures_dir / resolved_task_id
        if not setup_workspace(fixture_path, self.workspace_root):
            return SREObservation(stderr="Error: Could not setup workspace.")

        # Start fresh state
        self.state = SREState(
            episode_id=str(uuid4()),
            task_id=resolved_task_id,
            task_name=task_config.name,
            max_steps=task_config.max_steps,
            workspace_root=str(self.workspace_root),
        )
        
        # Reset rewarder for new episode
        self.rewarder = SREStepRewarder()

        # Get initial alert information
        alert_data = await self.alert_provider.get_alert(resolved_task_id)

        file_tree = get_file_tree(self.workspace_root)
        self.rewarder.seed_initial_files(file_tree)

        return SREObservation(
            alert_message=alert_data.get("message", ""),
            file_tree=file_tree,
        )

    async def step(self, action: SREAction) -> SREStepResult:
        """Handle one agent action.

        Returns:
            SREStepResult: typed result of the agent's command.
        """
        if not self.state or self.state.done:
            return SREStepResult(
                observation=SREObservation(stderr="Error: No active episode. Call reset() first."),
                reward=SREReward(value=0.0),
                done=True,
                info=SREStepInfo(last_action_error="Error: No active episode. Call reset() first."),
            )

        task_config = self.registry.get_task(self.state.task_id)
        if not task_config:
            return SREStepResult(
                observation=SREObservation(stderr="Error: Active task configuration lost."),
                reward=SREReward(value=0.0),
                done=True,
                info=SREStepInfo(last_action_error="Error: Active task configuration lost."),
            )

        # 1. Update step count
        self.state.step_count += 1
        reached_step_limit = self.state.step_count >= self.state.max_steps

        # 2. Execute tool
        observation = SREObservation()
        last_action_error: Optional[str] = None
        info_message = ""

        final_score: Optional[float] = None

        if action.tool == "terminal":
            timeout = 30 if "pytest" in action.command.lower() else 10
            stdout, stderr, exit_code = await self.executor.execute(
                action.command, self.workspace_root, timeout=timeout
            )
            observation.stdout = stdout
            observation.stderr = stderr
            observation.exit_code = exit_code
            last_action_error = stderr or None

        elif action.tool == "editor":
            # Write to file
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
            except Exception as e:
                observation.stderr = f"FAIL: Error writing file: {str(e)}"
                last_action_error = observation.stderr

        elif action.tool == "submit":
            # Signalling end of episode
            self.state.done = True
            
            # RUN GRADING!
            fixture_path = self.fixtures_dir / self.state.task_id
            final_score = await self.grader.grade_episode(
                task_config, fixture_path, self.workspace_root
            )
            observation.stdout = f"Episode submitted for grading. FINAL SCORE: {final_score:.2f}"
            info_message = observation.stdout
        else:
            observation.stderr = f"Error: Unsupported tool {action.tool}"
            last_action_error = observation.stderr

        if reached_step_limit and action.tool != "submit":
            self.state.done = True
            fixture_path = self.fixtures_dir / self.state.task_id
            final_score = await self.grader.grade_episode(
                task_config, fixture_path, self.workspace_root
            )
            info_message = (
                f"Step budget exhausted at {self.state.step_count}/{self.state.max_steps}. "
                f"Workspace auto-graded with FINAL SCORE: {final_score:.2f}"
            )

        # 3. Always refresh file tree
        observation.file_tree = get_file_tree(self.workspace_root)

        # 4. Calculate Step Reward
        reward_value = 0.0
        if final_score is not None:
            reward_value = final_score
        elif action.tool != "submit":
            reward_value = self.rewarder.calculate_reward(
                action,
                observation,
                task_config.expected_fix_files,
            )
        else:
            reward_value = final_score or 0.0
        self.state.cumulative_reward += reward_value

        info = SREStepInfo(
            score=final_score,
            message=info_message,
            last_action_error=last_action_error,
        )
        return SREStepResult(
            observation=observation,
            reward=SREReward(value=reward_value),
            done=self.state.done,
            info=info,
        )
