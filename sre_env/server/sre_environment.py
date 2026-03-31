"""Core SRE Environment controller."""

import logging
from pathlib import Path
from typing import Optional, Tuple

from ..models import SREAction, SREObservation, SREState
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

    async def reset(self, task_id: str) -> SREObservation:
        """Initialize a new incident episode.

        1. Setup a fresh workspace from the task fixture.
        2. Create the initial SREState.
        3. Fetch the first observation (alert + file tree).
        """
        task_config = self.registry.get_task(task_id)
        if not task_config:
            return SREObservation(stderr=f"Error: Task {task_id} not found.")

        # Prepare clear workspace
        fixture_path = self.fixtures_dir / task_id
        if not setup_workspace(fixture_path, self.workspace_root):
            return SREObservation(stderr="Error: Could not setup workspace.")

        # Start fresh state
        self.state = SREState(
            task_id=task_id,
            task_name=task_config.name,
            workspace_root=str(self.workspace_root)
        )
        
        # Reset rewarder for new episode
        self.rewarder = SREStepRewarder()

        # Get initial alert information
        alert_data = await self.alert_provider.get_alert(task_id)

        return SREObservation(
            alert_message=alert_data.get("message", ""),
            file_tree=get_file_tree(self.workspace_root),
            reward=0.0,
            done=False
        )

    async def step(self, action: SREAction) -> SREObservation:
        """Handle one agent action.

        Returns:
            Observation: result of the agent's command.
        """
        if not self.state or self.state.done:
            return SREObservation(stderr="Error: No active episode. Call reset() first.", done=True)

        task_config = self.registry.get_task(self.state.task_id)
        if not task_config:
            return SREObservation(stderr="Error: Active task configuration lost.")

        # 1. Update step count
        self.state.step_count += 1
        if self.state.step_count >= self.state.max_steps:
             self.state.done = True

        # 2. Execute tool
        observation = SREObservation(done=self.state.done)
        
        if action.tool == "terminal":
            stdout, stderr, exit_code = await self.executor.execute(
                action.command, self.workspace_root
            )
            observation.stdout = stdout
            observation.stderr = stderr
            observation.exit_code = exit_code

        elif action.tool == "editor":
            # Write to file
            try:
                target_path = self.workspace_root / action.file_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(action.file_content)
                observation.stdout = f"SUCCESS: Wrote content to {action.file_path}"
            except Exception as e:
                observation.stderr = f"FAIL: Error writing file: {str(e)}"

        elif action.tool == "submit":
            # Signalling end of episode
            self.state.done = True
            observation.done = True
            
            # RUN GRADING!
            fixture_path = self.fixtures_dir / self.state.task_id
            score = await self.grader.grade_episode(
                task_config, fixture_path, self.workspace_root
            )
            observation.score = score
            observation.stdout = f"Episode submitted for grading. FINAL SCORE: {score:.2f}"

        # 3. Always refresh file tree
        observation.file_tree = get_file_tree(self.workspace_root)
        
        # 4. Calculate Step Reward
        observation.reward = self.rewarder.calculate_reward(action)
        self.state.cumulative_reward += observation.reward

        return observation
