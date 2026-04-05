"""Safe file operations for setting up episode workspaces."""

import shutil
from pathlib import Path
from typing import Any, Dict


def setup_workspace(fixture_path: Path, workspace_path: Path) -> bool:
    """Prepare a fresh workspace from a task fixture.

    Args:
        fixture_path: The template directory.
        workspace_path: Where the episode's sandbox will be.

    Returns:
        bool: True if successful.
    """
    try:
        # 1. Clean existing workspace
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
            
        # 2. Re-create workspace root
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. Copy fixture contents (minus config files)
        # We don't want the agent seeing the task_config.json!
        shutil.copytree(
            fixture_path, 
            workspace_path, 
            ignore=shutil.ignore_patterns("task_config.json", "*.pyc", "__pycache__")
        )
        return True
    except (IOError, OSError) as e:
        print(f"Error setting up workspace: {e}")
        return False


def get_file_tree(root: Path) -> list[str]:
    """Recursively list all files in the workspace for the agent to see.

    Returns:
        list[str]: Relative paths of all files.
    """
    if not root.exists():
        return []
        
    return [
        f.relative_to(root).as_posix()
        for f in root.rglob("*")
        if f.is_file() and "__pycache__" not in str(f)
    ]
