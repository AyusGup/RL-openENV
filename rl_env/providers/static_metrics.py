"""Provides performance metrics from the task's workspace."""

import json
from pathlib import Path
from typing import Any, Dict

from .base import MetricsProvider


class StaticMetricsProvider(MetricsProvider):
    """Fetches performance metrics from JSON files in the task's fixture folder."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.metrics_dir = workspace_root / "metrics"

    async def get_metrics(self, metric_name: str) -> Dict[str, Any]:
        """Fetch metric data from a JSON file.

        Args:
            metric_name: Name of the metric (e.g., latency, error_rate).

        Returns:
            dict: Metric data.
        """
        # We append .json if it's missing
        if not metric_name.endswith(".json"):
            metric_name += ".json"

        metric_path = self.metrics_dir / metric_name
        if not metric_path.exists():
            return {"error": f"Metric '{metric_name}' not available."}

        try:
            with open(metric_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return {"error": f"Error parsing metric JSON: {str(e)}"}
