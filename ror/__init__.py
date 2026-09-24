"""Reason or Recall? — package root.

Framework modules (config, registry, results, logging, utils, sandbox, metrics,
faithfulness, experiment) are implemented. ML modules (data, models, training,
inference) are contract stubs to be implemented — see their docstrings and
IMPLEMENTATION_PLAN.md.
"""
from .config import ExperimentConfig, load_experiment_config
from .registry import Registry, Status
from .results import RunResult, append_result, load_results

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "Registry",
    "Status",
    "RunResult",
    "append_result",
    "load_results",
]

__version__ = "0.1.0"
