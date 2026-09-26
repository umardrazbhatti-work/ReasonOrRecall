"""Reason or Recall? — package root.

Framework, data, training, inference and report modules are implemented; the
remaining stubs raise NotImplementedError with their contract in the docstring.
The plan and tracker is docs/ROADMAP.md.
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
