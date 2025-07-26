"""Base interface for all Callbacks."""

from abc import ABC
from typing import Any


class Callback(ABC):
    """Abstract base class for a Callback.

    Callbacks can be used to monitor the optimization process, log metrics,
    and even control the flow of the optimization (e.g., early stopping).
    Each method is passed a `state` dictionary containing the current state
    of the optimizer.
    """

    def on_run_start(self, state: dict[str, Any]):
        """Called at the beginning of an optimization run."""
        pass

    def on_refinement_start(self, state: dict[str, Any]):
        """Called at the beginning of a refinement attempt for a single sample."""
        pass

    def on_refinement_end(self, state: dict[str, Any]):
        """Called at the end of a refinement attempt."""
        pass

    def on_validation_start(self, state: dict[str, Any]):
        """Called at the beginning of a validation run."""
        pass

    def on_validation_end(self, state: dict[str, Any]):
        """Called at the end of a validation run."""
        pass

    def on_merge_success(self, state: dict[str, Any]):
        """Called after a prompt has been successfully validated and merged."""
        pass

    def on_merge_failure(self, state: dict[str, Any]):
        """Called after a prompt has failed validation and was not merged."""
        pass

    def on_run_end(self, state: dict[str, Any]):
        """Called at the end of an optimization run."""
        pass
