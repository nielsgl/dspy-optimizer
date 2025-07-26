"""A callback that logs the history of the optimization process."""

from typing import Any

from .base import Callback


class HistoryCallback(Callback):
    """A callback that records the history of the optimization process.

    This callback stores a chronological log of all events and state changes
    that occur during an optimization run. The history is stored in a list
    of dictionaries, where each dictionary represents a snapshot of the state
    at a specific event.

    Attributes:
        history: A list of dictionaries, where each dictionary is a log entry.
    """

    def __init__(self):
        """Initializes the HistoryCallback."""
        self.history: list[dict[str, Any]] = []

    def _log_event(self, event_name: str, state: dict[str, Any]):
        """Logs an event to the history."""
        log_entry = {"event": event_name, **state}
        self.history.append(log_entry)

    def on_run_start(self, state: dict[str, Any]):
        """Called at the beginning of an optimization run."""
        self._log_event("run_start", state)

    def on_refinement_start(self, state: dict[str, Any]):
        """Called at the beginning of a refinement attempt for a single sample."""
        self._log_event("refinement_start", state)

    def on_refinement_end(self, state: dict[str, Any]):
        """Called at the end of a refinement attempt."""
        self._log_event("refinement_end", state)

    def on_validation_start(self, state: dict[str, Any]):
        """Called at the beginning of a validation run."""
        self._log_event("validation_start", state)

    def on_validation_end(self, state: dict[str, Any]):
        """Called at the end of a validation run."""
        self._log_event("validation_end", state)

    def on_merge_success(self, state: dict[str, Any]):
        """Called after a prompt has been successfully validated and merged."""
        self._log_event("merge_success", state)

    def on_merge_failure(self, state: dict[str, Any]):
        """Called after a prompt has failed validation and was not merged."""
        self._log_event("merge_failure", state)

    def on_run_end(self, state: dict[str, Any]):
        """Called at the end of an optimization run."""
        self._log_event("run_end", state)
