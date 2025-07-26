"""Tests for the HistoryCallback."""

from dspy_optimizer.callback.history_callback import HistoryCallback


def test_history_callback_logging():
    """Test that HistoryCallback correctly logs events and state."""
    # Arrange
    callback = HistoryCallback()
    initial_state = {"prompt": "Initial Prompt"}
    refinement_state = {"sample": "sample1", "patch": "patch1"}
    validation_state = {"is_valid": True}
    merge_state = {"new_prompt": "Updated Prompt"}
    final_state = {"final_prompt": "Final Prompt"}

    # Act
    callback.on_run_start(initial_state)
    callback.on_refinement_start(refinement_state)
    callback.on_refinement_end(refinement_state)
    callback.on_validation_start(validation_state)
    callback.on_validation_end(validation_state)
    callback.on_merge_success(merge_state)
    callback.on_run_end(final_state)

    # Assert
    assert len(callback.history) == 7

    assert callback.history[0] == {"event": "run_start", **initial_state}
    assert callback.history[1] == {"event": "refinement_start", **refinement_state}
    assert callback.history[2] == {"event": "refinement_end", **refinement_state}
    assert callback.history[3] == {"event": "validation_start", **validation_state}
    assert callback.history[4] == {"event": "validation_end", **validation_state}
    assert callback.history[5] == {"event": "merge_success", **merge_state}
    assert callback.history[6] == {"event": "run_end", **final_state}


def test_history_callback_initialization():
    """Test that the HistoryCallback initializes with an empty history."""
    # Arrange & Act
    callback = HistoryCallback()

    # Assert
    assert callback.history == []
