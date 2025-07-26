"""Unit tests for the Callback base class."""

from dspy_optimizer.callback.base import Callback


class DummyCallback(Callback):
    """A concrete implementation of the Callback for testing purposes."""

    pass


def test_callback_interface_methods():
    """
    Tests that all methods of the Callback interface can be called without error.
    This ensures the interface is well-defined and usable.
    """
    # Arrange
    callback = DummyCallback()
    dummy_state = {"key": "value"}

    # Act & Assert
    # Simply call each method to ensure it exists and runs without raising an exception.
    try:
        callback.on_run_start(dummy_state)
        callback.on_refinement_start(dummy_state)
        callback.on_refinement_end(dummy_state)
        callback.on_validation_start(dummy_state)
        callback.on_validation_end(dummy_state)
        callback.on_merge_success(dummy_state)
        callback.on_merge_failure(dummy_state)
        callback.on_run_end(dummy_state)
    except Exception as e:
        assert False, f"Callback method raised an unexpected exception: {e}"
