"""Tests for the MLflowCallback."""

from unittest.mock import MagicMock, patch

import pytest

from dspy_optimizer.callback.mlflow_callback import MLflowCallback
from dspy_optimizer.models import PatchOperation, PromptPatch


@pytest.fixture
def mock_mlflow():
    """Fixture to mock the mlflow module."""
    # We patch the mlflow module within the callback's namespace
    with patch("dspy_optimizer.callback.mlflow_callback.mlflow") as mock:
        # Mock the dspy flavor as an attribute
        mock.dspy = MagicMock()
        yield mock


def test_mlflow_callback_initialization(mock_mlflow):
    """Test that MLflow is configured correctly on initialization."""
    # Arrange
    experiment_name = "test_experiment"
    tracking_uri = "http://localhost:5000"

    # Act
    MLflowCallback(experiment_name=experiment_name, tracking_uri=tracking_uri)

    # Assert
    mock_mlflow.set_tracking_uri.assert_called_once_with(tracking_uri)
    mock_mlflow.set_experiment.assert_called_once_with(experiment_name)


def test_mlflow_import_error():
    """Test that an ImportError is raised if mlflow is not installed."""
    with patch("dspy_optimizer.callback.mlflow_callback.mlflow", None):
        with pytest.raises(ImportError, match="MLflow is not installed"):
            MLflowCallback(experiment_name="test")


def test_on_run_start(mock_mlflow):
    """Test logging on the start of a run."""
    # Arrange
    callback = MLflowCallback(experiment_name="test")
    state = {
        "initial_prompt": "Initial",
        "merger_strategy": "block",
        "validation_strategy": "full",
        "scorer": "exact_match",
    }

    # Act
    callback.on_run_start(state)

    # Assert
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once_with(
        {
            "initial_prompt": "Initial",
            "merger_strategy": "block",
            "validation_strategy": "full",
            "scorer": "exact_match",
        }
    )


def test_on_validation_end(mock_mlflow):
    """Test logging of validation metrics."""
    # Arrange
    callback = MLflowCallback(experiment_name="test")
    state = {"score": 0.85, "is_valid": True, "total_evaluations": 5}

    # Act
    callback.on_validation_end(state)

    # Assert
    mock_mlflow.log_metrics.assert_called_once_with(
        {"validation_score": 0.85, "is_valid": 1}, step=5
    )


def test_on_merge_success(mock_mlflow):
    """Test logging of artifacts on merge success."""
    # Arrange
    callback = MLflowCallback(experiment_name="test")
    patch_obj = PromptPatch(
        operation=PatchOperation.REPLACE, target_block="OLD_BLOCK", content="NEW_CONTENT"
    )
    state = {
        "patch": patch_obj,
        "new_prompt": "This is the new prompt.",
        "total_evaluations": 1,
    }

    # Act
    callback.on_merge_success(state)

    # Assert
    mock_mlflow.log_params.assert_called_once_with(
        {
            "step_1_patch_op": "replace",
            "step_1_patch_target": "OLD_BLOCK",
        }
    )
    assert mock_mlflow.log_text.call_count == 2
    mock_mlflow.log_text.assert_any_call("NEW_CONTENT", artifact_file="patches/step_1.txt")
    mock_mlflow.log_text.assert_any_call(
        "This is the new prompt.", artifact_file="prompts/prompt_v1.txt"
    )


def test_on_run_end(mock_mlflow):
    """Test logging the final model and ending the run."""
    # Arrange
    callback = MLflowCallback(experiment_name="test")
    mock_optimizer = MagicMock()
    state = {"optimizer": mock_optimizer}
    # Simulate an active run
    callback.run = MagicMock()

    # Act
    callback.on_run_end(state)

    # Assert
    mock_mlflow.dspy.log_model.assert_called_once_with(
        dspy_model=mock_optimizer,
        artifact_path="optimized_model",
    )
    mock_mlflow.end_run.assert_called_once()
