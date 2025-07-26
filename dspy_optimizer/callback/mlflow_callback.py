"""A callback for logging optimization runs to MLflow."""

from typing import Any

from .base import Callback

# Lazy import of mlflow
try:
    import mlflow
    import mlflow.dspy
    from mlflow.exceptions import MlflowException
except ImportError:
    mlflow = None
    MlflowException = None


class MLflowCallback(Callback):
    """
    A callback to log the entire optimization process to MLflow.

    This callback logs hyperparameters, metrics, and artifacts to an MLflow
    tracking server, providing a comprehensive audit trail. It leverages the
    `mlflow.dspy` flavor to log the final optimized `dspy.Module`.

    If `mlflow` is not installed, this callback will raise an ImportError
    upon initialization.

    Attributes:
        experiment_name: The name of the MLflow experiment to use.
        tracking_uri: The URI of the MLflow tracking server.
        run_name: An optional name for the MLflow run.
        run: The active MLflow run object.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
    ):
        """Initializes the MLflowCallback."""
        if mlflow is None:
            raise ImportError(
                "MLflow is not installed. Please install it with `pip install mlflow` "
                "to use the MLflowCallback."
            )

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.run = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        assert MlflowException is not None  # For static analysis
        try:
            # Check if the experiment exists, otherwise create it
            if not mlflow.get_experiment_by_name(self.experiment_name):
                mlflow.create_experiment(self.experiment_name)
        except MlflowException as e:
            # Handle cases where the tracking URI is not set or invalid
            raise ConnectionError(
                f"Could not connect to MLflow tracking server at {self.tracking_uri}. "
                "Please ensure the server is running and the URI is correct."
            ) from e

        mlflow.set_experiment(self.experiment_name)

    def on_run_start(self, state: dict[str, Any]):
        """Starts an MLflow run and logs initial parameters."""
        assert mlflow is not None  # For static analysis
        self.run = mlflow.start_run(run_name=self.run_name)
        initial_params = {
            "initial_prompt": state.get("initial_prompt"),
            "merger_strategy": state.get("merger_strategy"),
            "validation_strategy": state.get("validation_strategy"),
            "scorer": state.get("scorer"),
        }
        mlflow.log_params({k: v for k, v in initial_params.items() if v is not None})

    def on_validation_end(self, state: dict[str, Any]):
        """Logs validation metrics."""
        assert mlflow is not None  # For static analysis
        metrics = {
            "validation_score": state.get("score"),
            "is_valid": 1 if state.get("is_valid") else 0,
        }
        # MLflow requires a step for time-series metrics
        step = state.get("total_evaluations", 0)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None}, step=step)

    def on_merge_success(self, state: dict[str, Any]):
        """Logs the prompt patch and the new prompt text."""
        assert mlflow is not None  # For static analysis
        patch = state.get("patch")
        if patch:
            patch_params = {
                f"step_{state['total_evaluations']}_patch_op": patch.operation.value,
                f"step_{state['total_evaluations']}_patch_target": patch.target_block,
            }
            mlflow.log_params(patch_params)
            mlflow.log_text(
                patch.content,
                artifact_file=f"patches/step_{state['total_evaluations']}.txt",
            )

        new_prompt = state.get("new_prompt")
        if new_prompt:
            mlflow.log_text(
                new_prompt,
                artifact_file=f"prompts/prompt_v{state['total_evaluations']}.txt",
            )

    def on_run_end(self, state: dict[str, Any]):
        """Logs the final dspy.Module and ends the MLflow run."""
        assert mlflow is not None  # For static analysis
        optimizer = state.get("optimizer")
        if optimizer:
            mlflow.dspy.log_model(
                dspy_model=optimizer,
                artifact_path="optimized_model",
            )
        if self.run:
            mlflow.end_run()
