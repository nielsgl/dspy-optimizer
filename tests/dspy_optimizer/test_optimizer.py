"""Integration tests for the PromptOptimizer."""

import dspy
import pytest

from dspy_optimizer.optimizer import PromptOptimizer
from dspy_optimizer.strategies.scoring.common import exact_match_scorer


class SimpleSignature(dspy.Signature):
    """A simple signature for testing."""

    input = dspy.InputField()
    output = dspy.OutputField()


def test_optimizer_full_run(mock_llm, monkeypatch):
    """Tests a full run of the PromptOptimizer.

    This is an integration test that checks if the main loop, evaluator,
    refiner, merger, and validator all work together.
    """
    # Arrange
    initial_prompt = "### Instructions\nInitial instructions."
    dataset = [dspy.Example(input="q1", output="correct").with_inputs("input")]

    # Configure the mock LLM to simulate a refinement
    # 1. First call to evaluator will fail.
    # 2. Call to refiner will propose a patch.
    # 3. Subsequent calls to evaluator (in the validator) will pass.
    mock_llm.response_text = '{"reasoning": "reasoning", "output": "wrong"}'

    def mock_refiner_call(*args, **kwargs):
        return dspy.Prediction(
            target_block="### Instructions",
            operation="replace",
            content="Refined instructions.",
        )

    monkeypatch.setattr("dspy_optimizer.refiner.refiner.Refiner.__call__", mock_refiner_call)

    def mock_validator_call(*args, **kwargs):
        # Make the validator pass after the first refinement.
        mock_llm.response_text = '{"reasoning": "reasoning", "output": "correct"}'
        return True

    monkeypatch.setattr(
        "dspy_optimizer.strategies.validation.full.FullValidationStrategy.__call__",
        mock_validator_call,
    )

    optimizer = PromptOptimizer(
        signature=SimpleSignature,
        initial_prompt=initial_prompt,
    )

    # Act
    optimizer.optimize(dataset, scorer=exact_match_scorer)

    # Assert
    assert "Refined instructions." in optimizer.prompt


def test_optimizer_invalid_scorer_name():
    """Test that a ValueError is raised for a non-existent scorer."""
    # Arrange
    optimizer = PromptOptimizer(
        signature=SimpleSignature,
        initial_prompt="Initial prompt",
    )
    dataset = [dspy.Example(input="q1", output="correct").with_inputs("input")]

    # Act & Assert
    with pytest.raises(ValueError, match="Scorer 'non_existent_scorer' not found in registry."):
        optimizer.optimize(dataset, scorer="non_existent_scorer")
