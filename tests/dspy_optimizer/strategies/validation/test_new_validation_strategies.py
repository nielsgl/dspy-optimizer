"""Unit tests for the new validation strategies."""

import dspy
import pytest

from dspy_optimizer.strategies.validation.sample import SampleValidationStrategy
from dspy_optimizer.strategies.validation.single_example import (
    SingleExampleValidationStrategy,
)


def test_single_example_validation_strategy_success(mock_evaluator, mock_scorer):
    """Tests the SingleExampleValidationStrategy for a successful validation."""
    # Arrange
    strategy = SingleExampleValidationStrategy()
    candidate_prompt = "New and improved prompt"
    example = dspy.Example(text="This should now pass.", label="pass").with_inputs("text")

    # Act
    is_valid = strategy(
        candidate_prompt=candidate_prompt,
        evaluator=mock_evaluator,
        scorer=mock_scorer,
        dataset=[example],
        example=example,
    )

    # Assert
    assert is_valid is True


def test_single_example_validation_strategy_failure(mock_evaluator, mock_scorer):
    """Tests the SingleExampleValidationStrategy for a failed validation."""
    # Arrange
    strategy = SingleExampleValidationStrategy()
    candidate_prompt = "This prompt will fail"
    example = dspy.Example(text="This should fail.", label="fail").with_inputs("text")

    # Act
    is_valid = strategy(
        candidate_prompt=candidate_prompt,
        evaluator=mock_evaluator,
        scorer=mock_scorer,
        dataset=[example],
        example=dspy.Example(text="This should fail.", label="pass").with_inputs("text"),
    )

    # Assert
    assert is_valid is False


def test_single_example_validation_strategy_missing_example(mock_evaluator, mock_scorer):
    """Tests that SingleExampleValidationStrategy raises an error if the example is missing."""
    # Arrange
    strategy = SingleExampleValidationStrategy()
    candidate_prompt = "A prompt"

    # Act & Assert
    with pytest.raises(ValueError, match="requires 'example' in kwargs"):
        strategy(
            candidate_prompt=candidate_prompt,
            evaluator=mock_evaluator,
            scorer=mock_scorer,
            dataset=[],
        )


def test_sample_validation_strategy_success(mock_evaluator, mock_scorer):
    """Tests the SampleValidationStrategy for a successful validation."""
    # Arrange
    strategy = SampleValidationStrategy(sample_size=2, threshold=1.0)
    candidate_prompt = "New and improved prompt"
    dataset = [
        dspy.Example(text="pass 1", label="pass").with_inputs("text"),
        dspy.Example(text="pass 2", label="pass").with_inputs("text"),
        dspy.Example(text="pass 3", label="pass").with_inputs("text"),
    ]

    # Act
    result = strategy(
        candidate_prompt=candidate_prompt,
        evaluator=mock_evaluator,
        scorer=mock_scorer,
        dataset=dataset,
    )

    # Assert
    assert result["is_valid"] is True
    assert result["score"] == 1.0


def test_sample_validation_strategy_failure(mock_evaluator, mock_scorer):
    """Tests the SampleValidationStrategy for a failed validation."""
    # Arrange
    strategy = SampleValidationStrategy(sample_size=3, threshold=0.7)
    candidate_prompt = "A prompt that will mostly fail"
    dataset = [
        dspy.Example(text="pass 1", label="pass").with_inputs("text"),
        dspy.Example(text="fail 1", label="fail").with_inputs("text"),
        dspy.Example(text="fail 2", label="fail").with_inputs("text"),
    ]

    # Act
    result = strategy(
        candidate_prompt=candidate_prompt,
        evaluator=mock_evaluator,
        scorer=mock_scorer,
        dataset=dataset,
    )

    # Assert
    assert result["is_valid"] is False
    assert result["score"] < 0.7
