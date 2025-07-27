"""Unit tests for the ValidationStrategy implementations."""

import dspy
import pytest

from dspy_optimizer.evaluator import Evaluator
from dspy_optimizer.strategies.validation.batched import (
    BatchedTrainingSetValidationStrategy,
)
from dspy_optimizer.strategies.validation.full import FullValidationStrategy


# A simple signature for evaluator tests
class SimpleSignature(dspy.Signature):
    input = dspy.InputField()
    output = dspy.OutputField()


# A mock scorer function for testing
def mock_scorer(example, prediction):
    return example.output == prediction.output


@pytest.fixture
def mock_evaluator(mock_llm):
    """Fixture to provide a mocked Evaluator."""
    # The mock LLM will return a JSON that ChainOfThought can parse.
    # It must match the fields in SimpleSignature (i.e., 'output').
    mock_llm.response_text = '{"reasoning": "mocked reasoning", "output": "pass"}'
    return Evaluator(signature=SimpleSignature)


def test_full_validation_all_pass(mock_evaluator):
    """Tests FullValidationStrategy when all examples should pass."""
    strategy = FullValidationStrategy()
    dataset = [
        dspy.Example(input="q1", output="pass").with_inputs("input"),
        dspy.Example(input="q2", output="pass").with_inputs("input"),
    ]
    is_valid = strategy("prompt", mock_evaluator, mock_scorer, dataset)
    assert is_valid is True


def test_full_validation_one_fail(mock_evaluator):
    """Tests FullValidationStrategy when one example fails."""
    strategy = FullValidationStrategy()
    dataset = [
        dspy.Example(input="q1", output="pass").with_inputs("input"),
        dspy.Example(input="q2", output="fail").with_inputs("input"),
    ]
    is_valid = strategy("prompt", mock_evaluator, mock_scorer, dataset)
    assert is_valid is False


def test_batched_validation_all_pass(mock_evaluator):
    """Tests BatchedTrainingSetValidationStrategy when the batch passes."""
    strategy = BatchedTrainingSetValidationStrategy(batch_size=2)
    dataset = [
        dspy.Example(input="q1", output="pass").with_inputs("input"),
        dspy.Example(input="q2", output="pass").with_inputs("input"),
        dspy.Example(input="q3", output="pass").with_inputs("input"),
    ]
    is_valid = strategy("prompt", mock_evaluator, mock_scorer, dataset)
    assert is_valid is True


def test_batched_validation_one_fail(mock_evaluator, monkeypatch):
    """Tests BatchedTrainingSetValidationStrategy when a failure is in the batch."""
    # Ensure the failing example is chosen
    dataset = [
        dspy.Example(input="q1", output="pass").with_inputs("input"),
        dspy.Example(input="q2", output="fail").with_inputs("input"),
    ]
    monkeypatch.setattr("random.sample", lambda data, size: data)  # Mock random.sample

    strategy = BatchedTrainingSetValidationStrategy(batch_size=2)
    is_valid = strategy("prompt", mock_evaluator, mock_scorer, dataset)
    assert is_valid is False


def test_batched_validation_invalid_batch_size():
    """Tests that BatchedTrainingSetValidationStrategy raises error for invalid batch_size."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        BatchedTrainingSetValidationStrategy(batch_size=0)
