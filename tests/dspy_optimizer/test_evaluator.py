"""Unit tests for the Evaluator module."""

import dspy

from dspy_optimizer.evaluator import Evaluator
from tests.conftest import MockLLM


class SimpleSignature(dspy.Signature):
    """A simple signature for testing."""

    input = dspy.InputField()
    output = dspy.OutputField()


def test_evaluator_with_instructions(mock_llm: MockLLM):
    """Tests that the Evaluator correctly uses with_instructions.

    Args:
        mock_llm: The mocked language model fixture.
    """
    # Arrange
    evaluator = Evaluator(signature=SimpleSignature)
    test_prompt = "This is a dynamic prompt."
    expected_output = "mocked output"
    # dspy.ChainOfThought expects a JSON response with "reasoning" and the
    # signature's output fields.
    mock_llm.response_text = f'{{"reasoning": "mocked reasoning", "output": "{expected_output}"}}'

    # Act
    result = evaluator(prompt=test_prompt, input="test_input")

    # Assert
    # 1. Check that the result from the evaluator is correct.
    assert isinstance(result, dspy.Prediction)
    assert result.output == expected_output

    # 2. Check that the LLM was called with the correct, dynamic prompt.
    # The dspy.ChainOfThought module will format the signature and prompt.
    # We need to check that our dynamic prompt is part of the prompt sent to the LM.
    assert len(mock_llm.history) > 0
    # For chat-optimized modules, dspy sends the prompt via `messages`.
    last_messages = mock_llm.history[-1]["messages"]
    assert last_messages is not None
    # The instructions are typically at the start of the system message.
    system_prompt = next((m["content"] for m in last_messages if m["role"] == "system"), "")
    assert test_prompt in system_prompt
