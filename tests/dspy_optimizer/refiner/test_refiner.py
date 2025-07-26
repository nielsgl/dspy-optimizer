"""Unit tests for the Refiner module."""

import dspy

from dspy_optimizer.refiner.refiner import Refiner
from tests.conftest import MockLLM


def test_refiner_forward_pass(mock_llm: MockLLM):
    """Tests the forward pass of the Refiner module.

    Args:
        mock_llm: The mocked language model fixture.
    """
    # Arrange
    refiner = Refiner()
    test_prompt = "Original prompt"
    test_example = dspy.Example(input="some input", output="expected output")
    test_error = "It failed."

    # The RefinerSignature expects a JSON-like string with specific keys.
    # We will mock the LLM to return a string that can be parsed into the
    # expected dspy.Prediction object. The mock response must now include
    # all fields from the updated RefinerSignature.
    mock_response = (
        '{"reasoning": "The reasoning.", "analysis": "The analysis.", '
        '"suggestion": "The suggestion.", "target_block": "### Heuristics", '
        '"operation": "append", "content": "New content."}'
    )
    mock_llm.response_text = mock_response

    # Act
    result = refiner(prompt=test_prompt, example=test_example, error=test_error)

    # Assert
    # 1. Check that the result object has the correct, parsed fields.
    assert isinstance(result, dspy.Prediction)
    assert result.analysis == "The analysis."
    assert result.suggestion == "The suggestion."
    assert result.target_block == "### Heuristics"
    assert result.operation == "append"
    assert result.content == "New content."

    # 2. Check that the LLM was called with a prompt containing the inputs.
    assert len(mock_llm.history) > 0
    # For chat-optimized modules, dspy sends the prompt via `messages`.
    last_messages = mock_llm.history[-1]["messages"]
    assert last_messages is not None
    # The full prompt is typically in the last user message.
    full_prompt_str = last_messages[-1]["content"]
    assert test_prompt in full_prompt_str
    assert str(test_example) in full_prompt_str
    assert test_error in full_prompt_str
