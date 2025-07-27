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
    test_example = "Inputs: {'file': 'test.pdf'}, Outputs: 123.45"
    test_reasoning = "The model failed because..."
    test_prediction = "N/A"
    test_expected_output = "123.45"
    test_history = ["Failed Attempt 1: ..."]

    # The RefinerSignature expects a JSON-like string with specific keys.
    # We will mock the LLM to return a string that can be parsed into the
    # expected dspy.Prediction object.
    mock_response = (
        '{"reasoning": "The mock reasoning.", "analysis": "The analysis.", '
        '"suggestion": "The suggestion.", "target_block": "### Heuristics", '
        '"operation": "append", "content": "New content."}'
    )
    mock_llm.response_text = mock_response

    # Act
    result = refiner(
        prompt=test_prompt,
        example=test_example,
        error_reasoning=test_reasoning,
        prediction=test_prediction,
        expected_output=test_expected_output,
        history=test_history,
    )

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
    assert test_example in full_prompt_str
    assert test_reasoning in full_prompt_str
    assert test_prediction in full_prompt_str
    assert test_expected_output in full_prompt_str
    assert str(test_history) in full_prompt_str
