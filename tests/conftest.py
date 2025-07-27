"""Configuration and fixtures for pytest."""

from types import SimpleNamespace

import dspy
import pytest


class MockLLM(dspy.BaseLM):
    """A mock language model for testing that adheres to dspy.BaseLM interface.

    This mock overrides the `forward` method as specified in the dspy
    documentation for creating custom LMs. It does not make any real API calls.
    Instead, it returns a pre-configured response in a structure that mimics
    the OpenAI API response format, as expected by the base class.
    """

    def __init__(self, response_text: str = "mocked response"):
        """Initializes the MockLLM."""
        # Use a dummy model name for the base class.
        super().__init__(model="mock-model")
        self.response_text = response_text
        # The history is now captured by the dspy.BaseLM itself.

    def forward(self, prompt=None, messages=None, **kwargs):
        """Mocks the forward pass, returning a mock OpenAI-like response.

        This method is called by the `__call__` method of the `dspy.BaseLM`
        base class.

        Args:
            prompt: The prompt to be "sent" to the LLM.
            messages: The messages to be "sent" to the LLM.
            **kwargs: Additional arguments.

        Returns:
            A SimpleNamespace object mimicking the OpenAI response structure.
        """
        # Create a mock response object that mimics the OpenAI structure
        # as expected by BaseLM._process_lm_response.
        mock_choice = SimpleNamespace(
            message=SimpleNamespace(content=self.response_text, tool_calls=None),
            logprobs=None,
        )
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model,
        )
        # The _process_lm_response method also expects a _hidden_params attribute
        # for cost calculation, so we add a dummy one.
        setattr(mock_response, "_hidden_params", {"response_cost": 0.0})
        return mock_response


@pytest.fixture
def mock_llm() -> MockLLM:
    """A pytest fixture that provides a configured mock LLM.

    This fixture also configures dspy to use this mock LLM for the
    duration of the test.
    """
    llm = MockLLM()
    dspy.settings.configure(lm=llm)
    return llm


@pytest.fixture
def mock_evaluator(mock_llm: MockLLM) -> dspy.Module:
    """A pytest fixture that provides a mock evaluator."""

    class MockPredict(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("text -> label")

        def forward(self, text, prompt=None):
            if prompt and "fail" in prompt:
                return dspy.Prediction(label="fail")
            if "fail" in text:
                return dspy.Prediction(label="fail")
            return dspy.Prediction(label="pass")

    return MockPredict()


@pytest.fixture
def mock_scorer():
    """A pytest fixture that provides a simple mock scorer."""

    def scorer(example, prediction):
        return example.label == prediction.label

    return scorer
