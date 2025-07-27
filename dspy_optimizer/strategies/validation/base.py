"""Base interface for all Validation Strategies."""

from collections.abc import Callable

import dspy


class ValidationStrategy(dspy.Module):
    """Abstract base class for a Validation Strategy.

    A Validation Strategy defines the method for checking a candidate prompt
    for regressions before it is permanently accepted. Subclasses must
    implement the `forward` method.
    """

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> bool | dict:
        """Checks the candidate prompt for regressions.

        Args:
            candidate_prompt: The new prompt to be validated.
            evaluator: The Evaluator module to run predictions.
            scorer: The function to score the predictions.
            dataset: The dataset to validate against.
            **kwargs: Additional arguments for specific strategies.

        Returns:
            True if the prompt is considered safe and accepted, False otherwise.
        """
        raise NotImplementedError("Each ValidationStrategy must implement the `forward` method.")
