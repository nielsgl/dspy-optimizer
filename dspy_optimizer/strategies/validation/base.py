"""Base interface for all Validation Strategies."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import dspy


class ValidationStrategy(ABC):
    """Abstract base class for a Validation Strategy.

    A Validation Strategy defines the method for checking a candidate prompt
    for regressions before it is permanently accepted.
    """

    @abstractmethod
    def __call__(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
    ) -> bool:
        """Checks the candidate prompt for regressions.

        Args:
            candidate_prompt: The new prompt to be validated.
            evaluator: The Evaluator module to run predictions.
            scorer: The function to score the predictions.
            dataset: The dataset to validate against.

        Returns:
            True if the prompt is considered safe and accepted, False otherwise.
        """
        pass
