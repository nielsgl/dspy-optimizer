"""A validation strategy that always passes."""

from collections.abc import Callable

import dspy

from ..registry import validators
from .base import ValidationStrategy


@validators.register("none")
class NoValidationStrategy(ValidationStrategy):
    """
    A validation strategy that performs no validation and always returns True.

    This is useful for debugging the Refiner, as it allows you to see the
    Refiner's proposed changes without the Validator rejecting them.
    """

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> bool:
        """
        Performs no validation and always returns True.
        """
        return True
