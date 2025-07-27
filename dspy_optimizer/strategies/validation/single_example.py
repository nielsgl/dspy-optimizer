"""A validation strategy that checks only the single failing example."""

from collections.abc import Callable

import dspy

from ..registry import validators
from .base import ValidationStrategy


@validators.register("single_example")
class SingleExampleValidationStrategy(ValidationStrategy):
    """
    A validation strategy that checks if a candidate prompt corrects the single
    example that triggered the refinement.

    This strategy is useful for rapid iteration and debugging, as it accepts a
    patch as soon as it fixes the immediate problem. However, it provides no
    guarantee that the change does not negatively impact other examples (i.e.,
    it does not protect against regressions).
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
        Validates the candidate prompt against the single failing example.

        Args:
            candidate_prompt: The new prompt to validate.
            evaluator: The dspy.Module used for evaluation.
            scorer: The scoring function to determine correctness.
            dataset: The full dataset (ignored by this strategy).
            **kwargs: Must contain the 'example' that failed.

        Returns:
            True if the candidate prompt correctly handles the failing example,
            False otherwise.
        """
        example = kwargs.get("example")
        if example is None:
            raise ValueError("SingleExampleValidationStrategy requires 'example' in kwargs.")

        # The evaluator is a dspy.Predict module; we can override its prompt for one call.
        # This will automatically use any LM set in the surrounding dspy.context.
        prediction = evaluator(prompt=candidate_prompt, **example.inputs())
        return scorer(example, prediction)
