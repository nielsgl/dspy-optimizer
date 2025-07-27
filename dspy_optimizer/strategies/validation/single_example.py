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
        example: dspy.Example,
        **kwargs,
    ) -> bool:
        """
        Validates the candidate prompt against the single failing example.

        Args:
            candidate_prompt: The new prompt to validate.
            evaluator: The dspy.Module used for evaluation.
            scorer: The scoring function to determine correctness.
            dataset: The full dataset (ignored by this strategy).
            example: The specific dspy.Example that failed with the original prompt.

        Returns:
            True if the candidate prompt correctly handles the failing example,
            False otherwise.
        """
        # Create a temporary evaluator with the new prompt
        with dspy.context(lm=evaluator.lm):
            temp_evaluator = evaluator.with_signature(dspy.Predict(evaluator.signature))
            temp_evaluator.prompt = candidate_prompt

        prediction = temp_evaluator(**example.inputs())
        return scorer(example, prediction)
