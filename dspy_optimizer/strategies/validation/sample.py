"""A validation strategy that checks a candidate prompt against a single sample."""

from collections.abc import Callable

import dspy

from ..registry import validators
from .base import ValidationStrategy


@validators.register("sample")
class SampleValidationStrategy(ValidationStrategy):
    """
    A validation strategy that validates a candidate prompt against the single
    failing example that triggered the refinement.

    This is a fast and targeted strategy, useful for ensuring that a proposed
    patch actually fixes the specific problem it was designed for.
    """

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> dict:
        """
        Validates the candidate prompt against the single failing example.

        Args:
            candidate_prompt: The prompt to validate.
            evaluator: The evaluator module to use.
            scorer: The scoring function.
            dataset: The full dataset (ignored in this strategy).
            **kwargs: Must contain 'example', the single dspy.Example to test against.

        Returns:
            A dictionary containing the validation result and the score.
        """
        if "example" not in kwargs:
            raise ValueError("SampleValidationStrategy requires 'example' in kwargs.")

        example = kwargs["example"]
        prediction = evaluator(prompt=candidate_prompt, **example.inputs())
        score = scorer(example, prediction)
        print(f"      Prediction: {prediction}")
        print(f"      Score: {score}")
        return {"is_valid": score, "score": score}
