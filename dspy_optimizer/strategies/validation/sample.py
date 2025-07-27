"""A validation strategy that checks a candidate prompt against a random sample of the dataset."""

from collections.abc import Callable
import random

import dspy

from ..registry import validators
from .base import ValidationStrategy


@validators.register("sample")
class SampleValidationStrategy(ValidationStrategy):
    """
    A validation strategy that validates a candidate prompt against a random
    sample of the dataset.

    This offers a balance between the speed of SingleExampleValidationStrategy
    and the thoroughness of FullValidationStrategy.
    """

    def __init__(self, sample_size: int = 3, threshold: float = 1.0):
        """Initializes the SampleValidationStrategy.

        Args:
            sample_size: The number of examples to randomly sample from the dataset.
            threshold: The minimum proportion of samples that must pass for the
                       prompt to be considered valid. Defaults to 1.0 (all
                       samples must pass).
        """
        super().__init__()
        if not 0 < sample_size:
            raise ValueError("Sample size must be greater than 0.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")
        self.sample_size = sample_size
        self.threshold = threshold

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> dict:
        """
        Validates the candidate prompt against a random sample of the dataset.

        Args:
            candidate_prompt: The prompt to validate.
            evaluator: The evaluator module to use.
            scorer: The scoring function.
            dataset: The full dataset to sample from.
            **kwargs: Not used in this strategy.

        Returns:
            A dictionary containing the validation result and the score.
        """
        if len(dataset) < self.sample_size:
            sample_set = dataset
        else:
            sample_set = random.sample(dataset, self.sample_size)

        if not sample_set:
            return {"is_valid": True, "score": 1.0}  # Vacuously true

        score = sum(
            scorer(example, evaluator(prompt=candidate_prompt, **example.inputs()))
            for example in sample_set
        ) / len(sample_set)

        is_valid = score >= self.threshold
        return {"is_valid": is_valid, "score": score}
