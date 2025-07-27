"""A ValidationStrategy that checks a random batch of the dataset."""

from collections.abc import Callable
import random

import dspy
from tqdm import tqdm

from dspy_optimizer.strategies.registry import validators
from dspy_optimizer.strategies.validation.base import ValidationStrategy


@validators.register("batched")
class BatchedTrainingSetValidationStrategy(ValidationStrategy):
    """Validates a prompt against a random batch of the training set.

    This strategy provides a faster, stochastic alternative to full validation.
    It's useful when the validation dataset is large and full validation is
    too time-consuming.

    Attributes:
        batch_size: The number of samples to draw from the dataset.
    """

    def __init__(self, batch_size: int = 10):
        """Initializes the BatchedTrainingSetValidationStrategy.

        Args:
            batch_size: The number of samples to randomly select for validation.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> bool:
        """Checks the candidate prompt against a random batch of the dataset.

        Args:
            candidate_prompt: The new prompt to be validated.
            evaluator: The Evaluator module to run predictions.
            scorer: The function to score the predictions.
            dataset: The dataset to validate against.

        Returns:
            True if the prompt passes for all examples in the batch, False otherwise.
        """
        if not dataset:
            return True  # An empty dataset trivially passes.

        effective_batch_size = min(self.batch_size, len(dataset))
        validation_batch = random.sample(dataset, effective_batch_size)

        num_correct = 0
        with tqdm(total=effective_batch_size, desc="Batched Validation", unit="example") as pbar:
            for example in validation_batch:
                prediction = evaluator(prompt=candidate_prompt, **example.inputs())
                is_correct = scorer(example, prediction)

                if is_correct:
                    num_correct += 1
                else:
                    pbar.set_postfix_str("Failed!", refresh=True)
                    return False

                pbar.update(1)
                accuracy = num_correct / pbar.n
                pbar.set_postfix_str(f"Accuracy: {accuracy:.2%}", refresh=True)

        return True
