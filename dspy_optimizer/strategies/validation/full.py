"""A ValidationStrategy that checks the entire dataset."""

from collections.abc import Callable

import dspy
from tqdm import tqdm

from dspy_optimizer.strategies.registry import validators
from dspy_optimizer.strategies.validation.base import ValidationStrategy


@validators.register("full")
class FullValidationStrategy(ValidationStrategy):
    """Validates a prompt against the full dataset.

    This strategy ensures that a candidate prompt does not cause any regressions
    by running it against every example in the provided validation dataset.
    It is thorough but can be slow for large datasets.
    """

    def forward(
        self,
        candidate_prompt: str,
        evaluator: dspy.Module,
        scorer: Callable,
        dataset: list[dspy.Example],
        **kwargs,
    ) -> bool:
        """Checks the candidate prompt against the full dataset.

        Args:
            candidate_prompt: The new prompt to be validated.
            evaluator: The Evaluator module to run predictions.
            scorer: The function to score the predictions.
            dataset: The dataset to validate against.

        Returns:
            True if the prompt passes for all examples, False otherwise.
        """
        num_correct = 0
        total = len(dataset)

        with tqdm(total=total, desc="Full Validation", unit="example") as pbar:
            for example in dataset:
                prediction = evaluator(prompt=candidate_prompt, **example.inputs())
                is_correct = scorer(example, prediction)

                if is_correct:
                    num_correct += 1
                else:
                    # As soon as one example fails, we can reject the prompt.
                    pbar.set_postfix_str("Failed!", refresh=True)
                    return False

                pbar.update(1)
                accuracy = num_correct / pbar.n
                pbar.set_postfix_str(f"Accuracy: {accuracy:.2%}", refresh=True)

        # If all examples passed, the prompt is accepted.
        return True
