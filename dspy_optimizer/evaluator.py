"""The Evaluator module for the DSPy Optimizer."""

import dspy


class Evaluator(dspy.Module):
    """A module for evaluating a prompt against a given signature.

    This module is designed to be thread-safe, allowing for parallel execution
    of the evaluation logic.
    """

    def __init__(self, signature: type[dspy.Signature], callbacks: list | None = None) -> None:
        """Initializes the Evaluator.

        Args:
            signature: The DSPy signature class to use for evaluation.
        """
        super().__init__(callbacks)
        self._base_signature = signature

    def forward(self, prompt: str, **kwargs) -> dspy.Prediction:
        """Evaluates the prompt with the given arguments in a thread-safe manner.

        This method uses the official `with_instructions` API to create a new,
        temporary signature for each call. This is the correct, stateless,
        and thread-safe way to use dynamic prompts in DSPy.

        Args:
            prompt: The prompt instructions to use for this evaluation.
            **kwargs: The input fields for the signature.

        Returns:
            A dspy.Prediction object containing the model's output.
        """
        # Use the official, immutable API to create a new signature with the
        # desired instructions. This is thread-safe and robust.
        dynamic_signature = self._base_signature.with_instructions(prompt)

        # Instantiate a predictor with the new, temporary signature.
        predictor = dspy.ChainOfThought(dynamic_signature)

        return predictor(**kwargs)
