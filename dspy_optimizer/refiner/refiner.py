"""The Refiner module for the DSPy Optimizer."""

import dspy

from .signature import RefinerSignature


class Refiner(dspy.Module):
    """A module that analyzes a failed prompt and proposes a fix.

    This module uses a ChainOfThought predictor with the RefinerSignature
    to guide a language model in analyzing a prompt, a failing example,
    and an error description to produce a refined prompt.
    """

    def __init__(self):
        """Initializes the Refiner module."""
        super().__init__()
        self._predictor = dspy.ChainOfThought(RefinerSignature)

    def forward(self, prompt: str, example: dspy.Example, error: str) -> dspy.Prediction:
        """Analyzes the prompt and proposes a refined version.

        Args:
            prompt: The original prompt that needs to be improved.
            example: The dspy.Example that caused the prompt to fail.
            error: A description of the error.

        Returns:
            A dspy.Prediction object containing the analysis, suggestion,
            and the full text of the modified prompt.
        """
        return self._predictor(prompt=prompt, example=example, error=error)
