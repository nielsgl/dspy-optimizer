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

    def forward(
        self,
        prompt: str,
        example: dspy.Example,
        error_reasoning: str,
        prediction: str,
        expected_output: str,
        history: list[str],
    ) -> dspy.Prediction:
        """Analyzes the prompt and proposes a refined version.

        Args:
            prompt: The original prompt that needs to be improved.
            example: The dspy.Example that caused the prompt to fail.
            error_reasoning: The flawed reasoning from the model.
            prediction: The incorrect final prediction from the model.
            expected_output: The correct output that was expected.
            history: A list of previously attempted suggestions that have failed.

        Returns:
            A dspy.Prediction object containing the analysis, suggestion,
            and the proposed patch.
        """
        print("---in refiner")
        print(f"{expected_output=}, {type(expected_output)=}")
        print(f"{prediction=}, {type(prediction)=}")
        return self._predictor(
            prompt=prompt,
            example=example,
            error_reasoning=error_reasoning,
            prediction=prediction,
            expected_output=expected_output,
            history=str(history),  # Pass history as a string for the LLM
        )
