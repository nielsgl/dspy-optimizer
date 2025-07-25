"""The Evaluator module for the DSPy Optimizer."""

import dspy


class Evaluator(dspy.Module):
    """A module for evaluating a prompt against a given signature."""

    def __init__(self, signature: type[dspy.Signature]):
        """Initializes the Evaluator.

        Args:
            signature: The DSPy signature to use for evaluation.
        """
        super().__init__()
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, prompt: str, **kwargs) -> dspy.Prediction:
        """Evaluates the prompt with the given arguments.

        This method dynamically sets the instructions on the predictor's signature
        before calling it.

        Args:
            prompt: The prompt instructions to use for this evaluation.
            **kwargs: The input fields for the signature.

        Returns:
            A dspy.Prediction object containing the model's output.
        """

        # Create a new signature class with the updated prompt
        class DynamicSignature(self.predictor.signature):
            """{prompt}"""

            pass

        DynamicSignature.__doc__ = DynamicSignature.__doc__.format(prompt=prompt)

        self.predictor.predict.signature = DynamicSignature

        return self.predictor(**kwargs)
