"""Common, reusable scorer functions."""

from typing import Any

import dspy

from dspy_optimizer.strategies.registry import scorers


@scorers.register("exact_match")
def exact_match_scorer(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    """A flexible scorer that checks for exact match of the first output field.

    This scorer is designed to be signature-agnostic. It dynamically identifies
    the first output field from the example and performs a case-sensitive,
    string-based comparison against the corresponding field in the prediction.

    Args:
        example: The ground truth dspy.Example.
        prediction: The model's dspy.Prediction.

    Returns:
        True if the prediction is an exact match, False otherwise.
    """
    # 1. Identify the output key from the example.
    try:
        input_keys = example.inputs()
        output_keys = [k for k in example.keys() if k not in input_keys]
        if not output_keys:
            return False  # No output fields to score.
        output_key = output_keys[0]
    except (AttributeError, IndexError):
        return False

    # 2. Extract the values.
    try:
        expected_value = getattr(example, output_key)
        predicted_value = prediction[output_key]
    except (AttributeError, KeyError):
        return False

    # 3. Perform a robust, string-based comparison.
    return str(expected_value) == str(predicted_value)


@scorers.register("numeric")
def numeric_scorer(
    example: dspy.Example, prediction: dspy.Prediction, tolerance: float = 1e-6
) -> bool:
    """A flexible scorer for numeric comparisons with a tolerance.

    This scorer attempts to parse the first output field of both the example
    and the prediction as floats. It then checks if they are close within a
    given tolerance.

    Args:
        example: The ground truth dspy.Example.
        prediction: The model's dspy.Prediction.
        tolerance: The tolerance for the float comparison.

    Returns:
        True if the numeric values are close, False otherwise.
    """
    # 1. Identify the output key from the example.
    try:
        input_keys = example.inputs()
        output_keys = [k for k in example.keys() if k not in input_keys]
        if not output_keys:
            return False
        output_key = output_keys[0]
    except (AttributeError, IndexError):
        return False

    # 2. Extract and parse the values.
    try:
        # A more robust parser for different number formats
        def parse_numeric(value: Any) -> float:
            s = str(value).strip()
            # Handle thousands separators (both comma and period) by removing them
            s = s.replace(",", "")
            # Handle decimal comma by replacing it with a period
            # This is safe because all commas have been removed
            s = s.replace(",", ".")
            return float(s)

        expected_value = getattr(example, output_key)
        expected_float = parse_numeric(expected_value)
        predicted_float = parse_numeric(prediction[output_key])
    except (AttributeError, KeyError, ValueError, TypeError):
        return False

    # 3. Perform a robust, floating-point comparison.
    import math

    return math.isclose(expected_float, predicted_float, rel_tol=tolerance)
