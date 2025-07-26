"""Unit tests for common scorer functions."""

import dspy

from dspy_optimizer.strategies.scoring.common import exact_match_scorer


def test_exact_match_scorer_pass():
    """Tests exact_match_scorer for a passing case."""
    example = dspy.Example(input="question", output="correct").with_inputs("input")
    prediction = dspy.Prediction(output="correct")
    assert exact_match_scorer(example, prediction) is True


def test_exact_match_scorer_fail():
    """Tests exact_match_scorer for a failing case."""
    example = dspy.Example(input="question", output="correct").with_inputs("input")
    prediction = dspy.Prediction(output="incorrect")
    assert exact_match_scorer(example, prediction) is False


def test_exact_match_scorer_type_coercion():
    """Tests that the scorer coerces values to strings for comparison."""
    example = dspy.Example(input="question", output=123).with_inputs("input")
    prediction = dspy.Prediction(output="123")
    assert exact_match_scorer(example, prediction) is True


def test_exact_match_scorer_multiple_fields():
    """Tests that the scorer only compares the first output field."""
    example = dspy.Example(input="question", output="correct", other_field="a").with_inputs("input")
    prediction = dspy.Prediction(output="correct", other_field="b")
    assert exact_match_scorer(example, prediction) is True


def test_numeric_scorer_pass():
    """Tests numeric_scorer for a passing case."""
    from dspy_optimizer.strategies.scoring.common import numeric_scorer

    example = dspy.Example(input="question", output=123.45).with_inputs("input")
    prediction = dspy.Prediction(output="123.45")
    assert numeric_scorer(example, prediction) is True


def test_numeric_scorer_with_comma_pass():
    """Tests numeric_scorer with a comma in the string."""
    from dspy_optimizer.strategies.scoring.common import numeric_scorer

    example = dspy.Example(input="question", output=1234.56).with_inputs("input")
    prediction = dspy.Prediction(output="1,234.56")
    assert numeric_scorer(example, prediction) is True


def test_numeric_scorer_fail():
    """Tests numeric_scorer for a failing case."""
    from dspy_optimizer.strategies.scoring.common import numeric_scorer

    example = dspy.Example(input="question", output=123.45).with_inputs("input")
    prediction = dspy.Prediction(output="123.46")
    assert numeric_scorer(example, prediction) is False


def test_numeric_scorer_invalid_string():
    """Tests numeric_scorer with a non-numeric string."""
    from dspy_optimizer.strategies.scoring.common import numeric_scorer

    example = dspy.Example(input="question", output=123.45).with_inputs("input")
    prediction = dspy.Prediction(output="N/A")
    assert numeric_scorer(example, prediction) is False
