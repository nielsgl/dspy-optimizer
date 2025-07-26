"""Signature for the Refiner module."""

import dspy


class RefinerSignature(dspy.Signature):
    """
    You are an expert prompt engineer. Your task is to analyze a prompt that
    failed on a specific example, determine the root cause of the failure,
    and propose a modification to the prompt to fix the issue.
    """

    prompt = dspy.InputField(desc="The original prompt that needs to be improved.")
    example = dspy.InputField(desc="The dspy.Example that caused the prompt to fail.")
    error = dspy.InputField(
        desc="A description of the error, e.g., 'Expected '123.45', got 'N/A''."
    )

    analysis = dspy.OutputField(
        desc=(
            "A step-by-step analysis of why the prompt failed for the given "
            "example. Be very specific and reference parts of the prompt and "
            "example."
        )
    )
    suggestion = dspy.OutputField(
        desc=(
            "A concrete suggestion for how to modify the prompt to address the "
            "root cause identified in the analysis."
        )
    )
    target_block = dspy.OutputField(desc="The target block to modify (e.g., '### Heuristics').")
    operation = dspy.OutputField(desc="The operation to perform: 'append' or 'replace'.")
    content = dspy.OutputField(desc="The new content for the operation.")
