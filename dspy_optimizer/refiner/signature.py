"""Signature for the Refiner module."""

import dspy


class RefinerSignature(dspy.Signature):
    """
    You are an expert prompt engineer. Your task is to analyze a prompt that
    failed on a specific example and propose a targeted modification to fix it.
    Think step-by-step. Critically evaluate the prompt, the reasoning, and the
    history of failed attempts. Propose a modification that is both specific
    and generalizable. Follow the format of the examples below exactly.

    ---

    **Example 1: Adding a Heuristic**

    **Prompt:**
    ### TASK
    Given a file containing a Dutch invoice, extract the amount excluding taxes (btw).
    ### OUTPUT FORMAT
    Single floating point number
    ### EXAMPLES
    ### HEURISTICS

    **Example:**
    Inputs: {'file': 'invoice_with_total_and_subtotal.pdf'}, Outputs: 80.0

    **Reasoning:**
    The model found both a "Totaal" of 100.0 and a "Subtotaal" of 80.0. It incorrectly chose the
      "Totaal".

    **Prediction:**
    100.0

    **Expected Output:**
    80.0

    **History:**
    ["Failed Attempt 1: Operation: append, Target: '### HEURISTICS', Content: '- Be more careful
    when choosing between total and subtotal.'"]

    **Analysis:**
    The model incorrectly extracted the total amount (`100.0`) instead of the subtotal (`80.0`).
    The reasoning shows it found both numbers but chose the wrong one.
    The prompt lacks a rule to resolve this ambiguity.
    The history shows that a generic "be more careful" rule already failed.
    A more specific rule is needed.

    **Suggestion:**
    I will add a new, specific heuristic to prioritize the subtotal.

    **Target Block:**
    ### HEURISTICS

    **Operation:**
    append

    **Content:**
    - If both a "Totaal" and "Subtotaal" are present, always use the "Subtotaal".

    ---

    **Example 2: Adding an Example**

    **Prompt:**
    ### TASK
    Given a file containing a Dutch invoice, extract the amount excluding taxes (btw).
    ### OUTPUT FORMAT
    Single floating point number
    ### EXAMPLES
    ### HEURISTICS
    - If both a "Totaal" and "Subtotaal" are present, always use the "Subtotaal".

    **Example:**
    Inputs: {'file': 'invoice_with_shipping_costs.pdf'}, Outputs: 125.99

    **Reasoning:**
    The model correctly subtracted the VAT from the total, but it did not account for the
    shipping costs, which should also be excluded.

    **Prediction:**
    146.94

    **Expected Output:**
    125.99

    **History:**
    []

    **Analysis:**
    The model failed because it did not understand how to handle shipping costs ("verzendkosten").
    The current heuristics don't cover this case.
    This is a specific, tricky case that is best handled by adding a concrete example.

    **Suggestion:**
    I will add a new example to demonstrate the correct handling of shipping costs.

    **Target Block:**
    ### EXAMPLES

    **Operation:**
    append

    **Content:**
    - Input: An invoice with a subtotal of 146.94, shipping of 5.50, and VAT
        of 26.45 -> Output: 125.99
    """

    prompt: str = dspy.InputField(desc="The original prompt that needs to be improved.")
    example = dspy.InputField(desc="The dspy.Example that caused the prompt to fail.")
    error_reasoning: str = dspy.InputField(
        desc="The flawed reasoning (chain of thought) from the model."
    )
    prediction = dspy.InputField(desc="The incorrect final prediction from the model.")
    expected_output = dspy.InputField(desc="The correct output that was expected.")
    history: list[str] = dspy.InputField(
        desc="A list of previously attempted suggestions that have already failed for this example."
    )

    analysis: str = dspy.OutputField(
        desc=(
            "A step-by-step analysis of why the prompt failed for the given "
            "example. Be very specific and reference the prompt, reasoning, "
            "example, and history."
        )
    )
    suggestion: str = dspy.OutputField(
        desc=(
            "A concrete suggestion for how to modify the prompt to address the "
            "root cause identified in the analysis."
        )
    )
    target_block: str = dspy.OutputField(
        desc="The target block to modify (e.g., '### HEURISTICS' or '### EXAMPLES')."
    )
    operation: str = dspy.OutputField(desc="The operation to perform: 'append' or 'replace'.")
    content: str = dspy.OutputField(desc="The new content for the operation.")
