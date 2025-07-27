"""The main PromptOptimizer class."""

from collections.abc import Callable

import dspy

from .evaluator import Evaluator
from .models import Config, PromptPatch
from .refiner.refiner import Refiner
from .strategies.registry import mergers, scorers, validators


class PromptOptimizer(dspy.Module):
    """The main orchestrator for the DSPy Optimizer framework.

    This class integrates the Evaluator, Refiner, and pluggable strategies
    to iteratively refine a prompt based on a given dataset.
    """

    def __init__(
        self,
        signature: type[dspy.Signature],
        initial_prompt: str,
        merger_strategy: str = "block_based",
        validation_strategy: str = "full",
        config: Config = Config(),
    ):
        """Initializes the PromptOptimizer.

        Args:
            signature: The DSPy signature for the task.
            initial_prompt: The initial prompt to be optimized.
            merger_strategy: The name of the merger strategy to use.
            validation_strategy: The name of the validation strategy to use.
            config: The configuration object for the optimizer.
        """
        super().__init__()
        self.signature = signature
        self.prompt = initial_prompt
        self.config = config

        # Instantiate core components
        self.evaluator = Evaluator(signature=self.signature)
        self.refiner = Refiner()

        # Instantiate strategies from the registry
        merger_class = mergers.get(merger_strategy)
        self.merger = merger_class()

        validator_class = validators.get(validation_strategy)
        self.validator = validator_class()

    def optimize(
        self,
        dataset: list[dspy.Example],
        scorer: str | Callable,
        callbacks: list | None = None,
    ):
        """The main optimization loop.

        This method iterates through the dataset, identifies failing examples,
        uses the Refiner to propose fixes, and validates them before merging.

        Args:
            dataset: The dataset to use for optimization.
            scorer: The scoring function (or its registered name) to evaluate predictions.
            callbacks: A list of callbacks to be called at various stages.
        """
        # Resolve scorer if it's a string
        if isinstance(scorer, str):
            try:
                scorer_fn = scorers.get(scorer)
            except KeyError:
                raise ValueError(f"Scorer '{scorer}' not found in registry.")
        else:
            scorer_fn = scorer
        from tqdm import tqdm

        if callbacks is None:
            callbacks = []

        state = {"prompt": self.prompt, "dataset": dataset, "total_evaluations": 0}
        for cb in callbacks:
            cb.on_run_start(state)

        for i, example in tqdm(enumerate(dataset), desc="Optimizing Prompt", total=len(dataset)):
            state["example_id"] = i
            state["example"] = example
            prediction = self.evaluator(prompt=self.prompt, **example.inputs())
            state["total_evaluations"] += 1
            state["prediction"] = prediction
            is_correct = scorer_fn(example, prediction)

            if is_correct:
                print(
                    f"{i=} is correct, EXAMPLE:{getattr(example, 'amount', '')}"
                    + f"\nPREDICTION:{getattr(prediction, 'amount', '')}"
                )
            if not is_correct:
                print(
                    f"{i=} is not correct\n{getattr(example, 'amount', '')}"
                    + f"\n{getattr(prediction, 'amount', '')}\n---"
                )

            if not is_correct:
                history = []
                for refine_iter in range(self.config.max_refine_iters):
                    print(f"Refining attempt: {refine_iter}")
                    state["refinement_attempt"] = refine_iter + 1
                    for cb in callbacks:
                        cb.on_refinement_start(state)

                    expected_outputs = {
                        k: v for k, v in example.items() if k not in example.inputs()
                    }
                    # Extract the reasoning and predicted value from the prediction object
                    reasoning = getattr(prediction, "reasoning", "")
                    predicted_value = prediction.get(list(expected_outputs.keys())[0], "")

                    print(f"{reasoning=}")
                    print(f"{predicted_value=}")

                    # Create a string summary of the example for the refiner
                    example_summary = (
                        f"Inputs: {example.inputs()}, Outputs: {list(expected_outputs.values())[0]}"
                    )

                    refiner_output = self.refiner(
                        prompt=self.prompt,
                        example=example_summary,
                        error_reasoning=reasoning,
                        prediction=str(predicted_value),
                        expected_output=str(list(expected_outputs.values())[0]),
                        history=history,
                    )

                    from .models import PatchOperation

                    patch = PromptPatch(
                        target_block=refiner_output.target_block,
                        operation=PatchOperation(refiner_output.operation),
                        content=refiner_output.content,
                    )
                    state["patch"] = patch

                    candidate_prompt = self.merger(self.prompt, patch)
                    state["candidate_prompt"] = candidate_prompt

                    for cb in callbacks:
                        cb.on_validation_start(state)

                    validation_result = self.validator(
                        candidate_prompt=candidate_prompt,
                        evaluator=self.evaluator,
                        scorer=scorer_fn,
                        dataset=dataset,
                        example=example,
                    )
                    # TODO: This is a simplification. A better approach would be for the
                    # validator to return the number of evaluations it performed.
                    state["total_evaluations"] += 1

                    if isinstance(validation_result, dict):
                        is_valid = validation_result.get("is_valid", False)
                        state["validation_score"] = validation_result.get("score")
                    else:
                        is_valid = validation_result
                        state["validation_score"] = is_valid

                    state["is_valid"] = is_valid

                    for cb in callbacks:
                        cb.on_validation_end(state)

                    if is_valid:
                        self.prompt = candidate_prompt
                        state["prompt"] = self.prompt
                        for cb in callbacks:
                            cb.on_merge_success(state)
                        break
                    else:
                        for cb in callbacks:
                            cb.on_merge_failure(state)
                        history.append(
                            f"Failed Attempt {i + 1}: Operation: {patch.operation.value}, "
                            f"Target: '{patch.target_block}', Content: '{patch.content}'"
                        )

                    for cb in callbacks:
                        cb.on_refinement_end(state)

        for cb in callbacks:
            cb.on_run_end(state)

        return self.prompt
