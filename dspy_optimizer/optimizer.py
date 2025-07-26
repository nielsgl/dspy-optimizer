"""The main PromptOptimizer class."""

from collections.abc import Callable

import dspy

from .evaluator import Evaluator
from .models import Config, PromptPatch
from .refiner.refiner import Refiner
from .strategies.registry import mergers, validators


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
        self, dataset: list[dspy.Example], scorer: Callable, callbacks: list | None = None
    ):
        """The main optimization loop.

        This method iterates through the dataset, identifies failing examples,
        uses the Refiner to propose fixes, and validates them before merging.

        Args:
            dataset: The dataset to use for optimization.
            scorer: The scoring function to evaluate predictions.
            callbacks: A list of callbacks to be called at various stages.
        """
        from tqdm import tqdm

        if callbacks is None:
            callbacks = []

        state = {"prompt": self.prompt, "dataset": dataset}
        for cb in callbacks:
            cb.on_run_start(state)

        for example in tqdm(dataset, desc="Optimizing Prompt"):
            state["example"] = example
            prediction = self.evaluator(prompt=self.prompt, **example.inputs())
            state["prediction"] = prediction
            is_correct = scorer(example, prediction)

            if not is_correct:
                for i in range(self.config.max_refine_iters):
                    state["refinement_attempt"] = i + 1
                    for cb in callbacks:
                        cb.on_refinement_start(state)

                    expected_outputs = {
                        k: v for k, v in example.items() if k not in example.inputs()
                    }
                    refiner_output = self.refiner(
                        prompt=self.prompt,
                        example=example,
                        error=f"Scorer returned False. Expected: {expected_outputs},"
                        + f"Got: {prediction._store}",
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

                    is_valid = self.validator(
                        candidate_prompt=candidate_prompt,
                        evaluator=self.evaluator,
                        scorer=scorer,
                        dataset=dataset,
                    )
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

                    for cb in callbacks:
                        cb.on_refinement_end(state)

        for cb in callbacks:
            cb.on_run_end(state)
