"""Core data models for the DSPy Optimizer."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Config:
    """Configuration settings for the PromptOptimiser.

    Attributes:
        max_refine_iters: The maximum number of refinement attempts for a single failing sample.
        temperature: The temperature to use for LLM calls during refinement.
        parallel_workers: The number of parallel threads for evaluation.
    """

    max_refine_iters: int = 5
    temperature: float = 0.0
    parallel_workers: int = 8


class PatchOperation(Enum):
    """Enum for the type of operation in a PromptPatch."""

    APPEND = "append"
    REPLACE = "replace"


@dataclass(frozen=True)
class PromptPatch:
    """A structured command to modify a prompt.

    This object is the output of the Refiner and the input to the Merger.
    It represents a single, atomic change to be applied to the prompt.

    Attributes:
        target_block: The named block in the prompt to modify (e.g., "### Heuristics").
        operation: The operation to perform (e.g., APPEND, REPLACE).
        content: The new text content for the operation.
    """

    target_block: str
    operation: PatchOperation
    content: str
