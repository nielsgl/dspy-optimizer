"""Base interface for all Merger Strategies."""

from abc import ABC, abstractmethod

from dspy_optimizer.models import PromptPatch


class MergerStrategy(ABC):
    """Abstract base class for a Merger Strategy.

    A Merger Strategy defines the method for applying a PromptPatch to a base prompt
    to create a new, candidate prompt.
    """

    @abstractmethod
    def __call__(self, base_prompt: str, patch: PromptPatch) -> str:
        """Applies the patch to the base prompt.

        Args:
            base_prompt: The original prompt text.
            patch: The PromptPatch object containing the proposed change.

        Returns:
            A new string representing the candidate prompt.
        """
        pass
