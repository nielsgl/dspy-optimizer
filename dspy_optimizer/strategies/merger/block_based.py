"""A MergerStrategy that operates on named blocks in a prompt."""

import re

from dspy_optimizer.models import PatchOperation, PromptPatch
from dspy_optimizer.strategies.merger.base import MergerStrategy
from dspy_optimizer.strategies.registry import mergers


@mergers.register("block_based")
class BlockBasedMerger(MergerStrategy):
    """A merger that appends to or replaces named blocks in a prompt.

    A named block is a section of the prompt that starts with a line like
    '### Block Name'.
    """

    def __call__(self, base_prompt: str, patch: PromptPatch) -> str:
        """Applies the patch to the base prompt.

        Args:
            base_prompt: The original prompt text.
            patch: The PromptPatch object containing the proposed change.

        Returns:
            A new string representing the candidate prompt.

        Raises:
            ValueError: If the target block is not found in the prompt.
        """
        # Find the start of the target block
        block_start_regex = re.compile(f"^{re.escape(patch.target_block)}.*$", re.MULTILINE)
        match = block_start_regex.search(base_prompt)

        if not match:
            raise ValueError(f"Target block '{patch.target_block}' not found in prompt.")

        start_pos = match.end()

        # Find the end of the block (next block header or end of string)
        next_block_regex = re.compile(r"^###\s", re.MULTILINE)
        next_match = next_block_regex.search(base_prompt, start_pos)
        end_pos = next_match.start() if next_match else len(base_prompt)

        if patch.operation == PatchOperation.APPEND:
            # Append content to the end of the block
            new_prompt = (
                base_prompt[:end_pos].rstrip() + "\n" + patch.content + "\n" + base_prompt[end_pos:]
            )
        elif patch.operation == PatchOperation.REPLACE:
            # Replace the content of the block
            new_prompt = (
                base_prompt[:start_pos] + "\n" + patch.content + "\n" + base_prompt[end_pos:]
            )
        else:
            # This should be unreachable if PatchOperation is used correctly
            raise ValueError(f"Unsupported operation: {patch.operation}")

        return new_prompt.strip()
