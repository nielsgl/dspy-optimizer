"""Unit tests for the BlockBasedMerger."""

import pytest

from dspy_optimizer.models import PatchOperation, PromptPatch
from dspy_optimizer.strategies.merger.block_based import BlockBasedMerger

BASE_PROMPT = """
### Instructions
This is the instruction block.

### Heuristics
- Heuristic 1
- Heuristic 2

### Examples
- Example 1
"""


def test_block_based_merger_append():
    """Tests appending content to a block."""
    merger = BlockBasedMerger()
    patch = PromptPatch(
        target_block="### Heuristics",
        operation=PatchOperation.APPEND,
        content="- Heuristic 3",
    )
    new_prompt = merger(BASE_PROMPT, patch)
    assert "- Heuristic 1" in new_prompt
    assert "- Heuristic 2" in new_prompt
    assert "- Heuristic 3" in new_prompt
    assert "### Examples" in new_prompt


def test_block_based_merger_replace():
    """Tests replacing the content of a block."""
    merger = BlockBasedMerger()
    patch = PromptPatch(
        target_block="### Heuristics",
        operation=PatchOperation.REPLACE,
        content="- New Heuristic",
    )
    new_prompt = merger(BASE_PROMPT, patch)
    assert "- Heuristic 1" not in new_prompt
    assert "- Heuristic 2" not in new_prompt
    assert "- New Heuristic" in new_prompt
    assert "### Examples" in new_prompt


def test_block_based_merger_append_to_last_block():
    """Tests appending to the last block in the prompt."""
    merger = BlockBasedMerger()
    patch = PromptPatch(
        target_block="### Examples",
        operation=PatchOperation.APPEND,
        content="- Example 2",
    )
    new_prompt = merger(BASE_PROMPT, patch)
    assert "- Example 1" in new_prompt
    assert new_prompt.endswith("- Example 2")


def test_block_based_merger_nonexistent_block():
    """Tests that a ValueError is raised for a non-existent block."""
    merger = BlockBasedMerger()
    patch = PromptPatch(
        target_block="### NonExistent",
        operation=PatchOperation.APPEND,
        content="some content",
    )
    with pytest.raises(ValueError, match="Target block '### NonExistent' not found in prompt."):
        merger(BASE_PROMPT, patch)
