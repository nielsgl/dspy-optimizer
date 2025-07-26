"""Unit tests for the data models."""

import pytest

from dspy_optimizer.models import Config, PatchOperation, PromptPatch


def test_config_defaults():
    """Tests that the Config dataclass has the correct default values."""
    config = Config()
    assert config.max_refine_iters == 5
    assert config.temperature == 0.0
    assert config.parallel_workers == 8


def test_config_custom_values():
    """Tests that the Config dataclass can be instantiated with custom values."""
    config = Config(max_refine_iters=10, temperature=0.7, parallel_workers=1)
    assert config.max_refine_iters == 10
    assert config.temperature == 0.7
    assert config.parallel_workers == 1


def test_patch_operation_enum():
    """Tests the members of the PatchOperation enum."""
    assert PatchOperation.APPEND.value == "append"
    assert PatchOperation.REPLACE.value == "replace"


def test_prompt_patch_instantiation():
    """Tests the instantiation of the PromptPatch dataclass."""
    patch = PromptPatch(
        target_block="### Heuristics",
        operation=PatchOperation.APPEND,
        content="New heuristic.",
    )
    assert patch.target_block == "### Heuristics"
    assert patch.operation == PatchOperation.APPEND
    assert patch.content == "New heuristic."


def test_frozen_dataclasses():
    """Tests that the dataclasses are frozen and raise an error on modification."""
    config = Config()
    with pytest.raises(AttributeError):
        config.max_refine_iters = 10

    patch = PromptPatch(target_block="test", operation=PatchOperation.APPEND, content="test")
    with pytest.raises(AttributeError):
        patch.content = "new content"
