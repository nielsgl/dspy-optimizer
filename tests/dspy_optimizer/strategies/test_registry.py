"""Unit tests for the Registry."""

import pytest

from dspy_optimizer.strategies.registry import Registry


def test_registry_register_and_get():
    """Tests basic registration and retrieval of classes."""
    registry = Registry("test_registry")

    @registry.register("TestClass")
    class MyTestClass:
        pass

    retrieved_class = registry.get("TestClass")
    assert retrieved_class is MyTestClass


def test_registry_get_nonexistent():
    """Tests that getting a non-existent key raises a KeyError."""
    registry = Registry("test_registry")
    with pytest.raises(KeyError, match="'NonExistent' not found in 'test_registry' registry."):
        registry.get("NonExistent")


def test_registry_register_duplicate():
    """Tests that registering a duplicate name raises a ValueError."""
    registry = Registry("test_registry")

    @registry.register("DuplicateName")
    class FirstClass:
        pass

    with pytest.raises(
        ValueError, match="'DuplicateName' is already registered in 'test_registry'."
    ):

        @registry.register("DuplicateName")
        class SecondClass:
            pass
