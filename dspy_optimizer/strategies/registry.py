"""A registry for pluggable strategies."""

from collections.abc import Callable


class Registry:
    """A registry for storing and retrieving classes by name."""

    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, type] = {}

    def register(self, name: str) -> Callable:
        """A decorator to register a class."""

        def decorator(cls: type) -> type:
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in '{self._name}'.")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Get a class from the registry by name."""
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in '{self._name}' registry.")
        return self._registry[name]


# Global registries for different strategy types
mergers = Registry("mergers")
validators = Registry("validators")
scorers = Registry("scorers")
