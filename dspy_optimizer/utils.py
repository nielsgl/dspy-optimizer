from pathlib import Path


def get_root() -> Path:
    """Returns the root path of the project."""
    return Path(__file__).parent
