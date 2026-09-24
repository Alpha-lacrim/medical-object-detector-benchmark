"""Keep portable CI independent of licensed data without relaxing local checks."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose an explicit opt-in for the unchanged full-data scientific tests."""
    parser.addoption(
        "--run-scientific",
        action="store_true",
        help="Include scientific_data tests; missing authorized inputs are errors, not skips.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect only marked integration tests in the default portable suite."""
    if config.getoption("--run-scientific"):
        return
    portable, scientific = [], []
    for item in items:
        (scientific if item.get_closest_marker("scientific_data") else portable).append(item)
    items[:] = portable
    config.hook.pytest_deselected(items=scientific)
