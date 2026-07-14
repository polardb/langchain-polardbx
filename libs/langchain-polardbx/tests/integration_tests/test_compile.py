"""Placeholder test for compile marker used by CI.

CI runs ``uv run pytest -m compile tests/integration_tests`` to verify
that integration tests can be collected without running them.
"""

import pytest


@pytest.mark.compile
def test_compile() -> None:
    """Placeholder test to verify integration tests can be imported."""
    assert True
