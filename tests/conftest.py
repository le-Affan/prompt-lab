"""
conftest.py — shared fixtures for the prompt versioning test suite.

All fixtures produce fresh, isolated state per test.
No fixture touches the module-level _default_store.
"""

import uuid

import pytest

from src.prompt_versioning import Version, VersionStore, create_version


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> VersionStore:
    """Return a fresh, empty VersionStore for each test."""
    return VersionStore()


@pytest.fixture()
def prompt_id() -> str:
    """Return a unique prompt UUID for each test."""
    return str(uuid.uuid4())


@pytest.fixture()
def prompt_id_b() -> str:
    """Second unique prompt UUID — used for cross-prompt isolation tests."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Version chain builder
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_chain(store: VersionStore):
    """
    Factory fixture: build a linear chain of versions on a single branch.

    Usage::

        def test_something(make_chain, prompt_id):
            v1, v2, v3 = make_chain(prompt_id, ["Content A", "Content B", "Content C"])

    Parameters
    ----------
    prompt_id : str
    contents : list[str]
    branch_name : str, optional  (default "main")

    Returns
    -------
    list[Version]
    """

    def _build(
        pid: str,
        contents: list[str],
        branch_name: str = "main",
    ) -> list[Version]:
        versions: list[Version] = []
        for i, content in enumerate(contents):
            v = create_version(
                pid,
                content,
                f"commit {i + 1}",
                branch_name=branch_name,
                store=store,
            )
            versions.append(v)
        return versions

    return _build
