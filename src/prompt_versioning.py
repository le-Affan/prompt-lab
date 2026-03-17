"""
Prompt Versioning Engine
========================
In-memory, Git-style version control for prompts.

This module is built incrementally. This file currently contains:

- ``Version``              — immutable prompt snapshot (data model)
- ``VersionStore``         — in-memory storage container
- Storage functions        — ``_store_version``, ``_get_version``,
  ``_get_versions_by_prompt``, ``_reset_store``
- Hashing                  — ``compute_prompt_hash``
- Version creation         — ``create_version``, ``branch_version``,
  ``rollback_version``
- Tree traversal           — ``get_version``, ``get_children``,
  ``get_version_lineage``, ``get_root_version``,
  ``get_branch_versions``, ``get_version_tree``
- Diff engine              — ``compute_char_diff``, ``compute_line_diff``

Thread Safety
-------------
This module does not implement internal locking. The caller is responsible
for serialising concurrent access to any shared state.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from diff_match_patch import diff_match_patch as DMP


@dataclass(frozen=True)
class Version:
    """An immutable snapshot of a prompt at a point in time.

    Models a single node in the prompt version DAG, analogous to a Git
    commit. Every field is set at creation and cannot be changed thereafter
    (``frozen=True``).

    Attributes
    ----------
    id : str
        UUID4 string. Unique identifier for this version across all prompts
        and branches.

    prompt_id : str
        UUID4 string. Identifies which prompt this version belongs to. All
        versions that share a ``prompt_id`` form one version tree.

    content : str
        The full text of the prompt at this version. Stored directly on the
        node — no indirection through a content pool.

    parent_version_id : str or None
        UUID4 string of the parent version, or ``None`` for a root node (the
        first commit on a branch that was not branched from an existing
        version).

    branch_name : str
        The branch this version was committed on (e.g. ``"main"``,
        ``"experiment-tone"``, ``"feature-v2"``).

    commit_message : str
        A short human-readable description of what changed in this version
        (e.g. ``"Add safety constraint to system prompt"``).

    hash : str
        SHA-256 hex digest (64 characters) of ``content``. Computed
        externally by ``hash_content()`` and passed in at creation. Used for
        duplicate-content detection and integrity verification.

    created_at : datetime
        UTC timestamp set at the moment the version is created. Should always
        be ``datetime.now(timezone.utc)``; never back-dated.

    metadata : dict
        Arbitrary key/value pairs for tagging, provenance, or tooling data.
        Examples: ``{"author": "alice"}``, ``{"rollback_from": "<uuid>"}``,
        ``{"branched_from": "<uuid>"}``.

        .. note::
            Although the ``Version`` dataclass is frozen (attribute
            reassignment is blocked), the ``metadata`` dict itself is
            mutable. Callers must not mutate it after the version is stored.
            By convention, the factory functions in this module always store
            a *copy* of the caller-supplied dict.

    Examples
    --------
    Typical construction (done internally by ``create_version``):

    >>> import uuid
    >>> from datetime import datetime, timezone
    >>> v = Version(
    ...     id=str(uuid.uuid4()),
    ...     prompt_id=str(uuid.uuid4()),
    ...     content="You are a helpful assistant.",
    ...     parent_version_id=None,
    ...     branch_name="main",
    ...     commit_message="Initial version",
    ...     hash="a" * 64,          # placeholder; real value from hash_content()
    ...     created_at=datetime.now(timezone.utc),
    ...     metadata={},
    ... )
    >>> v.branch_name
    'main'
    """

    id: str
    prompt_id: str
    content: str
    parent_version_id: Optional[str]
    branch_name: str
    commit_message: str
    hash: str
    created_at: datetime
    metadata: dict[str, Any] = field(
        default_factory=dict,
        # Exclude from equality checks and hashing — dict is not hashable and
        # two versions differing only in metadata extras should still compare
        # as different objects (identity is determined by `id`).
        compare=False,
        hash=False,
    )


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------


class VersionStore:
    """Container for all in-memory version state.

    Holds two plain dicts that together represent the full version graph for
    all prompts within a session:

    ``_versions``
        Maps ``version_id -> Version``. Every committed version lives here.
        Entries are never deleted or mutated — the store is append-only.

    ``_branches``
        Maps ``(prompt_id, branch_name) -> version_id``. Points each branch
        to its current HEAD. Updated on every new commit to that branch.

    This class is intentionally minimal — it owns only state, no behaviour.
    All read/write logic lives in the module-level storage functions below.

    Thread Safety
    -------------
    ``VersionStore`` uses plain Python dicts and provides **no internal
    locking**. If multiple threads share one instance, the caller must
    serialise access using an external lock (e.g. ``threading.Lock``).

    Example
    -------
    >>> store = VersionStore()
    >>> len(store._versions)
    0
    >>> len(store._branches)
    0
    """

    def __init__(self) -> None:
        # version_id (str) -> Version
        self._versions: dict[str, Version] = {}
        # (prompt_id, branch_name) -> HEAD version_id
        self._branches: dict[tuple[str, str], str] = {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VersionStore("
            f"versions={len(self._versions)}, "
            f"branches={len(self._branches)})"
        )


# ---------------------------------------------------------------------------
# Module-level default store
# ---------------------------------------------------------------------------

# A single shared store used by all storage functions when no explicit store
# is passed. Callers that need isolation (e.g. tests) should either call
# _reset_store() before each test or pass their own VersionStore instance.
_default_store: VersionStore = VersionStore()


# ---------------------------------------------------------------------------
# Storage functions
# ---------------------------------------------------------------------------


def _store_version(
    version: Version,
    store: Optional[VersionStore] = None,
) -> None:
    """Persist a ``Version`` object into the store.

    Inserts *version* into ``store._versions`` keyed by ``version.id``. If a
    version with the same ``id`` already exists it is silently overwritten
    (idempotent — safe to call twice with an identical object).

    This is the **only** function that writes to ``_versions``. All version
    creation paths must go through here.

    Parameters
    ----------
    version : Version
        The version to store. Must be a fully constructed, immutable
        :class:`Version` instance.
    store : VersionStore, optional
        Target store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    None

    Examples
    --------
    >>> import uuid
    >>> from datetime import datetime, timezone
    >>> store = VersionStore()
    >>> v = Version(
    ...     id=str(uuid.uuid4()), prompt_id=str(uuid.uuid4()),
    ...     content="Hello", parent_version_id=None, branch_name="main",
    ...     commit_message="init", hash="a" * 64,
    ...     created_at=datetime.now(timezone.utc),
    ... )
    >>> _store_version(v, store=store)
    >>> len(store._versions)
    1
    """
    target = store if store is not None else _default_store
    target._versions[version.id] = version


def _get_version(
    version_id: str,
    store: Optional[VersionStore] = None,
) -> Version:
    """Retrieve a single ``Version`` by its ID.

    Parameters
    ----------
    version_id : str
        UUID4 string identifying the version to retrieve.
    store : VersionStore, optional
        Source store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    Version
        The version object stored under *version_id*.

    Raises
    ------
    KeyError
        If no version with *version_id* exists in the store. The error
        message includes the unknown ID for easier debugging.

    Examples
    --------
    >>> store = VersionStore()
    >>> _get_version("nonexistent", store=store)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    KeyError: "version_id 'nonexistent' not found in store"
    """
    target = store if store is not None else _default_store
    try:
        return target._versions[version_id]
    except KeyError:
        raise KeyError(
            f"version_id '{version_id}' not found in store"
        ) from None


def _get_versions_by_prompt(
    prompt_id: str,
    store: Optional[VersionStore] = None,
) -> list[Version]:
    """Return all versions belonging to a given prompt, sorted oldest-first.

    Performs a linear scan over the store (O(n) in total versions). For the
    in-memory use case this is acceptable. A future database-backed
    implementation would replace this with an indexed query on ``prompt_id``.

    Parameters
    ----------
    prompt_id : str
        UUID4 string identifying the prompt whose versions to retrieve.
    store : VersionStore, optional
        Source store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    list[Version]
        All versions for *prompt_id*, sorted by ``created_at`` ascending
        (oldest first). Returns an empty list if no versions exist for the
        prompt.

    Examples
    --------
    >>> store = VersionStore()
    >>> _get_versions_by_prompt("unknown-prompt", store=store)
    []
    """
    target = store if store is not None else _default_store
    results = [
        v for v in target._versions.values()
        if v.prompt_id == prompt_id
    ]
    results.sort(key=lambda v: v.created_at)
    return results


def _reset_store(store: Optional[VersionStore] = None) -> None:
    """Clear all versions and branch HEADs from a store.

    Intended for use in tests that need a clean slate before each test case.
    Empties both ``_versions`` and ``_branches`` in-place so the same store
    object can be reused without recreation.

    Parameters
    ----------
    store : VersionStore, optional
        The store to reset. Defaults to the module-level ``_default_store``.

    Returns
    -------
    None

    Examples
    --------
    >>> store = VersionStore()
    >>> _reset_store(store=store)
    >>> len(store._versions)
    0
    >>> len(store._branches)
    0
    """
    target = store if store is not None else _default_store
    target._versions.clear()
    target._branches.clear()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_prompt_hash(content: str) -> str:
    """Compute a deterministic SHA-256 hash of prompt content.

    Encodes *content* as UTF-8 bytes before hashing so that identical text
    always produces the same digest regardless of the Python runtime or
    platform. The returned hex digest is 64 lowercase hexadecimal characters.

    This function is **pure** — it has no side effects and does not access
    the version store. It is called by version-creation functions to populate
    the ``Version.hash`` field, and is exposed publicly so callers can
    pre-check for duplicate content before committing.

    Parameters
    ----------
    content : str
        The prompt text to hash. Must be a ``str`` (not bytes). Empty strings
        are valid and produce a deterministic digest.

    Returns
    -------
    str
        A 64-character lowercase hexadecimal string representing the
        SHA-256 digest of *content* encoded as UTF-8.

    Raises
    ------
    TypeError
        If *content* is not a ``str``.

    Examples
    --------
    >>> h = compute_prompt_hash("You are a helpful assistant.")
    >>> len(h)
    64
    >>> all(c in "0123456789abcdef" for c in h)
    True

    The result is deterministic — calling twice yields the same digest:

    >>> compute_prompt_hash("hello") == compute_prompt_hash("hello")
    True

    Different inputs produce different digests:

    >>> compute_prompt_hash("hello") == compute_prompt_hash("Hello")
    False

    Empty string is valid:

    >>> len(compute_prompt_hash(""))
    64
    """
    if not isinstance(content, str):
        raise TypeError(
            f"compute_prompt_hash() expects a str, got {type(content).__name__!r}"
        )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Version creation
# ---------------------------------------------------------------------------


def create_version(
    prompt_id: str,
    content: str,
    commit_message: str,
    branch_name: str = "main",
    parent_version_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    store: Optional[VersionStore] = None,
) -> Version:
    """Create and store a new immutable prompt version.

    This is the primary entry point for adding a version to the store. It
    handles all four responsibilities automatically:

    1. **UUID generation** — a UUID4 is generated for ``Version.id``.
    2. **Hash computation** — ``compute_prompt_hash(content)`` is called to
       populate ``Version.hash``.
    3. **Timestamp** — the current UTC time is recorded in
       ``Version.created_at``.
    4. **Storage** — the version is written to the store via
       ``_store_version`` and the branch HEAD is advanced.

    Parent resolution
    -----------------
    If *parent_version_id* is ``None`` **and** the target branch already has
    a HEAD, that HEAD version becomes the parent automatically. This mirrors
    Git's behaviour of implicitly parenting new commits on the current HEAD.
    If the branch has no HEAD yet (first commit on the branch), the new
    version becomes a root node (``parent_version_id = None``).

    Parameters
    ----------
    prompt_id : str
        UUID4 string identifying the prompt being versioned.
    content : str
        Full prompt text for this version.
    commit_message : str
        Short description of the change (e.g.
        ``"Add safety constraint to system prompt"``).
    branch_name : str, optional
        Branch to commit on. Defaults to ``"main"``.
    parent_version_id : str or None, optional
        Explicit parent version ID. When ``None`` (the default), the current
        HEAD of *branch_name* is used if one exists.
    metadata : dict or None, optional
        Arbitrary key/value pairs to attach. A shallow copy is stored so
        the caller’s dict is not aliased.
    store : VersionStore, optional
        Target store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    Version
        The newly created, immutable :class:`Version` object.

    Raises
    ------
    TypeError
        If *content* is not a ``str`` (propagated from
        ``compute_prompt_hash``).
    KeyError
        If an explicit *parent_version_id* is provided but does not exist in
        the store.

    Examples
    --------
    >>> store = VersionStore()
    >>> v1 = create_version(
    ...     prompt_id="p1",
    ...     content="You are a helpful assistant.",
    ...     commit_message="Initial version",
    ...     store=store,
    ... )
    >>> v1.branch_name
    'main'
    >>> v1.parent_version_id is None
    True
    >>> len(v1.hash)
    64

    A second commit on the same branch inherits the first as its parent:

    >>> v2 = create_version(
    ...     prompt_id="p1",
    ...     content="You are a concise, helpful assistant.",
    ...     commit_message="Improve tone",
    ...     store=store,
    ... )
    >>> v2.parent_version_id == v1.id
    True
    """
    target = store if store is not None else _default_store

    # Resolve parent: use explicit value, or fall back to current branch HEAD.
    resolved_parent = parent_version_id
    if resolved_parent is None:
        resolved_parent = target._branches.get((prompt_id, branch_name))

    # Validate explicit parent exists (guards against dangling references).
    if resolved_parent is not None and resolved_parent not in target._versions:
        raise KeyError(
            f"parent_version_id '{resolved_parent}' not found in store"
        )

    version = Version(
        id=str(uuid.uuid4()),
        prompt_id=prompt_id,
        content=content,
        parent_version_id=resolved_parent,
        branch_name=branch_name,
        commit_message=commit_message,
        hash=compute_prompt_hash(content),
        created_at=datetime.now(timezone.utc),
        # Shallow-copy metadata to prevent caller aliasing.
        metadata=dict(metadata) if metadata is not None else {},
    )

    # Write to store and advance branch HEAD.
    _store_version(version, store=target)
    target._branches[(prompt_id, branch_name)] = version.id

    return version


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------


def branch_version(
    base_version_id: str,
    new_branch_name: str,
    commit_message: str,
    metadata: Optional[dict] = None,
    store: Optional[VersionStore] = None,
) -> Version:
    """Create a new version on a new branch, forked from an existing version.

    The new version:

    - copies *content* from the base version (no text change at fork point)
    - sets ``parent_version_id`` to *base_version_id* (preserves lineage)
    - sets ``branch_name`` to *new_branch_name*
    - records ``metadata["branched_from"] = base_version_id`` automatically

    The new branch HEAD is registered in the store so that subsequent calls
    to ``create_version(..., branch_name=new_branch_name)`` will
    automatically chain from this fork point.

    Parameters
    ----------
    base_version_id : str
        UUID4 string of the version to fork from. Must exist in the store.
    new_branch_name : str
        Name of the new branch to create (e.g. ``"experiment-tone"``).
    commit_message : str
        Description of the branch point
        (e.g. ``"Branch: try more concise tone"``).
    metadata : dict or None, optional
        Additional key/value pairs to attach. A shallow copy is stored.
        ``"branched_from"`` is always set automatically and cannot be
        overridden via this argument.
    store : VersionStore, optional
        Target store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    Version
        The first version on the new branch.

    Raises
    ------
    KeyError
        If *base_version_id* does not exist in the store.
    ValueError
        If *new_branch_name* already has a HEAD for the same ``prompt_id``.
        Branching into an existing branch would silently move its HEAD,
        which is almost always a mistake.

    Examples
    --------
    >>> store = VersionStore()
    >>> v1 = create_version("p1", "Hello world", "init", store=store)
    >>> bv = branch_version(v1.id, "experiment", "Try new tone", store=store)
    >>> bv.branch_name
    'experiment'
    >>> bv.parent_version_id == v1.id
    True
    >>> bv.content == v1.content
    True
    >>> bv.metadata["branched_from"] == v1.id
    True
    """
    target = store if store is not None else _default_store

    # Fetch and validate the base version.
    if base_version_id not in target._versions:
        raise KeyError(
            f"base_version_id '{base_version_id}' not found in store"
        )
    base = target._versions[base_version_id]

    # Guard: prevent silently overwriting an existing branch HEAD.
    if (base.prompt_id, new_branch_name) in target._branches:
        raise ValueError(
            f"Branch '{new_branch_name}' already exists for prompt "
            f"'{base.prompt_id}'. Use create_version() to add commits to "
            f"an existing branch."
        )

    # Build metadata: caller-supplied values plus automatic provenance key.
    merged_meta: dict = dict(metadata) if metadata is not None else {}
    merged_meta["branched_from"] = base_version_id

    version = Version(
        id=str(uuid.uuid4()),
        prompt_id=base.prompt_id,
        content=base.content,              # copy content from base
        parent_version_id=base_version_id, # maintain lineage
        branch_name=new_branch_name,
        commit_message=commit_message,
        hash=compute_prompt_hash(base.content),
        created_at=datetime.now(timezone.utc),
        metadata=merged_meta,
    )

    # Write to store and register new branch HEAD.
    _store_version(version, store=target)
    target._branches[(base.prompt_id, new_branch_name)] = version.id

    return version


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_version(
    target_version_id: str,
    commit_message: str,
    branch_name: Optional[str] = None,
    store: Optional[VersionStore] = None,
) -> Version:
    """Restore a prompt to the content of an earlier version.

    Creates a **new** version whose content is copied from *target_version_id*.
    The original version and all intermediate versions are untouched — history
    is never mutated.

    This mirrors Git's ``git revert`` semantics: the rollback is expressed as
    a forward commit that happens to carry old content, not as a deletion of
    newer commits.

    Parent resolution
    -----------------
    The new version's ``parent_version_id`` is set to the **current HEAD** of
    the target branch (not to *target_version_id* itself). This keeps the
    rollback commit correctly placed at the tip of the branch timeline.

    Branch resolution
    -----------------
    If *branch_name* is ``None``, the branch is inherited from the target
    version (``target_version.branch_name``). Pass an explicit *branch_name*
    to roll back onto a different branch.

    Parameters
    ----------
    target_version_id : str
        UUID4 string of the version whose content to restore.
    commit_message : str
        Description of the rollback
        (e.g. ``"Revert: remove hallucination-prone addendum"``).
    branch_name : str or None, optional
        Branch to commit the rollback on. Inherits from the target version
        when ``None``.
    store : VersionStore, optional
        Target store. Defaults to the module-level ``_default_store``.

    Returns
    -------
    Version
        The newly created rollback version.

    Raises
    ------
    KeyError
        If *target_version_id* does not exist in the store.

    Examples
    --------
    >>> store = VersionStore()
    >>> v1 = create_version("p1", "Good prompt", "init", store=store)
    >>> v2 = create_version("p1", "Bad prompt", "mistake", store=store)
    >>> rb = rollback_version(v1.id, "Revert to v1", store=store)
    >>> rb.content == v1.content
    True
    >>> rb.parent_version_id == v2.id      # parent is current HEAD, not v1
    True
    >>> rb.metadata["rollback_from"] == v1.id
    True
    """
    target = store if store is not None else _default_store

    # Fetch and validate the target version.
    if target_version_id not in target._versions:
        raise KeyError(
            f"target_version_id '{target_version_id}' not found in store"
        )
    target_v = target._versions[target_version_id]

    # Resolve branch: explicit arg wins, else inherit from the target version.
    effective_branch = branch_name if branch_name is not None else target_v.branch_name

    # Parent = current HEAD of the effective branch (tip of the timeline).
    current_head = target._branches.get((target_v.prompt_id, effective_branch))

    version = Version(
        id=str(uuid.uuid4()),
        prompt_id=target_v.prompt_id,
        content=target_v.content,           # restore content from target
        parent_version_id=current_head,     # place at tip of branch
        branch_name=effective_branch,
        commit_message=commit_message,
        hash=compute_prompt_hash(target_v.content),
        created_at=datetime.now(timezone.utc),
        metadata={"rollback_from": target_version_id},
    )

    # Write to store and advance branch HEAD.
    _store_version(version, store=target)
    target._branches[(target_v.prompt_id, effective_branch)] = version.id

    return version


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------


def get_version(
    version_id: str,
    store: Optional[VersionStore] = None,
) -> Version:
    """Return the Version with the given ID.

    A thin public wrapper over ``_get_version`` suitable for use by
    callers outside this module.

    Parameters
    ----------
    version_id : str
        UUID4 string of the version to retrieve.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    Version

    Raises
    ------
    KeyError
        If *version_id* does not exist.
    """
    return _get_version(version_id, store=store)


def get_children(
    version_id: str,
    store: Optional[VersionStore] = None,
) -> list[Version]:
    """Return all direct child versions of *version_id*.

    A child is any version whose ``parent_version_id`` equals *version_id*.
    Children may live on any branch, so this correctly handles fork points
    where a single version has children on multiple branches.

    Parameters
    ----------
    version_id : str
        UUID4 string of the parent version.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    list[Version]
        Direct children sorted by ``created_at`` ascending. Empty list if
        *version_id* has no children.
    """
    target = store if store is not None else _default_store
    children = [
        v for v in target._versions.values()
        if v.parent_version_id == version_id
    ]
    children.sort(key=lambda v: v.created_at)
    return children


def get_version_lineage(
    version_id: str,
    store: Optional[VersionStore] = None,
) -> list[Version]:
    """Return the ordered path from the root version to *version_id*.

    Walks the ``parent_version_id`` chain from *version_id* back to the
    root (where ``parent_version_id is None``), then reverses the result
    so the list reads **root → target** (oldest first).

    Parameters
    ----------
    version_id : str
        UUID4 string of the target version.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    list[Version]
        ``[root, ..., parent, target]`` — oldest ancestor first.

    Raises
    ------
    KeyError
        If *version_id* does not exist.
    RuntimeError
        If a cycle is detected in the parent chain.
    """
    target = store if store is not None else _default_store
    lineage: list[Version] = []
    current_id: Optional[str] = version_id
    visited: set[str] = set()

    while current_id is not None:
        if current_id in visited:
            raise RuntimeError(
                f"Cycle detected in version history at '{current_id}'"
            )
        visited.add(current_id)
        v = _get_version(current_id, store=target)
        lineage.append(v)
        current_id = v.parent_version_id

    lineage.reverse()          # root → target
    return lineage


def get_root_version(
    prompt_id: str,
    store: Optional[VersionStore] = None,
) -> Version:
    """Return the root version for a given prompt.

    The root is the version belonging to *prompt_id* whose
    ``parent_version_id`` is ``None``. If multiple roots exist (e.g. due to
    manual store manipulation or multiple disconnected import operations),
    the oldest one by ``created_at`` is returned.

    Parameters
    ----------
    prompt_id : str
        UUID4 string identifying the prompt.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    Version
        The root version.

    Raises
    ------
    KeyError
        If no versions exist for *prompt_id* or none have
        ``parent_version_id == None``.
    """
    target = store if store is not None else _default_store
    roots = [
        v for v in target._versions.values()
        if v.prompt_id == prompt_id and v.parent_version_id is None
    ]
    if not roots:
        raise KeyError(
            f"No root version found for prompt_id '{prompt_id}'"
        )
    roots.sort(key=lambda v: v.created_at)
    return roots[0]


def get_branch_versions(
    branch_name: str,
    prompt_id: str,
    store: Optional[VersionStore] = None,
) -> list[Version]:
    """Return all versions on *branch_name* for a given prompt, oldest first.

    Parameters
    ----------
    branch_name : str
        Name of the branch to query.
    prompt_id : str
        UUID4 string scoping the query to a single prompt.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    list[Version]
        Versions on the branch sorted by ``created_at`` ascending.
        Empty list if no versions exist for the given prompt/branch
        combination.
    """
    target = store if store is not None else _default_store
    results = [
        v for v in target._versions.values()
        if v.prompt_id == prompt_id and v.branch_name == branch_name
    ]
    results.sort(key=lambda v: v.created_at)
    return results


def get_version_tree(
    prompt_id: str,
    store: Optional[VersionStore] = None,
) -> list[dict]:
    """Return a nested, JSON-serialisable tree of all versions for *prompt_id*.

    The internal store is a flat dict; this function dynamically constructs
    a nested parent→children tree on demand using the ``parent_version_id``
    relationships. Multiple branches are naturally represented as sibling
    children at their fork points.

    Each node in the returned tree is a plain dict:

    .. code-block:: python

        {
            "id":                str,
            "prompt_id":         str,
            "branch_name":       str,
            "commit_message":    str,
            "hash":              str,
            "created_at":        str,   # ISO-8601
            "parent_version_id": str | None,
            "metadata":          dict,
            "children":          list[dict],   # recursive
        }

    Parameters
    ----------
    prompt_id : str
        Scope the tree to versions belonging to this prompt.
    store : VersionStore, optional
        Source store. Defaults to ``_default_store``.

    Returns
    -------
    list[dict]
        One dict per root node. Typically a list of one element, but may
        contain multiple roots in edge cases. Returns an empty list if no
        versions exist for *prompt_id*.
    """
    target = store if store is not None else _default_store

    nodes = [
        v for v in target._versions.values()
        if v.prompt_id == prompt_id
    ]
    if not nodes:
        return []

    ids_in_scope: set[str] = {n.id for n in nodes}

    # Build children adjacency map (invert parent pointers).
    children_map: dict[str, list[str]] = {n.id: [] for n in nodes}
    for node in nodes:
        if node.parent_version_id and node.parent_version_id in ids_in_scope:
            children_map[node.parent_version_id].append(node.id)

    # Sort children lists by created_at for deterministic output.
    versions_by_id = {n.id: n for n in nodes}
    for clist in children_map.values():
        clist.sort(key=lambda vid: versions_by_id[vid].created_at)

    def _build_node(vid: str) -> dict:
        v = versions_by_id[vid]
        return {
            "id":                v.id,
            "prompt_id":         v.prompt_id,
            "branch_name":       v.branch_name,
            "commit_message":    v.commit_message,
            "hash":              v.hash,
            "created_at":        v.created_at.isoformat(),
            "parent_version_id": v.parent_version_id,
            "metadata":          dict(v.metadata),
            "children":          [_build_node(c) for c in children_map[vid]],
        }

    # Roots: nodes with no parent inside this prompt scope.
    roots = [
        n for n in nodes
        if n.parent_version_id is None or n.parent_version_id not in ids_in_scope
    ]
    roots.sort(key=lambda v: v.created_at)
    return [_build_node(r.id) for r in roots]


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------


def _dmp_ops_to_list(diffs: list[tuple[int, str]]) -> list[list[str]]:
    """Convert diff-match-patch native ops to the canonical ``[type, text]`` format.

    diff-match-patch uses integer constants:

    - ``DMP.DIFF_EQUAL   ==  0``  →  ``"="``
    - ``DMP.DIFF_INSERT  ==  1``  →  ``"+"``
    - ``DMP.DIFF_DELETE  == -1``  →  ``"-"``

    Parameters
    ----------
    diffs : list[tuple[int, str]]
        Raw diff-match-patch output.

    Returns
    -------
    list[list[str]]
        ``[["=", "text"], ["+", "text"], ["-", "text"], ...]``
    """
    op_map = {DMP.DIFF_EQUAL: "=", DMP.DIFF_INSERT: "+", DMP.DIFF_DELETE: "-"}
    return [[op_map[op], text] for op, text in diffs]


def compute_char_diff(
    old: str,
    new: str,
) -> list[list[str]]:
    """Compute a character-level diff between two strings.

    Uses ``diff-match-patch`` for character-granularity diffing followed by
    a semantic cleanup pass that makes the result more human-readable (e.g.
    prefers word boundaries over arbitrary mid-word splits).

    Output format::

        [
          ["=", "unchanged text"],
          ["+", "added text"],
          ["-", "removed text"],
          ...
        ]

    The output is deterministic: identical inputs always produce identical
    output.

    Parameters
    ----------
    old : str
        The base (original) text.
    new : str
        The target (modified) text.

    Returns
    -------
    list[list[str]]
        Ordered list of ``[op, text]`` operations where *op* is
        ``"="``, ``"+"``, or ``"-"``.

    Examples
    --------
    >>> compute_char_diff("hello world", "hello there")
    [['=', 'hello '], ['-', 'world'], ['+', 'there']]
    """
    dmp = DMP()
    diffs = dmp.diff_main(old, new)
    dmp.diff_cleanupSemantic(diffs)
    return _dmp_ops_to_list(diffs)


def compute_line_diff(
    old: str,
    new: str,
) -> list[list[str]]:
    """Compute a line-level diff between two strings.

    Uses ``diff-match-patch``’s ``diff_linesToChars`` /
    ``diff_charsToLines`` technique to operate at line granularity rather
    than character granularity. Each operation covers one or more complete
    lines. Suitable for the visual diff view described in the RoadMap.

    Output format is identical to :func:`compute_char_diff`::

        [
          ["=", "unchanged line\\n"],
          ["+", "added line\\n"],
          ["-", "removed line\\n"],
          ...
        ]

    Parameters
    ----------
    old : str
        The base (original) text.
    new : str
        The target (modified) text.

    Returns
    -------
    list[list[str]]
        Ordered list of ``[op, text]`` operations where *op* is
        ``"="``, ``"+"``, or ``"-"``.

    Examples
    --------
    >>> result = compute_line_diff("line1\\nline2\\n", "line1\\nline3\\n")
    >>> result[0]
    ['=', 'line1\\n']
    """
    dmp = DMP()
    chars_old, chars_new, line_array = dmp.diff_linesToChars(old, new)
    diffs = dmp.diff_main(chars_old, chars_new, checklines=False)
    dmp.diff_charsToLines(diffs, line_array)
    dmp.diff_cleanupSemantic(diffs)
    return _dmp_ops_to_list(diffs)
