"""
Unit tests for src/prompt_versioning.py

All tests use isolated VersionStore instances; the module-level _default_store
is never touched. Run with:

    pytest tests/test_prompt_versioning.py -v
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.prompt_versioning import (
    Version,
    VersionStore,
    _get_versions_by_prompt,
    _reset_store,
    branch_version,
    compute_char_diff,
    compute_line_diff,
    compute_prompt_hash,
    create_version,
    get_branch_versions,
    get_children,
    get_root_version,
    get_version,
    get_version_lineage,
    get_version_tree,
    rollback_version,
)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

PROMPT_A = "prompt-aaa"
PROMPT_B = "prompt-bbb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> VersionStore:
    """Fresh isolated store for each test."""
    return VersionStore()


@pytest.fixture()
def v1(store: VersionStore) -> Version:
    return create_version(PROMPT_A, "Hello world", "Initial commit", store=store)


@pytest.fixture()
def v2(store: VersionStore, v1: Version) -> Version:
    return create_version(PROMPT_A, "Hello world v2", "Second commit", store=store)


# ---------------------------------------------------------------------------
# compute_prompt_hash
# ---------------------------------------------------------------------------


class TestComputePromptHash:
    def test_returns_64_char_hex(self):
        h = compute_prompt_hash("some text")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        assert compute_prompt_hash("abc") == compute_prompt_hash("abc")

    def test_different_input_different_hash(self):
        assert compute_prompt_hash("abc") != compute_prompt_hash("xyz")

    def test_empty_string_valid(self):
        h = compute_prompt_hash("")
        assert len(h) == 64

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            compute_prompt_hash(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# VersionStore
# ---------------------------------------------------------------------------


class TestVersionStore:
    def test_empty_on_creation(self):
        s = VersionStore()
        assert len(s._versions) == 0
        assert len(s._branches) == 0

    def test_reset_clears_store(self, store, v1, v2):
        _reset_store(store=store)
        assert len(store._versions) == 0
        assert len(store._branches) == 0


# ---------------------------------------------------------------------------
# create_version
# ---------------------------------------------------------------------------


class TestCreateVersion:
    def test_returns_version_object(self, store):
        v = create_version(PROMPT_A, "First", "msg", store=store)
        assert isinstance(v, Version)

    def test_version_stored(self, store):
        v = create_version(PROMPT_A, "First", "msg", store=store)
        assert v.id in store._versions

    def test_fields_set_correctly(self, store):
        v = create_version(PROMPT_A, "content here", "my msg", branch_name="dev", store=store)
        assert v.prompt_id == PROMPT_A
        assert v.content == "content here"
        assert v.commit_message == "my msg"
        assert v.branch_name == "dev"
        assert v.hash == compute_prompt_hash("content here")
        assert v.parent_version_id is None  # first on branch

    def test_hash_equals_compute_prompt_hash(self, store):
        content = "Hello, Prompt!"
        v = create_version(PROMPT_A, content, "test hash", store=store)
        assert v.hash == compute_prompt_hash(content)

    def test_second_version_auto_parents_first(self, store, v1, v2):
        assert v2.parent_version_id == v1.id

    def test_branch_head_advanced(self, store):
        v = create_version(PROMPT_A, "text", "msg", store=store)
        assert store._branches[(PROMPT_A, "main")] == v.id

    def test_first_commit_on_new_branch_has_no_parent(self, store, v1):
        v = create_version(PROMPT_A, "feat text", "branch", branch_name="feat", store=store)
        assert v.parent_version_id is None

    def test_metadata_stored(self, store):
        v = create_version(PROMPT_A, "t", "m", metadata={"author": "alice"}, store=store)
        assert v.metadata["author"] == "alice"

    def test_metadata_defaults_to_empty_dict(self, store):
        v = create_version(PROMPT_A, "t", "m", store=store)
        assert v.metadata == {}

    def test_metadata_shallow_copied(self, store):
        m = {"k": "v"}
        v = create_version(PROMPT_A, "t", "m", metadata=m, store=store)
        m["k"] = "mutated"
        assert v.metadata["k"] == "v"

    def test_separate_prompts_independent(self, store):
        va = create_version(PROMPT_A, "a", "msg", store=store)
        vb = create_version(PROMPT_B, "b", "msg", store=store)
        assert va.parent_version_id is None
        assert vb.parent_version_id is None

    def test_explicit_parent_version_id_honoured(self, store, v1, v2):
        v3 = create_version(PROMPT_A, "v3", "third", parent_version_id=v1.id, store=store)
        assert v3.parent_version_id == v1.id

    def test_bad_parent_version_id_raises_key_error(self, store):
        with pytest.raises(KeyError, match="not found"):
            create_version(PROMPT_A, "x", "msg", parent_version_id="bad-id", store=store)

    def test_version_is_immutable(self, store, v1):
        with pytest.raises(Exception):  # FrozenInstanceError
            v1.content = "mutated"  # type: ignore[misc]

    def test_created_at_is_tz_aware_utc(self, store, v1):
        assert v1.created_at.tzinfo is not None

    def test_uuid4_id(self, store, v1):
        # Should not raise
        uuid.UUID(v1.id, version=4)


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_retrieves_correct_version(self, store, v1):
        assert get_version(v1.id, store=store) is v1

    def test_raises_key_error_on_missing(self, store):
        with pytest.raises(KeyError, match="not found"):
            get_version("nonexistent-id", store=store)


# ---------------------------------------------------------------------------
# _get_versions_by_prompt  (internal, used as list_versions equivalent)
# ---------------------------------------------------------------------------


class TestGetVersionsByPrompt:
    def test_returns_all_for_prompt(self, store, v1, v2):
        versions = _get_versions_by_prompt(PROMPT_A, store=store)
        ids = [v.id for v in versions]
        assert v1.id in ids and v2.id in ids

    def test_empty_for_unknown_prompt(self, store):
        assert _get_versions_by_prompt("unknown", store=store) == []

    def test_sorted_chronologically(self, store, v1, v2):
        versions = _get_versions_by_prompt(PROMPT_A, store=store)
        times = [v.created_at for v in versions]
        assert times == sorted(times)

    def test_does_not_mix_prompts(self, store):
        create_version(PROMPT_A, "a", "msg", store=store)
        create_version(PROMPT_B, "b", "msg", store=store)
        assert all(v.prompt_id == PROMPT_A for v in _get_versions_by_prompt(PROMPT_A, store=store))
        assert all(v.prompt_id == PROMPT_B for v in _get_versions_by_prompt(PROMPT_B, store=store))


# ---------------------------------------------------------------------------
# rollback_version
# ---------------------------------------------------------------------------


class TestRollbackVersion:
    def test_creates_new_version(self, store, v1, v2):
        rb = rollback_version(v1.id, "Rollback to v1", store=store)
        assert rb.id != v1.id
        assert rb.id != v2.id

    def test_content_matches_target(self, store, v1, v2):
        rb = rollback_version(v1.id, "Rollback to v1", store=store)
        assert rb.content == v1.content

    def test_hash_matches_content(self, store, v1, v2):
        rb = rollback_version(v1.id, "Rollback", store=store)
        assert rb.hash == compute_prompt_hash(v1.content)

    def test_parent_is_current_head(self, store, v1, v2):
        rb = rollback_version(v1.id, "Rollback", store=store)
        assert rb.parent_version_id == v2.id

    def test_inherits_branch_from_target_when_none(self, store):
        v = create_version(PROMPT_A, "text", "msg", branch_name="feat", store=store)
        rb = rollback_version(v.id, "Rollback on feat", store=store)
        assert rb.branch_name == "feat"

    def test_explicit_branch_overrides(self, store, v1):
        create_version(PROMPT_A, "feat text", "msg", branch_name="feat", store=store)
        rb = rollback_version(v1.id, "Rollback to feat", branch_name="feat", store=store)
        assert rb.branch_name == "feat"

    def test_metadata_records_rollback_from(self, store, v1, v2):
        rb = rollback_version(v1.id, "Rollback", store=store)
        assert rb.metadata["rollback_from"] == v1.id

    def test_raises_key_error_on_missing_target(self, store):
        with pytest.raises(KeyError):
            rollback_version("bad-id", "msg", store=store)

    def test_history_is_not_mutated(self, store, v1, v2):
        orig_content = v1.content
        rollback_version(v1.id, "Rollback", store=store)
        assert store._versions[v1.id].content == orig_content

    def test_rollback_version_is_immutable(self, store, v1):
        rb = rollback_version(v1.id, "Rollback", store=store)
        with pytest.raises(Exception):  # FrozenInstanceError
            rb.content = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# branch_version
# ---------------------------------------------------------------------------


class TestBranchVersion:
    def test_returns_version(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert isinstance(bv, Version)

    def test_content_copied_from_base(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert bv.content == v1.content

    def test_parent_is_base_version(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert bv.parent_version_id == v1.id

    def test_branch_name_set(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert bv.branch_name == "feature-x"

    def test_new_branch_head_registered(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert store._branches[(PROMPT_A, "feature-x")] == bv.id

    def test_metadata_contains_branched_from(self, store, v1):
        bv = branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert bv.metadata["branched_from"] == v1.id

    def test_raises_key_error_on_missing_base(self, store):
        with pytest.raises(KeyError):
            branch_version("bad-id", "feat", "msg", store=store)

    def test_raises_value_error_if_branch_exists(self, store, v1):
        branch_version(v1.id, "feat", "first branch", store=store)
        with pytest.raises(ValueError, match="already exists"):
            branch_version(v1.id, "feat", "duplicate branch", store=store)

    def test_original_branch_head_unchanged(self, store, v1, v2):
        branch_version(v1.id, "feature-x", "Branch from v1", store=store)
        assert store._branches[(PROMPT_A, "main")] == v2.id

    def test_caller_metadata_preserved(self, store, v1):
        bv = branch_version(v1.id, "feat", "msg", metadata={"author": "bob"}, store=store)
        assert bv.metadata["author"] == "bob"
        assert bv.metadata["branched_from"] == v1.id


# ---------------------------------------------------------------------------
# get_version_lineage
# ---------------------------------------------------------------------------


class TestGetVersionLineage:
    def test_single_root(self, store, v1):
        lineage = get_version_lineage(v1.id, store=store)
        assert lineage == [v1]

    def test_two_versions_root_to_target(self, store, v1, v2):
        lineage = get_version_lineage(v2.id, store=store)
        assert lineage[0] is v1
        assert lineage[1] is v2

    def test_lineage_ends_at_root(self, store, v1, v2):
        lineage = get_version_lineage(v2.id, store=store)
        assert lineage[0].parent_version_id is None

    def test_cross_branch_lineage(self, store, v1):
        bv = branch_version(v1.id, "feat", "branch", store=store)
        v_feat = create_version(PROMPT_A, "feat content", "on feat",
                                branch_name="feat", store=store)
        lineage = get_version_lineage(v_feat.id, store=store)
        ids = [v.id for v in lineage]
        assert ids == [v1.id, bv.id, v_feat.id]

    def test_raises_key_error_on_unknown(self, store):
        with pytest.raises(KeyError):
            get_version_lineage("nonexistent", store=store)


# ---------------------------------------------------------------------------
# get_root_version
# ---------------------------------------------------------------------------


class TestGetRootVersion:
    def test_returns_root(self, store, v1, v2):
        root = get_root_version(PROMPT_A, store=store)
        assert root is v1

    def test_root_has_no_parent(self, store, v1):
        root = get_root_version(PROMPT_A, store=store)
        assert root.parent_version_id is None

    def test_raises_key_error_for_unknown_prompt(self, store):
        with pytest.raises(KeyError):
            get_root_version("unknown-prompt", store=store)


# ---------------------------------------------------------------------------
# get_children
# ---------------------------------------------------------------------------


class TestGetChildren:
    def test_single_child(self, store, v1, v2):
        children = get_children(v1.id, store=store)
        assert len(children) == 1
        assert children[0] is v2

    def test_no_children_for_leaf(self, store, v1, v2):
        assert get_children(v2.id, store=store) == []

    def test_cross_branch_children(self, store, v1, v2):
        bv = branch_version(v1.id, "feat", "fork", store=store)
        children = get_children(v1.id, store=store)
        child_ids = {c.id for c in children}
        assert {v2.id, bv.id} == child_ids

    def test_children_sorted_oldest_first(self, store, v1, v2):
        bv = branch_version(v1.id, "feat", "fork", store=store)
        children = get_children(v1.id, store=store)
        assert children == sorted(children, key=lambda v: v.created_at)


# ---------------------------------------------------------------------------
# get_branch_versions
# ---------------------------------------------------------------------------


class TestGetBranchVersions:
    def test_returns_all_on_branch(self, store, v1, v2):
        versions = get_branch_versions("main", PROMPT_A, store=store)
        ids = {v.id for v in versions}
        assert {v1.id, v2.id} == ids

    def test_empty_for_unknown_branch(self, store, v1):
        assert get_branch_versions("nonexistent", PROMPT_A, store=store) == []

    def test_sorted_oldest_first(self, store, v1, v2):
        versions = get_branch_versions("main", PROMPT_A, store=store)
        assert versions == sorted(versions, key=lambda v: v.created_at)

    def test_scoped_to_prompt(self, store, v1):
        create_version(PROMPT_B, "b", "msg", store=store)
        versions = get_branch_versions("main", PROMPT_A, store=store)
        assert all(v.prompt_id == PROMPT_A for v in versions)


# ---------------------------------------------------------------------------
# get_version_tree
# ---------------------------------------------------------------------------


class TestGetVersionTree:
    def test_empty_for_unknown_prompt(self, store):
        assert get_version_tree("unknown", store=store) == []

    def test_single_root_no_children(self, store, v1):
        tree = get_version_tree(PROMPT_A, store=store)
        assert len(tree) == 1
        assert tree[0]["id"] == v1.id
        assert tree[0]["children"] == []

    def test_linear_chain(self, store, v1, v2):
        tree = get_version_tree(PROMPT_A, store=store)
        root = tree[0]
        assert root["id"] == v1.id
        assert len(root["children"]) == 1
        assert root["children"][0]["id"] == v2.id

    def test_branch_shows_two_children(self, store, v1):
        v2 = create_version(PROMPT_A, "main text", "main v2", store=store)
        bv = branch_version(v1.id, "feat", "branch", store=store)
        tree = get_version_tree(PROMPT_A, store=store)
        root = tree[0]
        child_ids = {c["id"] for c in root["children"]}
        assert {v2.id, bv.id} == child_ids

    def test_scoped_to_prompt(self, store):
        create_version(PROMPT_A, "a", "msg", store=store)
        create_version(PROMPT_B, "b", "msg", store=store)
        tree_a = get_version_tree(PROMPT_A, store=store)
        assert all(n["prompt_id"] == PROMPT_A for n in tree_a)

    def test_node_has_expected_keys(self, store, v1):
        node = get_version_tree(PROMPT_A, store=store)[0]
        for key in ("id", "prompt_id", "branch_name", "commit_message",
                    "hash", "created_at", "parent_version_id", "metadata", "children"):
            assert key in node

    def test_created_at_is_iso8601_string(self, store, v1):
        node = get_version_tree(PROMPT_A, store=store)[0]
        assert isinstance(node["created_at"], str)
        assert "T" in node["created_at"]

    def test_json_serialisable(self, store, v1, v2):
        import json
        tree = get_version_tree(PROMPT_A, store=store)
        json.dumps(tree)  # must not raise


# ---------------------------------------------------------------------------
# compute_char_diff
# ---------------------------------------------------------------------------


class TestComputeCharDiff:
    def test_returns_list(self):
        result = compute_char_diff("hello world", "hello there")
        assert isinstance(result, list)

    def test_each_op_is_two_element_list(self):
        result = compute_char_diff("hello world", "hello there")
        for op in result:
            assert isinstance(op, list)
            assert len(op) == 2
            assert op[0] in ("+", "-", "=")
            assert isinstance(op[1], str)

    def test_reconstruct_old(self):
        old, new = "Hello world", "Hello there"
        diff = compute_char_diff(old, new)
        assert "".join(t for o, t in diff if o in ("=", "-")) == old

    def test_reconstruct_new(self):
        old, new = "Hello world", "Hello there"
        diff = compute_char_diff(old, new)
        assert "".join(t for o, t in diff if o in ("=", "+")) == new

    def test_identical_inputs_equals_only(self):
        diff = compute_char_diff("same", "same")
        assert all(op == "=" for op, _ in diff)

    def test_both_empty(self):
        assert compute_char_diff("", "") == []

    def test_deterministic(self):
        assert compute_char_diff("a", "b") == compute_char_diff("a", "b")


# ---------------------------------------------------------------------------
# compute_line_diff
# ---------------------------------------------------------------------------


class TestComputeLineDiff:
    def test_returns_list(self):
        result = compute_line_diff("line1\nline2\n", "line1\nline3\n")
        assert isinstance(result, list)

    def test_op_types_valid(self):
        result = compute_line_diff("a\nb\n", "a\nc\n")
        for op, text in result:
            assert op in ("+", "-", "=")
            assert isinstance(text, str)

    def test_reconstruct_old(self):
        old, new = "line1\nline2\n", "line1\nline3\n"
        diff = compute_line_diff(old, new)
        assert "".join(t for o, t in diff if o in ("=", "-")) == old

    def test_reconstruct_new(self):
        old, new = "line1\nline2\n", "line1\nline3\n"
        diff = compute_line_diff(old, new)
        assert "".join(t for o, t in diff if o in ("=", "+")) == new

    def test_identical_inputs_equals_only(self):
        diff = compute_line_diff("x\n", "x\n")
        assert all(op == "=" for op, _ in diff)

    def test_deterministic(self):
        assert compute_line_diff("a\n", "b\n") == compute_line_diff("a\n", "b\n")

    def test_pure_no_store_access(self):
        # Should work with no store at all — completely independent
        result1 = compute_line_diff("alpha\n", "beta\n")
        result2 = compute_line_diff("alpha\n", "beta\n")
        assert result1 == result2
