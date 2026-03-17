"""
tests/test_rollback.py — Tests for rollback_version.
"""

import uuid
import pytest

from src.prompt_versioning import (
    Version, VersionStore, branch_version, compute_prompt_hash,
    create_version, get_branch_versions, rollback_version,
)


class TestRollbackVersionReturnType:
    def test_returns_version_instance(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert isinstance(rb, Version)

    def test_id_is_new_uuid4(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.id != v1.id
        assert rb.id != v2.id
        uuid.UUID(rb.id, version=4)


class TestRollbackVersionFields:
    def test_content_copied_from_target(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["target content", "newer content"])
        rb = rollback_version(v1.id, "revert to v1", store=store)
        assert rb.content == v1.content

    def test_hash_matches_target_content(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.hash == compute_prompt_hash(v1.content)

    def test_parent_is_branch_head_not_target(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        rb = rollback_version(v1.id, "revert", store=store)
        # Parent must be v3 (HEAD at time of rollback), NOT v1
        assert rb.parent_version_id == v3.id
        assert rb.parent_version_id != v1.id

    def test_commit_message_preserved(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "My rollback message", store=store)
        assert rb.commit_message == "My rollback message"

    def test_prompt_id_inherited(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.prompt_id == prompt_id

    def test_created_at_is_timezone_aware(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.created_at.tzinfo is not None

    def test_immutable_frozen_dataclass(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        with pytest.raises(Exception):
            rb.content = "mutated"  # type: ignore[misc]


class TestRollbackVersionBranchResolution:
    def test_inherits_branch_from_target_when_none(self, store, prompt_id):
        v1 = create_version(prompt_id, "A", "init", branch_name="feat", store=store)
        rb = rollback_version(v1.id, "revert on feat", store=store)
        assert rb.branch_name == "feat"

    def test_explicit_branch_name_overrides_target(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        rb = rollback_version(v1.id, "rollback onto feat", branch_name="feat", store=store)
        assert rb.branch_name == "feat"

    def test_explicit_branch_parent_is_that_branch_head(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        feat_head_before = store._branches[(prompt_id, "feat")]
        rb = rollback_version(v1.id, "rollback onto feat", branch_name="feat", store=store)
        assert rb.parent_version_id == feat_head_before

    def test_branch_head_advanced_after_rollback(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert store._branches[(prompt_id, "main")] == rb.id


class TestRollbackVersionMetadata:
    def test_rollback_from_auto_set(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.metadata["rollback_from"] == v1.id

    def test_rollback_from_points_to_target_not_parent(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert rb.metadata["rollback_from"] == v1.id
        assert rb.metadata["rollback_from"] != v3.id


class TestRollbackVersionHistoryImmutability:
    def test_original_version_content_unchanged(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["original", "newer"])
        rollback_version(v1.id, "revert", store=store)
        assert store._versions[v1.id].content == "original"

    def test_intermediate_versions_unchanged(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        rollback_version(v1.id, "revert", store=store)
        assert store._versions[v2.id].content == "B"
        assert store._versions[v3.id].content == "C"

    def test_version_count_increases_by_one(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        count_before = len(store._versions)
        rollback_version(v1.id, "revert", store=store)
        assert len(store._versions) == count_before + 1

    def test_original_version_objects_same_identity(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rollback_version(v1.id, "revert", store=store)
        assert store._versions[v1.id] is v1
        assert store._versions[v2.id] is v2

    def test_rollback_version_stored(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        assert store._versions[rb.id] is rb


class TestRollbackVersionChaining:
    def test_sequential_rollbacks_chain_correctly(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb1 = rollback_version(v1.id, "first revert", store=store)
        rb2 = rollback_version(v2.id, "second revert", store=store)
        assert rb2.parent_version_id == rb1.id

    def test_rollback_to_current_head_creates_new_node(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v2.id, "rollback to head", store=store)
        assert rb.id != v2.id
        assert rb.content == v2.content

    def test_rollback_to_root_from_deep_chain(self, store, prompt_id):
        versions = []
        for i in range(10):
            v = create_version(prompt_id, f"version {i}", f"commit {i}", store=store)
            versions.append(v)
        rb = rollback_version(versions[0].id, "rollback to root", store=store)
        assert rb.content == versions[0].content
        assert rb.parent_version_id == versions[-1].id

    def test_rollback_after_branch_onto_main(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        v2 = create_version(prompt_id, "main v2", "m2", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        # Rollback v1's content onto main (where HEAD is v2)
        rb = rollback_version(v1.id, "revert main to root content", store=store)
        assert rb.content == v1.content
        assert rb.branch_name == "main"
        assert rb.parent_version_id == v2.id

    def test_rollback_on_branch_isolation(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        create_version(prompt_id, "feat v2", "fv2", branch_name="feat", store=store)
        # Rollback on feat branch
        rb = rollback_version(bv.id, "revert feat", branch_name="feat", store=store)
        main_ids = {v.id for v in get_branch_versions("main", prompt_id, store=store)}
        assert rb.id not in main_ids


class TestRollbackVersionErrors:
    def test_raises_key_error_on_unknown_target(self, store, prompt_id):
        with pytest.raises(KeyError):
            rollback_version("nonexistent-id", "revert", store=store)

    def test_raises_key_error_on_empty_string(self, store):
        with pytest.raises(KeyError):
            rollback_version("", "revert", store=store)


class TestRollbackVersionEdgeCases:
    def test_rollback_empty_content_version(self, store, prompt_id):
        v1 = create_version(prompt_id, "", "empty", store=store)
        create_version(prompt_id, "non-empty", "m2", store=store)
        rb = rollback_version(v1.id, "revert to empty", store=store)
        assert rb.content == ""
        assert rb.hash == compute_prompt_hash("")

    def test_rollback_large_content_version(self, store, prompt_id):
        large = "B" * 50_000
        v1 = create_version(prompt_id, large, "large init", store=store)
        create_version(prompt_id, "small", "m2", store=store)
        rb = rollback_version(v1.id, "revert to large", store=store)
        assert rb.content == large

    def test_rollback_unicode_content_version(self, store, prompt_id):
        content = "こんにちは 🌸"
        v1 = create_version(prompt_id, content, "unicode init", store=store)
        create_version(prompt_id, "ascii", "m2", store=store)
        rb = rollback_version(v1.id, "revert to unicode", store=store)
        assert rb.content == content

    def test_prompt_isolation_rollback_not_cross_prompt(self, store, prompt_id, prompt_id_b):
        va = create_version(prompt_id, "A root", "init", store=store)
        create_version(prompt_id, "A v2", "m2", store=store)
        create_version(prompt_id_b, "B root", "init", store=store)
        rollback_version(va.id, "revert A", store=store)
        b_versions = get_branch_versions("main", prompt_id_b, store=store)
        assert all(v.prompt_id == prompt_id_b for v in b_versions)
