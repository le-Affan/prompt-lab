"""
tests/test_branching.py — Tests for branch_version.
"""

import uuid
import pytest

from src.prompt_versioning import (
    Version, VersionStore, branch_version, compute_prompt_hash,
    create_version, get_branch_versions, get_children,
)


class TestBranchVersionReturnType:
    def test_returns_version_instance(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert isinstance(bv, Version)

    def test_id_is_new_uuid4(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.id != v1.id
        uuid.UUID(bv.id, version=4)


class TestBranchVersionFields:
    def test_content_copied_from_base(self, store, prompt_id):
        v1 = create_version(prompt_id, "Original content", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.content == v1.content

    def test_content_copied_from_non_head_base(self, store, prompt_id):
        v1 = create_version(prompt_id, "Root content", "init", store=store)
        create_version(prompt_id, "Main v2", "m2", store=store)
        bv = branch_version(v1.id, "feat", "fork from root", store=store)
        assert bv.content == v1.content

    def test_parent_version_id_is_base(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.parent_version_id == v1.id

    def test_branch_name_assigned(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "my-feature-branch", "fork", store=store)
        assert bv.branch_name == "my-feature-branch"

    def test_prompt_id_inherited(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.prompt_id == prompt_id

    def test_hash_matches_base_content(self, store, prompt_id):
        v1 = create_version(prompt_id, "Some content", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.hash == compute_prompt_hash(v1.content)

    def test_commit_message_preserved(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "My fork commit message", store=store)
        assert bv.commit_message == "My fork commit message"

    def test_created_at_is_timezone_aware(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.created_at.tzinfo is not None

    def test_immutable_frozen_dataclass(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        with pytest.raises(Exception):
            bv.content = "mutated"  # type: ignore[misc]


class TestBranchVersionMetadata:
    def test_branched_from_auto_set(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv.metadata["branched_from"] == v1.id

    def test_caller_metadata_preserved(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork",
                            metadata={"author": "alice"}, store=store)
        assert bv.metadata["author"] == "alice"
        assert bv.metadata["branched_from"] == v1.id

    def test_metadata_shallow_copy_isolates_caller(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        m = {"key": "original"}
        bv = branch_version(v1.id, "feat", "fork", metadata=m, store=store)
        m["key"] = "mutated"
        assert bv.metadata["key"] == "original"

    def test_no_metadata_arg_only_branched_from(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert list(bv.metadata.keys()) == ["branched_from"]


class TestBranchVersionStorage:
    def test_new_branch_head_registered(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert store._branches[(prompt_id, "feat")] == bv.id

    def test_original_branch_head_unchanged(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        v2 = create_version(prompt_id, "main v2", "m2", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        assert store._branches[(prompt_id, "main")] == v2.id

    def test_branch_version_stored_and_retrievable(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert store._versions[bv.id] is bv

    def test_branch_version_appears_as_child_of_base(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        assert bv in get_children(v1.id, store=store)


class TestBranchVersionGraph:
    def test_commit_on_new_branch_chains_from_fork(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        bv2 = create_version(prompt_id, "feat v2", "fv2", branch_name="feat", store=store)
        assert bv2.parent_version_id == bv.id

    def test_multiple_branches_from_same_node(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv_a = branch_version(v1.id, "branch-a", "fork-a", store=store)
        bv_b = branch_version(v1.id, "branch-b", "fork-b", store=store)
        bv_c = branch_version(v1.id, "branch-c", "fork-c", store=store)
        child_ids = {c.id for c in get_children(v1.id, store=store)}
        assert {bv_a.id, bv_b.id, bv_c.id} == child_ids

    def test_branch_from_mid_history_not_head(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        v2 = create_version(prompt_id, "v2", "second", store=store)
        create_version(prompt_id, "v3", "third", store=store)
        bv = branch_version(v2.id, "mid-branch", "mid fork", store=store)
        assert bv.parent_version_id == v2.id
        assert bv.content == v2.content

    def test_branches_independent_content_evolution(self, store, prompt_id):
        v1 = create_version(prompt_id, "shared root", "init", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        create_version(prompt_id, "main evolved", "main v2", store=store)
        create_version(prompt_id, "feat evolved", "feat v2", branch_name="feat", store=store)
        main_ids = {v.id for v in get_branch_versions("main", prompt_id, store=store)}
        feat_ids = {v.id for v in get_branch_versions("feat", prompt_id, store=store)}
        assert main_ids.isdisjoint(feat_ids)


class TestBranchVersionErrors:
    def test_raises_key_error_on_unknown_base(self, store):
        with pytest.raises(KeyError):
            branch_version("nonexistent-id", "feat", "fork", store=store)

    def test_raises_value_error_on_duplicate_branch(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        branch_version(v1.id, "feat", "first", store=store)
        with pytest.raises(ValueError, match="already exists"):
            branch_version(v1.id, "feat", "duplicate", store=store)

    def test_store_intact_after_error(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        branch_version(v1.id, "feat", "first", store=store)
        count_before = len(store._versions)
        with pytest.raises(ValueError):
            branch_version(v1.id, "feat", "duplicate", store=store)
        assert len(store._versions) == count_before


class TestBranchVersionEdgeCases:
    def test_empty_content_branched_correctly(self, store, prompt_id):
        v1 = create_version(prompt_id, "", "empty init", store=store)
        bv = branch_version(v1.id, "feat", "fork empty", store=store)
        assert bv.content == ""

    def test_large_content_branched_correctly(self, store, prompt_id):
        large = "A" * 50_000
        v1 = create_version(prompt_id, large, "large init", store=store)
        bv = branch_version(v1.id, "feat", "fork large", store=store)
        assert bv.content == large
        assert bv.hash == compute_prompt_hash(large)

    def test_unicode_content_branched_correctly(self, store, prompt_id):
        content = "你好世界 🌍 Привет"
        v1 = create_version(prompt_id, content, "unicode init", store=store)
        bv = branch_version(v1.id, "feat", "fork unicode", store=store)
        assert bv.content == content
