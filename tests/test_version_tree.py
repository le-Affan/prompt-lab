"""
tests/test_version_tree.py — Tests for tree traversal functions.

Covers: get_version, get_children, get_version_lineage,
        get_root_version, get_branch_versions, get_version_tree.
"""

import json
import uuid

import pytest

from src.prompt_versioning import (
    VersionStore,
    branch_version,
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
# get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_correct_version(self, store, prompt_id):
        v = create_version(prompt_id, "content", "init", store=store)
        assert get_version(v.id, store=store) is v

    def test_returns_same_object_identity(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        assert get_version(v1.id, store=store) is v1
        assert get_version(v2.id, store=store) is v2

    def test_raises_key_error_on_unknown_id(self, store):
        with pytest.raises(KeyError, match="not found"):
            get_version("nonexistent-id", store=store)

    def test_raises_key_error_on_empty_string(self, store):
        with pytest.raises(KeyError):
            get_version("", store=store)

    def test_raises_key_error_on_valid_uuid_not_in_store(self, store):
        with pytest.raises(KeyError):
            get_version(str(uuid.uuid4()), store=store)


# ---------------------------------------------------------------------------
# get_children
# ---------------------------------------------------------------------------


class TestGetChildren:
    def test_single_child(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        children = get_children(v1.id, store=store)
        assert len(children) == 1
        assert children[0] is v2

    def test_leaf_has_no_children(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        assert get_children(v2.id, store=store) == []

    def test_cross_branch_children_all_returned(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        v2 = create_version(prompt_id, "main v2", "commit 2", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        children = get_children(v1.id, store=store)
        child_ids = {c.id for c in children}
        assert {v2.id, bv.id} == child_ids

    def test_three_branches_from_single_node(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv_a = branch_version(v1.id, "branch-a", "a", store=store)
        bv_b = branch_version(v1.id, "branch-b", "b", store=store)
        bv_c = branch_version(v1.id, "branch-c", "c", store=store)
        children = get_children(v1.id, store=store)
        child_ids = {c.id for c in children}
        assert {bv_a.id, bv_b.id, bv_c.id} == child_ids

    def test_children_sorted_oldest_first(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        children = get_children(v1.id, store=store)
        assert children == sorted(children, key=lambda v: v.created_at)

    def test_root_of_other_prompt_not_included(self, store, prompt_id, prompt_id_b):
        v1 = create_version(prompt_id, "A root", "init", store=store)
        create_version(prompt_id, "A child", "c2", store=store)
        vb = create_version(prompt_id_b, "B root", "init", store=store)
        # v1's children should not include vb
        children = get_children(v1.id, store=store)
        assert vb not in children

    def test_empty_for_version_with_no_successors(self, store, prompt_id):
        v1 = create_version(prompt_id, "solo", "only", store=store)
        assert get_children(v1.id, store=store) == []


# ---------------------------------------------------------------------------
# get_version_lineage
# ---------------------------------------------------------------------------


class TestGetVersionLineage:
    def test_single_root_returns_list_of_one(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        lineage = get_version_lineage(v1.id, store=store)
        assert lineage == [v1]

    def test_two_versions_root_to_target_order(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        lineage = get_version_lineage(v2.id, store=store)
        assert [v.id for v in lineage] == [v1.id, v2.id]

    def test_three_version_chain_root_to_target(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        lineage = get_version_lineage(v3.id, store=store)
        assert [v.id for v in lineage] == [v1.id, v2.id, v3.id]

    def test_first_element_is_root(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        lineage = get_version_lineage(v3.id, store=store)
        assert lineage[0].parent_version_id is None

    def test_last_element_is_target(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        lineage = get_version_lineage(v3.id, store=store)
        assert lineage[-1] is v3

    def test_cross_branch_lineage(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        v_feat = create_version(prompt_id, "feat v2", "fv2",
                                branch_name="feat", store=store)
        lineage = get_version_lineage(v_feat.id, store=store)
        assert [v.id for v in lineage] == [v1.id, bv.id, v_feat.id]

    def test_lineage_after_rollback(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        rb = rollback_version(v1.id, "revert", store=store)
        lineage = get_version_lineage(rb.id, store=store)
        assert lineage[-1] is rb
        assert lineage[0] is v1

    def test_raises_key_error_on_unknown_id(self, store):
        with pytest.raises(KeyError):
            get_version_lineage("bad-id", store=store)

    def test_deep_chain_length(self, store, prompt_id):
        contents = [f"version {i}" for i in range(20)]
        versions = []
        for i, content in enumerate(contents):
            v = create_version(prompt_id, content, f"commit {i}", store=store)
            versions.append(v)
        lineage = get_version_lineage(versions[-1].id, store=store)
        assert len(lineage) == 20
        assert lineage[0] is versions[0]
        assert lineage[-1] is versions[-1]


# ---------------------------------------------------------------------------
# get_root_version
# ---------------------------------------------------------------------------


class TestGetRootVersion:
    def test_returns_root_single_chain(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        root = get_root_version(prompt_id, store=store)
        assert root is v1

    def test_root_has_no_parent(self, store, prompt_id):
        create_version(prompt_id, "root content", "init", store=store)
        root = get_root_version(prompt_id, store=store)
        assert root.parent_version_id is None

    def test_root_same_after_branch(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        root = get_root_version(prompt_id, store=store)
        assert root is v1

    def test_root_isolation_per_prompt(self, store, prompt_id, prompt_id_b):
        va = create_version(prompt_id, "A root", "init", store=store)
        vb = create_version(prompt_id_b, "B root", "init", store=store)
        assert get_root_version(prompt_id, store=store) is va
        assert get_root_version(prompt_id_b, store=store) is vb

    def test_raises_key_error_for_unknown_prompt(self, store):
        with pytest.raises(KeyError):
            get_root_version("no-such-prompt", store=store)

    def test_raises_key_error_on_empty_store(self, store, prompt_id):
        with pytest.raises(KeyError):
            get_root_version(prompt_id, store=store)


# ---------------------------------------------------------------------------
# get_branch_versions
# ---------------------------------------------------------------------------


class TestGetBranchVersions:
    def test_returns_all_versions_on_branch(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        versions = get_branch_versions("main", prompt_id, store=store)
        ids = {v.id for v in versions}
        assert ids == {v1.id, v2.id, v3.id}

    def test_empty_for_unknown_branch(self, store, prompt_id):
        create_version(prompt_id, "content", "init", store=store)
        assert get_branch_versions("nonexistent", prompt_id, store=store) == []

    def test_empty_for_unknown_prompt(self, store, prompt_id):
        assert get_branch_versions("main", "no-such-prompt", store=store) == []

    def test_sorted_oldest_first(self, store, prompt_id, make_chain):
        make_chain(prompt_id, ["A", "B", "C"])
        versions = get_branch_versions("main", prompt_id, store=store)
        assert versions == sorted(versions, key=lambda v: v.created_at)

    def test_filters_by_prompt_id(self, store, prompt_id, prompt_id_b):
        create_version(prompt_id, "A", "init", store=store)
        create_version(prompt_id_b, "B", "init", store=store)
        versions = get_branch_versions("main", prompt_id, store=store)
        assert all(v.prompt_id == prompt_id for v in versions)

    def test_separate_branches_independent(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "main v2", "m2", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        create_version(prompt_id, "feat v2", "fv2", branch_name="feat", store=store)
        main_versions = get_branch_versions("main", prompt_id, store=store)
        feat_versions = get_branch_versions("feat", prompt_id, store=store)
        main_ids = {v.id for v in main_versions}
        feat_ids = {v.id for v in feat_versions}
        assert main_ids.isdisjoint(feat_ids)

    def test_rollback_included_in_branch(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        rb = rollback_version(v1.id, "revert", store=store)
        versions = get_branch_versions("main", prompt_id, store=store)
        assert rb in versions


# ---------------------------------------------------------------------------
# get_version_tree
# ---------------------------------------------------------------------------


class TestGetVersionTree:
    def test_empty_for_unknown_prompt(self, store):
        assert get_version_tree("unknown-prompt", store=store) == []

    def test_single_node_no_children(self, store, prompt_id):
        v1 = create_version(prompt_id, "solo", "init", store=store)
        tree = get_version_tree(prompt_id, store=store)
        assert len(tree) == 1
        assert tree[0]["id"] == v1.id
        assert tree[0]["children"] == []

    def test_linear_chain_nested_correctly(self, store, prompt_id, make_chain):
        v1, v2, v3 = make_chain(prompt_id, ["A", "B", "C"])
        tree = get_version_tree(prompt_id, store=store)
        assert len(tree) == 1
        root_node = tree[0]
        assert root_node["id"] == v1.id
        assert len(root_node["children"]) == 1
        assert root_node["children"][0]["id"] == v2.id
        assert root_node["children"][0]["children"][0]["id"] == v3.id

    def test_branch_creates_sibling_children(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        v2 = create_version(prompt_id, "main v2", "m2", store=store)
        bv = branch_version(v1.id, "feat", "fork", store=store)
        tree = get_version_tree(prompt_id, store=store)
        child_ids = {c["id"] for c in tree[0]["children"]}
        assert {v2.id, bv.id} == child_ids

    def test_node_has_all_required_fields(self, store, prompt_id):
        create_version(prompt_id, "content", "init", store=store)
        node = get_version_tree(prompt_id, store=store)[0]
        required = {"id", "prompt_id", "branch_name", "commit_message",
                    "hash", "created_at", "parent_version_id", "metadata", "children"}
        assert required.issubset(node.keys())

    def test_created_at_is_iso8601_string(self, store, prompt_id):
        create_version(prompt_id, "content", "init", store=store)
        node = get_version_tree(prompt_id, store=store)[0]
        assert isinstance(node["created_at"], str)
        assert "T" in node["created_at"]

    def test_json_serialisable(self, store, prompt_id):
        v1 = create_version(prompt_id, "root", "init", store=store)
        create_version(prompt_id, "v2", "second", store=store)
        branch_version(v1.id, "feat", "fork", store=store)
        tree = get_version_tree(prompt_id, store=store)
        json.dumps(tree)  # must not raise

    def test_scoped_to_prompt_no_cross_leakage(self, store, prompt_id, prompt_id_b):
        create_version(prompt_id, "A", "init", store=store)
        create_version(prompt_id_b, "B", "init", store=store)
        tree_a = get_version_tree(prompt_id, store=store)
        assert all(n["prompt_id"] == prompt_id for n in tree_a)

    def test_multi_branch_prompt_isolation(self, store, prompt_id, prompt_id_b):
        v1a = create_version(prompt_id, "A root", "init", store=store)
        branch_version(v1a.id, "feat-a", "fork-a", store=store)
        create_version(prompt_id_b, "B root", "init", store=store)
        tree_a = get_version_tree(prompt_id, store=store)
        all_ids = set()
        def collect_ids(nodes):
            for n in nodes:
                all_ids.add(n["id"])
                collect_ids(n["children"])
        collect_ids(tree_a)
        b_versions_store = [v for v in store._versions.values()
                            if v.prompt_id == prompt_id_b]
        for bv in b_versions_store:
            assert bv.id not in all_ids

    def test_metadata_preserved_in_tree_node(self, store, prompt_id):
        create_version(prompt_id, "content", "init",
                       metadata={"author": "alice"}, store=store)
        node = get_version_tree(prompt_id, store=store)[0]
        assert node["metadata"]["author"] == "alice"

    def test_parent_version_id_correct_in_node(self, store, prompt_id, make_chain):
        v1, v2 = make_chain(prompt_id, ["A", "B"])
        node = get_version_tree(prompt_id, store=store)
        child_node = node[0]["children"][0]
        assert child_node["parent_version_id"] == v1.id
