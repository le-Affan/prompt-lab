"""
tests/test_diff_engine.py — Tests for compute_char_diff and compute_line_diff.

Both functions are pure (no store dependency).
Covers: output format, ops correctness, reconstruction, determinism, edge cases.
"""

import pytest

from src.prompt_versioning import compute_char_diff, compute_line_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reconstruct_old(diff: list[list[str]]) -> str:
    return "".join(text for op, text in diff if op in ("=", "-"))


def reconstruct_new(diff: list[list[str]]) -> str:
    return "".join(text for op, text in diff if op in ("=", "+"))


def ops_set(diff: list[list[str]]) -> set[str]:
    return {op for op, _ in diff}


# ---------------------------------------------------------------------------
# compute_char_diff
# ---------------------------------------------------------------------------


class TestComputeCharDiff:
    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_list(self):
        result = compute_char_diff("hello", "hello")
        assert isinstance(result, list)

    def test_each_element_is_two_item_list(self):
        for op, text in compute_char_diff("abc", "axc"):
            assert isinstance(op, str)
            assert isinstance(text, str)
            assert len([op, text]) == 2

    def test_ops_limited_to_valid_symbols(self):
        result = compute_char_diff("old text", "new text")
        for op, _ in result:
            assert op in ("+", "-", "=")

    # ── Identical strings ─────────────────────────────────────────────────

    def test_identical_strings_only_equal_ops(self):
        diff = compute_char_diff("same text", "same text")
        assert all(op == "=" for op, _ in diff)

    def test_identical_single_char(self):
        diff = compute_char_diff("a", "a")
        assert all(op == "=" for op, _ in diff)

    def test_identical_empty(self):
        assert compute_char_diff("", "") == []

    # ── Empty string cases ────────────────────────────────────────────────

    def test_empty_old_only_insertions(self):
        diff = compute_char_diff("", "hello")
        assert all(op == "+" for op, _ in diff)

    def test_empty_new_only_deletions(self):
        diff = compute_char_diff("hello", "")
        assert all(op == "-" for op, _ in diff)

    # ── Reconstruction ────────────────────────────────────────────────────

    def test_reconstruct_old_simple(self):
        old, new = "hello world", "hello there"
        assert reconstruct_old(compute_char_diff(old, new)) == old

    def test_reconstruct_new_simple(self):
        old, new = "hello world", "hello there"
        assert reconstruct_new(compute_char_diff(old, new)) == new

    def test_reconstruct_old_complete_replacement(self):
        old, new = "aaa", "bbb"
        assert reconstruct_old(compute_char_diff(old, new)) == old

    def test_reconstruct_new_complete_replacement(self):
        old, new = "aaa", "bbb"
        assert reconstruct_new(compute_char_diff(old, new)) == new

    def test_reconstruct_old_with_unicode(self):
        old, new = "Héllo", "Hello"
        assert reconstruct_old(compute_char_diff(old, new)) == old

    def test_reconstruct_new_with_unicode(self):
        old, new = "Héllo", "Hello"
        assert reconstruct_new(compute_char_diff(old, new)) == new

    def test_reconstruct_old_empty_to_text(self):
        assert reconstruct_old(compute_char_diff("", "content")) == ""

    def test_reconstruct_new_empty_to_text(self):
        assert reconstruct_new(compute_char_diff("", "content")) == "content"

    # ── Op presence ───────────────────────────────────────────────────────

    def test_addition_present_when_new_is_longer(self):
        diff = compute_char_diff("Hi", "Hi there")
        assert "+" in ops_set(diff)

    def test_deletion_present_when_old_is_longer(self):
        diff = compute_char_diff("Hi there", "Hi")
        assert "-" in ops_set(diff)

    def test_mixed_edit_has_all_op_types(self):
        diff = compute_char_diff("The quick fox", "A quick dog")
        types = ops_set(diff)
        assert "=" in types   # "quick " is common
        assert "+" in types
        assert "-" in types

    # ── Determinism ───────────────────────────────────────────────────────

    def test_deterministic_across_calls(self):
        args = ("The quick brown fox", "The slow green fox")
        assert compute_char_diff(*args) == compute_char_diff(*args)

    def test_deterministic_with_special_chars(self):
        args = ("line1\nline2\n", "line1\nchanged\n")
        assert compute_char_diff(*args) == compute_char_diff(*args)

    # ── Pure function — no store ──────────────────────────────────────────

    def test_no_store_argument_needed(self):
        # Must work with zero side effects, no VersionStore
        result = compute_char_diff("a", "b")
        assert isinstance(result, list)

    # ── Large content ─────────────────────────────────────────────────────

    def test_large_identical_content(self):
        text = "x" * 5_000
        diff = compute_char_diff(text, text)
        assert all(op == "=" for op, _ in diff)

    def test_large_content_reconstruction(self):
        old = "A" * 2_000 + "B" * 2_000
        new = "A" * 2_000 + "C" * 2_000
        diff = compute_char_diff(old, new)
        assert reconstruct_old(diff) == old
        assert reconstruct_new(diff) == new


# ---------------------------------------------------------------------------
# compute_line_diff
# ---------------------------------------------------------------------------


class TestComputeLineDiff:
    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_list(self):
        assert isinstance(compute_line_diff("a\n", "b\n"), list)

    def test_each_element_is_two_item_list(self):
        for op, text in compute_line_diff("line1\n", "line2\n"):
            assert isinstance(op, str)
            assert isinstance(text, str)

    def test_ops_limited_to_valid_symbols(self):
        result = compute_line_diff("old\ncontent\n", "new\ncontent\n")
        for op, _ in result:
            assert op in ("+", "-", "=")

    # ── Identical strings ─────────────────────────────────────────────────

    def test_identical_lines_only_equal_ops(self):
        diff = compute_line_diff("same\ncontent\n", "same\ncontent\n")
        assert all(op == "=" for op, _ in diff)

    def test_identical_empty(self):
        assert compute_line_diff("", "") == []

    # ── Empty string cases ────────────────────────────────────────────────

    def test_empty_old_only_insertions(self):
        diff = compute_line_diff("", "hello\n")
        assert all(op == "+" for op, _ in diff)

    def test_empty_new_only_deletions(self):
        diff = compute_line_diff("hello\n", "")
        assert all(op == "-" for op, _ in diff)

    # ── Reconstruction ────────────────────────────────────────────────────

    def test_reconstruct_old(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_old(diff) == old

    def test_reconstruct_new(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_new(diff) == new

    def test_reconstruct_old_add_lines(self):
        old = "a\n"
        new = "a\nb\nc\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_old(diff) == old

    def test_reconstruct_new_add_lines(self):
        old = "a\n"
        new = "a\nb\nc\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_new(diff) == new

    def test_reconstruct_old_remove_lines(self):
        old = "a\nb\nc\n"
        new = "a\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_old(diff) == old

    def test_reconstruct_new_remove_lines(self):
        old = "a\nb\nc\n"
        new = "a\n"
        diff = compute_line_diff(old, new)
        assert reconstruct_new(diff) == new

    # ── Line-granularity behaviour ────────────────────────────────────────

    def test_unchanged_line_present_as_equal(self):
        old = "same\n" + "old line\n"
        new = "same\n" + "new line\n"
        diff = compute_line_diff(old, new)
        equal_texts = [t for op, t in diff if op == "="]
        assert any("same" in t for t in equal_texts)

    # ── Determinism ───────────────────────────────────────────────────────

    def test_deterministic_across_calls(self):
        args = ("line1\nline2\n", "line1\nchanged\n")
        assert compute_line_diff(*args) == compute_line_diff(*args)

    # ── Pure function ─────────────────────────────────────────────────────

    def test_no_store_needed(self):
        result = compute_line_diff("x\n", "y\n")
        assert isinstance(result, list)

    # ── Large content ─────────────────────────────────────────────────────

    def test_large_identical_content(self):
        text = "line\n" * 2_000
        diff = compute_line_diff(text, text)
        assert all(op == "=" for op, _ in diff)

    def test_large_content_reconstruction(self):
        old = "".join(f"line{i}\n" for i in range(500))
        new = "".join(f"line{i}\n" for i in range(250)) + "".join(f"new{i}\n" for i in range(250))
        diff = compute_line_diff(old, new)
        assert reconstruct_old(diff) == old
        assert reconstruct_new(diff) == new
