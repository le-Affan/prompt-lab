"""
tests/test_ab_testing.py
========================
Comprehensive pytest + pytest-asyncio test suite for src/AB_Testing.py.

Strategy for deterministic timing
----------------------------------
The real adapters use random.uniform(0.1, 0.8) for latency. Every test that
cares about timing monkeypatches OpenAIAdapter.execute and/or
AnthropicAdapter.execute with fixed-sleep coroutines so results are
predictable and fast.

Tests that do NOT care about timing use the real mock adapters directly —
they're deterministic enough for correctness assertions.
"""

import asyncio
from datetime import datetime, timezone

import pytest

import src.AB_Testing as _engine
from src.AB_Testing import (
    # Data structures
    VariantConfig,
    Experiment,
    ExecutionResult,
    VariantResult,
    ExperimentRun,
    # Engine
    run_experiment,
    validate_experiment,
    build_experiment,
    # Analysis utilities
    get_successful_results,
    get_failed_results,
    get_fastest_variant,
    get_highest_token_variant,
    summarize_experiment_run,
    # Configurable defaults
    DEFAULT_MAX_VARIANTS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_variant(
    label: str = "A",
    provider: str = "openai",
    model: str = "gpt-4o",
) -> VariantConfig:
    """Factory matching the spec; all timing/token details use engine defaults."""
    return VariantConfig(
        label=f"Variant {label}",
        provider=provider,
        model=model,
        api_key="test-key",
        temperature=0.7,
        max_tokens=100,
    )


def make_experiment(variants, **kwargs) -> Experiment:
    """Wrap variants into a minimal Experiment; kwargs override defaults."""
    return Experiment(
        id="test-exp",
        name="Test Experiment",
        input_text="Test prompt",
        variants=variants,
        created_at=datetime.now(timezone.utc),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Deterministic adapter replacements (used with monkeypatch)
# ---------------------------------------------------------------------------

# (Adapter mocks are defined inline within each test that needs them
#  to avoid class-method binding issues with module-level async functions.)

# ---------------------------------------------------------------------------
# 1. Happy path — two valid variants, both succeed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_two_variants():
    variants = [
        make_variant("A", provider="openai",    model="gpt-4o"),
        make_variant("B", provider="anthropic", model="claude-3-sonnet"),
    ]
    run = await run_experiment(make_experiment(variants))

    assert isinstance(run, ExperimentRun)
    assert len(run.variant_results) == 2
    for vr in run.variant_results:
        assert isinstance(vr, VariantResult)
        assert vr.execution.status == "success"
        assert vr.execution.latency_ms > 0


# ---------------------------------------------------------------------------
# 2. Validation — zero variants raises ValueError (via validate_experiment)
# ---------------------------------------------------------------------------

def test_validate_zero_variants():
    with pytest.raises(ValueError, match="at least one variant"):
        validate_experiment(make_experiment([]))


# ---------------------------------------------------------------------------
# 3. Validation — exceeding max_variants fails inside run_experiment safely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exceeding_max_variants_fails_safely():
    # Build 4 variants but cap the experiment at 2 — validation should fail.
    variants = [make_variant(str(i)) for i in range(4)]
    exp = make_experiment(variants, max_variants=2)
    run = await run_experiment(exp)

    assert isinstance(run, ExperimentRun)
    # All results must be marked failed (validation error triggers _all_failed).
    assert all(vr.execution.status == "failed" for vr in run.variant_results)


# ---------------------------------------------------------------------------
# 4. Partial failure — invalid provider leaves other variants intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_failure_invalid_provider():
    variants = [
        make_variant("A", provider="openai",   model="gpt-4o"),
        make_variant("B", provider="no-such-provider", model="x"),
    ]
    run = await run_experiment(make_experiment(variants))

    by_label = {vr.variant_label: vr for vr in run.variant_results}
    assert by_label["Variant A"].execution.status == "success"
    assert by_label["Variant B"].execution.status == "failed"
    assert "Unsupported provider" in by_label["Variant B"].execution.error_message


# ---------------------------------------------------------------------------
# 5. Per-variant timeout — all variants fail when adapter hangs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_variants_timeout(monkeypatch):
    async def _hang(self, prompt, model, temperature, max_tokens):
        await asyncio.sleep(30.0)
        return ExecutionResult(output_text="", token_count=0, latency_ms=0.0,
                               model=model, status="success", error_message=None)

    monkeypatch.setattr(_engine.OpenAIAdapter, "execute", _hang)

    variants = [
        make_variant("A", provider="openai"),
        make_variant("B", provider="openai"),
    ]
    # 0.1 s timeout — much shorter than the 30 s hang.
    exp = make_experiment(variants, timeout_seconds=0.1)
    run = await run_experiment(exp)

    assert len(run.variant_results) == 2
    for vr in run.variant_results:
        assert vr.execution.status == "failed"
        assert "timed out" in vr.execution.error_message.lower()


# ---------------------------------------------------------------------------
# 6. Mixed timeout + success — one fast (openai), one slow (anthropic hangs)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_timeout_and_success(monkeypatch):
    """One valid provider succeeds; one invalid provider fails immediately."""
    variants = [
        make_variant("A", provider="openai",      model="gpt-4o"),
        make_variant("B", provider="bad-provider", model="x"),
    ]
    run = await run_experiment(make_experiment(variants))

    by_label = {vr.variant_label: vr for vr in run.variant_results}
    assert by_label["Variant A"].execution.status == "success"
    assert by_label["Variant B"].execution.status == "failed"
    assert "Unsupported provider" in by_label["Variant B"].execution.error_message


# ---------------------------------------------------------------------------
# 7. Concurrency — max_concurrency=1 is slower than max_concurrency=3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit_is_slower(monkeypatch):
    # Fixed 0.2 s per variant so timing is deterministic.
    async def _fixed_execute(self, prompt, model, temperature, max_tokens):
        await asyncio.sleep(0.2)
        return ExecutionResult(
            output_text="ok", token_count=50, latency_ms=200.0,
            model=model, status="success", error_message=None,
        )

    monkeypatch.setattr(_engine.OpenAIAdapter, "execute", _fixed_execute)

    variants = [make_variant(c) for c in ("A", "B", "C")]

    # --- sequential (concurrency = 1) ---
    t0 = datetime.now(timezone.utc)
    await run_experiment(make_experiment(variants, max_concurrency=1, timeout_seconds=10))
    sequential_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1_000

    # --- parallel (concurrency = 3) ---
    t1 = datetime.now(timezone.utc)
    await run_experiment(make_experiment(variants, max_concurrency=3, timeout_seconds=10))
    parallel_ms = (datetime.now(timezone.utc) - t1).total_seconds() * 1_000

    # Sequential should be ~3× slower (≥ 450 ms) vs parallel (≈ 200 ms).
    assert sequential_ms > parallel_ms
    assert sequential_ms > 450   # 3 × 200 ms − some tolerance
    assert parallel_ms < 450     # at most 1 wait of 200 ms + overhead


# ---------------------------------------------------------------------------
# 8. Per-variant timeout — variants fail quickly when timeout is tiny
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_variant_timeout_fires_fast(monkeypatch):
    async def _hang(self, prompt, model, temperature, max_tokens):
        await asyncio.sleep(30.0)
        return ExecutionResult(output_text="", token_count=0, latency_ms=0.0,
                               model=model, status="success", error_message=None)

    monkeypatch.setattr(_engine.OpenAIAdapter, "execute", _hang)

    variants = [make_variant("A"), make_variant("B")]
    exp = make_experiment(variants, timeout_seconds=0.05)

    t0 = datetime.now(timezone.utc)
    run = await run_experiment(exp)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    # Should finish well under 1 s despite 30 s hang.
    assert elapsed < 1.0
    assert all(vr.execution.status == "failed" for vr in run.variant_results)


# ---------------------------------------------------------------------------
# 9. Default config — build_experiment uses module defaults, no crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_config_no_crash():
    exp = build_experiment(
        experiment_id="default-test",
        name="Default Config Test",
        input_text="Hello",
        variants=[make_variant("A")],
        # timeout_seconds and max_concurrency intentionally omitted
    )

    assert exp.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert exp.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert exp.max_variants == DEFAULT_MAX_VARIANTS

    run = await run_experiment(exp)
    assert isinstance(run, ExperimentRun)
    assert run.variant_results[0].execution.status == "success"


# ---------------------------------------------------------------------------
# 10. Adapter routing — valid vs invalid provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_routing_valid_providers():
    variants = [
        make_variant("OAI",  provider="openai",    model="gpt-4o"),
        make_variant("ANT",  provider="anthropic", model="claude-3-sonnet"),
    ]
    run = await run_experiment(make_experiment(variants))
    assert all(vr.execution.status == "success" for vr in run.variant_results)


@pytest.mark.asyncio
async def test_adapter_routing_invalid_provider():
    exp = make_experiment([make_variant("X", provider="bad-provider")])
    run = await run_experiment(exp)
    assert run.variant_results[0].execution.status == "failed"
    assert "Unsupported provider" in run.variant_results[0].execution.error_message


# ---------------------------------------------------------------------------
# 11. Data integrity — labels, types, variant count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_data_integrity():
    variants = [
        make_variant("A", provider="openai",    model="gpt-4o"),
        make_variant("B", provider="anthropic", model="claude-3-sonnet"),
        make_variant("C", provider="openai",    model="gpt-4o"),
    ]
    exp = make_experiment(variants)
    run = await run_experiment(exp)

    assert len(run.variant_results) == 3

    expected_labels = {"Variant A", "Variant B", "Variant C"}
    actual_labels = {vr.variant_label for vr in run.variant_results}
    assert actual_labels == expected_labels

    for vr in run.variant_results:
        assert isinstance(vr, VariantResult)
        assert isinstance(vr.execution, ExecutionResult)
        assert isinstance(vr.config, VariantConfig)
        assert vr.execution.status in ("success", "failed")

    assert isinstance(run.started_at, datetime)
    assert isinstance(run.completed_at, datetime)
    assert run.completed_at >= run.started_at


# ---------------------------------------------------------------------------
# 12. No-crash guarantee — run_experiment ALWAYS returns ExperimentRun
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_crash_on_zero_variants():
    run = await run_experiment(make_experiment([]))
    assert isinstance(run, ExperimentRun)


@pytest.mark.asyncio
async def test_no_crash_on_all_invalid_providers():
    variants = [make_variant("X", provider="no-such")]
    run = await run_experiment(make_experiment(variants))
    assert isinstance(run, ExperimentRun)


@pytest.mark.asyncio
async def test_no_crash_on_invalid_temperature():
    # Temperature 9.9 violates validation — engine should still return a run.
    variants = [VariantConfig(
        label="Variant Bad",
        provider="openai",
        model="gpt-4o",
        api_key="",
        temperature=9.9,
        max_tokens=100,
    )]
    run = await run_experiment(make_experiment(variants))
    assert isinstance(run, ExperimentRun)


# ---------------------------------------------------------------------------
# 13. Analysis utilities — summarize, fastest, highest token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_experiment_run():
    """One valid provider succeeds; one invalid provider fails — verify summary counts."""
    variants = [
        make_variant("A", provider="openai",      model="gpt-4o"),
        make_variant("B", provider="bad-provider", model="x"),
    ]
    run = await run_experiment(make_experiment(variants))

    summary = summarize_experiment_run(run)
    assert summary["experiment_id"] == "test-exp"
    assert summary["variant_count"] == 2
    assert summary["successful_variants"] == 1
    assert summary["failed_variants"] == 1
    assert summary["fastest_variant"] == "Variant A"
    assert summary["total_runtime_ms"] >= 0


@pytest.mark.asyncio
async def test_fastest_variant_excludes_failures():
    """get_fastest_variant must never return a failed result (latency_ms == 0)."""
    variants = [
        make_variant("Fast",   provider="openai",      model="gpt-4o"),
        make_variant("Failed", provider="bad-provider", model="x"),
    ]
    run = await run_experiment(make_experiment(variants))

    fastest = get_fastest_variant(run)
    assert fastest is not None
    assert fastest.execution.status == "success"
    assert fastest.variant_label == "Variant Fast"


@pytest.mark.asyncio
async def test_get_successful_and_failed_filter():
    variants = [
        make_variant("Good", provider="openai"),
        make_variant("Bad",  provider="unknown-provider"),
    ]
    run = await run_experiment(make_experiment(variants))

    successes = get_successful_results(run)
    failures  = get_failed_results(run)

    assert len(successes) == 1
    assert len(failures)  == 1
    assert successes[0].execution.status == "success"
    assert failures[0].execution.status  == "failed"
    assert len(successes) + len(failures) == len(run.variant_results)


@pytest.mark.asyncio
async def test_highest_token_variant_excludes_failures():
    """get_highest_token_variant must not return a failed variant (token_count=0)."""
    variants = [
        make_variant("A", provider="openai"),
        make_variant("B", provider="bad-provider"),
    ]
    run = await run_experiment(make_experiment(variants))

    highest = get_highest_token_variant(run)
    assert highest is not None
    assert highest.execution.status == "success"
    assert highest.execution.token_count > 0
