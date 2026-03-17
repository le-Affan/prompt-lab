import asyncio
from datetime import datetime, timezone

import pytest

from src.AB_Testing import (
    VariantConfig,
    Experiment,
    ExecutionResult,
    VariantResult,
    ExperimentRun,
    run_experiment,
    validate_experiment,
    get_successful_results,
    get_failed_results,
    get_fastest_variant,
    get_highest_token_variant,
    summarize_experiment_run,
)


def create_base_experiment(variants) -> Experiment:
    """Helper to create test experiments easily."""
    return Experiment(
        id="test-exp",
        name="Test Experiment",
        input_text="Test prompt",
        variants=variants,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# 1. Test Experiment Validation
# ---------------------------------------------------------------------------

def test_validate_experiment_no_variants():
    exp = create_base_experiment([])
    with pytest.raises(ValueError, match="at least one variant"):
        validate_experiment(exp)

def test_validate_experiment_too_many_variants():
    variants = [
        VariantConfig(label=f"V{i}", model="gpt-4o", temperature=0.7, max_tokens=100)
        for i in range(7)
    ]
    exp = create_base_experiment(variants)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        validate_experiment(exp)

def test_validate_experiment_duplicate_labels():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
    ]
    exp = create_base_experiment(variants)
    with pytest.raises(ValueError, match="Duplicate variant label"):
        validate_experiment(exp)

def test_validate_experiment_invalid_temperature():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=2.5, max_tokens=100)
    ]
    exp = create_base_experiment(variants)
    with pytest.raises(ValueError, match="invalid temperature"):
        validate_experiment(exp)

def test_validate_experiment_invalid_max_tokens():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=-10)
    ]
    exp = create_base_experiment(variants)
    with pytest.raises(ValueError, match="invalid max_tokens"):
        validate_experiment(exp)


# ---------------------------------------------------------------------------
# 2. Test Successful Experiment Run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_experiment_success():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
        VariantConfig(label="B", model="claude-3-sonnet", temperature=0.7, max_tokens=100),
    ]
    exp = create_base_experiment(variants)
    
    run = await run_experiment(exp)
    
    assert isinstance(run, ExperimentRun)
    assert len(run.variant_results) == 2
    for vr in run.variant_results:
        assert vr.execution.status == "success"
        assert getattr(vr.execution, "latency_ms") > 0


# ---------------------------------------------------------------------------
# 3. Test Failure Handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_experiment_failure():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
        VariantConfig(label="B", model="invalid-model", temperature=0.7, max_tokens=100),  # Will fail
    ]
    exp = create_base_experiment(variants)
    
    run = await run_experiment(exp)
    
    assert len(run.variant_results) == 2
    
    # Check failure status
    results_by_label = {vr.variant_label: vr for vr in run.variant_results}
    assert results_by_label["A"].execution.status == "success"
    assert results_by_label["B"].execution.status == "failed"
    assert "Unsupported model" in results_by_label["B"].execution.error_message


# ---------------------------------------------------------------------------
# 4. Test Concurrency Behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_behavior():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
        VariantConfig(label="B", model="claude-3-sonnet", temperature=0.7, max_tokens=100),
        VariantConfig(label="C", model="gpt-4o", temperature=0.7, max_tokens=100),
    ]
    exp = create_base_experiment(variants)
    
    start = datetime.now(timezone.utc)
    run = await run_experiment(exp)
    end = datetime.now(timezone.utc)
    
    runtime_seconds = (end - start).total_seconds()
    
    # Since each mock adapter sleeps between 0.1s and 0.8s, 
    # parallel execution should complete in about 0.8s. 
    # Sequential execution would take at least ~1.5 - 2.4s.
    assert runtime_seconds < 2.0
    assert len(run.variant_results) == 3


# ---------------------------------------------------------------------------
# 5. Test Result Analysis Utilities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_analysis_utilities():
    variants = [
        VariantConfig(label="A", model="gpt-4o", temperature=0.7, max_tokens=100),
        VariantConfig(label="B", model="invalid-model", temperature=0.7, max_tokens=100), # failed
        VariantConfig(label="C", model="claude-3-sonnet", temperature=0.7, max_tokens=100),
    ]
    exp = create_base_experiment(variants)
    
    run = await run_experiment(exp)
    
    successful = get_successful_results(run)
    failed = get_failed_results(run)
    fastest = get_fastest_variant(run)
    highest = get_highest_token_variant(run)
    summary = summarize_experiment_run(run)
    
    assert len(successful) + len(failed) == len(variants)
    assert len(successful) == 2
    assert len(failed) == 1
    
    assert fastest is not None
    assert isinstance(fastest, VariantResult)
    assert fastest.execution.status == "success"
    
    assert highest is not None
    assert isinstance(highest, VariantResult)
    assert highest.execution.status == "success"
    
    assert summary["experiment_id"] == exp.id
    assert summary["variant_count"] == 3
    assert summary["successful_variants"] == 2
    assert summary["failed_variants"] == 1
    assert summary["fastest_variant"] == fastest.variant_label
