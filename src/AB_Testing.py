"""
AB_Testing.py — A/B Testing Engine
====================================
Step 1: Core data structures.

This module is framework-independent. It defines only the data model
used throughout the A/B testing engine. No execution logic lives here yet.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, cast


# ---------------------------------------------------------------------------
# Public API — symbols re-exported when another module does
# `from AB_Testing import *`, and surfaced as the stable interface
# for the services layer and FastAPI routers.
# ---------------------------------------------------------------------------

__all__ = [
    # --- Data structures ---
    "VariantConfig",
    "Experiment",
    "ExecutionResult",
    "VariantResult",
    "ExperimentRun",
    # --- Core engine ---
    "run_experiment",
    "validate_experiment",
    "build_experiment",
    "ABTestingEngine",
    # --- Analysis utilities ---
    "get_successful_results",
    "get_failed_results",
    "get_fastest_variant",
    "get_highest_token_variant",
    "summarize_experiment_run",
]


# ---------------------------------------------------------------------------
# 1. VariantConfig
#    Represents a single experiment variant configuration.
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    """Configuration for a single A/B test variant."""

    label: str
    model: str
    temperature: float
    max_tokens: int
    prompt_version_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Experiment
#    Represents the full A/B test definition.
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """An A/B test experiment containing one or more variants."""

    id: str
    name: str
    input_text: str
    variants: List[VariantConfig]
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 3. ExecutionResult
#    Represents the raw result of a single LLM call.
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """The outcome of executing a prompt against an LLM."""

    output_text: str
    token_count: int
    latency_ms: float
    model: str
    status: str                        # e.g. "success" | "error"
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. VariantResult
#    Pairs a VariantConfig with its ExecutionResult.
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """The result of executing one specific variant within an experiment."""

    variant_label: str
    config: VariantConfig
    execution: ExecutionResult
    run_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 5. ExperimentRun
#    The top-level record produced when a full experiment is executed.
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRun:
    """The aggregated results of running all variants in an experiment."""

    experiment_id: str
    started_at: datetime
    completed_at: datetime
    variant_results: List[VariantResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 6. LLMAdapter
#    Abstract interface that every LLM provider adapter must implement.
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.

    This class defines the contract that every concrete provider adapter
    (e.g. OpenAIAdapter, AnthropicAdapter) must fulfil.  It deliberately
    contains no implementation logic — all provider-specific API calls,
    authentication, and error handling belong in the concrete subclasses.

    Usage
    -----
    Subclass ``LLMAdapter`` and implement ``execute`` for a specific
    provider::

        class OpenAIAdapter(LLMAdapter):
            async def execute(self, prompt, model, temperature, max_tokens):
                ...  # call the OpenAI API and return an ExecutionResult
    """

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ExecutionResult:
        """
        Execute a single LLM call and return a structured result.

        Parameters
        ----------
        prompt : str
            The fully-rendered prompt string to send to the model.
        model : str
            The provider-specific model identifier (e.g. ``"gpt-4o"``).
        temperature : float
            Sampling temperature; controls output randomness (0.0 – 2.0).
        max_tokens : int
            Upper bound on the number of tokens in the completion.

        Returns
        -------
        ExecutionResult
            Contains the output text, token count, latency, model used,
            status (``"success"`` or ``"error"``), and an optional error
            message if the call failed.
        """
        ...


# ---------------------------------------------------------------------------
# 7. OpenAIAdapter
#    Mock concrete adapter simulating an OpenAI API call.
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """
    Simulated OpenAI adapter.

    Mimics the latency and output shape of a real OpenAI API call without
    using the SDK or making any network requests.  Swap out the body of
    ``execute`` with the real ``openai.AsyncOpenAI`` client in a later step.
    """

    async def execute(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ExecutionResult:
        """Simulate an OpenAI completion call and return an ExecutionResult."""
        try:
            # --- Simulate network + inference latency ---
            latency_seconds = random.uniform(0.1, 0.8)
            t_start = time.perf_counter()
            await asyncio.sleep(latency_seconds)
            latency_ms = (time.perf_counter() - t_start) * 1_000

            # --- Simulate model output ---
            preview = prompt[slice(50)].replace("\n", " ")
            output_text = f"OpenAI simulated response to: {preview}..."

            # --- Simulate token usage ---
            token_count = random.randint(50, max_tokens)

            return ExecutionResult(
                output_text=output_text,
                token_count=token_count,
                latency_ms=int(latency_ms * 100) / 100.0,
                model=model,
                status="success",
                error_message=None,
            )

        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                output_text="",
                token_count=0,
                latency_ms=0.0,
                model=model,
                status="failed",
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# 8. AnthropicAdapter
#    Mock concrete adapter simulating an Anthropic API call.
# ---------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):
    """
    Simulated Anthropic adapter.

    Mimics the latency and output shape of a real Anthropic Messages API
    call without using the SDK or making any network requests.  Swap out
    the body of ``execute`` with the real ``anthropic.AsyncAnthropic``
    client in a later step.
    """

    async def execute(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ExecutionResult:
        """Simulate an Anthropic completion call and return an ExecutionResult."""
        try:
            # --- Simulate network + inference latency ---
            latency_seconds = random.uniform(0.1, 0.8)
            t_start = time.perf_counter()
            await asyncio.sleep(latency_seconds)
            latency_ms = (time.perf_counter() - t_start) * 1_000

            # --- Simulate model output ---
            preview = prompt[slice(50)].replace("\n", " ")
            output_text = f"Anthropic simulated response to: {preview}..."

            # --- Simulate token usage ---
            token_count = random.randint(50, max_tokens)

            return ExecutionResult(
                output_text=output_text,
                token_count=token_count,
                latency_ms=int(latency_ms * 100) / 100.0,
                model=model,
                status="success",
                error_message=None,
            )

        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                output_text="",
                token_count=0,
                latency_ms=0.0,
                model=model,
                status="failed",
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# 9. ADAPTER_REGISTRY
#    Maps provider keys to their concrete LLMAdapter classes (not instances).
#    Add new providers here as they are implemented.
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


# ---------------------------------------------------------------------------
# 10. get_adapter_for_model
#     Inspects the model name prefix and returns the appropriate adapter
#     instance. Raises ValueError for unrecognised models.
# ---------------------------------------------------------------------------

def get_adapter_for_model(model: str) -> LLMAdapter:
    """
    Return a concrete ``LLMAdapter`` instance for the given model name.

    Routing rules
    -------------
    - ``"gpt-*"``    → :class:`OpenAIAdapter`
    - ``"claude-*"`` → :class:`AnthropicAdapter`

    Parameters
    ----------
    model : str
        Provider-specific model identifier, e.g. ``"gpt-4o"`` or
        ``"claude-3-5-sonnet-20241022"``.

    Returns
    -------
    LLMAdapter
        A freshly instantiated adapter ready to call ``execute``.

    Raises
    ------
    ValueError
        If the model prefix does not match any registered provider.
    """
    if model.startswith("gpt"):
        return OpenAIAdapter()
    if model.startswith("claude"):
        return AnthropicAdapter()
    raise ValueError(f"Unsupported model: {model}")


# ---------------------------------------------------------------------------
# 11. execute_variant
#     Async router: resolves the adapter and runs a single variant execution.
#     Always returns an ExecutionResult — never raises.
# ---------------------------------------------------------------------------

async def execute_variant(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> ExecutionResult:
    """
    Route a single variant execution to the correct LLM adapter.

    This function is the primary entry-point for running one variant.
    It handles adapter selection and wraps the call in a top-level
    safety net so the caller is guaranteed to receive an
    ``ExecutionResult`` regardless of what goes wrong.

    Parameters
    ----------
    prompt : str
        The fully-rendered prompt string to send to the model.
    model : str
        Provider-specific model identifier used to select the adapter.
    temperature : float
        Sampling temperature forwarded to the adapter.
    max_tokens : int
        Token limit forwarded to the adapter.

    Returns
    -------
    ExecutionResult
        On success: populated with output, token count, and latency.
        On failure: ``status="failed"``, ``error_message`` set, numeric
        fields zeroed out.
    """
    try:
        adapter = get_adapter_for_model(model)
        return await adapter.execute(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            output_text="",
            token_count=0,
            latency_ms=0.0,
            model=model,
            status="failed",
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# 12. validate_experiment
#     Enforces safety constraints before any API calls are made.
# ---------------------------------------------------------------------------

_MAX_VARIANTS = 6
_TIMEOUT_SECONDS = 30.0


def validate_experiment(experiment: Experiment) -> None:
    """
    Validate an :class:`Experiment` against safety and correctness rules.

    Rules
    -----
    - At least one variant must be present.
    - No more than ``_MAX_VARIANTS`` (6) variants are allowed.
    - Every variant label must be unique within the experiment.
    - Each variant's ``temperature`` must be in the closed interval [0, 2].
    - Each variant's ``max_tokens`` must be a positive integer (> 0).

    Raises
    ------
    ValueError
        With a descriptive message for the first failing rule encountered.
    """
    if not experiment.variants:
        raise ValueError("Experiment must have at least one variant.")

    if len(experiment.variants) > _MAX_VARIANTS:
        raise ValueError(
            f"Experiment exceeds the maximum of {_MAX_VARIANTS} variants "
            f"(got {len(experiment.variants)})."
        )

    seen: set[str] = set()
    for v in experiment.variants:
        if v.label in seen:
            raise ValueError(
                f"Duplicate variant label detected: '{v.label}'. "
                "Each variant label must be unique within an experiment."
            )
        seen.add(v.label)

    for v in experiment.variants:
        if not (0.0 <= v.temperature <= 2.0):
            raise ValueError(
                f"Variant '{v.label}' has an invalid temperature "
                f"{v.temperature!r}. Must be between 0 and 2 (inclusive)."
            )
        if v.max_tokens <= 0:
            raise ValueError(
                f"Variant '{v.label}' has an invalid max_tokens value "
                f"{v.max_tokens!r}. Must be greater than 0."
            )


# ---------------------------------------------------------------------------
# 13. run_experiment
#     Validates, executes all variants concurrently with a global timeout,
#     and always returns an ExperimentRun — never raises.
# ---------------------------------------------------------------------------

async def run_experiment(experiment: Experiment) -> ExperimentRun:
    """
    Validate and run all variants of an A/B experiment concurrently.

    Execution flow
    --------------
    1. ``validate_experiment`` is called first — invalid experiments are
       rejected immediately before any adapter is touched.
    2. All variant coroutines are gathered under a 30-second global timeout.
       ``return_exceptions=True`` ensures one failing variant cannot cancel
       the rest.
    3. A ``TimeoutError`` causes every variant to be marked failed; an
       ``ExperimentRun`` is still returned so the caller has a complete record.
    4. The outermost ``except`` block guarantees an ``ExperimentRun`` is
       returned even if validation or an unexpected error prevents execution.

    Returns
    -------
    ExperimentRun
        Always returned — never raises. Check each
        :attr:`VariantResult.execution.status` for per-variant outcomes.
    """
    started_at = datetime.now(timezone.utc)

    def _failed_result(model: str, reason: str) -> ExecutionResult:
        """Zero-value ExecutionResult for any failure path."""
        return ExecutionResult(
            output_text="",
            token_count=0,
            latency_ms=0.0,
            model=model,
            status="failed",
            error_message=reason,
        )

    def _all_failed(variants: List[VariantConfig], reason: str) -> List[VariantResult]:
        """Build a full failed VariantResult list when the run cannot proceed."""
        return [
            VariantResult(
                variant_label=v.label,
                config=v,
                execution=_failed_result(v.model, reason),
                run_at=datetime.now(timezone.utc),
            )
            for v in variants
        ]

    try:
        # --- Step 1: validate before touching any adapter ---
        validate_experiment(experiment)

        # --- Step 2: build one coroutine per variant ---
        tasks = [
            execute_variant(
                prompt=experiment.input_text,
                model=variant.model,
                temperature=variant.temperature,
                max_tokens=variant.max_tokens,
            )
            for variant in experiment.variants
        ]

        # --- Step 3: run concurrently under global timeout ---
        try:
            raw_results: list[ExecutionResult | BaseException] = list(
                cast(
                    "tuple[ExecutionResult | BaseException, ...]",
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=_TIMEOUT_SECONDS,
                    ),
                )
            )
        except asyncio.TimeoutError:
            completed_at = datetime.now(timezone.utc)
            return ExperimentRun(
                experiment_id=experiment.id,
                started_at=started_at,
                completed_at=completed_at,
                variant_results=_all_failed(
                    experiment.variants,
                    "Experiment execution timed out",
                ),
            )

        # --- Step 4: classify each raw result ---
        variant_results: list[VariantResult] = []
        for variant, raw in zip(experiment.variants, raw_results):
            if isinstance(raw, BaseException):
                execution: ExecutionResult = _failed_result(variant.model, str(raw))
            else:
                assert isinstance(raw, ExecutionResult)
                execution = raw

            variant_results.append(
                VariantResult(
                    variant_label=variant.label,
                    config=variant,
                    execution=execution,
                    run_at=datetime.now(timezone.utc),
                )
            )

        completed_at = datetime.now(timezone.utc)
        return ExperimentRun(
            experiment_id=experiment.id,
            started_at=started_at,
            completed_at=completed_at,
            variant_results=variant_results,
        )

    except Exception as exc:  # noqa: BLE001 — top-level safety net
        # Validation failures, unexpected errors, or anything not caught above
        # are boxed here so the caller always receives a complete ExperimentRun.
        completed_at = datetime.now(timezone.utc)
        return ExperimentRun(
            experiment_id=experiment.id,
            started_at=started_at,
            completed_at=completed_at,
            variant_results=_all_failed(
                experiment.variants,
                f"Experiment failed: {exc}",
            ),
        )


# ---------------------------------------------------------------------------
# 14. get_successful_results
# ---------------------------------------------------------------------------

def get_successful_results(run: ExperimentRun) -> List[VariantResult]:
    """
    Return only the :class:`VariantResult` objects whose execution succeeded.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to filter.

    Returns
    -------
    List[VariantResult]
        All variant results where ``execution.status == "success"``.
        Empty list if none succeeded.
    """
    return [vr for vr in run.variant_results if vr.execution.status == "success"]


# ---------------------------------------------------------------------------
# 15. get_failed_results
# ---------------------------------------------------------------------------

def get_failed_results(run: ExperimentRun) -> List[VariantResult]:
    """
    Return only the :class:`VariantResult` objects whose execution failed.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to filter.

    Returns
    -------
    List[VariantResult]
        All variant results where ``execution.status == "failed"``.
        Empty list if none failed.
    """
    return [vr for vr in run.variant_results if vr.execution.status == "failed"]


# ---------------------------------------------------------------------------
# 16. get_fastest_variant
# ---------------------------------------------------------------------------

def get_fastest_variant(run: ExperimentRun) -> Optional[VariantResult]:
    """
    Return the successful variant with the lowest ``latency_ms``.

    Only successful results are considered; failed variants are excluded
    because their ``latency_ms`` is always 0.0 and would corrupt the ranking.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to inspect.

    Returns
    -------
    Optional[VariantResult]
        The fastest successful variant, or ``None`` if no variant succeeded.
    """
    successful = get_successful_results(run)
    if not successful:
        return None
    return min(successful, key=lambda vr: vr.execution.latency_ms)


# ---------------------------------------------------------------------------
# 17. get_highest_token_variant
# ---------------------------------------------------------------------------

def get_highest_token_variant(run: ExperimentRun) -> Optional[VariantResult]:
    """
    Return the successful variant with the highest ``token_count``.

    Only successful results are considered; failed variants always report
    ``token_count == 0`` and would corrupt the ranking.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to inspect.

    Returns
    -------
    Optional[VariantResult]
        The variant that used the most tokens, or ``None`` if none succeeded.
    """
    successful = get_successful_results(run)
    if not successful:
        return None
    return max(successful, key=lambda vr: vr.execution.token_count)


# ---------------------------------------------------------------------------
# 18. summarize_experiment_run
# ---------------------------------------------------------------------------

def summarize_experiment_run(run: ExperimentRun) -> dict:
    """
    Produce a concise summary dictionary for an :class:`ExperimentRun`.

    The summary is suitable for logging, API responses, or displaying a
    results overview in the UI without exposing the full nested structure.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to summarise.

    Returns
    -------
    dict
        Keys:

        - ``experiment_id``     — the experiment's unique identifier
        - ``variant_count``     — total number of variants executed
        - ``successful_variants`` — count of variants with status ``"success"``
        - ``failed_variants``   — count of variants with status ``"failed"``
        - ``fastest_variant``   — label of the fastest successful variant,
                                  or ``None`` if none succeeded
        - ``total_runtime_ms``  — wall-clock duration of the entire run in ms
    """
    successful = get_successful_results(run)
    failed = get_failed_results(run)
    fastest = get_fastest_variant(run)
    total_runtime_ms = (
        (run.completed_at - run.started_at).total_seconds() * 1_000
    )

    return {
        "experiment_id": run.experiment_id,
        "variant_count": len(run.variant_results),
        "successful_variants": len(successful),
        "failed_variants": len(failed),
        "fastest_variant": fastest.variant_label if fastest is not None else None,
        "total_runtime_ms": total_runtime_ms,
    }


# ---------------------------------------------------------------------------
# 19. format_variant_result
#     Returns a human-readable multi-line string for a single VariantResult.
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 40


def format_variant_result(result: VariantResult) -> str:
    """
    Format a single :class:`VariantResult` as a human-readable string.

    Parameters
    ----------
    result : VariantResult
        The variant result to format.

    Returns
    -------
    str
        A multi-line block showing label, model, status, latency, token
        count, and an 80-character output preview.  Failed variants also
        include the ``Error:`` line.

    Example
    -------
    ::

        ----------------------------------------
        Variant: Variant A
        Model: gpt-4o
        Status: success
        Latency: 312.45 ms
        Tokens: 187
        Output Preview: OpenAI simulated response to: Summarise the following...
        ----------------------------------------
    """
    exc = result.execution
    preview = exc.output_text[slice(80)].replace("\n", " ")

    lines = [
        _DIVIDER,
        f"Variant: {result.variant_label}",
        f"Model:   {exc.model}",
        f"Status:  {exc.status}",
        f"Latency: {exc.latency_ms} ms",
        f"Tokens:  {exc.token_count}",
        f"Output Preview: {preview}",
    ]

    if exc.status == "failed":
        lines.append(f"Error:   {exc.error_message}")

    lines.append(_DIVIDER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 20. print_experiment_run
#     Prints a structured per-variant report for an ExperimentRun.
# ---------------------------------------------------------------------------

_HEADER = "=" * 40


def print_experiment_run(run: ExperimentRun) -> None:
    """
    Print a full structured report for an :class:`ExperimentRun`.

    Outputs a header block with timing metadata followed by a formatted
    block for every variant in ``run.variant_results``.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to display.

    Example output
    --------------
    ::

        ========================================
        Experiment: exp-001
        Started:   2024-01-15 10:00:00.123456
        Completed: 2024-01-15 10:00:00.678901
        Runtime:   555.445 ms
        ========================================
        ----------------------------------------
        Variant: Variant A
        ...
    """
    runtime_ms = (run.completed_at - run.started_at).total_seconds() * 1_000

    print(_HEADER)
    print(f"Experiment: {run.experiment_id}")
    print(f"Started:    {run.started_at}")
    print(f"Completed:  {run.completed_at}")
    print(f"Runtime:    {runtime_ms:.2f} ms")
    print(_HEADER)

    for result in run.variant_results:
        print(format_variant_result(result))


# ---------------------------------------------------------------------------
# 21. print_experiment_summary
#     Prints the aggregated summary produced by summarize_experiment_run.
# ---------------------------------------------------------------------------

def print_experiment_summary(run: ExperimentRun) -> None:
    """
    Print a concise summary of an :class:`ExperimentRun`.

    Delegates to :func:`summarize_experiment_run` for the data layer so
    the display logic stays decoupled from the aggregation logic.

    Parameters
    ----------
    run : ExperimentRun
        The completed experiment run to summarise.

    Example output
    --------------
    ::

        Experiment Summary
        ------------------
        Total Variants : 3
        Successful     : 2
        Failed         : 1
        Fastest Variant: Variant A
        Total Runtime  : 555.45 ms
    """
    summary = summarize_experiment_run(run)
    fastest_label = summary["fastest_variant"] if summary["fastest_variant"] is not None else "None"

    print("Experiment Summary")
    print("------------------")
    print(f"Total Variants : {summary['variant_count']}")
    print(f"Successful     : {summary['successful_variants']}")
    print(f"Failed         : {summary['failed_variants']}")
    print(f"Fastest Variant: {fastest_label}")
    print(f"Total Runtime  : {summary['total_runtime_ms']:.2f} ms")


# ---------------------------------------------------------------------------
# 22. create_sample_experiment
#     Builds a ready-to-run demo Experiment with three variants.
# ---------------------------------------------------------------------------

def create_sample_experiment() -> Experiment:
    """
    Return a canonical demo :class:`Experiment` for testing and development.

    The experiment pits two OpenAI ``gpt-4o`` variants (different
    temperatures) against one Anthropic ``claude-3-sonnet`` variant on a
    simple factual question.  All three use the mock adapters so no API
    keys or network access are required.

    Returns
    -------
    Experiment
        A fully-configured experiment ready to pass to :func:`run_experiment`.
    """
    return Experiment(
        id="demo-experiment",
        name="Demo A/B Test",
        input_text="Explain the concept of black holes in simple terms.",
        variants=[
            VariantConfig(
                label="Variant A",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=300,
                prompt_version_id=None,
            ),
            VariantConfig(
                label="Variant B",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=300,
                prompt_version_id=None,
            ),
            VariantConfig(
                label="Variant C",
                model="gpt-4o",
                temperature=0.3,
                max_tokens=300,
                prompt_version_id=None,
            ),

        ],
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# 23. run_demo_experiment
#     Async driver that exercises the full pipeline end-to-end.
# ---------------------------------------------------------------------------

async def run_demo_experiment() -> None:
    """
    Run the demo experiment and print a full report plus summary.

    Pipeline
    --------
    1. Build the experiment via :func:`create_sample_experiment`.
    2. Execute all variants concurrently via :func:`run_experiment`.
    3. Display per-variant details via :func:`print_experiment_run`.
    4. Display aggregated metrics via :func:`print_experiment_summary`.

    This function is intentionally side-effect-only (returns ``None``) so
    it can be called safely from a ``__main__`` guard or a test harness
    without needing to inspect a return value.
    """
    experiment = create_sample_experiment()

    print(f"\nStarting demo experiment: '{experiment.name}'")
    print(f"Input: {experiment.input_text!r}")
    print(f"Variants: {len(experiment.variants)}\n")

    run = await run_experiment(experiment)

    print_experiment_run(run)
    print()
    print_experiment_summary(run)


# ---------------------------------------------------------------------------
# 24. build_experiment
#     Convenience constructor called by API request handlers.
# ---------------------------------------------------------------------------

def build_experiment(
    experiment_id: str,
    name: str,
    input_text: str,
    variants: List[VariantConfig],
) -> Experiment:
    """
    Construct an :class:`Experiment` with ``created_at`` set automatically.

    This thin wrapper is the canonical factory for creating experiments from
    API request payloads.  Callers supply the four business-logic fields;
    the timestamp is always sourced here so the value is consistent and
    timezone-aware.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment (e.g. a UUID from the database).
    name : str
        Human-readable experiment name.
    input_text : str
        The shared prompt input sent to every variant.
    variants : List[VariantConfig]
        The variant configurations to test.  Will be validated by
        :func:`run_experiment` before any execution begins.

    Returns
    -------
    Experiment
        A fully-constructed experiment ready to pass to
        :func:`run_experiment` or :func:`validate_experiment`.
    """
    return Experiment(
        id=experiment_id,
        name=name,
        input_text=input_text,
        variants=variants,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# 25. ABTestingEngine
#     Thin class wrapper around run_experiment for dependency injection.
# ---------------------------------------------------------------------------

class ABTestingEngine:
    """
    Stable injectable wrapper around the A/B testing engine.

    FastAPI services and routers should depend on *this class* rather than
    calling the free functions directly.  This makes the engine swappable
    in tests (subclass or mock ``ABTestingEngine``) and keeps the routers
    decoupled from the module's internal function names.

    Usage (FastAPI)
    ---------------
    ::

        engine = ABTestingEngine()

        @router.post("/experiments/{id}/run")
        async def run_experiment_endpoint(id: str):
            experiment = build_experiment(id, ...)
            return await engine.run(experiment)
    """

    async def run(self, experiment: Experiment) -> ExperimentRun:
        """
        Validate and run all variants of ``experiment`` concurrently.

        Delegates entirely to :func:`run_experiment`.  The method signature
        is intentionally identical so the class is a transparent proxy —
        any future cross-cutting concern (logging, metrics, rate limiting)
        can be added here without touching the underlying engine function.

        Parameters
        ----------
        experiment : Experiment
            A fully-configured experiment (use :func:`build_experiment` to
            construct one from a request payload).

        Returns
        -------
        ExperimentRun
            Always returned — never raises.  See :func:`run_experiment`.
        """
        return await run_experiment(experiment)


# ---------------------------------------------------------------------------
# 26. run_engine_self_test
#     Runs a stress test simulating multiple experiment executions.
# ---------------------------------------------------------------------------

async def run_engine_self_test() -> None:
    """
    Execute the demo experiment multiple times to validate engine stability.

    Runs the demo experiment 5 times sequentially, aggregating statistics
    across all runs, and prints a final diagnostic report.
    """
    print("Starting AB Testing Engine Self Test...\n")
    
    total_runs = 0
    total_variants = 0
    total_successes = 0
    total_failures = 0
    total_runtime_ms = 0.0

    for i in range(1, 6):
        experiment = create_sample_experiment()
        run = await run_experiment(experiment)
        summary = summarize_experiment_run(run)
        
        total_runs += 1
        total_variants += summary["variant_count"]
        total_successes += summary["successful_variants"]
        total_failures += summary["failed_variants"]
        total_runtime_ms += summary["total_runtime_ms"]
        
        fastest = summary["fastest_variant"] if summary["fastest_variant"] else "None"
        
        print(
            f"Run {i} \u2192 Success: {summary['successful_variants']} | "
            f"Failed: {summary['failed_variants']} | "
            f"Fastest: {fastest} | "
            f"Runtime: {summary['total_runtime_ms']:.0f} ms"
        )

    avg_runtime = total_runtime_ms / total_runs if total_runs > 0 else 0.0

    print("\n" + "-" * 40)
    print("ENGINE SELF TEST RESULTS")
    print("-" * 40)
    print(f"Runs executed: {total_runs}")
    print(f"Total variants: {total_variants}")
    print(f"Successful variants: {total_successes}")
    print(f"Failed variants: {total_failures}")
    print(f"Average runtime: {avg_runtime:.0f} ms")
    print("-" * 40)


# ---------------------------------------------------------------------------
# 27. run_cli
#     Interactive CLI entry point — collects user config, then runs.
# ---------------------------------------------------------------------------

async def run_cli() -> None:
    """
    Interactive command-line interface for the A/B testing engine.

    Prompts the user for:
    - A shared input prompt
    - The number of variants and their configs (model, temperature, max_tokens)
    - How many times to repeat the experiment

    Then builds and runs the experiment ``n`` times, printing a summary
    after each run using the existing :func:`print_experiment_summary`.
    """
    _LABELS = list("ABCDEFGHIJKLMNOP")  # auto-labelling pool

    print("\n" + "=" * 42)
    print("  Prompt-Lab A/B Testing Engine — CLI")
    print("=" * 42 + "\n")

    # --- Step 1: shared input prompt ---
    input_text = input("Enter the input prompt:\n> ").strip()
    if not input_text:
        print("Error: input prompt cannot be empty.")
        return

    # --- Step 2: variants ---
    try:
        num_variants = int(input("\nHow many variants? (1–6): ").strip())
    except ValueError:
        print("Error: please enter a whole number.")
        return

    if not (1 <= num_variants <= 6):
        print("Error: number of variants must be between 1 and 6.")
        return

    variants: List[VariantConfig] = []
    for i in range(num_variants):
        label = f"Variant {_LABELS[i]}"
        print(f"\n--- {label} ---")
        model = input("  Model (e.g. gpt-4o / claude-3-sonnet): ").strip()
        try:
            temperature = float(input("  Temperature (0.0 – 2.0): ").strip())
            max_tokens = int(input("  Max tokens: ").strip())
        except ValueError:
            print("  Error: invalid number. Aborting.")
            return

        variants.append(
            VariantConfig(
                label=label,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    # --- Step 3: number of runs ---
    try:
        num_runs = int(input("\nHow many times should the experiment run? ").strip())
    except ValueError:
        print("Error: please enter a whole number.")
        return

    if num_runs < 1:
        print("Error: must run at least once.")
        return

    # --- Step 4: build and execute ---
    experiment = build_experiment(
        experiment_id="cli-experiment",
        name="CLI A/B Test",
        input_text=input_text,
        variants=variants,
    )

    print(f"\nRunning experiment {num_runs} time(s) with {len(variants)} variant(s)...\n")

    for run_number in range(1, num_runs + 1):
        print(f"--- Run {run_number} of {num_runs} ---")
        run = await run_experiment(experiment)
        print_experiment_summary(run)
        print()

    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point — run as: python src/AB_Testing.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio  # local alias keeps the global namespace tidy
    _asyncio.run(run_cli())
