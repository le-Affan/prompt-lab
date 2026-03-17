"""
AB_Testing.py — A/B Testing Engine
====================================
Step 1: Core data structures.

This module is framework-independent. It defines only the data model
used throughout the A/B testing engine. No execution logic lives here yet.
"""

from __future__ import annotations

import asyncio
import contextvars
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, cast


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
# Engine configuration defaults
# These are referenced by the Experiment dataclass field defaults and by
# build_experiment / run_cli, so they must be defined before the dataclasses.
# ---------------------------------------------------------------------------

DEFAULT_MAX_VARIANTS: int = 6
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_CONCURRENCY: int = 5

_current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_run_id", default="")


# ---------------------------------------------------------------------------
# 1. VariantConfig
#    Represents a single experiment variant configuration.
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    """Configuration for a single A/B test variant."""

    label: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: str = ""
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
    max_variants: int = DEFAULT_MAX_VARIANTS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY


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
    cost_usd: float = 0.0
    run_id: Optional[str] = None


def estimate_cost(model: str, tokens: int) -> float:
    """Estimate cost in USD based on model and token usage."""
    model_lower = model.lower()
    if "gpt-4" in model_lower:
        rate = 0.00001
    elif "claude" in model_lower:
        rate = 0.000008
    else:
        rate = 0.00001
    return tokens * rate


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
    run_id: str
    started_at: datetime
    completed_at: datetime
    variant_results: List[VariantResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5b. MultiRunSummary
#     Aggregated analytics across multiple ExperimentRun executions.
# ---------------------------------------------------------------------------

@dataclass
class MultiRunSummary:
    """
    Cross-run analytics produced by :func:`aggregate_runs`.

    Attributes
    ----------
    total_runs : int
        Number of :class:`ExperimentRun` objects that were aggregated.
    total_variants : int
        Sum of all variant executions across every run.
    success_rate_per_variant : dict[str, float]
        Per-label ratio of successful executions to total runs (0.0–1.0).
    avg_latency_per_variant : dict[str, float]
        Per-label mean ``latency_ms`` across successful executions only.
        Labels with zero successes are omitted.
    fastest_variant_overall : Optional[str]
        Label of the variant with the lowest ``avg_latency_per_variant``,
        or ``None`` if no variant succeeded across any run.
    """

    total_runs: int
    total_variants: int
    success_rate_per_variant: dict
    avg_latency_per_variant: dict
    fastest_variant_overall: Optional[str]


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
#    Real OpenAI adapter with mock fallback when api_key is absent.
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """
    OpenAI adapter with automatic mock fallback.

    Behaviour
    ---------
    - **Real mode** (``api_key`` is non-empty): initialises an
      ``openai.AsyncOpenAI`` client and calls
      ``chat.completions.create``.  Requires the ``openai`` package
      (``pip install openai``).
    - **Mock mode** (``api_key`` is empty or ``openai`` is not
      installed): falls back to the original simulated response so
      tests and offline demos continue to work without any keys.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Mock helper (unchanged from original implementation)
    # ------------------------------------------------------------------

    async def _mock_execute(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> ExecutionResult:
        """Original simulated response — used when no real API key is provided."""
        latency_seconds = random.uniform(0.1, 0.8)
        t_start = time.perf_counter()
        await asyncio.sleep(latency_seconds)
        latency_ms = (time.perf_counter() - t_start) * 1_000
        preview = prompt[slice(50)].replace("\n", " ")
        token_count = random.randint(50, max_tokens)
        return ExecutionResult(
            output_text=f"OpenAI simulated response to: {preview}...",
            token_count=token_count,
            latency_ms=round(latency_ms, 2),
            model=model,
            status="success",
            error_message=None,
            cost_usd=estimate_cost(model, token_count),
        )

    # ------------------------------------------------------------------
    # Real API call
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ExecutionResult:
        """
        Run a real OpenAI completion, or fall back to mock if no key is set.

        Falls back to mock when api_key is empty or the openai package is
        not installed.  In real mode, retries up to 3 times on rate-limit
        and transient network errors with linear back-off (1 s, 2 s, 3 s).
        """
        # --- Fallback path: no key or SDK unavailable ---
        if not self.api_key.strip():
            return await self._mock_execute(prompt, model, temperature, max_tokens)

        try:
            import openai  # noqa: PLC0415
        except ImportError:
            return await self._mock_execute(prompt, model, temperature, max_tokens)

        # --- Real API path with retry ---
        client = openai.AsyncOpenAI(api_key=self.api_key)
        last_result: ExecutionResult | None = None

        for attempt in range(3):
            try:
                t_start = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = round((time.perf_counter() - t_start) * 1_000, 2)

                output_text = (
                    response.choices[0].message.content or ""
                    if response.choices else ""
                )
                token_count = (
                    response.usage.completion_tokens if response.usage else 0
                )
                return ExecutionResult(
                    output_text=output_text,
                    token_count=token_count,
                    latency_ms=latency_ms,
                    model=model,
                    status="success",
                    error_message=None,
                    cost_usd=estimate_cost(model, token_count),
                )

            except openai.AuthenticationError as exc:
                # Auth errors are permanent — no point retrying.
                return ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="auth_error",
                    error_message=f"auth_error: invalid API key — {exc}",
                )

            except openai.RateLimitError as exc:
                delay = 1.0 * (attempt + 1)
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="rate_limited",
                    error_message=f"rate_limit: too many requests — {exc}",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] openai/{model} rate-limited — attempt {attempt + 1}/3, "
                          f"waiting {delay:.0f}s")
                    await asyncio.sleep(delay)

            except asyncio.TimeoutError:
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="timeout",
                    error_message="timeout: request timed out",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] openai/{model} timed out — attempt {attempt + 1}/3")

            except (ConnectionError, OSError) as exc:
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="network_error",
                    error_message=f"network_error: connection issue — {exc}",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] openai/{model} network error — attempt {attempt + 1}/3")
                    await asyncio.sleep(1.0 * (attempt + 1))

            except Exception as exc:  # noqa: BLE001
                # Unknown errors are not retried.
                return ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="failed",
                    error_message=f"unknown_error: {exc}",
                )

        # All 3 attempts exhausted — return last classified result.
        return last_result or ExecutionResult(
            output_text="", token_count=0, latency_ms=0.0, model=model,
            status="failed", error_message="unknown_error: all attempts failed",
        )


# ---------------------------------------------------------------------------
# 8. AnthropicAdapter
#    Real Anthropic adapter with mock fallback when api_key is absent.
# ---------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):
    """
    Anthropic adapter with automatic mock fallback.

    Behaviour
    ---------
    - **Real mode** (``api_key`` is non-empty): initialises an
      ``anthropic.AsyncAnthropic`` client and calls
      ``messages.create``.  Requires the ``anthropic`` package
      (``pip install anthropic``).
    - **Mock mode** (``api_key`` is empty or ``anthropic`` is not
      installed): falls back to the original simulated response so
      tests and offline demos continue to work without any keys.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Mock helper (unchanged from original implementation)
    # ------------------------------------------------------------------

    async def _mock_execute(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> ExecutionResult:
        """Original simulated response — used when no real API key is provided."""
        latency_seconds = random.uniform(0.1, 0.8)
        t_start = time.perf_counter()
        await asyncio.sleep(latency_seconds)
        latency_ms = (time.perf_counter() - t_start) * 1_000
        preview = prompt[slice(50)].replace("\n", " ")
        token_count = random.randint(50, max_tokens)
        return ExecutionResult(
            output_text=f"Anthropic simulated response to: {preview}...",
            token_count=token_count,
            latency_ms=round(latency_ms, 2),
            model=model,
            status="success",
            error_message=None,
            cost_usd=estimate_cost(model, token_count),
        )

    # ------------------------------------------------------------------
    # Real API call
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ExecutionResult:
        """
        Run a real Anthropic message, or fall back to mock if no key is set.

        Falls back to mock when api_key is empty or the anthropic package is
        not installed.  In real mode, retries up to 3 times on rate-limit
        and transient network errors with linear back-off (1 s, 2 s, 3 s).
        """
        # --- Fallback path: no key or SDK unavailable ---
        if not self.api_key.strip():
            return await self._mock_execute(prompt, model, temperature, max_tokens)

        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            return await self._mock_execute(prompt, model, temperature, max_tokens)

        # --- Real API path with retry ---
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        last_result: ExecutionResult | None = None

        for attempt in range(3):
            try:
                t_start = time.perf_counter()
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency_ms = round((time.perf_counter() - t_start) * 1_000, 2)

                output_text = (
                    response.content[0].text
                    if response.content and hasattr(response.content[0], "text")
                    else ""
                )
                token_count = (
                    response.usage.output_tokens if response.usage else 0
                )
                return ExecutionResult(
                    output_text=output_text,
                    token_count=token_count,
                    latency_ms=latency_ms,
                    model=model,
                    status="success",
                    error_message=None,
                    cost_usd=estimate_cost(model, token_count),
                )

            except anthropic.AuthenticationError as exc:
                # Auth errors are permanent — no point retrying.
                return ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="auth_error",
                    error_message=f"auth_error: invalid API key — {exc}",
                )

            except anthropic.RateLimitError as exc:
                delay = 1.0 * (attempt + 1)
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="rate_limited",
                    error_message=f"rate_limit: too many requests — {exc}",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] anthropic/{model} rate-limited — attempt {attempt + 1}/3, "
                          f"waiting {delay:.0f}s")
                    await asyncio.sleep(delay)

            except asyncio.TimeoutError:
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="timeout",
                    error_message="timeout: request timed out",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] anthropic/{model} timed out — attempt {attempt + 1}/3")

            except (ConnectionError, OSError) as exc:
                last_result = ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="network_error",
                    error_message=f"network_error: connection issue — {exc}",
                )
                if attempt < 2:
                    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
                    print(f"{run_prefix}[RETRY] anthropic/{model} network error — attempt {attempt + 1}/3")
                    await asyncio.sleep(1.0 * (attempt + 1))

            except Exception as exc:  # noqa: BLE001
                # Unknown errors are not retried.
                return ExecutionResult(
                    output_text="", token_count=0, latency_ms=0.0, model=model,
                    status="failed",
                    error_message=f"unknown_error: {exc}",
                )

        # All 3 attempts exhausted — return last classified result.
        return last_result or ExecutionResult(
            output_text="", token_count=0, latency_ms=0.0, model=model,
            status="failed", error_message="unknown_error: all attempts failed",
        )


# ---------------------------------------------------------------------------
# 9. ADAPTER_REGISTRY
#    Maps provider keys to their concrete LLMAdapter classes (not instances).
#    Add new providers here as they are implemented.
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, Callable[[str], LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


# ---------------------------------------------------------------------------
# 10. get_adapter
#     Looks up the correct adapter by explicit provider name and wires
#     in the API key.  Raises ValueError for unregistered providers.
# ---------------------------------------------------------------------------

def get_adapter(provider: str, api_key: str) -> LLMAdapter:
    """
    Return a concrete ``LLMAdapter`` instance for the given provider.

    Parameters
    ----------
    provider : str
        Provider name, e.g. ``"openai"`` or ``"anthropic"``.
        Must be a key in :data:`ADAPTER_REGISTRY`.
    api_key : str
        The API key forwarded to the adapter constructor.  In mock mode
        this is stored but never used for actual requests.

    Returns
    -------
    LLMAdapter
        A freshly instantiated adapter ready to call ``execute``.

    Raises
    ------
    ValueError
        If ``provider`` is not present in :data:`ADAPTER_REGISTRY`.
    """
    adapter_cls = ADAPTER_REGISTRY.get(provider.lower())
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Registered providers: {list(ADAPTER_REGISTRY)}"
        )
    return adapter_cls(api_key)


# ---------------------------------------------------------------------------
# 11. execute_variant
#     Async router: resolves the adapter and runs a single variant execution.
#     Always returns an ExecutionResult — never raises.
# ---------------------------------------------------------------------------

async def execute_variant(
    config: VariantConfig,
    prompt: str,
) -> ExecutionResult:
    """
    Route a single variant execution to the correct LLM adapter.

    Logs [START] before the call and [END] / [ERROR] after it.
    Always returns an :class:`ExecutionResult` — never raises.
    """
    tag = f"{config.label} ({config.provider}/{config.model})"
    run_prefix = f"[{_current_run_id.get()}] " if _current_run_id.get() else ""
    print(f"{run_prefix}[START] {tag}")
    try:
        adapter = get_adapter(config.provider, config.api_key)
        result = await adapter.execute(
            prompt=prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        result = ExecutionResult(
            output_text="",
            token_count=0,
            latency_ms=0.0,
            model=config.model,
            status="failed",
            error_message=str(exc),
        )

    if result.status == "success":
        print(f"{run_prefix}[END]   {tag} → success ({result.latency_ms} ms, "
              f"{result.token_count} tokens)")
    else:
        print(f"{run_prefix}[ERROR] {tag} → {result.status}: {result.error_message}")

    return result


# ---------------------------------------------------------------------------
# 12. validate_experiment
#     Enforces safety constraints before any API calls are made.
# ---------------------------------------------------------------------------

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

    if len(experiment.variants) > experiment.max_variants:
        raise ValueError(
            f"Experiment exceeds the maximum of {experiment.max_variants} variants "
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
    run_id = f"run_{int(time.time() * 1000)}"
    token = _current_run_id.set(run_id)

    print(f"[{run_id}] Starting experiment ({len(experiment.variants)} variants)")

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

        # --- Step 2: build one coroutine per variant with semaphore + per-variant timeout ---
        semaphore = asyncio.Semaphore(experiment.max_concurrency)

        async def run_with_limit(config: VariantConfig) -> ExecutionResult:
            """Acquire semaphore slot, then run the variant with a per-variant timeout."""
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        execute_variant(config=config, prompt=experiment.input_text),
                        timeout=experiment.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    return ExecutionResult(
                        output_text="",
                        token_count=0,
                        latency_ms=0.0,
                        model=config.model,
                        status="failed",
                        error_message=(
                            f"Variant '{config.label}' timed out after "
                            f"{experiment.timeout_seconds}s"
                        ),
                    )

        tasks = [run_with_limit(variant) for variant in experiment.variants]

        # --- Step 3: run concurrently under global timeout ---
        try:
            raw_results: list[ExecutionResult | BaseException] = list(
                cast(
                    "tuple[ExecutionResult | BaseException, ...]",
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=experiment.timeout_seconds,
                    ),
                )
            )
        except asyncio.TimeoutError:
            completed_at = datetime.now(timezone.utc)
            _current_run_id.reset(token)
            return ExperimentRun(
                experiment_id=experiment.id,
                run_id=run_id,
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

            execution.run_id = run_id

            variant_results.append(
                VariantResult(
                    variant_label=variant.label,
                    config=variant,
                    execution=execution,
                    run_at=datetime.now(timezone.utc),
                )
            )

        completed_at = datetime.now(timezone.utc)
        print(f"[{run_id}] Completed experiment "
              f"in {(completed_at - started_at).total_seconds() * 1000:.0f} ms")
        _current_run_id.reset(token)
        return ExperimentRun(
            experiment_id=experiment.id,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            variant_results=variant_results,
        )

    except Exception as exc:  # noqa: BLE001 — top-level safety net
        # Validation failures, unexpected errors, or anything not caught above
        # are boxed here so the caller always receives a complete ExperimentRun.
        completed_at = datetime.now(timezone.utc)
        print(f"[{run_id}] Completed experiment with error "
              f"in {(completed_at - started_at).total_seconds() * 1000:.0f} ms")
        _current_run_id.reset(token)
        return ExperimentRun(
            experiment_id=experiment.id,
            run_id=run_id,
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

    total_cost_usd = sum(vr.execution.cost_usd for vr in run.variant_results)

    return {
        "experiment_id": run.experiment_id,
        "variant_count": len(run.variant_results),
        "successful_variants": len(successful),
        "failed_variants": len(failed),
        "fastest_variant": fastest.variant_label if fastest is not None else None,
        "total_runtime_ms": total_runtime_ms,
        "total_cost_usd": total_cost_usd,
    }


# ---------------------------------------------------------------------------
# 18b. aggregate_runs
#      Compute cross-run analytics from a list of ExperimentRun objects.
# ---------------------------------------------------------------------------

def aggregate_runs(runs: List[ExperimentRun]) -> MultiRunSummary:
    """
    Aggregate analytics across multiple :class:`ExperimentRun` executions.

    Parameters
    ----------
    runs : List[ExperimentRun]
        One or more completed experiment runs to aggregate.  An empty list
        returns a zeroed-out :class:`MultiRunSummary`.

    Returns
    -------
    MultiRunSummary
        - ``success_rate_per_variant`` — successes / total_runs per label
        - ``avg_latency_per_variant``  — mean latency of successful runs per label
        - ``fastest_variant_overall``  — label with the lowest average latency,
          or ``None`` when no variant ever succeeded
    """
    if not runs:
        return MultiRunSummary(
            total_runs=0,
            total_variants=0,
            success_rate_per_variant={},
            avg_latency_per_variant={},
            fastest_variant_overall=None,
        )

    total_runs = len(runs)
    total_variants = sum(len(r.variant_results) for r in runs)

    # Accumulate per-label success counts and latencies.
    success_counts = {}     # label -> int
    latency_sums = {}       # label -> float (successful runs only)
    latency_counts = {}     # label -> int

    for run in runs:
        for vr in run.variant_results:
            label = vr.variant_label
            success_counts.setdefault(label, 0)
            latency_sums.setdefault(label, 0.0)
            latency_counts.setdefault(label, 0)

            if vr.execution.status == "success":
                success_counts[label] += 1
                latency_sums[label] += vr.execution.latency_ms
                latency_counts[label] += 1

    # Derive rates and averages — guard against division by zero.
    success_rate_per_variant = {
        label: (success_counts[label] / total_runs)
        for label in success_counts
    }

    avg_latency_per_variant = {
        label: (latency_sums[label] / latency_counts[label])
        for label in latency_counts
        if latency_counts[label] > 0
    }

    # Fastest overall = label with lowest average latency.
    fastest_variant_overall = (
        min(avg_latency_per_variant, key=avg_latency_per_variant.get)  # type: ignore[arg-type]
        if avg_latency_per_variant
        else None
    )

    return MultiRunSummary(
        total_runs=total_runs,
        total_variants=total_variants,
        success_rate_per_variant=success_rate_per_variant,
        avg_latency_per_variant=avg_latency_per_variant,
        fastest_variant_overall=fastest_variant_overall,
    )


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
                provider="openai",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=300,
                api_key="",
                prompt_version_id=None,
            ),
            VariantConfig(
                label="Variant B",
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=300,
                api_key="",
                prompt_version_id=None,
            ),
            VariantConfig(
                label="Variant C",
                provider="openai",
                model="gpt-4o",
                temperature=0.3,
                max_tokens=300,
                api_key="",
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
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Experiment:
    """
    Construct an :class:`Experiment` with ``created_at`` set automatically.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment.
    name : str
        Human-readable experiment name.
    input_text : str
        The shared prompt sent to every variant.
    variants : List[VariantConfig]
        Variant configurations; validated by :func:`run_experiment`.
    timeout_seconds : float
        Global (and per-variant) timeout in seconds.  Defaults to
        ``DEFAULT_TIMEOUT_SECONDS`` (30 s).
    max_concurrency : int
        Maximum number of variants that execute simultaneously.  Defaults
        to ``DEFAULT_MAX_CONCURRENCY`` (5).

    Returns
    -------
    Experiment
        Ready to pass to :func:`run_experiment` or :func:`validate_experiment`.
    """
    return Experiment(
        id=experiment_id,
        name=name,
        input_text=input_text,
        variants=variants,
        created_at=datetime.now(timezone.utc),
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
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
    if len(input_text) > 10000:
        print(f"Error: input prompt is too long ({len(input_text)} > 10000 characters).")
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
        provider = input("  Provider (e.g. openai / anthropic): ").strip()
        if not provider:
            print("  Error: provider cannot be empty. Aborting.")
            return

        model = input("  Model (e.g. gpt-4o / claude-3-sonnet): ").strip()
        if not model:
            print("  Error: model cannot be empty. Aborting.")
            return

        api_key = input("  API key (leave blank for mock mode): ").strip()

        try:
            temperature = float(input("  Temperature (0.0 – 2.0): ").strip())
            if not (0.0 <= temperature <= 2.0):
                print("  Error: temperature must be between 0.0 and 2.0. Aborting.")
                return

            max_tokens = int(input("  Max tokens: ").strip())
            if max_tokens <= 0:
                print("  Error: max tokens must be > 0. Aborting.")
                return
        except ValueError:
            print("  Error: invalid number. Aborting.")
            return

        variants.append(
            VariantConfig(
                label=label,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        )

    # --- Step 3: experiment-level config ---
    print("\nExperiment settings (press Enter to use defaults):")

    _timeout_raw = input(
        f"  Timeout in seconds [{DEFAULT_TIMEOUT_SECONDS}]: "
    ).strip()
    try:
        timeout_seconds = float(_timeout_raw) if _timeout_raw else DEFAULT_TIMEOUT_SECONDS
        if timeout_seconds <= 0:
            print("  Error: timeout must be > 0. Aborting.")
            return
    except ValueError:
        print("  Error: invalid timeout. Aborting.")
        return

    _concurrency_raw = input(
        f"  Max concurrent variants [{DEFAULT_MAX_CONCURRENCY}]: "
    ).strip()
    try:
        max_concurrency = int(_concurrency_raw) if _concurrency_raw else DEFAULT_MAX_CONCURRENCY
        if max_concurrency < 1:
            print("  Error: concurrency must be >= 1. Aborting.")
            return
    except ValueError:
        print("  Error: invalid concurrency limit. Aborting.")
        return

    # --- Step 4: number of runs ---
    try:
        num_runs = int(input("\nHow many times should the experiment run? ").strip())
    except ValueError:
        print("Error: please enter a whole number.")
        return

    if num_runs < 1:
        print("Error: must run at least once.")
        return

    # --- Step 5: build and execute ---
    experiment = build_experiment(
        experiment_id="cli-experiment",
        name="CLI A/B Test",
        input_text=input_text,
        variants=variants,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
    )

    print(
        f"\nRunning experiment {num_runs} time(s) | "
        f"{len(variants)} variant(s) | "
        f"timeout={timeout_seconds}s | "
        f"concurrency={max_concurrency}\n"
    )

    all_runs: List[ExperimentRun] = []
    for run_number in range(1, num_runs + 1):
        print(f"--- Run {run_number} of {num_runs} ---")
        run = await run_experiment(experiment)
        all_runs.append(run)
        
        run_id_val = run.variant_results[0].execution.run_id if run.variant_results else "unknown"
        cost_val = sum(v.execution.cost_usd for v in run.variant_results)
        print(f"Run ID: {run_id_val}")
        print(f"Total cost: ${cost_val:.4f}")

        print_experiment_summary(run)
        print()

    # --- Step 6: final cross-run summary ---
    multi = aggregate_runs(all_runs)

    print("=" * 42)
    print("  FINAL SUMMARY")
    print("=" * 42)
    print(f"Total runs    : {multi.total_runs}")
    print(f"Total variants: {multi.total_variants}")
    print()
    print("Success rate per variant:")
    for label, rate in multi.success_rate_per_variant.items():
        print(f"  {label:<20} {rate * 100:.1f}%")
    print()
    print("Avg latency per variant (successful runs only):")
    if multi.avg_latency_per_variant:
        for label, avg_ms in multi.avg_latency_per_variant.items():
            print(f"  {label:<20} {avg_ms:.2f} ms")
    else:
        print("  (no successful runs)")
    print()
    fastest_label = multi.fastest_variant_overall or "None"
    print(f"Fastest variant overall: {fastest_label}")
    print("=" * 42)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI entry point — run as: python src/AB_Testing.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio  # local alias keeps the global namespace tidy
    _asyncio.run(run_cli())
