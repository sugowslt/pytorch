"""Structured compile/cache instrumentation for ``torch._native`` ops.

Native DSL ops compile device kernels lazily on first call. Those compiles
are the dominant first-call latency, and a silent cache miss is a common
"why is this slow again?" question. This module surfaces both, with no
runtime cost when neither ``TORCH_LOGS`` nor structured tracing is enabled.

Two sinks, both fed by a single :class:`CompileEvent`:

* The ``native_dsl`` logger (``TORCH_LOGS=+native_dsl``): a one-line
  human-readable summary per compile -- outcome, wall time, and running
  hit/miss totals.
* ``trace_structured`` artifacts (tlparse): a JSON record per compile, for
  production jobs where only the structured trace is retrievable.

Two DSLs compile through different machinery, so there are two entry points.
Both reduce to the same shared core (:func:`_make_wrapper`): snapshot the
cache, time the call, snapshot again, and flag ``compiled`` when the miss
counter advanced. They differ only in how a snapshot is sampled:

* :func:`instrument_cutedsl_compile` -- for CuTeDSL, stacked *above* the
  vendored ``quack`` ``@jit_cache`` decorator. It reads the cache wrapper's
  ``cache_info()`` and times the wrapped ``cute.compile`` call::

      @instrument_cutedsl_compile("aten::topk")
      @jit_cache
      def _compile_topk_radix(N, K, deterministic): ...

  A ``cache_info().misses`` delta means the cache ran a real
  ``cute.compile``, so the measured wall time *is* the compile time;
  otherwise the key was served from the in-memory or on-disk ``.o`` cache.

* :func:`instrument_triton_launch` -- for Triton ``@triton.jit`` kernels,
  which compile *and* launch in one ``kernel[grid](...)`` call and keep
  their own per-kernel cache (``JITFunction.device_caches``). The wrapper
  watches that cache's running variant count, which plays the role of the
  miss counter: a new entry means a fresh Triton compile fired this call.
  Because compile and launch are fused, ``wall_ms`` on a miss is compile +
  host-launch latency (compile dominates); on a hit it is just host-launch
  latency.

Both DSLs only expose miss-side signal directly (CuTeDSL's vendored cache
reports aggregate counters; Triton's cache only grows), so finer reasons
(disk-hit vs lock-timeout, Triton's on-disk cache) are not distinguished
here -- the boolean ``compiled`` flag plus wall time covers the common case.

Neither entry point touches the underlying DSL/vendored code, and neither
hijacks Triton's process-global ``knobs.runtime`` hooks (those would also
capture Inductor's unrelated Triton compiles).
"""

from __future__ import annotations

import functools
import logging
import time
import weakref
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "CompileEvent",
    "instrument_cutedsl_compile",
    "instrument_triton_launch",
]

log = logging.getLogger(__name__)

# tlparse artifact name. The "artifact" envelope (see trace_structured_artifact)
# is the well-supported transport; the name lets tlparse group these events.
_ARTIFACT_NAME = "native_dsl_compile"

R = TypeVar("R")


@dataclass(frozen=True)
class CompileEvent:
    """One compile-function invocation, as recorded to logs and tlparse.

    ``compiled`` is the ground truth (did the cache run a real compile);
    ``outcome`` is its human-readable form. ``hits`` / ``misses`` are the
    cache's running totals *after* this call, useful for spotting churn
    (e.g. misses climbing across calls of the same shape => keys not
    stable, or the persistent cache is disabled).
    """

    op: str
    dsl: str
    outcome: str
    compiled: bool
    wall_ms: float
    key: str
    hits: int
    misses: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _emit(event: CompileEvent) -> None:
    """Fan the event out to the native_dsl logger and tlparse.

    Both sinks self-gate (logging on level, trace_structured on
    ``trace_log.handlers``), so this is cheap when nothing is listening.
    """
    log.info(
        "%s [%s] %s in %.1fms (key=%s, cache hits=%d misses=%d)",
        event.op,
        event.dsl,
        event.outcome,
        event.wall_ms,
        event.key,
        event.hits,
        event.misses,
    )

    # Local import keeps `import torch._native` from pulling torch._logging's
    # heavier transitive imports at registration time.
    from torch._logging._internal import trace_structured

    # Same "artifact" envelope as Dynamo's trace_structured_artifact(); we call
    # trace_structured directly only to pass expect_trace_id=False, since that
    # helper would capture an expensive stack on every eager event (no live
    # CompileContext). trace_structured still reads the live trace id, so a
    # native op compiling inside torch.compile is auto-tagged with the ambient
    # frame ids and nests under that compile in tlparse.
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": _ARTIFACT_NAME, "encoding": "json"},
        expect_trace_id=False,
        payload_fn=lambda: _json_payload(event),
    )


def _json_payload(event: CompileEvent) -> str:
    import json

    return json.dumps(event.as_dict(), sort_keys=True)


def _format_key(args: tuple, kwargs: dict, key_fn: Callable | None) -> str:
    if key_fn is not None:
        try:
            return key_fn(*args, **kwargs)
        except Exception:
            pass
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    return "(" + ", ".join(parts) + ")"


def _make_wrapper(
    fn: Callable[..., R],
    op: str,
    dsl: str,
    key_fn: Callable[..., str] | None,
    sample: Callable[[], tuple[int | None, int | None]],
) -> Callable[..., R]:
    """Shared instrumentation core for both DSL entry points.

    ``sample()`` returns a ``(hits, misses)`` snapshot of the relevant cache;
    a ``misses`` increase across the call means a real compile fired. Timing,
    error handling, classification, and emission are identical across DSLs --
    only ``sample`` (and the reported ``dsl``) differ.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        _, misses_before = sample()
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            outcome_is_error = False
        except Exception:
            outcome_is_error = True
            raise
        finally:
            wall_ms = (time.perf_counter() - start) * 1e3
            hits_after, misses_after = sample()
            compiled = (
                not outcome_is_error
                and misses_before is not None
                and misses_after is not None
                and misses_after > misses_before
            )
            if outcome_is_error:
                outcome = "error"
            else:
                outcome = "compiled" if compiled else "cache_hit"
            _emit(
                CompileEvent(
                    op=op,
                    dsl=dsl,
                    outcome=outcome,
                    compiled=compiled,
                    wall_ms=wall_ms,
                    key=_format_key(args, kwargs, key_fn),
                    hits=hits_after or 0,
                    misses=misses_after or 0,
                )
            )
        return result

    return wrapper


def _cache_info_sampler(fn: Any) -> Callable[[], tuple[int | None, int | None]]:
    """Sampler reading ``(hits, misses)`` from a ``jit_cache`` wrapper.

    Defensive: the wrapped callable may not be ``jit_cache``-decorated (e.g.
    a plain function in tests). Then we report ``(None, None)`` and the core
    still times the call, just without compile/hit classification.
    """

    def sample() -> tuple[int | None, int | None]:
        info = getattr(fn, "cache_info", None)
        if info is None:
            return None, None
        try:
            ci = info()
            return ci.hits, ci.misses
        except Exception:
            return None, None

    return sample


def instrument_cutedsl_compile(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Instrument a CuTeDSL (``@jit_cache``-decorated) compile function.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::topk"``.
        key_fn: Optional callable with the wrapped function's signature
            returning a short string describing the compile key for logs.
            Defaults to a repr of the args/kwargs.

    Returns a decorator. The decorated function behaves identically to the
    original (same return value, same caching); it only adds a log line and
    a tlparse artifact per call. Errors raised by the wrapped compile are
    timed, reported with ``outcome="error"``, and re-raised unchanged.
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        wrapper = _make_wrapper(fn, op, "cutedsl", key_fn, _cache_info_sampler(fn))
        # Forward jit_cache's bespoke attributes (functools.wraps doesn't copy
        # them) so the instrumented function stays a drop-in for callers that
        # introspect the cache.
        for attr in ("cache", "cache_clear", "cache_info"):
            if hasattr(fn, attr):
                setattr(wrapper, attr, getattr(fn, attr))
        return wrapper

    return decorator


def _triton_cache_size(kernel: Any) -> int | None:
    """Total compiled-variant count across a JITFunction's per-device caches.

    Triton stores one entry per specialized variant in
    ``JITFunction.device_caches[device]``, a ``(kernel_cache, ...)`` tuple
    whose first element is the dict of compiled kernels. The count grows by
    one each time a new (signature, constexpr, options) variant compiles, so
    a delta across a launch tells us whether this call triggered a compile.
    Returns None if the object doesn't expose ``device_caches`` (e.g. a
    plain callable in tests, or a future Triton that renames this).
    """
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    try:
        return sum(len(per_device[0]) for per_device in caches.values())
    except Exception:
        return None


def instrument_triton_launch(
    op: str,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Instrument a function that launches a Triton kernel.

    Unlike CuTeDSL, a Triton ``@triton.jit`` kernel compiles lazily *inside*
    its ``kernel[grid](...)`` launch and caches variants on the kernel
    object itself. So rather than wrapping a separate compile function, wrap
    the op's launch helper and pass the ``@triton.jit`` kernel(s) whose cache
    should be watched.

    Args:
        op: Operator symbol being compiled for, e.g. ``"aten::bmm"``.
        key_fn: Optional callable with the wrapped launcher's signature
            returning a short string describing the launch for logs.

    The kernel(s) to watch are discovered lazily from the wrapped function's
    module globals on first call (every ``@triton.jit`` object defined
    there). This keeps the decorator import-light: no Triton import happens
    at registration time, only on first launch.

    Errors raised by the launch are timed, reported with ``outcome="error"``,
    and re-raised unchanged.
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        # Resolved once on first call and cached. weakref so we never keep a
        # JITFunction alive past its module.
        watched: list[weakref.ref] = []
        resolved = False

        def sample() -> tuple[int | None, int | None]:
            # Triton has no hit counter; the total variant count across watched
            # kernels stands in for `misses`, so a delta means a fresh compile.
            nonlocal resolved
            if not resolved:
                resolved = True
                module = __import__(fn.__module__, fromlist=["*"])
                for value in vars(module).values():
                    if _triton_cache_size(value) is not None:
                        watched.append(weakref.ref(value))
            total = 0
            saw_any = False
            for ref in watched:
                kernel = ref()
                if kernel is None:
                    continue
                size = _triton_cache_size(kernel)
                if size is not None:
                    saw_any = True
                    total += size
            return None, (total if saw_any else None)

        return _make_wrapper(fn, op, "triton", key_fn, sample)

    return decorator
