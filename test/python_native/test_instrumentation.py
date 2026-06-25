# Owner(s): ["module: dsl-native-ops"]

import json
import logging
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple

from torch._logging._internal import TorchLogsFormatter, trace_log
from torch._native.instrumentation import (
    CompileEvent,
    instrument_cutedsl_compile,
    instrument_triton_launch,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# No shared tlparse harness exists in torch (see test/dynamo/test_structured_trace.py,
# which defines these locally too), so mirror its minimal pattern here.
HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")


# Mirror of torch._vendor.quack.cache.CacheInfo. Defined locally so the test
# doesn't import quack (which pulls in cutlass, absent on CPU-only builds).
_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _FakeJITFunction:
    """Stand-in for a ``@triton.jit`` kernel's cache surface.

    ``instrument_triton_launch`` watches ``device_caches[dev][0]`` (the dict
    of compiled variants). We model a single fake device and grow the dict
    on each distinct ``key`` to simulate a Triton compile (miss); repeated
    keys leave it unchanged (hit). GPU- and Triton-free.
    """

    def __init__(self):
        # defaultdict-like: one device "cuda:0", value is a (cache_dict, ...) tuple.
        self._cache: dict = {}
        self.device_caches = {"cuda:0": (self._cache, None)}

    def launch(self, key):
        if key not in self._cache:
            self._cache[key] = object()


# Module-level kernel + launcher: instrument_triton_launch resolves watched
# kernels from the launcher's module globals, so both must live here.
_FAKE_KERNEL = _FakeJITFunction()


def _fake_launch(key="v0"):
    _FAKE_KERNEL.launch(key)
    return "launched"


class _FakeJitCache:
    """Stand-in for a ``@jit_cache``-decorated compile function.

    Mimics the bits ``instrument_cutedsl_compile`` observes: a ``cache_info()``
    whose ``misses`` advances on a cold key, plus a controllable wall time
    via the compiled callable. Keeps the test GPU- and CuTeDSL-free.
    """

    def __init__(self):
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0
        self.raise_on_call = False

    def __call__(self, *args, **kwargs):
        if self.raise_on_call:
            raise RuntimeError("boom")
        key = args + tuple(sorted(kwargs.items()))
        if key in self._cache:
            self.hits += 1
        else:
            self.misses += 1
            self._cache[key] = object()
        return self._cache[key]

    def cache_info(self):
        return _CacheInfo(
            hits=self.hits,
            misses=self.misses,
            maxsize=None,
            currsize=len(self._cache),
        )


class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class _LoggerCaptureTest(TestCase):
    """Captures the native_dsl logger so tests can assert on emitted lines."""

    def setUp(self):
        super().setUp()
        self.log = logging.getLogger("torch._native.instrumentation")
        self._orig_level = self.log.level
        self._orig_propagate = self.log.propagate
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self.handler = _CapturingHandler()
        self.log.addHandler(self.handler)

    def tearDown(self):
        self.log.removeHandler(self.handler)
        self.log.setLevel(self._orig_level)
        self.log.propagate = self._orig_propagate
        super().tearDown()

    @property
    def messages(self):
        return [r.getMessage() for r in self.handler.records]


class TestInstrumentation(_LoggerCaptureTest):
    def test_first_call_reports_compiled(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::topk", msg)
        self.assertIn("[cutedsl]", msg)
        self.assertIn("compiled", msg)
        self.assertIn("misses=1", msg)

    def test_second_call_reports_cache_hit(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 2)
        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])
        self.assertIn("hits=1", self.messages[1])

    def test_distinct_keys_each_compile(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(512, 128)

        self.assertEqual(fake.misses, 2)
        for msg in self.messages:
            self.assertIn("compiled", msg)

    def test_key_fn_used_in_log(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)

        compile_fn(256, 64)

        self.assertIn("radix N=256 K=64", self.messages[0])

    def test_error_is_reported_and_reraised(self):
        fake = _FakeJitCache()
        fake.raise_on_call = True
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        with self.assertRaises(RuntimeError):
            compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_cache_attrs_forwarded(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        # jit_cache exposes cache_info / cache_clear; the wrapper must keep
        # them reachable so it's a drop-in replacement.
        self.assertTrue(hasattr(compile_fn, "cache_info"))
        self.assertEqual(compile_fn.cache_info().misses, 0)

    def test_works_without_cache_info(self):
        # A plain callable (no cache_info) must still be timed and reported,
        # just without compiled/cache_hit ground truth (defaults to cache_hit).
        calls = []

        def plain(N, K):
            calls.append((N, K))
            return "ok"

        compile_fn = instrument_cutedsl_compile("aten::topk")(plain)
        self.assertEqual(compile_fn(256, 64), "ok")
        self.assertEqual(calls, [(256, 64)])
        self.assertEqual(len(self.messages), 1)

    def test_compile_event_json_roundtrip(self):
        event = CompileEvent(
            op="aten::topk",
            dsl="cutedsl",
            outcome="compiled",
            compiled=True,
            wall_ms=12.5,
            key="radix N=256 K=64",
            hits=0,
            misses=1,
        )
        loaded = json.loads(json.dumps(event.as_dict(), sort_keys=True))
        self.assertEqual(loaded["op"], "aten::topk")
        self.assertEqual(loaded["compiled"], True)
        self.assertEqual(loaded["misses"], 1)


class TestTritonLaunchInstrumentation(_LoggerCaptureTest):
    def setUp(self):
        super().setUp()
        _FAKE_KERNEL._cache.clear()

    def tearDown(self):
        _FAKE_KERNEL._cache.clear()
        super().tearDown()

    def test_first_launch_reports_compiled(self):
        launch = instrument_triton_launch("aten::bmm")(_fake_launch)

        self.assertEqual(launch(), "launched")

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::bmm", msg)
        self.assertIn("[triton]", msg)
        self.assertIn("compiled", msg)

    def test_repeated_launch_reports_cache_hit(self):
        launch = instrument_triton_launch("aten::bmm")(_fake_launch)

        launch(key="v0")
        launch(key="v0")

        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])

    def test_new_variant_recompiles(self):
        launch = instrument_triton_launch("aten::bmm")(_fake_launch)

        launch(key="v0")
        launch(key="v1")

        for msg in self.messages:
            self.assertIn("compiled", msg)
        # Running variant count surfaces as misses=2 on the second compile.
        self.assertIn("misses=2", self.messages[1])

    def test_error_is_reported_and_reraised(self):
        def boom():
            raise RuntimeError("kaboom")

        launch = instrument_triton_launch("aten::bmm")(boom)
        with self.assertRaises(RuntimeError):
            launch()

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_key_fn_used_in_log(self):
        launch = instrument_triton_launch(
            "aten::bmm", key_fn=lambda key="v0": f"variant={key}"
        )(_fake_launch)

        launch(key="abc")

        self.assertIn("variant=abc", self.messages[0])


class TestTlparseOutput(TestCase):
    """The instrumentation's whole point on production jobs is tlparse-
    retrievable artifacts. Assert the structured-trace plumbing actually
    fires and that tlparse parses the emitted artifact in --strict mode.
    """

    def setUp(self):
        super().setUp()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Raw trace file in the on-disk format tlparse consumes, written via
        # the same TorchLogsFormatter(trace=True) that TORCH_TRACE installs.
        # NB: this handler must be registered BEFORE the capture handler --
        # TorchLogsFormatter(trace=True) populates record.metadata as a side
        # effect of formatting, and the capture handler reads that field.
        self.raw_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", delete=True
        )
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

        # Capture the records so we can assert on metadata/payload without
        # re-parsing the raw file.
        self.records: list[logging.LogRecord] = []
        self.capture = _CapturingHandler()
        self.capture.records = self.records
        trace_log.addHandler(self.capture)

    def tearDown(self):
        trace_log.removeHandler(self.capture)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        trace_log.setLevel(self.old_level)
        super().tearDown()

    def _emit_one(self):
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)
        compile_fn(256, 64)
        self.raw_file.flush()

    def _artifact_records(self):
        return [
            r
            for r in self.records
            if getattr(r, "metadata", {}).get("artifact", {}).get("name")
            == "native_dsl_compile"
        ]

    def test_emits_artifact_record(self):
        self._emit_one()

        recs = self._artifact_records()
        self.assertEqual(len(recs), 1)
        meta = recs[0].metadata["artifact"]
        self.assertEqual(meta["encoding"], "json")
        # Payload must be valid JSON carrying the CompileEvent fields.
        payload = json.loads(recs[0].payload)
        self.assertEqual(payload["op"], "aten::topk")
        self.assertEqual(payload["dsl"], "cutedsl")
        self.assertTrue(payload["compiled"])
        self.assertEqual(payload["key"], "radix N=256 K=64")

    def test_eager_record_has_no_compile_context(self):
        # In eager dispatch there's no live CompileContext, so the record
        # carries no frame id (and -- via expect_trace_id=False -- no
        # diagnostic stack either).
        self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertNotIn("frame_id", meta)
        self.assertNotIn("stack", meta)

    def test_picks_up_live_compile_context(self):
        # When a native op compiles inside a torch.compile (CompileContext is
        # live), the artifact is auto-tagged with the ambient frame ids so it
        # nests under that compile in tlparse -- like a Dynamo artifact.
        from torch._guards import compile_context, CompileContext, CompileId

        cid = CompileId(frame_id=7, frame_compile_id=3)
        with compile_context(CompileContext(cid)):
            self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertEqual(meta["frame_id"], 7)
        self.assertEqual(meta["frame_compile_id"], 3)
        self.assertEqual(meta["attempt"], 0)

    @requires_tlparse
    def test_tlparse_parses_artifact(self):
        self._emit_one()

        # Guard against a false pass: --strict over an empty file exits 0, so
        # assert the artifact was actually written before parsing it.
        with open(self.raw_file.name) as f:
            raw = f.read()
        self.assertIn("native_dsl_compile", raw)

        out = tempfile.mkdtemp()
        try:
            # --strict makes tlparse exit non-zero on any unparsable line, so
            # check_call alone is the assertion.
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
