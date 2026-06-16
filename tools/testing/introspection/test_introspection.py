# Owner(s): ["module: ci"]
"""Tests for the test-introspection collector.

Fast tests cover the platform/descriptor logic. The collector smoke tests import a
real (small) test file in a subprocess and are marked slow.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

from tools.testing.introspection import (
    collector,
    diff,
    platforms,
    post_comment,
    render_comment,
    where,
)

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase


INDUCTOR = "test/inductor/test_torchinductor.py"


class TestPlatforms(TestCase):
    def test_registry_get(self):
        self.assertEqual(platforms.get("linux-cpu").device_type, "cpu")
        with self.assertRaises(KeyError):
            platforms.get("does-not-exist")

    def test_cuda_caps_sm_derived(self):
        sm80 = platforms.get("linux-cuda-sm80")
        sm90 = platforms.get("linux-cuda-sm90")
        # FP8 needs SM89+, so it is off for SM80 and on for SM90.
        self.assertFalse(sm80.caps["PLATFORM_SUPPORTS_FP8"])
        self.assertTrue(sm90.caps["PLATFORM_SUPPORTS_FP8"])
        # Flash attention is SM80+.
        self.assertTrue(sm80.caps["PLATFORM_SUPPORTS_FLASH_ATTENTION"])

    def test_subprocess_env_hides_accelerators(self):
        env = platforms.get("linux-rocm").subprocess_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTORCH_TEST_WITH_ROCM"], "1")


class TestCollector(TestCase):
    @slowTest
    def test_worker_does_not_shadow_torch_from_cwd(self):
        # In CI torch is wheel-installed (site-packages) while the repo tree still has
        # a torch/ source dir. The worker must import the installed torch, not the repo
        # source. Reproduce by running the worker from a cwd containing a poison torch/
        # package; it must still import the real torch (worker runs by path, so cwd is
        # not on sys.path[0]).
        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / "torch").mkdir()
            (pathlib.Path(td) / "torch" / "__init__.py").write_text(
                "raise RuntimeError('poison torch on path')\n"
            )
            env = dict(os.environ)
            env.update(platforms.get_job("linux-cpu").subprocess_env())
            proc = subprocess.run(
                [
                    sys.executable,
                    collector._COLLECTOR,
                    "linux-cpu/default",
                    "enumerate",
                    "test/test_bundled_inputs.py",
                ],
                cwd=td,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertNotIn("poison torch", proc.stderr)
            self.assertTrue(
                any(
                    line.startswith(collector._SENTINEL)
                    for line in proc.stdout.splitlines()
                ),
                msg=f"worker failed:\n{proc.stderr[-2000:]}",
            )

    @slowTest
    def test_device_gating(self):
        # GPU classes appear only on the cuda platform, not on cpu.
        cpu = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cpu"), use_cache=False
        )
        cuda = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cuda-sm80"), use_cache=False
        )
        self.assertNotIn("GPUTests", cpu)
        self.assertIn("GPUTests", cuda)
        self.assertIn("CpuTests", cpu)

    @slowTest
    def test_status_consistency(self):
        # ran union skipped must equal the enumerated set.
        job = platforms.get_job("linux-cpu")
        enum = collector.enumerate_tests(INDUCTOR, job, use_cache=False)
        enumerated = {f"{c}::{m}" for c, ms in enum.items() for m in ms}
        st = collector.status(INDUCTOR, job, use_cache=False)
        observed = set(st["ran"]) | {k for k, _ in st["skipped"]}
        self.assertEqual(enumerated, observed)


class TestDiff(TestCase):
    def test_is_test_py(self):
        self.assertTrue(diff._is_test_py("test/test_x.py"))
        self.assertTrue(diff._is_test_py("test/nn/test_pooling.py"))
        self.assertFalse(diff._is_test_py("test/helper.py"))
        self.assertFalse(diff._is_test_py("torch/x.py"))

    def test_is_broad(self):
        # Generation/selection surface + test infra are broad.
        self.assertTrue(diff._is_broad("torch/testing/_internal/common_utils.py"))
        self.assertTrue(diff._is_broad("tools/testing/discover_tests.py"))
        self.assertTrue(diff._is_broad("test/run_test.py"))
        self.assertTrue(diff._is_broad("test/conftest.py"))
        # Behavior-only source changes can't add/remove tests -> not broad.
        self.assertFalse(diff._is_broad("torch/csrc/foo.cpp"))
        self.assertFalse(diff._is_broad("torch/utils/flop_counter.py"))
        self.assertFalse(diff._is_broad("test/test_x.py"))
        self.assertFalse(diff._is_broad("test/nn/test_pooling.py"))
        # Non-.py data under test/ (xfail lists) marks xfails, not existence.
        self.assertFalse(
            diff._is_broad("test/inductor/pallas_expected_failures/CpuTests.test_foo")
        )

    def test_module_ids(self):
        self.assertEqual(
            diff._module_ids("test/inductor/test_x.py"), ("inductor.test_x", "test_x")
        )

    def test_scope_pulls_importers(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "test").mkdir()
            (root / "test" / "test_base.py").write_text("class A:\n    pass\n")
            (root / "test" / "test_dep.py").write_text("from test_base import A\n")
            (root / "test" / "test_other.py").write_text("import os\n")
            sel = ["test/test_base.py", "test/test_dep.py", "test/test_other.py"]
            graph = diff._build_import_graph(sel, root)
            aff = diff._scope(["test/test_base.py"], sel, graph)
            self.assertIn("test/test_dep.py", aff)  # synthetic dependent pulled in
            self.assertNotIn("test/test_other.py", aff)


class TestRenderComment(TestCase):
    # Synthetic diff() result: test_plain (same id) added on both jobs -> "all
    # platforms"; test_b_cuda only on the cuda job.
    RES = {
        "from": "a" * 40,
        "to": "b" * 40,
        "per_job": {
            "linux-cpu/default": {
                "scope_reason": "test-only change",
                "n_affected": 1,
                "uncomparable": [],
                "per_file": {
                    "test/test_x.py": {"added": ["C::test_plain"], "removed": []}
                },
            },
            "linux-cuda-sm80/default": {
                "scope_reason": "test-only change",
                "n_affected": 1,
                "uncomparable": ["test/inductor/test_y.py"],
                "per_file": {
                    "test/test_x.py": {
                        "added": ["C::test_plain", "C::test_b_cuda"],
                        "removed": [],
                    }
                },
            },
        },
    }

    def test_render_groups_and_marker(self):
        md = render_comment.render(self.RES)
        self.assertIn(render_comment.MARKER, md)
        self.assertIn("+2 added", md)  # test_plain + test_b_cuda (distinct ids)
        # test_plain exists on both jobs -> "all platforms"; test_b_cuda only on cuda.
        self.assertIn("all platforms", md)
        self.assertIn("linux-cuda-sm80/default", md)
        self.assertIn("`test/test_x.py::C::test_b_cuda`", md)
        self.assertIn("1 file(s) could not be compared", md)

    def test_render_broad_note(self):
        res = dict(self.RES, broad=True)
        md = render_comment.render(res)
        self.assertIn("limited to", md)
        self.assertIn("linux-cpu", md)

    def test_render_empty(self):
        empty = {
            "from": "a" * 40,
            "to": "b" * 40,
            "per_job": {
                "linux-cpu/default": {
                    "scope_reason": "",
                    "n_affected": 0,
                    "uncomparable": [],
                    "per_file": {},
                }
            },
        }
        self.assertIn("No tests added or removed", render_comment.render(empty))


class TestPostComment(TestCase):
    def test_skipped_is_noop(self):
        # A skipped result posts nothing and needs no token/network.
        post_comment.post({"pr": 1, "skipped": True})

    def test_upsert_posts_when_no_existing(self):
        calls = []

        def fake_api(method, url, token, data=None):
            calls.append((method, data))
            return [] if method == "GET" else {"id": 1}

        orig_api = post_comment._api
        orig_env = {k: os.environ.get(k) for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY")}
        post_comment._api = fake_api
        os.environ["GITHUB_TOKEN"] = "x"
        os.environ["GITHUB_REPOSITORY"] = "o/r"
        try:
            post_comment.post({**TestRenderComment.RES, "pr": 5, "skipped": False})
        finally:
            post_comment._api = orig_api
            for k, v in orig_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        methods = [m for m, _ in calls]
        self.assertIn("GET", methods)
        self.assertIn("POST", methods)
        posted = next(d for m, d in calls if m == "POST")
        self.assertIn(render_comment.MARKER, posted["body"])


class TestWhere(TestCase):
    def test_match(self):
        tid = "TestFooCUDA::test_bar_cuda_float32"
        self.assertTrue(where._match(tid, tid))  # full id
        self.assertTrue(where._match("test_bar_cuda_float32", tid))  # bare method
        self.assertTrue(where._match("test_bar", tid))  # substring
        self.assertFalse(where._match("test_baz", tid))


if __name__ == "__main__":
    run_tests()
