"""PEP 517 build backend wrapper around scikit-build-core.

Aliases PyTorch's historical MAX_JOBS knob to CMAKE_BUILD_PARALLEL_LEVEL, which
scikit-build-core and CMake natively honor for the `cmake --build` step. setup.py
used to apply MAX_JOBS to the build implicitly; the scikit-build-core backend does
not, so we set the alias here -- in the parent process, before scikit-build-core
snapshots os.environ for the build subprocess.

MAX_JOBS remains PyTorch's umbrella parallelism throttle (linters, cpp_extension
JIT, nccl/MKLDNN sub-builds); this only wires up its top-level build-parallelism
role. A user-set CMAKE_BUILD_PARALLEL_LEVEL always wins.

All other PEP 517 hooks are re-exported from scikit-build-core unchanged.
"""

import os

from scikit_build_core.build import (  # noqa: F401  # pyrefly: ignore[missing-import]
    build_editable as _build_editable,
    build_sdist,
    build_wheel as _build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)


def _alias_max_jobs() -> None:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = max_jobs


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _alias_max_jobs()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _alias_max_jobs()
    return _build_editable(wheel_directory, config_settings, metadata_directory)
