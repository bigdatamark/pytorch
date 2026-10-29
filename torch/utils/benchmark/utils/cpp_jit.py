# [MODULE DOC]
# Compiles C++ benchmark snippets: fills statement, setup and global setup into a C++ template
# and builds it with the torch cpp extension loader, either as a Python module (timeit
# template) or a standalone executable (callgrind template). Also provides back-testing shims
# (compiler flags and compat valgrind bindings). Import-time side effects: creates a global lock,
# computes the C++ flags (adding debug info, dropping warning flags) and include paths.
"""JIT C++ strings into executables."""
import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension


LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]

# We calculate uuid once at import time so that separate processes will have
# separate build roots, but threads will share the same build root.
# `cpp_extension` uses build root as part of the cache key, so per-invocation
# uuid's (e.g. different build root per _compile_template call) would lead to
# a 0% cache hit rate and spurious recompilation. Consider the following:
#   ```
#   setup = "auto x = torch::ones({1024, 1024});"
#   stmt = "torch::mm(x, x);"
#   for num_threads in [1, 2, 4, 8]:
#     print(Timer(stmt, setup, num_threads=num_threads, language="c++").blocked_autorange())
#   ````
# `setup` and `stmt` do not change, so we can reuse the executable from the
# first pass through the loop.
_BUILD_ROOT: str | None = None

def _get_build_root() -> str:
    # [FUNCTION DOC] HELPER: Lazily creates (once per process) a temporary build root directory,
    # registers its removal at interpreter exit, caches it in a global and returns its path.
    global _BUILD_ROOT
    if _BUILD_ROOT is None:
        _BUILD_ROOT = _make_temp_dir(prefix="benchmark_utils_jit_build")
        # pyrefly: ignore [missing-argument]
        atexit.register(shutil.rmtree, _BUILD_ROOT)
    return _BUILD_ROOT


# BACK_TESTING_NOTE:
#   There are two workflows where this code could be used. One is the obvious
#   case where someone simply builds or installs PyTorch and uses Timer.
#   The other is that the entire `torch/utils/benchmark` folder from a CURRENT
#   PyTorch checkout is copy-pasted into a much OLDER version of the PyTorch
#   source code. This is what we refer to here as "back testing". The rationale
#   is that we might want to use current tooling to study some aspect of an
#   earlier version of PyTorch. (e.g. a regression.)
#
#   The problem is that Timer relies on several aspects of core PyTorch, namely
#   some binding functions for Valgrind symbols in `torch._C` and the
#   `torch.__config__._cxx_flags()` method. If we were to naively copy code
#   around this wouldn't work as the symbols of interest aren't present in
#   earlier versions of PyTorch. In order to work around this, we must add back
#   testing shims. These shims will never activate during normal use, but will
#   allow Timer to function outside of the "correct" version of PyTorch by
#   emulating functionality that was added later.
#
#   These shims are temporary, and as Timer becomes more integrated with
#   PyTorch the cost and complexity of such shims will increase. Once back
#   testing is no longer required (which is to say we have done enough historic
#   analysis and the shims no longer justify their maintenance and code
#   complexity costs) back testing paths will be removed.

CXX_FLAGS: list[str] | None
if hasattr(torch.__config__, "_cxx_flags"):
    try:
        CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
        if CXX_FLAGS is not None and "-g" not in CXX_FLAGS:
            CXX_FLAGS.append("-g")
        # remove "-W" flags to allow build benchmarks
        # with a relaxed constraint of compiler versions
        if CXX_FLAGS is not None:
            CXX_FLAGS = list(filter(lambda x: not x.startswith("-W"), CXX_FLAGS))

    except RuntimeError:
        # We are in FBCode.
        CXX_FLAGS = None
else:
    # FIXME: Remove when back testing is no longer required.
    CXX_FLAGS = ["-O2", "-fPIC", "-g"]

EXTRA_INCLUDE_PATHS: list[str] = [os.path.join(SOURCE_ROOT, "valgrind_wrapper")]
CONDA_PREFIX = os.getenv("CONDA_PREFIX")
if CONDA_PREFIX is not None:
    # Load will automatically search /usr/include, but not conda include.
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, "include"))


COMPAT_CALLGRIND_BINDINGS: CallgrindModuleType | None = None
def get_compat_bindings() -> CallgrindModuleType:
    # [FUNCTION DOC] Builds (once, under the lock) and returns the compatibility valgrind
    # bindings extension from its bundled C++ source, caching it in a module global. Used when
    # the installed torch lacks native valgrind bindings.
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(
                name="callgrind_bindings",
                sources=[os.path.join(
                    SOURCE_ROOT,
                    "valgrind_wrapper",
                    "compat_bindings.cpp"
                )],
                extra_cflags=CXX_FLAGS,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
            )
    return COMPAT_CALLGRIND_BINDINGS


def _compile_template(
    *,
    stmt: str,
    setup: str,
    global_setup: str,
    src: str,
    is_standalone: bool
) -> Any:
    # [FUNCTION DOC] HELPER: Substitutes global setup, setup and statement text into the
    # marker lines of the template source, writes it under a build directory named from a hash of
    # the source (so identical code reuses the cached build), and builds it with the extension
    # loader. Returns a loaded Python module, or the path of an executable when standalone.
    # Writes files; the lock guards the file creation only.
    for before, after, indentation in (
        ("// GLOBAL_SETUP_TEMPLATE_LOCATION", global_setup, 0),
        ("// SETUP_TEMPLATE_LOCATION", setup, 4),
        ("// STMT_TEMPLATE_LOCATION", stmt, 8)
    ):
        # C++ doesn't care about indentation so this code isn't load
        # bearing the way it is with Python, but this makes the source
        # look nicer if a human has to look at it.
        src = re.sub(
            before,
            textwrap.indent(after, " " * indentation)[indentation:],
            src
        )

    # We want to isolate different Timers. However `cpp_extension` will
    # cache builds which will significantly reduce the cost of repeated
    # invocations.
    with LOCK:
        name = f"timer_cpp_{abs(hash(src))}"
        build_dir = os.path.join(_get_build_root(), name)
        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, "timer_src.cpp")
        with open(src_path, "w") as f:
            f.write(src)

    # `cpp_extension` has its own locking scheme, so we don't need our lock.
    return cpp_extension.load(
        name=name,
        sources=[src_path],
        build_directory=build_dir,
        extra_cflags=CXX_FLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
        is_python_module=not is_standalone,
        is_standalone=is_standalone,
    )


def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    # [FUNCTION DOC] Compiles the timeit C++ template with the given code into a Python module
    # and returns it. Raises an assertion error if the result does not provide timeit.
    template_path: str = os.path.join(SOURCE_ROOT, "timeit_template.cpp")
    with open(template_path) as f:
        src: str = f.read()

    module = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=False)
    if not isinstance(module, TimeitModuleType):
        raise AssertionError("compiled module is not a TimeitModuleType")
    return module


def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str:
    # [FUNCTION DOC] Compiles the callgrind C++ template with the given code into a standalone
    # executable and returns its path. Raises an assertion error if the result is not a string.
    template_path: str = os.path.join(SOURCE_ROOT, "valgrind_wrapper", "timer_callgrind_template.cpp")
    with open(template_path) as f:
        src: str = f.read()

    target = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=True)
    if not isinstance(target, str):
        raise AssertionError("compiled target path is not a string")
    return target
