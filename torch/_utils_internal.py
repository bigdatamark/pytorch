# mypy: allow-untyped-defs
# [MODULE DOC] Central, overridable hooks for behavior that differs between OSS and the FB build
# (file paths, logging, JustKnobs killswitches, profiling, GPU clock lookup, a deprecation decorator,
# and constants such as test master address/port and dlopen flags). OSS versions are mostly no-ops.
# Import-time side effect: if the TORCH_COMPILE_STROBELIGHT environment variable is set and the
# strobeclient binary exists, enables the Strobelight compile-time profiler; also computes torch_parent.
import functools
import logging
import os
import sys
import tempfile
import typing_extensions
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


_T = TypeVar("_T")
_P = ParamSpec("_P")

log = logging.getLogger(__name__)

if os.environ.get("TORCH_COMPILE_STROBELIGHT", False):
    import shutil

    if not shutil.which("strobeclient"):
        log.info(
            "TORCH_COMPILE_STROBELIGHT is true, but seems like you are not on a FB machine."
        )
    else:
        log.info("Strobelight profiler is enabled via environment variable")
        StrobelightCompileTimeProfiler.enable()

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overridable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.


def _compute_torch_parent() -> str:
    # [FUNCTION DOC] HELPER: Returns the directory that contains the torch package directory. If this file sits in a 'shared' directory, returns two levels up from it. If the package directory has no bin subdirectory, tries the installed torch distribution location and uses its parent when that has a bin subdirectory (editable-install case); any failure there is swallowed. Otherwise returns the parent of the package directory.
    torch_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(torch_dir) == "shared":
        return os.path.dirname(os.path.dirname(torch_dir))
    # In scikit-build-core editable installs with redirect mode, binary
    # artifacts (bin/, lib/) are installed to the dist package directory
    # rather than the source tree. Fall back to the installed package
    # location for get_file_path.
    if not os.path.isdir(os.path.join(torch_dir, "bin")):
        try:
            from importlib.metadata import distribution

            installed = str(distribution("torch").locate_file("torch"))
            if os.path.isdir(os.path.join(installed, "bin")):
                return os.path.dirname(installed)
        except Exception:
            pass
    return os.path.dirname(torch_dir)


torch_parent = _compute_torch_parent()


def get_file_path(*path_components: str) -> str:
    # [FUNCTION DOC] Joins the given path components onto the torch parent directory and returns the string path (overridable hook for FB builds).
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components: str) -> str:
    # [FUNCTION DOC] Joins the components with no base directory and returns the path (overridable hook).
    return os.path.join(*path_components)


def get_writable_path(path: str) -> str:
    # [FUNCTION DOC] Returns the path if it is writable; otherwise creates and returns a new temporary directory whose name ends with the path's basename (side effect: creates a directory).
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))


def prepare_multiprocessing_environment(path: str) -> None:
    # [FUNCTION DOC] No-op hook in OSS; FB builds may prepare the multiprocessing environment for the given path.
    pass


def resolve_library_path(path: str) -> str:
    # [FUNCTION DOC] Returns the canonical real path (symlinks resolved) of the given library path.
    return os.path.realpath(path)


def throw_abstract_impl_not_imported_error(opname, module, context):
    # [FUNCTION DOC] Always raises NotImplementedError saying the fake impl for operator opname was not found. If the named module is already imported the message is short; otherwise it also tells the user to import that module and appends the context text.
    if module in sys.modules:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
        )
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the fake impl. {context}"
        )


# NB!  This treats "skip" kwarg specially!!
def compile_time_strobelight_meta(
    phase_name: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory: returns a decorator that wraps a function so that, when the Strobelight compile-time profiler is enabled, the call is profiled under phase_name.
    # NB: bumps an integer keyword argument named skip by one so stack-skip counts stay right through the wrapper.
    def compile_time_strobelight_meta_inner(
        function: Callable[_P, _T],
    ) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Decorator produced by the factory; returns the wrapped version of the given function.
        @functools.wraps(function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # [FUNCTION DOC] HELPER: Increments an integer skip keyword argument if present, then calls the function directly when profiling is disabled, otherwise calls it through the Strobelight profiler. Returns the function's result.
            if "skip" in kwargs and isinstance(
                skip := kwargs["skip"],
                int,
            ):
                kwargs["skip"] = skip + 1

            # This is not needed but we have it here to avoid having profile_compile_time
            # in stack traces when profiling is not enabled.
            if not StrobelightCompileTimeProfiler.enabled:
                return function(*args, **kwargs)

            return StrobelightCompileTimeProfiler.profile_compile_time(
                function, phase_name, *args, **kwargs
            )

        return wrapper_function

    return compile_time_strobelight_meta_inner


# Meta only, see
# https://www.internalfb.com/intern/wiki/ML_Workflow_Observability/User_Guides/Adding_instrumentation_to_your_code/
#
# This will cause an event to get logged to Scuba via the signposts API.  You
# can view samples on the API at https://fburl.com/scuba/workflow_signpost/zh9wmpqs
# we log to subsystem "torch", and the category and name you provide here.
# Each of the arguments translate into a Scuba column.  We're still figuring
# out local conventions in PyTorch, but category should be something like
# "dynamo" or "inductor", and name should be a specific string describing what
# kind of event happened.
#
# Killswitch is at
# https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fsignpost#event
def signpost_event(category: str, name: str, parameters: dict[str, Any]):
    # [FUNCTION DOC] OSS version: logs category, name and parameters at info level (FB version sends a Scuba signpost event). Returns nothing.
    log.info("%s %s: %r", category, name, parameters)


def add_mlhub_insight(category: str, insight: str, insight_description: str):
    # [FUNCTION DOC] No-op hook in OSS (FB records an MLHub insight).
    pass


def log_compilation_event(metrics):
    # [FUNCTION DOC] OSS version: logs the metrics object at info level.
    log.info("%s", metrics)


def upload_graph(graph):
    # [FUNCTION DOC] No-op hook in OSS (FB uploads the graph).
    pass


def set_pytorch_distributed_envs_from_justknobs():
    # [FUNCTION DOC] No-op hook in OSS (FB sets distributed environment variables from JustKnobs).
    pass


def log_export_usage(**kwargs):
    # [FUNCTION DOC] No-op hook in OSS; accepts and ignores keyword arguments.
    pass


def log_draft_export_usage(**kwargs):
    # [FUNCTION DOC] No-op hook in OSS; accepts and ignores keyword arguments.
    pass


def log_trace_structured_event(*args, **kwargs) -> None:
    # [FUNCTION DOC] No-op hook in OSS; ignores all arguments.
    pass


def log_cache_bypass(*args, **kwargs) -> None:
    # [FUNCTION DOC] No-op hook in OSS; ignores all arguments.
    pass


def log_torchscript_usage(api: str, **kwargs):
    # [FUNCTION DOC] No-op hook in OSS; ignores the api name and keyword arguments.
    _ = api
    return


def check_if_torch_exportable():
    # [FUNCTION DOC] Returns false in OSS (FB checks whether torch is exportable).
    return False


def export_training_ir_rollout_check() -> bool:
    # [FUNCTION DOC] Returns true in OSS: the training IR rollout is always considered enabled.
    return True


def full_aoti_runtime_assert() -> bool:
    # [FUNCTION DOC] Returns true in OSS: full AOTI runtime asserts are always on.
    return True


def log_torch_jit_trace_exportability(
    api: str,
    type_of_export: str,
    export_outcome: str,
    result: str,
):
    # [FUNCTION DOC] No-op hook in OSS; ignores all four arguments (api, export type, outcome, result).
    _, _, _, _ = api, type_of_export, export_outcome, result
    return


DISABLE_JUSTKNOBS = True


def justknobs_check(name: str, default: bool = True) -> bool:
    # [FUNCTION DOC] Killswitch query. In OSS always returns the given default (true unless overridden); name is ignored. Do not call at import time (JustKnobs is not fork safe in FB).
    """
    This function can be used to killswitch functionality in FB prod,
    where you can toggle this value to False in JK without having to
    do a code push.  In OSS, we always have everything turned on all
    the time, because downstream users can simply choose to not update
    PyTorch.  (If more fine-grained enable/disable is needed, we could
    potentially have a map we lookup name in to toggle behavior.  But
    the point is that it's all tied to source code in OSS, since there's
    no live server to query.)

    This is the bare minimum functionality I needed to do some killswitches.
    We have a more detailed plan at
    https://docs.google.com/document/d/1Ukerh9_42SeGh89J-tGtecpHBPwGlkQ043pddkKb3PU/edit
    In particular, in some circumstances it may be necessary to read in
    a knob once at process start, and then use it consistently for the
    rest of the process.  Future functionality will codify these patterns
    into a better high level API.

    WARNING: Do NOT call this function at module import time, JK is not
    fork safe and you will break anyone who forks the process and then
    hits JK again.
    """
    return default


def justknobs_getval_int(name: str) -> int:
    # [FUNCTION DOC] Returns an integer knob value; in OSS always 0 and name is ignored.
    """
    Read warning on justknobs_check
    """
    return 0


def is_fb_unit_test() -> bool:
    # [FUNCTION DOC] Returns false in OSS (FB detects FB unit-test environment).
    return False


@functools.cache
def max_clock_rate():
    # [FUNCTION DOC] Cached. Returns the maximum SM clock rate in MHz of the current GPU. On CUDA asks triton's nvidia-smi helper, falling back on FileNotFoundError to pynvml via the torch handler (shutting NVML down afterwards). On ROCm returns a hard-coded value by GPU architecture name of device 0 (default 1100); the gfx95 value is a placeholder. Needs a GPU and triton.
    """
    unit: MHz
    """
    if not torch.version.hip:
        from triton.testing import nvsmi

        try:
            return nvsmi(["clocks.max.sm"])[0]
        except FileNotFoundError:
            import pynvml  # type: ignore[import]

            handle = torch.cuda._get_pynvml_handler()
            try:
                return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            finally:
                pynvml.nvmlShutdown()
    else:
        # Manually set max-clock speeds on ROCm until equivalent nvmsi
        # functionality in triton.testing or via pyamdsmi enablement. Required
        # for test_snode_runtime unit tests.
        gcn_arch = str(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0])
        if "gfx94" in gcn_arch:
            return 1700
        elif "gfx90a" in gcn_arch:
            return 1700
        elif "gfx908" in gcn_arch:
            return 1502
        elif "gfx12" in gcn_arch:
            return 1700
        elif "gfx11" in gcn_arch:
            return 1700
        elif "gfx103" in gcn_arch:
            return 1967
        elif "gfx101" in gcn_arch:
            return 1144
        elif "gfx95" in gcn_arch:
            return 1700  # TODO: placeholder, get actual value
        else:
            return 1100


def get_mast_job_name_version() -> tuple[str, int] | None:
    # [FUNCTION DOC] Returns None in OSS; FB returns the MAST job name and version pair.
    return None


TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
# If an op was defined in C++ and extended from Python using the
# torch.library.register_fake, returns if we require that there be a
# m.set_python_module("mylib.ops") call from C++ that associates
# the C++ op with a python module.
REQUIRES_SET_PYTHON_MODULE = False


def maybe_upload_prof_stats_to_manifold(profile_path: str) -> str | None:
    # [FUNCTION DOC] OSS version: prints a notice and returns None; FB uploads profile stats and returns a location.
    print("Uploading profile stats (fb-only otherwise no-op)")
    return None


def log_chromium_event_internal(
    event: dict[str, Any],
    stack: list[str],
    logger_uuid: str,
    start_time_ns: int,
):
    # [FUNCTION DOC] No-op hook in OSS; ignores the event, stack, logger id and start time and returns None.
    return None


def record_chromium_event_internal(
    event: dict[str, Any],
):
    # [FUNCTION DOC] No-op hook in OSS; ignores the event and returns None.
    return None


def profiler_allow_cudagraph_cupti_lazy_reinit_cuda12():
    # [FUNCTION DOC] Returns true in OSS: allows lazy CUPTI reinitialization with CUDA graphs on CUDA 12.
    return True


def deprecated():
    # [FUNCTION DOC] Decorator factory for deprecating an internal underscore-prefixed function. The returned decorator checks the name starts with an underscore (else ValueError), refuses to run if the module already has the public name (RuntimeError), creates a public alias without the underscore wrapped with a deprecation UserWarning, fixes its name and qualified name, installs it into the function's module, and returns the original function unchanged.
    """
    When we deprecate a function that might still be in use, we make it internal
    by adding a leading underscore. This decorator is used with a private function,
    and creates a public alias without the leading underscore, but has a deprecation
    warning. This tells users "THIS FUNCTION IS DEPRECATED, please use something else"
    without breaking them, however, if they still really really want to use the
    deprecated function without the warning, they can do so by using the internal
    function name.
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: The decorator described by the factory; side effect is setting the public alias attribute on the function's module.
        # Validate naming convention - single leading underscore, not dunder
        if not (func.__name__.startswith("_")):
            raise ValueError(
                "@deprecate must decorate a function whose name "
                "starts with a single leading underscore (e.g. '_foo') as the api should be considered internal for deprecation."
            )

        public_name = func.__name__[1:]  # drop exactly one leading underscore
        module = sys.modules[func.__module__]

        # Don't clobber an existing symbol accidentally.
        if hasattr(module, public_name):
            raise RuntimeError(
                f"Cannot create alias '{public_name}' -> symbol already exists in {module.__name__}. \
                 Please rename it or consult a pytorch developer on what to do"
            )

        warning_msg = f"{func.__name__[1:]} is DEPRECATED, please consider using an alternative API(s). "

        # public deprecated alias
        alias = typing_extensions.deprecated(
            # pyrefly: ignore [bad-argument-type]
            warning_msg,
            category=UserWarning,
            stacklevel=1,
        )(func)

        alias.__name__ = public_name

        # Adjust qualname if nested inside a class or another function
        if "." in func.__qualname__:
            alias.__qualname__ = func.__qualname__.rsplit(".", 1)[0] + "." + public_name
        else:
            alias.__qualname__ = public_name

        setattr(module, public_name, alias)

        return func

    return decorator


def get_default_numa_options():
    # [FUNCTION DOC] Returns None in OSS meaning no NUMA binding by default (FB returns NUMA options).
    """
    When using elastic agent, if no numa options are provided, we will use these
    as the default.

    For external use cases, we return None, i.e. no numa binding. If you would like
    to use torch's automatic numa binding capabilities, you should provide
    NumaOptions to your launch config directly or use the numa binding option
    available in torchrun.

    Must return None or NumaOptions, but not specifying to avoid circular import.
    """
    return None


def log_triton_builds(fail: str | None):
    # [FUNCTION DOC] No-op hook in OSS; ignores the failure text.
    pass


def find_compile_subproc_binary() -> str | None:
    # [FUNCTION DOC] Returns None in OSS, meaning no override of the compile subprocess binary.
    """
    Allows overriding the binary used for subprocesses
    """
    return None


def get_torch_source_version() -> str:
    # [FUNCTION DOC] Returns the git commit hash recorded in torch.version, or the empty string if not available.
    """Return the source commit hash for the current PyTorch build."""
    import torch.version as torch_version

    return getattr(torch_version, "git_version", "")
