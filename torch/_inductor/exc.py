# [MODULE DOC] Exception types raised by Inductor: lowering/operator issues, compiler errors
# (C++/CUDA/XPU), Triton availability errors and the top-level InductorError wrapper.
# Import-time side effect: if env var TORCHINDUCTOR_WRITE_MISSING_OPS is 1, _record_missing_op
# is a cached function appending op names to missing_ops.txt in the temp dir; otherwise a no-op.
from __future__ import annotations

import os
import tempfile
import textwrap
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from torch._dynamo.exc import BackendCompilerFailed, ShortenTraceback


if TYPE_CHECKING:
    import types

    from torch.cuda import _CudaDeviceProperties

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":

    @lru_cache(None)
    def _record_missing_op(target: Any) -> None:
        # [FUNCTION DOC] HELPER: Appends the string form of an unsupported op target to missing_ops.txt
        # in the system temp directory. Cached so each distinct target is written once. Side effect: file I/O.
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            fd.write(str(target) + "\n")

else:

    def _record_missing_op(target: Any) -> None:  # type: ignore[misc]
        # [FUNCTION DOC] HELPER: No-op replacement used when missing-op recording is disabled.
        pass


class OperatorIssue(RuntimeError):
    # [CLASS DOC] Base RuntimeError for problems with a specific operator during lowering;
    # provides a helper that formats an operator target with its args and kwargs.
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str:
        # [FUNCTION DOC] Static. Builds an indented multi-line string listing the target, each positional
        # argument by index, and the keyword arguments (only if any). Returns the string.
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    # [CLASS DOC] Raised when an operator has neither a lowering nor a decomposition.
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Records the target as missing (if enabled) and raises-ready error with a
        # 'missing lowering' message including the formatted target, args and kwargs.
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    # [CLASS DOC] Raised when an operator lacks a lowering but a decomposition exists in the
    # global decomposition table; the message tells the user to register it in Inductor decompositions.
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] Records the target as missing (if enabled) and builds a 'missing decomposition'
        # message with the operator details and instructions to add it to the decompositions list.
        _record_missing_op(target)
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""

                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decomposition
                """
            )
        )


class LoweringException(OperatorIssue):
    # [CLASS DOC] Wraps an exception that occurred while lowering an operator, carrying the
    # original error type/message and the operator description.
    def __init__(
        self,
        exc: Exception,
        target: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        stack_trace: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a message from the wrapped exception's type name and text plus the
        # formatted operator. If a stack_trace is given it is appended.
        # KNOWN ISSUE: the append step adds the whole message again (self-concatenation), so the message
        # is duplicated when a stack trace is supplied.
        msg = f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        if stack_trace:
            msg += f"{msg}\nFound from : \n {stack_trace}"
        super().__init__(msg)


class SubgraphLoweringException(RuntimeError):
    # [CLASS DOC] Error raised when lowering a subgraph fails; carries no extra state.
    pass


class InvalidCxxCompiler(RuntimeError):
    # [CLASS DOC] Raised when no usable C++ compiler is available.
    def __init__(self, compiler: str | None = None) -> None:
        # [FUNCTION DOC] If compiler is None, message reports the configured cpp.cxx setting as having no
        # working compiler; otherwise reports that the named compiler was not found.
        from . import config

        if compiler is None:
            msg = (
                f"No working C++ compiler found in {config.__name__}.cpp.cxx: "
                f"{config.cpp.cxx}"
            )
        else:
            msg = f"Compiler: {compiler} is not found."
        super().__init__(msg)


class CppWrapperCodegenError(RuntimeError):
    # [CLASS DOC] Error for failures during C++ wrapper code generation; message gets a fixed prefix.
    def __init__(self, msg: str) -> None:
        # [FUNCTION DOC] Prefixes the given message with a C++ wrapper codegen error label.
        super().__init__(f"C++ wrapper codegen error: {msg}")


class CppCompileError(RuntimeError):
    # [CLASS DOC] Error for a failed C++ compilation. Stores cmd (argument list) and output (text)
    # and renders both in the message.
    def __init__(self, cmd: list[str], output: str) -> None:
        # [FUNCTION DOC] Decodes output from bytes if needed, stores cmd and output, and builds a message
        # containing the space-joined command and the compiler output.
        if isinstance(output, bytes):
            output = output.decode("utf-8")

        self.cmd = cmd
        self.output = output

        super().__init__(
            textwrap.dedent(
                """
                    C++ compile error

                    Command:
                    {cmd}

                    Output:
                    {output}
                """
            )
            .strip()
            .format(cmd=" ".join(cmd), output=output)
        )

    def __reduce__(self) -> tuple[type, tuple[list[str], str]]:
        # [FUNCTION DOC] Pickle support: reconstructs the exception from its class, cmd and output.
        return (self.__class__, (self.cmd, self.output))


class CUDACompileError(CppCompileError):
    # [CLASS DOC] CppCompileError subtype for CUDA compilation failures.
    pass


class XPUCompileError(CppCompileError):
    # [CLASS DOC] CppCompileError subtype for XPU compilation failures.
    pass


class TritonMissing(ShortenTraceback):
    # [CLASS DOC] Error (with shortened traceback) raised when Triton is missing or too old.
    def __init__(self, first_useful_frame: types.FrameType | None) -> None:
        # [FUNCTION DOC] Initializes with a fixed message about the missing Triton install and the
        # first useful stack frame used to trim the traceback.
        super().__init__(
            "Cannot find a working triton installation. "
            "Either the package is not installed or it is too old. "
            "More information on installing Triton can be found at: https://github.com/triton-lang/triton",
            first_useful_frame=first_useful_frame,
        )


class GPUTooOldForTriton(ShortenTraceback):
    # [CLASS DOC] Error (with shortened traceback) raised when the GPU's CUDA capability is below 7.0.
    def __init__(
        self,
        # pyrefly: ignore [not-a-type]
        device_props: _CudaDeviceProperties,
        first_useful_frame: types.FrameType | None,
    ) -> None:
        # [FUNCTION DOC] Builds a message from the device name and its compute capability major/minor,
        # passing the first useful frame for traceback shortening.
        super().__init__(
            f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, "
            "which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, "
            f"but your device is of CUDA capability {device_props.major}.{device_props.minor}",
            first_useful_frame=first_useful_frame,
        )


class InductorError(BackendCompilerFailed):
    # [CLASS DOC] Backend-compiler failure for the inductor backend. Stores the inner exception
    # and presents its type name and text as the message.
    backend_name = "inductor"

    def __init__(
        self,
        inner_exception: Exception,
        first_useful_frame: types.FrameType | None,
    ) -> None:
        # [FUNCTION DOC] Stores inner_exception and calls the ShortenTraceback initializer directly,
        # skipping the BackendCompilerFailed initializer, with the inner error rendered as the message.
        self.inner_exception = inner_exception
        ShortenTraceback.__init__(
            self,
            f"{type(inner_exception).__name__}: {inner_exception}",
            first_useful_frame=first_useful_frame,
        )
