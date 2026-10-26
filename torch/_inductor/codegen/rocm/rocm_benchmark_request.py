# mypy: allow-untyped-defs
# [MODULE DOC] Benchmark request type for ROCm (CK) template kernels used during autotuning.
# Defines ROCmBenchmarkRequest, which compiles/loads a shared library and builds a callable that runs it.
from __future__ import annotations

import functools
import logging
from ctypes import byref, c_int, c_size_t, c_void_p
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codecache import DLLWrapper, ROCmCodeCache


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


log = logging.getLogger(__name__)


class ROCmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    # [CLASS DOC] Serializable BenchmarkRequest for a ROCm kernel built from generated C++ source. Holds the
    # source code, cache hash key and source file, a lazily loaded DLL wrapper, and the workspace size and
    # tensor. The workspace size is queried once by calling the kernel in size-query mode. Must not hold
    # device tensors in its constructor state because it crosses process boundaries.
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put CUDA Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        # [FUNCTION DOC] Stores the source and initial state (no DLL, zero workspace) and writes the source to the ROCm
        # code cache as a shared library, recording the resulting hash key and source file path.
        # Side effect: writes a file to the cache. Args mirror BenchmarkRequest plus source_code.
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: torch.Tensor | None = None
        self.DLL: DLLWrapper | None = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = ROCmCodeCache.write(self.source_code, "so")

    def precompile(self):
        # [FUNCTION DOC] Compiles the source into the code cache ahead of time (shared library, plus a standalone
        # executable if config.rocm.generate_test_runner is set). Side effect: compilation and logging.
        # Prepopulate code cache
        # may happen in separate Threadpool
        log.debug("Precompiling %s", self)
        ROCmCodeCache.compile(self.source_code, "so")
        if config.rocm.generate_test_runner:
            ROCmCodeCache.compile(self.source_code, "exe")
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # [FUNCTION DOC] Loads the library, ensures the workspace size is known, and returns a zero-argument callable that
        # runs the kernel with the given input tensors and output tensor. Allocates and zero-fills a workspace
        # tensor on the output device when needed. Pointer args come from tensor data pointers, size args from
        # extra_args as ints, followed by a null workspace size pointer, workspace pointer and the current
        # stream. Side effects: loads DLL, stores self.workspace.
        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [out]]
        size_args = [c_int(arg) for arg in self.extra_args]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=out.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *size_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

    def update_workspace_size(self) -> None:
        # [FUNCTION DOC] Queries the kernel's required workspace size once (cached by a flag): calls the kernel with null
        # data pointers and a size output pointer, synchronizes CUDA to surface errors, and stores the size
        # in self.workspace_size. Number of pointers is distinct input names plus one output.
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len(
            dict.fromkeys(meta.name for meta in self.input_tensor_meta)
        )
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        size_args = [c_int(arg) for arg in self.extra_args]
        run_method(
            *args,  # input ptrs and output ptrs
            *size_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        torch.cuda.synchronize()  # shake out any CUDA errors
        self.workspace_size = c_workspace_size.value
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        # [FUNCTION DOC] Lazily loads the compiled shared library from the code cache, setting DLL, hash_key and
        # source_file if not yet loaded.
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = ROCmCodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        # [FUNCTION DOC] Closes the loaded DLL if any and releases the workspace tensor.
        if self.DLL is not None:
            self.DLL.close()
        self.workspace = None

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns a short debug string with kernel name, source file and hash key.
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"
