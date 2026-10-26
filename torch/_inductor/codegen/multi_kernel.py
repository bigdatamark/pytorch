# [MODULE DOC] Multi-kernel support for Inductor codegen: MultiKernelState/MultiKernel/SizeHintMultiKernel generate wrapper definitions that group alternative sub-kernels at compile time;
# MultiKernelCall/SizeHintMultiKernelCall are the run-time classes that benchmark or shape-select the best sub-kernel, with an on-disk pick cache and cpp-wrapper choice recording.
# No import-time side effects besides creating the module logger.
# mypy: allow-untyped-defs
import functools
import logging
import math
import os
import pathlib
from typing import Any

from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..codecache import code_hash, CodeCacheFuture, get_path, write_atomic
from ..runtime.benchmarking import benchmarker
from ..utils import cache_on_self, IndentedBuffer
from ..virtualized import V
from .common import TensorArg, WorkspaceArg


log = logging.getLogger(__name__)


class MultiKernelState:
    # [CLASS DOC] Compile-time registry of multi-kernel definitions, referenced by the wrapper codegen (wrapper_code.multi_kernel_state). State: a mapping from tuple of sub-kernel
    # names to the multi-kernel name (prevents duplicate definitions) and an IndentedBuffer accumulating the generated definition code. Key method: define_kernel.
    """
    Maintain state of multi-kernel compilation so we don't define duplicated
    multi-kernel for the same set of sub-kernels.

    V.graph.wrapper_code has a reference to MultiKernelState instance.
    """

    def __init__(self):
        # [FUNCTION DOC] Creates the empty name mapping and the empty definition code buffer.
        self.subkernel_to_kernel_name = {}
        self.kernel_defs = IndentedBuffer()

    def define_kernel(
        self,
        kernels: list[Any],
        kernel_shape_keys: list[None | tuple[tuple[int, ...], ...]] | None = None,
    ) -> str:
        # [FUNCTION DOC] Registers a multi-kernel for the given sub-kernels and returns its name (multi_kernel_N, numbered by registration count); returns the existing name if the same
        # sub-kernel names were registered before. Unless in the second cpp-wrapper pass (returns early with no code), writes into kernel_defs an arg_index mapping (per kernel, slices of the call
        # arguments; template multi-buffers get extra per-kernel grid argument slices, otherwise numel args are added to the call args of the first kernel), then an async_compile call:
        # a plain multi_kernel over the list of kernel names, or, when kernel_shape_keys is given, a size-hint multi-kernel keyed by shape (requires Triton template kernels and equal lengths).
        # With autotune_at_compile_time, also records the name in the wrapper's source-to-kernel map. Raises AssertionError on inconsistent size-hint arguments.
        """
        Previously we name the multi kernel as "multi_kernel_{kernel_names[0]}".
        This has some minor issue.

        E.g. for persistent reduction https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca ,
        there are 2 flavors of non-persistent reduction:
          https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4
        and
          https://gist.github.com/shunting314/02ee753b65c513c54e695626afe682bd

        The only different is cache eviction policy.

        We should name the multi-kernel differently in these 2 cases.

        kernels:
            A list of kernels
        kernel_shape_keys:
            Specified for size-hint multi-kernels.
            Each list element is a shape key, corresponding to the concrete input & output size hints each kernel was tuned for.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        kernel_names = tuple(k.kernel_name for k in kernels)
        if kernel_names in self.subkernel_to_kernel_name:
            return self.subkernel_to_kernel_name[kernel_names]

        # name the multi kernel based on the first kernel
        multi_kernel_name = f"multi_kernel_{len(self.subkernel_to_kernel_name)}"
        self.subkernel_to_kernel_name[kernel_names] = multi_kernel_name

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # we should not generate any python code for multi-kernel during
            # the second pass of cpp-wrapper.
            return multi_kernel_name

        arg_index: dict[int, list[slice]] = {}
        _, call_args, _, arg_types = kernels[0].args.python_argdefs()
        if isinstance(kernels[0], TritonTemplateKernel) and isinstance(
            kernels[0].output_node, MultiTemplateBuffer
        ):
            for i, kernel in enumerate(kernels):
                additional_call_args, _ = kernel.additional_call_args_and_types()
                if i not in arg_index:
                    arg_index[i] = []
                arg_index[i].append(slice(0, len(call_args)))
                arg_index[i].append(
                    slice(
                        len(call_args) + i * len(additional_call_args),
                        len(call_args) + (i + 1) * len(additional_call_args),
                    )
                )
        else:
            kernels[0].add_numel_to_call_args(multi_kernel_name, call_args, arg_types)
            for i in range(len(kernels)):
                arg_index[i] = [slice(0, len(call_args))]

        keyed_by_sizes = kernel_shape_keys is not None
        buf = self.kernel_defs
        buf.writeline("")
        buf.writeline("arg_index = {")
        for key, slice_list in arg_index.items():
            slice_reprs = ", ".join(repr(s) for s in slice_list)
            buf.writeline(f"    {key}: [{slice_reprs}],")
        buf.writeline("}")

        if not keyed_by_sizes:  # no size hint keys, just call with list of kernels
            buf.writeline(
                f"{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, ["
            )
            with buf.indent():
                for name in kernel_names:
                    buf.writeline(f"{name},")
            buf.writeline("], arg_index=arg_index)")
        else:  # call with dict[size hint key, kernel]
            if not isinstance(kernels[0], TritonTemplateKernel):
                raise AssertionError("expected kernels[0] to be a TritonTemplateKernel")
            if not isinstance(kernel_shape_keys, list):
                raise AssertionError("expected kernel_shape_keys to be a list")
            if len(kernels) != len(kernel_shape_keys):
                raise AssertionError(
                    f"expected len(kernels) == len(kernel_shape_keys), "
                    f"got {len(kernels)} != {len(kernel_shape_keys)}"
                )
            buf.writeline(
                f"{multi_kernel_name} = async_compile.size_hint_multi_kernel({multi_kernel_name!r}, {{"
            )
            with buf.indent():
                for shape_key, name in zip(kernel_shape_keys, kernel_names):
                    buf.writeline(f"{shape_key}: {name},")
            buf.writeline("}, arg_index=arg_index)")

        if config.triton.autotune_at_compile_time:
            V.graph.wrapper_code.src_to_kernel["\n".join(kernel_names)] = (
                multi_kernel_name
            )

        return multi_kernel_name


class MultiKernel:
    # [CLASS DOC] Compile-time handle for a multi-kernel wrapping two or more equivalent sub-kernels (alternatives to be benchmarked at runtime). Attributes: kernels, kernel_name (assigned by
    # MultiKernelState), and a placeholder args object so that code checking for an args attribute treats it as a non-null kernel. Key methods: call_kernel, workspace merging, nan checks,
    # and properties that combine the sub-kernels' removed/inplace buffers.
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall(
        [kernel1, kernel2], multi_kernel_definition_code
    )
    ```

    Here is a concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
    """

    def __init__(self, kernels):
        # [FUNCTION DOC] Stores the sub-kernels (raises AssertionError if fewer than 2) and defines the multi-kernel through the wrapper's multi-kernel state to get its name. Sets a dummy args object.
        if len(kernels) < 2:
            raise AssertionError(f"expected at least 2 kernels, got {len(kernels)}")

        self.kernels = kernels
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()

    @staticmethod
    def _merge_workspace_args(left: list[WorkspaceArg], right: list[WorkspaceArg]):
        # [FUNCTION DOC] HELPER: Merges two lists of workspace args by inner name, keeping the larger (maximum) workspace for names present in both; returns left unchanged if the lists are equal.
        if left == right:
            return left
        result = {x.inner_name: x for x in left}
        for arg in right:
            if arg.inner_name in result:
                result[arg.inner_name] = WorkspaceArg.maximum(
                    result[arg.inner_name], arg
                )
            else:
                result[arg.inner_name] = arg
        return [*result.values()]

    @staticmethod
    def merge_workspaces_inplace(kernels):
        # [FUNCTION DOC] Makes all kernels share one merged workspace argument list: reduces the per-kernel workspace lists with the merge helper and assigns the result to every kernel's args. Returns the merged list, or None when given fewer than 2 kernels.
        if len(kernels) < 2:
            return
        # All kernels must share the same workspace
        workspace_args = functools.reduce(
            MultiKernel._merge_workspace_args,
            [kernel.args.workspace_args for kernel in kernels],
        )
        for kernel in kernels:
            kernel.args.workspace_args = workspace_args
        return workspace_args

    def call_kernel(self, kernel_name):
        # [FUNCTION DOC] Emits the wrapper code that launches the multi-kernel. Checks the sub-kernels have identical call args and arg types (AssertionError otherwise) and the given name matches. Builds the argument list
        # (for template multi-buffers the per-kernel grid args are appended; otherwise numel args from the first kernel), allocates and frees workspaces around the call, and in the cpp wrapper
        # calls the single already-chosen kernel directly (found via lookup_choice in the second pass).
        # KNOWN ISSUE: for template multi-buffers the extension of the call args list mutates the list obtained from the first kernel's argdefs in place (aliased as multi_call_args).
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        if kernel_name != self.kernel_name:
            raise AssertionError(
                f"expected kernel_name == self.kernel_name, "
                f"got {kernel_name} != {self.kernel_name}"
            )
        V.graph.wrapper_code.write_triton_header_once()
        _, call_args, _, arg_types = self.kernels[0].args.python_argdefs()
        for kernel in self.kernels[1:]:
            _, other_call_args, _, other_arg_types = kernel.args.python_argdefs()
            if call_args != other_call_args:
                raise AssertionError((call_args, other_call_args))
            if arg_types != other_arg_types:
                raise AssertionError(
                    f"expected arg_types == other_arg_types, "
                    f"got {arg_types} != {other_arg_types}"
                )

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # for the second pass of cpp-wrapper codegen, we should call
            # the fast kernel directly
            kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        if isinstance(self.kernels[0], TritonTemplateKernel) and isinstance(
            self.kernels[0].output_node, MultiTemplateBuffer
        ):
            # For matmuls the grid arguments are passed in as additional arguments
            # to the kernel run method. These grids change based on the various
            # parameters of the matmul. So we need to pass each kernel's grid into
            # the multi call kernel.
            multi_call_args = call_args
            multi_call_arg_types = arg_types
            for kernel in self.kernels:
                additional_call_args, additional_arg_types = (
                    kernel.additional_call_args_and_types()
                )
                multi_call_args.extend(list(additional_call_args))
                multi_call_arg_types.extend(list(additional_arg_types))
        else:
            # numels for all subkernels should be the same. Use kernels[0] here
            self.kernels[0].add_numel_to_call_args(kernel_name, call_args, arg_types)
            multi_call_args = call_args
            multi_call_arg_types = arg_types

        for ws in self.kernels[0].args.workspace_args:
            V.graph.wrapper_code.generate_workspace_allocation(ws)

        if V.graph.cpp_wrapper:
            # We have already selected the best kernel at compile time
            # so we only have one set of call args. NB: this currently
            # doesn't work with MultiTemplateBuffer kernels. @bobrenjc93
            # will add it in a subsequent PR.
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, call_args, arg_types=arg_types
            )
        else:
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, multi_call_args, arg_types=multi_call_arg_types
            )

        for ws in reversed(self.kernels[0].args.workspace_args):
            V.graph.wrapper_code.generate_workspace_deallocation(ws)

    def codegen_nan_check(self):
        # [FUNCTION DOC] Emits wrapper assertions that every distinct tensor argument of every sub-kernel contains no NaN or inf values.
        wrapper = V.graph.wrapper_code
        seen: OrderedSet[str] = OrderedSet()
        for k in self.kernels:
            _, call_args, precompile_args, _ = k.args.python_argdefs()
            for arg, precompile_arg in zip(call_args, precompile_args):
                if arg in seen:
                    continue
                seen.add(arg)
                if isinstance(precompile_arg, TensorArg):
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    @property
    def removed_buffers(self):
        # [FUNCTION DOC] Property: the set of buffers removed by all sub-kernels (intersection).
        return OrderedSet.intersection(*[k.removed_buffers for k in self.kernels])

    @property
    def inplaced_to_remove(self):
        # [FUNCTION DOC] Property: the set of inplaced buffers to remove for all sub-kernels (intersection).
        return OrderedSet.intersection(*[k.inplaced_to_remove for k in self.kernels])

    @property
    @cache_on_self
    def inplace_update_buffers(self):
        # [FUNCTION DOC] Cached property: the inplace-update mapping shared by all sub-kernels. Raises AssertionError if the kernels disagree.
        """
        Make sure all kernels have the same inplace update mappings.
        """
        for k in self.kernels[1:]:
            if k.inplace_update_buffers != self.kernels[0].inplace_update_buffers:
                raise AssertionError(
                    "expected all kernels to share the same inplace_update_buffers"
                )
        return self.kernels[0].inplace_update_buffers

    def warn_mix_layout(self, kernel_name: str):
        # [FUNCTION DOC] No-op hook kept for interface compatibility.
        pass


class MultiKernelCall:
    # [CLASS DOC] Run-time object for a multi-kernel: holds the sub-kernels (possibly still compile futures), the arg_index slices that select each kernel's arguments, the picked kernel index
    # (None until chosen), a flag for whether the choice was recorded, and whether the on-disk pick cache is disabled. On first run it benchmarks all sub-kernels, picks the fastest, persists the pick,
    # and afterwards forwards calls to the picked kernel.
    """
    This class is called at run time to actually run the kernel
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        # [FUNCTION DOC] Stores kernels, name and arg_index (raises AssertionError if no kernels). The cache is disabled by an environment variable or the persistent reduction metric table. If config forces a sub-kernel (multi_kernel above 1) it is picked by index (AssertionError if out of range); else the pick is loaded from the cache if enabled.
        if len(kernels) < 1:
            raise AssertionError(f"expected at least 1 kernel, got {len(kernels)}")
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

        self.disable_cache = os.environ.get(
            "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"
        ) == "1" or is_metric_table_enabled("persistent_red_perf")

        self.picked_kernel = None
        self.arg_index = arg_index
        if config.triton.multi_kernel > 1:
            # manually force a subkernel to ease perf testing
            picked_by_config = config.triton.multi_kernel - 2
            if picked_by_config >= len(self._kernels):
                raise AssertionError(
                    f"expected picked_by_config < len(self._kernels), "
                    f"got {picked_by_config} >= {len(self._kernels)}"
                )
            # pyrefly: ignore [bad-assignment]
            self.picked_kernel = picked_by_config
        elif not self.disable_cache:
            self.load_cache()

        self._recorded = False

    def cache_file_path(self):
        # [FUNCTION DOC] Returns the path of the pick cache file, keyed by a hash of each sub-kernel's cache key, size hints and triton meta.
        key = code_hash(
            ",".join(
                [
                    f"{k.fn.cache_key}{k.size_hints!r}{k.triton_meta!r}"
                    for k in self.kernels
                ]
            )
        )
        _, _, path = get_path(key, "picked_kernel")
        return pathlib.Path(path)

    def load_cache(self):
        # [FUNCTION DOC] Reads the picked kernel index from the cache file if it exists and validates it is in range (AssertionError otherwise; also if a pick already exists). Logs at debug level.
        if self.picked_kernel is not None:
            raise AssertionError("expected self.picked_kernel to be None")
        path = self.cache_file_path()
        if path.exists():
            with path.open() as fd:
                # pyrefly: ignore [bad-assignment]
                self.picked_kernel = int(fd.read())
                # pyrefly: ignore [unsupported-operation]
                if not (
                    self.picked_kernel >= 0 and self.picked_kernel < len(self._kernels)
                ):
                    raise AssertionError(
                        f"expected 0 <= picked_kernel < {len(self._kernels)}, "
                        f"got {self.picked_kernel}"
                    )
                log.debug(
                    "Load picked kernel %d from cache file %s", self.picked_kernel, path
                )

    def store_cache(self):
        # [FUNCTION DOC] Atomically writes the picked kernel index to the cache file, creating the directory. Raises AssertionError if nothing was picked yet.
        if self.picked_kernel is None:
            raise AssertionError("expected self.picked_kernel to not be None")
        path = self.cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        write_atomic(path, str(self.picked_kernel))
        log.debug("Store picked kernel %d to cache file %s", self.picked_kernel, path)

    @property
    def kernels(self):
        # [FUNCTION DOC] Property: returns the sub-kernel list, resolving any compile futures in place (blocks on pending compilation, so call after parallel compile finishes).
        """
        Read results from future.

        This should be called after parallel compilation is done.
        In case you call this before compilation is done,
        it may slow down the parallel compilation.
        """
        for i, kernel in enumerate(self._kernels):
            if isinstance(kernel, CodeCacheFuture):
                self._kernels[i] = kernel.result()

        return self._kernels

    def benchmark_sub_kernels(self, *args, **kwargs):
        # [FUNCTION DOC] Benchmarks every sub-kernel (cloning arguments per run, using only each kernel's own arguments) and returns the list of timings in milliseconds. Tests may mock this to force a pick.
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """

        def wrap_fn(kernel, index):
            # [FUNCTION DOC] HELPER: Returns a closure that benchmarks one sub-kernel.
            def inner():
                # [FUNCTION DOC] HELPER: Runs the kernel once on freshly cloned, filtered arguments.
                filtered_args = self._get_filtered_args(args, index)
                args_clone, kwargs_clone = kernel.clone_args(*filtered_args, **kwargs)
                return kernel.run(*args_clone, **kwargs_clone)

            return inner

        return [
            benchmarker.benchmark(
                wrap_fn(kernel, index),
                # Currently the kernel type must be a CachingAutotuner
                device=kernel.device_props.type,
                rep=40,
            )
            for index, kernel in enumerate(self.kernels)
        ]

    def _get_filtered_args(self, args, index):
        # [FUNCTION DOC] HELPER: Returns the subset of the all-kernel argument list that belongs to the kernel at index, via the arg_index slices; under the cpp wrapper returns args unchanged. Safe at runtime when the graph is a null handler.
        """
        We pass in all arguments to all kernels into the MultiKernelCall
        so when invoking a particular kernel we need to filter to only the
        arguments for that specific kernel.
        """

        # This is sometimes invoked at runtime where V.graph is
        # a NullHandler
        if hasattr(V.graph, "cpp_wrapper") and V.graph.cpp_wrapper:
            # for cpp-wrapper, we should not filter the args since
            # we already have chosen a single kernel and arg set.
            return args
        return [item for s in self.arg_index[index] for item in args[s]]

    # record_choice and lookup_choice are helper functions for cpp-wrapper
    # codegen. The first pass use record_choice to keep the choice and
    # the second pass do lookup by calling lookup_choice.
    #
    # An alternative that reused the multi-kernel cache does not work well
    # since during codegen of the second pass, it's very hard to know the
    # path for the cache file. Also reading the cache file need do some IO
    # which can be slower.
    @staticmethod
    def record_choice(multi_kernel_name: str, picked_kernel_name: str):
        # [FUNCTION DOC] Records the chosen kernel name for the multi-kernel in the graph's multi_kernel_to_choice map for the second cpp-wrapper pass. Does nothing outside codegen or when recording is off.
        """
        Record the multi-kernel choice for cpp-wrapper after autotuning

        We should do nothing if this function is not called during codegen.
        """
        from torch._inductor.graph import GraphLowering

        if not isinstance(V.graph, GraphLowering):
            return

        if not V.graph.record_multi_kernel_choice:
            return

        V.graph.multi_kernel_to_choice[multi_kernel_name] = picked_kernel_name

    @staticmethod
    def lookup_choice(multi_kernel_name: str) -> str:
        # [FUNCTION DOC] Returns the recorded chosen kernel name for a multi-kernel during cpp-wrapper codegen. Raises AssertionError if the choice was not recorded.
        # this should always been done during cpp-wrapper codegen
        if not (
            V.graph.record_multi_kernel_choice
            and multi_kernel_name in V.graph.multi_kernel_to_choice
        ):
            raise AssertionError(
                f"expected multi_kernel_name {multi_kernel_name} to be recorded "
                f"during cpp-wrapper codegen"
            )
        # there should be no miss
        return V.graph.multi_kernel_to_choice[multi_kernel_name]

    def run(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the multi-kernel. If no kernel is picked, benchmarks all sub-kernels, picks the fastest, logs, adds a row to the persistent reduction metric table, and stores the cache unless disabled. The first time
        # also records the picked kernel name (AssertionError if it has none). Then runs the picked kernel with its filtered arguments.
        if self.picked_kernel is None:
            timings = self.benchmark_sub_kernels(*args, **kwargs)
            self.picked_kernel = timings.index(min(timings))
            k0 = self.kernels[0]
            log.debug(
                "pick %dth sub-kernel in %s. Size hints %s. Reduction hint %s. Timings %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                k0.size_hints,
                k0.inductor_meta.get("reduction_hint"),
                timings,
            )
            get_metric_table("persistent_red_perf").add_row(
                functools.partial(self._metrics_table_row, timings)
            )

            if not self.disable_cache:
                self.store_cache()

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            if picked_kernel_name is None:
                raise AssertionError("expected picked_kernel_name to not be None")
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _metrics_table_row(self, timings):
        # [FUNCTION DOC] HELPER: Builds a metrics table row with size hints, reduction hint and, for up to 4 kernels, source file path and latency (empty strings for missing). Raises AssertionError if more than 4 timings.
        def get_kernel_path(k):
            # [FUNCTION DOC] HELPER: Returns the source file name of the kernel's underlying python function.
            return k.fn.fn.__code__.co_filename

        k0 = self.kernels[0]
        row = {
            "size_hints": k0.size_hints,
            "reduction_hint": k0.inductor_meta.get("reduction_hint"),
        }
        max_kernels = 4
        if len(timings) > max_kernels:
            raise AssertionError(
                f"expected len(timings) <= {max_kernels}, got {len(timings)}"
            )
        for i in range(max_kernels):
            if i < len(self.kernels):
                row[f"kernel{i}_path"] = get_kernel_path(self.kernels[i])
                row[f"kernel{i}_latency"] = timings[i]
            else:
                row[f"kernel{i}_path"] = ""
                row[f"kernel{i}_latency"] = ""
        return row


class SizeHintMultiKernel(MultiKernel):
    # [CLASS DOC] Compile-time multi-kernel whose sub-kernels are keyed by size-hint shapes (given as a mapping from shape key to kernel); only a one-dimensional search over hints is generated.
    # Attributes: kernels, kernel_shape_keys, kernel_name, placeholder args object.
    """
    Version of multi-kernel that generates kernels based on specified size hints.
    Currently only performs 1-d search over hints; doesn't perform combinatorial n-d search
    if n > 1 dynamic dimensions are specified.

    e.g. matmul([s0, s1], [s1, s2]) with size-hints [64, 256] only generates 2 kernels,
    based on tuning shapes ([64, 64], [64, 64]) and ([256, 256], [256, 256])
    """

    def __init__(self, kernels):
        # [FUNCTION DOC] Splits the non-empty mapping (AssertionError otherwise) into parallel lists of kernels and shape keys and defines the size-hint multi-kernel in the wrapper state. Does not call the parent constructor.
        if not (isinstance(kernels, dict) and len(kernels) >= 1):
            raise AssertionError("expected kernels to be a non-empty dict")

        self.kernels, self.kernel_shape_keys = [], []
        for shape_key, kernel in kernels.items():
            self.kernels.append(kernel)
            self.kernel_shape_keys.append(shape_key)
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            self.kernels, self.kernel_shape_keys
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()


class SizeHintMultiKernelCall(MultiKernelCall):
    # [CLASS DOC] Run-time size-hint multi-kernel: sub-kernels keyed by the input/output shapes they were tuned for. Chooses a kernel per runtime shape using a log2 L1 distance heuristic,
    # and caches the choice per distinct shape tuple (state: list of hint keys and a shape-to-index mapping).
    """
    Runtime class for size-hint multi-kernels.
    Instead of having a plain list of kernels to benchmark over, keys them by input & output shapes,
    and optionally perform shape-based selection. The pre-generated kernel is chosen based on the shape keys,
    with the heuristic being log2 l1 distance between the pre-generated / runtime input & output shapes.
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        # [FUNCTION DOC] Initializes the parent with the mapping's kernel values, remembers the shape keys as hints, and creates the empty per-shape choice cache.
        super().__init__(multi_kernel_name, list(kernels.values()), arg_index)
        self._kernel_hints = list(kernels.keys())

        # Caches results for unique shapes.
        self._shape_cache = {}

    def _get_shape_cache_key(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Returns a tuple of the shapes of all arguments that have a shape attribute.
        """
        Generate a cache key based on tensor shapes for shape-specialized dispatch.
        """
        shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                shapes.append(tuple(arg.shape))
        return tuple(shapes)

    def _get_cached_shape_choice(self, cache_key):
        # [FUNCTION DOC] HELPER: Returns the cached kernel index for the shape key, or None.
        """
        Get cached kernel choice for a specific shape.
        """
        return self._shape_cache.get(cache_key)

    def _cache_shape_choice(self, cache_key, kernel_idx):
        # [FUNCTION DOC] HELPER: Stores the kernel index for the shape key.
        """
        Cache kernel choice for a specific shape.
        """
        self._shape_cache[cache_key] = kernel_idx

    def _dist_heuristic(self, k1, k2):
        # [FUNCTION DOC] HELPER: Sums, over paired shapes and dimensions, the absolute difference of base-2 logs of sizes (non-positive sizes map to -1). Lower means closer.
        """
        log2 L1 distance heuristic for kernel selection.
        """

        def dist(x, y):
            # [FUNCTION DOC] HELPER: Absolute difference of log2 of two sizes, treating non-positive sizes as -1.
            lx = math.log2(x) if x > 0 else -1
            ly = math.log2(y) if y > 0 else -1
            return abs(lx - ly)

        out = 0
        for s1, s2 in zip(k1, k2):
            out += sum(dist(x, y) for x, y in zip(s1, s2))
        return out

    def run(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the multi-kernel: reuses the cached choice for this argument shape or selects by shape, records the picked kernel name once (AssertionError if missing), then runs it with its filtered arguments.
        cache_key = self._get_shape_cache_key(*args, **kwargs)
        cached_choice = self._get_cached_shape_choice(cache_key)
        if cached_choice is not None:
            self.picked_kernel = cached_choice
            log.debug(
                "using cached shape-specialized choice %dth sub-kernel in %s. Cache key: %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                cache_key,
            )
        else:
            self._select_kernel_by_shape(*args, **kwargs)

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            if picked_kernel_name is None:
                raise AssertionError("expected picked_kernel_name to not be None")
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _select_kernel_by_shape(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Picks the kernel whose shape key is closest to the runtime shapes (keys that are None get a huge distance), stores it as the picked kernel and caches it for this shape.
        # KNOWN ISSUE: the docstring says it benchmarks kernels, but the code only uses the distance heuristic and does no benchmarking.
        """
        Benchmark kernels for a particular shape and return the
        best kernel for this shape.
        """
        shape_key = self._get_shape_cache_key(*args, **kwargs)
        dists = [
            self._dist_heuristic(shape_key, key) if key is not None else 2**62
            for key in self._kernel_hints
        ]
        # pyrefly: ignore [bad-assignment]
        self.picked_kernel = dists.index(min(dists))
        self._cache_shape_choice(shape_key, self.picked_kernel)
