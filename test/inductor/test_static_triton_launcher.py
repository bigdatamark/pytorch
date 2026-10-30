# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the statically launched Triton kernel machinery in Inductor: the CUDA and
# XPU static launcher classes, the fast CUDA launcher C extension, and their integration with
# torch.compile and the autotuner. Unit tests use fake objects; GPU tests compile hand-written
# Triton kernels, dump their binaries to temporary files, and run them through the static
# launcher. On XPU with Triton, the JIT kernel launcher lookup is patched at test setup so native
# code is generated.
import gc
import os
import random
import tempfile
import weakref
from types import SimpleNamespace
from unittest import mock

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.static_triton_launcher import (
    statically_launched_kernel_by_device,
    StaticallyLaunchedCudaKernel,
    StaticallyLaunchedXpuKernel,
)
from torch._inductor.runtime.triton_compat import (
    CompiledKernel,
    JITFunction,
    tl,
    triton,
)
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    StaticTritonCompileResult,
)
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_XPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


if HAS_XPU_AND_TRITON:
    _orig_getitem = JITFunction.__getitem__


def _patched_getitem(self, grid):
    # [FUNCTION DOC] HELPER: Wraps the original JIT function grid lookup for XPU: returns a
    # launcher that defaults the generate_native_code keyword to true before delegating to the
    # original launcher.
    orig_launcher = _orig_getitem(self, grid)

    def launcher_with_native_code(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner launcher: sets the generate_native_code keyword default to
        # true and forwards all arguments to the original launcher, returning its result.
        kwargs.setdefault("generate_native_code", True)
        return orig_launcher(*args, **kwargs)

    return launcher_with_native_code


class TestStaticTritonLauncherUnit(TestCase):
    # [CLASS DOC] CPU-capable unit tests with fake objects for the XPU kernel loader, the fast
    # launcher builder, load device resolution and cubin retention at caching time.
    def test_xpu_load_kernel_uses_existing_three_tuple_abi(self):
        # [FUNCTION DOC] Builds an XPU static kernel object without running its constructor, sets
        # its fields by hand and calls load_kernel with device 3. Checks the fake C implementation
        # was called with path, name, shared memory and device, that the function, register and
        # spill counts are taken from the returned three-tuple, that module, cubin path and raw
        # cubin are cleared, and that close resets function and module.
        load_calls = []
        kernel_capsule = object()

        class FakeImpl:
            # [CLASS DOC] HELPER: Fake C implementation holder whose kernel loader records its
            # arguments and returns a capsule object with fixed register and spill counts of 7 and
            # 11.
            @staticmethod
            def _load_kernel(path, name, shared, device):
                # [FUNCTION DOC] HELPER: Records the path, name, shared size and device in the
                # enclosing list and returns the kernel capsule with register count 7 and spill
                # count 11.
                load_calls.append((path, name, shared, device))
                return kernel_capsule, 7, 11

        launcher = object.__new__(StaticallyLaunchedXpuKernel)
        launcher.function = None
        launcher.module = None
        launcher.device_agnostic = False
        launcher.functions = {}
        launcher.modules = {}
        launcher.cubin_path = "/tmp/kernel.zebin"
        launcher.cubin_raw = b"zebin"
        launcher.name = "kernel"
        launcher.shared = 13
        launcher.C_impl = FakeImpl

        launcher.load_kernel(3)

        self.assertEqual(load_calls, [("/tmp/kernel.zebin", "kernel", 13, 3)])
        self.assertIs(launcher.function, kernel_capsule)
        self.assertEqual(launcher.n_regs, 7)
        self.assertEqual(launcher.n_spills, 11)
        self.assertIsNone(launcher.module)
        self.assertIsNone(launcher.cubin_path)
        self.assertIsNone(launcher.cubin_raw)

        launcher.close()
        self.assertIsNone(launcher.function)
        self.assertIsNone(launcher.module)

    def test_fast_launcher_keeps_kernel_owner_alive(self):
        # [FUNCTION DOC] Verifies that building a fast launcher keeps the owning static kernel
        # alive. It builds a fake CUDA kernel and a launcher whose runner global refers to the
        # kernel run method, builds the fast launcher with a fake fast launcher class, then checks
        # the owner is stored on the fast launcher, the bound method is no longer in its globals,
        # the kernel survives garbage collection while the fast launcher exists, and is collected
        # after it is deleted.
        """
        _build_fast_launcher bakes kernel.function into a _FastCudaLauncher and
        replaces the "runner" global, dropping the launcher's only reference to
        the owning kernel. It must retain the owner so the kernel (and its
        loaded module) cannot be collected/unloaded while the fast launcher is
        still callable.
        """
        import types

        class FakeFastLauncher:
            # [CLASS DOC] HELPER: Fake replacement for the fast CUDA launcher class that just
            # stores the function handle it is constructed with.
            def __init__(self, func, num_warps, shared, arg_tys, n_scratch):
                # [FUNCTION DOC] HELPER: Stores the function argument; the other arguments are
                # accepted and ignored.
                self.func = func

        kernel = object.__new__(StaticallyLaunchedCudaKernel)
        kernel.function = 0xF00D
        kernel.num_warps = 4
        kernel.shared = 0
        kernel.arg_tys = "O"
        kernel.has_global_scratch = False
        kernel.has_profile_scratch = False

        launcher_globals = {"runner": kernel.run}

        def _launcher_body(grid_0, grid_1, grid_2, stream, *args):
            # [FUNCTION DOC] HELPER: Launcher body used only for its code object: forwards its
            # grid, stream and extra arguments to a global named runner. Never called directly.
            runner(grid_0, grid_1, grid_2, stream, *args)  # noqa: F821

        launcher = types.FunctionType(
            _launcher_body.__code__, launcher_globals, "launcher"
        )
        launcher._is_static = True

        autotuner = object.__new__(CachingAutotuner)
        autotuner.inductor_meta = {"use_fast_triton_launcher": True}
        autotuner.device_props = SimpleNamespace(type="cuda")

        with mock.patch("torch._C._FastCudaLauncher", FakeFastLauncher, create=True):
            fast_launcher = autotuner._build_fast_launcher(launcher)

        self.assertIsNotNone(fast_launcher)
        self.assertIs(fast_launcher._static_kernel_owner, kernel)
        self.assertNotIn(kernel.run, fast_launcher.__globals__.values())

        owner_ref = weakref.ref(kernel)
        del kernel, launcher, launcher_globals
        gc.collect()
        # Only fast_launcher._static_kernel_owner keeps the kernel alive now.
        self.assertIsNotNone(owner_ref())

        del fast_launcher
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_resolve_load_device_cpu_stays_none(self):
        # [FUNCTION DOC] Checks load-device resolution for CPU: resolving no device returns no
        # device, an explicit index is returned unchanged, and the result can be used in a device
        # guard for the CPU interface without raising.
        # CPU has no device index; its DeviceGuard is a no-op. Resolving its None to a
        # concrete index makes the guard call exchange_device, which CPU does not
        # implement, breaking the triton CPU backend's _make_launchers.
        from torch._dynamo.device_interface import DeviceGuard, get_interface_for_device
        from torch._inductor.runtime.triton_heuristics import _resolve_load_device

        self.assertIsNone(_resolve_load_device(None, "cpu"))
        self.assertEqual(_resolve_load_device(3, "cpu"), 3)
        # The resolved CPU value must be usable in a DeviceGuard without raising.
        iface = get_interface_for_device("cpu")
        with DeviceGuard(iface, _resolve_load_device(None, "cpu")):
            pass

    @staticmethod
    def _autotuner_with_static_cubin(cubin_raw):
        # [FUNCTION DOC] HELPER: Static helper that builds a caching autotuner and a static
        # compile result without running constructors. The result's kernel is a simple namespace
        # holding the given raw cubin, and the autotuner's compile results list holds that one
        # result. Returns both.
        autotuner = object.__new__(CachingAutotuner)
        result = object.__new__(StaticTritonCompileResult)
        result.kernel = SimpleNamespace(cubin_raw=cubin_raw)
        autotuner.compile_results = [result]
        return autotuner, result

    def test_prepare_for_caching_drops_cubin_raw_by_default(self):
        # [FUNCTION DOC] Checks that preparing the autotuner for caching drops the raw cubin on
        # the kernel by default (it becomes none).
        autotuner, result = self._autotuner_with_static_cubin(b"cubin")
        autotuner.prepare_for_caching()
        self.assertIsNone(result.kernel.cubin_raw)

    @torch._inductor.config.patch("keep_static_cubin_raw", True)
    def test_prepare_for_caching_keeps_cubin_raw_when_configured(self):
        # [FUNCTION DOC] Checks that with the keep_static_cubin_raw config enabled, preparing for
        # caching leaves the raw cubin bytes unchanged.
        autotuner, result = self._autotuner_with_static_cubin(b"cubin")
        autotuner.prepare_for_caching()
        self.assertEqual(result.kernel.cubin_raw, b"cubin")


@requires_gpu_and_triton
class TestStaticTritonLauncher(TestCase):
    # [CLASS DOC] GPU tests of the static launcher: each compiles a small Triton kernel, writes
    # its binary to a temporary file, creates a statically launched kernel from it and runs it
    # directly with a raw stream, checking results and argument type strings. Temporary files are
    # tracked and deleted in teardown.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup, resets the list of temporary files, and on XPU
        # with Triton replaces the JIT function getitem with the patched version.
        super().setUp()
        self.tmp_files = []
        if HAS_XPU_AND_TRITON:
            JITFunction.__getitem__ = _patched_getitem

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent teardown, removes every tracked temporary file ignoring
        # OS errors, and restores the original JIT function getitem on XPU with Triton.
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

        if HAS_XPU_AND_TRITON:
            JITFunction.__getitem__ = _orig_getitem

    def write_cubin_to_tmp(self, kernel: CompiledKernel) -> str:
        # [FUNCTION DOC] HELPER: Writes the compiled kernel's binary (cubin, hsaco on ROCm, or
        # zebin on XPU) from its assembly dictionary into a new temporary file, tracks the file
        # for cleanup and returns its name. Returns nothing if the kernel already has a cubin path
        # attribute.
        """
        Only used for tests where we don't have a cubin path.
        """
        if hasattr(kernel, "_cubin_path"):
            return
        # Just used by tests for now.
        # TODO: derive cubin_path from wherever triton stores the cubin file on disk.
        binary_key = ""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            if GPU_TYPE == "xpu":
                binary_key = "zebin"
            else:
                binary_key = "hsaco" if torch.version.hip else "cubin"

            tmp_file.write(kernel.asm[binary_key])
            self.tmp_files.append(tmp_file)
            return tmp_file.name

    def _make_launcher(
        self,
        compiled_kernel: CompiledKernel,
    ) -> StaticallyLaunchedCudaKernel | StaticallyLaunchedXpuKernel:
        # [FUNCTION DOC] HELPER: Builds a static launcher for a compiled kernel: writes the binary
        # to a temp file, sets the kernel cubin path, creates the launcher for the GPU type, then
        # clears the path and reloads the cubin from raw data to exercise that path (raises an
        # error if no path was set), and loads the kernel on the current device. Returns the
        # launcher.
        """
        Compiles a Triton kernel with the provided *args,
        writes its cubin to the temporary file, and returns the file path.
        """
        cubin_file = self.write_cubin_to_tmp(compiled_kernel)
        compiled_kernel._cubin_path = cubin_file
        result = statically_launched_kernel_by_device(compiled_kernel, GPU_TYPE)
        # Test reload cubin from raw here
        old_cubin_path = result.cubin_path
        if old_cubin_path is None:
            raise AssertionError
        result.cubin_path = None
        result.reload_cubin_from_raw(old_cubin_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        result.load_kernel(device_interface.current_device())
        return result

    def test_basic(self):
        # [FUNCTION DOC] Compiles a kernel that adds a scalar to a tensor, creates a static
        # launcher and checks the first run result, the argument type string for a pointer plus
        # int, and a second run through the launcher with a fresh tensor.
        """Verify _FastCudaLauncher correctly launches a kernel with tensor + scalar args.

        Compiles a simple kernel, wraps it in _FastCudaLauncher, then invokes
        via vectorcall and checks the output tensor has the expected value.
        """

        # [FUNCTION DOC] Triton kernel: loads a value from the first pointer, adds the scalar
        # argument and stores it back.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    # I wish I could macro all int types this into a single unit test on a loop, but
    # 1. variables aren't allowed as type annotations in python
    # 2. triton relies on inspect.get_source to get the type annotations
    # so I can't even use exec() to generate the test cases.
    # So we'll just make a few kernels by hand
    def test_unsigned_integers(self):
        # [FUNCTION DOC] Triton kernel: sums its four unsigned integer arguments and adds them to
        # the loaded value in the tensor.
        # [FUNCTION DOC] Compiles a kernel with unsigned 8, 16, 32 and 64 bit scalar arguments on
        # a uint64 tensor, checks the result of 200 and the argument type string for those types,
        # then runs the launcher on a fresh tensor.
        @triton.jit
        def unsigned_integers(
            arg0, arg1: tl.uint8, arg2: tl.uint16, arg3: tl.uint32, arg4: tl.uint64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.uint64, device=GPU_TYPE)
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = unsigned_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.uint64, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "OBHIK")
        new_arg0 = torch.zeros(1, dtype=torch.uint64, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    def test_signed_integers(self):
        # [FUNCTION DOC] Triton kernel: sums its four signed integer arguments and adds them to
        # the loaded value in the tensor.
        # [FUNCTION DOC] Same as the unsigned test for signed 8, 16, 32 and 64 bit integer scalar
        # arguments on an int64 tensor, checking the type string and launcher results.
        @triton.jit
        def signed_integers(
            arg0, arg1: tl.int8, arg2: tl.int16, arg3: tl.int32, arg4: tl.int64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int64, device=GPU_TYPE)
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = signed_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.int64, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Obhil")
        new_arg0 = torch.zeros(1, dtype=torch.int64, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    def test_basic_1arg(self):
        # [FUNCTION DOC] Triton kernel with a single pointer argument: loads a value and stores it
        # incremented by one.
        # [FUNCTION DOC] Compiles a kernel with a single pointer argument that increments the
        # value, and checks static launcher results and the one-character argument type string.
        @triton.jit
        def simple_kernel_1_arg(arg0):
            x = tl.load(arg0)
            tl.store(arg0, x + 1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel_1_arg[1,](arg0)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([1], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(
            1,
            1,
            1,
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    def test_constexpr(self):
        # [FUNCTION DOC] Compiles a kernel with a compile-time constant argument. The constant is
        # baked into the binary so the launcher only takes the pointer; checks the argument type
        # string and results.
        # Constexprs are compiled directly into the cubin file,
        # so we never need to pass it to StaticCudaLauncher.

        # [FUNCTION DOC] Triton kernel adding a compile-time constant to the loaded value and
        # storing it back.
        @triton.jit
        def kernel_constexpr(arg0, CONSTANT: tl.constexpr):
            x = tl.load(arg0)
            tl.store(arg0, x + CONSTANT)

        # Can't use make_launcher because constexpr needs to be constant
        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = kernel_constexpr[(1,)](arg0, CONSTANT=5)
        launcher = self._make_launcher(compiled_kernel)

        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(
            1,
            1,
            1,
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    def test_implied_constant(self):
        # [FUNCTION DOC] Compiles an Inductor-generated reduction kernel whose size argument is
        # unused and implied constant, then runs the launcher omitting that argument and checks
        # the result against the expected boolean output buffer.
        """xnumel is unused in this kernel, but isn't explicitly marked as a constexpr"""

        # This kernel was generated by inductor so it has a bunch of unused arguments. We don't change it
        # [FUNCTION DOC] Triton kernel produced by Inductor for an any-is-infinite reduction over
        # a float tensor, writing a single boolean. Contains unused arguments on purpose and is
        # not to be changed.
        @triton.jit
        def triton_red_fused_any_isinf_0(
            in_ptr0,
            out_ptr0,
            xnumel,
            r0_numel,
            XBLOCK: tl.constexpr,
            R0_BLOCK: tl.constexpr,
        ):
            xnumel = 1  # noqa: F841
            rnumel = r0_numel  # noqa: F841
            RBLOCK: tl.constexpr = R0_BLOCK  # noqa: F841
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # noqa: F841
            xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)  # noqa: F841
            r0_base = tl.arange(0, R0_BLOCK)[None, :]
            rbase = r0_base  # noqa: F841
            _tmp3 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
            for r0_offset in range(0, r0_numel, R0_BLOCK):
                r0_index = r0_offset + r0_base
                r0_mask = r0_index < r0_numel
                roffset = r0_offset  # noqa: F841
                rindex = r0_index  # noqa: F841
                r0_0 = r0_index
                tmp0 = tl.load(
                    in_ptr0 + (r0_0), r0_mask, eviction_policy="evict_first", other=0.0
                )
                tmp1 = libdevice.isinf(tmp0).to(tl.int1)
                tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
                tmp4 = _tmp3 | tmp2
                _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
            tmp3 = triton_helpers.any(_tmp3.to(tl.int8), 1)[:, None].to(tl.int1)
            tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)

        arg0 = torch.tensor([0.0, 0.5, float("inf"), 5], device=GPU_TYPE)
        arg1 = torch.tensor([False], device=GPU_TYPE)
        arg2 = torch.tensor([False], device=GPU_TYPE)
        compiled_kernel = triton_red_fused_any_isinf_0[1,](
            arg0, arg1, 1, 128, XBLOCK=1, R0_BLOCK=1
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        # Don't pass in xnumel, as it is a constant
        launcher.run(1, 1, 1, stream, arg0, arg2, 128)
        self.assertEqual(arg1, arg2)

    def test_kernel_no_args(self):
        # [FUNCTION DOC] Compiles a kernel with no arguments and runs the static launcher with no
        # arguments to check it works.
        # Just an easy way to test incompatible number of arguments
        # [FUNCTION DOC] Triton kernel that does nothing and takes no arguments.
        @triton.jit
        def kernel_no_op():
            pass

        compiled_kernel = kernel_no_op[(1,)]()
        launcher = self._make_launcher(compiled_kernel)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream)

    def test_high_shared_mem(self):
        # [FUNCTION DOC] Triton kernel: loads a value, adds a scalar and stores it back.
        # [FUNCTION DOC] Compiles a simple kernel, sets its shared memory size to 50000 bytes to
        # force the high shared memory path, enables the slow launch flag on the launcher, runs it
        # and checks the result.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        # Allocate 50 KB of memory
        compiled_kernel.shared = 50000
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.slow_launch_kernel = True
        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    @skipIfXpu(msg="Only testing CUDA OOM behavior")
    def test_too_high_shared_mem(self):
        # [FUNCTION DOC] Triton kernel: loads a value, adds a scalar and stores it back.
        # [FUNCTION DOC] Sets the kernel's shared memory to an absurdly large number and checks
        # that creating the launcher raises a RuntimeError mentioning the kernel is out of
        # resource. Skipped on XPU.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        # Allocate too much shared memory
        compiled_kernel.shared = 99999999
        self.assertRaisesRegex(
            RuntimeError,
            "out of resource: simple_kernel",
            lambda: self._make_launcher(compiled_kernel),
        )

    def test_kernel_empty_tensor(self):
        # [FUNCTION DOC] Runs an Inductor-style concatenation kernel on an empty first input and a
        # 20 element second input, with the first size argument zero, through the static launcher
        # and checks the output equals a second run's output.
        # Triton kernel generated by torch.compile of the following:
        # @torch.compile()
        # def foo(x, y):
        #   return torch.cat(((x * 4), y + 10))

        # Running with example input:
        # torch._dynamo.decorators.mark_unbacked(t, 0)
        # x = torch.rand(0, device=GPU_TYPE)
        # y = torch.rand(20, device=GPU_TYPE)

        # [FUNCTION DOC] Triton kernel generated for concatenating a scaled first input and an
        # offset second input into one output, with masks that handle an empty first input.
        @triton.jit
        def triton_poi_fused_cat_0(
            in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr
        ):
            xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = x0
            tmp3 = ks0
            tmp4 = tmp0 < tmp3
            tmp5 = tl.load(
                in_ptr0 + (x0), xmask & tmp4, eviction_policy="evict_last", other=0.0
            )
            tmp6 = 4.0
            tmp7 = tmp5 * tmp6
            tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
            tmp9 = tl.where(tmp4, tmp7, tmp8)
            tmp10 = tmp0 >= tmp3
            tmp13 = tl.load(
                in_ptr1 + (x0 + ((-1) * ks0)),
                xmask & tmp10,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp14 = 10.0
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
            tmp17 = tl.where(tmp10, tmp15, tmp16)
            tmp18 = tl.where(tmp4, tmp9, tmp17)
            tl.store(out_ptr0 + (x0), tmp18, xmask)

        arg0 = 0
        arg1 = torch.randn(0, device=GPU_TYPE)
        arg2 = torch.randn(20, device=GPU_TYPE)
        buf0 = torch.empty(20, device=GPU_TYPE)
        buf1 = torch.empty(20, device=GPU_TYPE)
        xnumel = 20 + arg0
        compiled_kernel = triton_poi_fused_cat_0[(1,)](
            arg1, arg2, buf0, arg0, xnumel, XBLOCK=32
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, arg1, arg2, buf1, arg0, xnumel)
        self.assertEqual(buf0, buf1)

    def test_kernel_many_args(self):
        # [FUNCTION DOC] Generates source for a Triton kernel with 200 scalar arguments that sum
        # into a tensor, loads it through the code cache, compiles it with random values and
        # checks that the static launcher gives the same result on a fresh buffer.
        N = 200
        # Make 200 arguments
        args = [f"arg_{i}" for i in range(N)]
        decl = ", ".join(args)
        sums = [f"    total += arg_{i}" for i in range(N)]
        sums_str = "\n".join(sums)

        template = f"""
from torch._inductor.runtime.triton_compat import tl, triton
@triton.jit
def kernel_many_args(out_tensor, {decl}):
    out = tl.load(out_tensor)
    total = out
{sums_str}
    tl.store(out_tensor, total)
        """

        result = PyCodeCache.load(template.lstrip())

        kernel_args = tuple(random.random() for _ in range(N))
        buf0 = torch.zeros(1, device=GPU_TYPE)
        compiled_kernel = result.kernel_many_args[1,](buf0, *kernel_args)
        launcher = self._make_launcher(compiled_kernel)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        buf1 = torch.zeros(1, device=GPU_TYPE)
        launcher.run(1, 1, 1, stream, buf1, *kernel_args)
        self.assertEqual(buf0, buf1)

    def test_launcher_keeps_module_owner_alive_until_release(self):
        # [FUNCTION DOC] Checks that a launcher holding the bound run method of a fake static
        # kernel keeps the owner alive until the launcher is deleted. After deleting the owner
        # variable the owner is still alive and nothing was unloaded; after deleting the launcher
        # the owner is collected and its module was unloaded once.
        """
        Generated launchers capture ``runner=self.kernel.run``.  That closure
        must keep the static kernel owner alive, otherwise its module could be
        unloaded while the launcher can still be called.
        """
        unloaded_modules = []

        class FakeImpl:
            # [CLASS DOC] HELPER: Fake C implementation whose module unloader records the module
            # handle in a list.
            @staticmethod
            def _unload_kernel(mod):
                # [FUNCTION DOC] HELPER: Appends the unloaded module handle to the enclosing list.
                unloaded_modules.append(mod)

        class FakeKernelOwner:
            # [CLASS DOC] HELPER: Fake kernel owner borrowing the real CUDA kernel destructor and
            # close behavior. It has a name, module and function handles, empty function and
            # module maps and the fake implementation.
            __del__ = StaticallyLaunchedCudaKernel.__del__

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the name, module handle, function handle, empty
                # function and module maps and the fake C implementation.
                self.name = "fake_kernel"
                self.module = 0xC0FFEE
                self.function = 0xF00D
                self.functions = {}
                self.modules = {}
                self.C_impl = FakeImpl

            def close(self):
                # [FUNCTION DOC] HELPER: Delegates to the real static CUDA kernel close method on
                # this object and returns its result.
                return StaticallyLaunchedCudaKernel.close(self)

            def run(self, *_args, **_kwargs):
                # [FUNCTION DOC] HELPER: Accepts any arguments and does nothing, returning none.
                return None

        owner = FakeKernelOwner()
        owner_ref = weakref.ref(owner)

        class FakeLauncher:
            # [CLASS DOC] HELPER: Fake launcher that stores a runner callable and calls it with a
            # fixed grid on the given stream.
            def __init__(self, runner):
                # [FUNCTION DOC] HELPER: Stores the runner callable.
                self.runner = runner

            def __call__(self, stream):
                # [FUNCTION DOC] HELPER: Calls the stored runner with grid 1, 1, 1 and the stream,
                # returning its result.
                return self.runner(1, 1, 1, stream)

        launcher = FakeLauncher(owner.run)

        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref())
        self.assertEqual(unloaded_modules, [])

        del launcher
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertEqual(unloaded_modules, [0xC0FFEE])


@requires_gpu_and_triton
@torch._inductor.config.patch(
    {"use_static_triton_launcher": True, "strict_static_triton_launcher": True}
)
class TestStaticTritonCompileResult(TestCase):
    # [CLASS DOC] End to end tests with the static launcher and strict mode enabled in config:
    # torch.compile on GPU must use the static Triton launcher or raise.
    """
    Tests static cuda launcher with torch.compile()
    """

    def test_basic_compile(self):
        # [FUNCTION DOC] Compiles an addition of two tensors and checks the result.
        @torch.compile
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Adds two tensors.
            return x + y

        x = torch.randn(10, device=GPU_TYPE)
        y = torch.randn(10, device=GPU_TYPE)
        self.assertEqual(foo(x, y), x + y)

    # The error gets raised on a worker, so we want to not use a separate process
    @torch._inductor.config.patch("compile_threads", 1)
    def test_incompatible_code(self):
        # [FUNCTION DOC] Uses a user-defined Triton kernel in a compiled function and checks that
        # strict static launching raises an Inductor error saying the kernel cannot be statically
        # launched because it is user defined. Compile threads are forced to 1 so the error occurs
        # in-process.
        # User defined triton kernel
        # [FUNCTION DOC] User-defined Triton kernel adding a scalar to a loaded value in place.
        @triton.jit
        def custom_kernel(arg_0, arg_1):
            x = tl.load(arg_0)
            y = arg_1
            tl.store(arg_0, x + y)

        @torch.compile
        def foo(x):
            # [FUNCTION DOC] HELPER: Launches the user-defined kernel on the input with scalar 5
            # and returns the input.
            custom_kernel[1,](x, 5)
            return x

        x = torch.randn(1, device=GPU_TYPE)
        self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "CannotStaticallyLaunchKernel: User defined triton kernel",
            lambda: foo(x),
        )

    # The error gets raised on a worker, so we want to not use a separate process
    @torch._inductor.config.patch(
        {"compile_threads": 1, "static_launch_user_defined_triton_kernels": True}
    )
    def test_static_launch_user_defined_triton_kernels(self):
        # [FUNCTION DOC] With static launching of user-defined kernels enabled, runs a compiled
        # function that launches a user-defined kernel and checks the input is increased by 5
        # compared to a clone.
        # User defined triton kernel
        # [FUNCTION DOC] User-defined Triton kernel adding a scalar to a loaded value in place.
        @triton.jit
        def custom_kernel(arg_0, arg_1):
            x = tl.load(arg_0)
            y = arg_1
            tl.store(arg_0, x + y)

        @torch.compile
        def foo(x):
            # [FUNCTION DOC] HELPER: Launches the user-defined kernel on the input with scalar 5
            # and returns the input.
            custom_kernel[1,](x, 5)
            return x

        x = torch.randn(1, device=GPU_TYPE)
        x2 = x.clone().detach_()
        self.assertEqual(foo(x), x2 + 5)

    @skipIfXpu(msg="Tests CUDA static launcher dtype validation")
    @torch._inductor.config.patch({"compile_threads": 1, "fx_graph_cache": False})
    def test_custom_op_bad_fake_dtype_fails_fast(self):
        # [FUNCTION DOC] Registers a custom op whose fake implementation reports bfloat16 but
        # whose real output is float32, compiles a function using it and checks that a
        # RuntimeError about the expected and actual dtype and an incorrect fake is raised.
        # Skipped on XPU; uses one compile thread and disables the FX graph cache.
        namespace = f"test_static_launcher_dtype_validation_{random.randint(0, 2**32)}"

        @torch.library.custom_op(f"{namespace}::bad_meta_mul", mutates_args=())
        def bad_meta_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation returning the elementwise product of
            # its two inputs.
            return a * b

        @bad_meta_mul.register_fake
        def _(a, b):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor shaped like the
            # first input but with dtype bfloat16, which intentionally mismatches the real output
            # dtype.
            return torch.empty_like(a, dtype=torch.bfloat16)

        op = getattr(torch.ops, namespace).bad_meta_mul

        @torch.compile(fullgraph=True)
        def f(a, b):
            # [FUNCTION DOC] HELPER: Applies the custom op to its two inputs and adds one.
            x = op(a, b)
            return x + 1

        a = torch.randn(1 << 20, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(1 << 20, device=GPU_TYPE, dtype=torch.float32)

        self.assertRaisesRegex(
            RuntimeError,
            r"(?s)expected dtype torch\.bfloat16 but got torch\.float32.*incorrect fake",
            lambda: f(a, b),
        )

    def test_empty_tensor(self):
        # [FUNCTION DOC] Compiles a concatenation with a first input of size zero marked as
        # unbacked and checks it matches eager.
        @torch.compile()
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Concatenates four times the first input with the second input
            # plus ten.
            return torch.cat(((x * 4), y + 10))

        x = torch.rand(0, device=GPU_TYPE)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        y = torch.rand(20, device=GPU_TYPE)
        result = foo(x, y)
        self.assertEqual(result, torch.cat(((x * 4), y + 10)))

    def test_any(self):
        # [FUNCTION DOC] Compiles a function with any and all reductions, including checks for
        # infinite values, on float64 input, and compares to eager, then repeats after setting one
        # element to infinity.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns four reductions: any over the last dim, whether any
            # element is infinite, all infinite along dim 0, and all not-infinite overall.
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        compiled_fn = torch.compile(fn)
        arg = -torch.rand(64, device=GPU_TYPE, dtype=torch.float64)
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)
        arg[1] = float("inf")
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)

    def test_disable_static_triton_launcher(self):
        # [FUNCTION DOC] Disables the static launcher in config and checks that the static compile
        # result's make_launcher method (replaced by a mock) is never called while the result
        # still matches eager.
        @torch.compile
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Concatenates four times the first input with the second input
            # plus ten.
            return torch.cat(((x * 4), y + 10))

        # Test that static cuda launcher is in fact disabled
        with torch._inductor.config.patch("use_static_triton_launcher", False):
            x = torch.rand(20, device=GPU_TYPE)
            y = torch.rand(20, device=GPU_TYPE)
            with mock.patch(
                "torch._inductor.runtime.triton_heuristics.StaticTritonCompileResult.make_launcher"
            ) as mocked:
                result = fn(x, y)
                mocked.assert_not_called()

            self.assertEqual(result, torch.cat(((x * 4), y + 10)))


@requires_gpu_and_triton
# _FastCudaLauncher hipModuleLaunchKernel path segfaults on ROCm
@skipIfRocm
@skipIfXpu
class TestFastCudaLauncher(TestCase):
    # [CLASS DOC] Tests of the fast CUDA launcher C extension built around a statically launched
    # kernel. Skipped on ROCm and XPU. Tracks temporary binary files for cleanup.
    """Tests for _FastCudaLauncher vectorcall C extension."""

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup and resets the list of temporary files.
        super().setUp()
        self.tmp_files = []

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent teardown and removes every tracked temporary file,
        # ignoring OS errors.
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

    def write_cubin_to_tmp(self, kernel: CompiledKernel) -> str:
        # [FUNCTION DOC] HELPER: Writes the kernel binary (hsaco on ROCm, otherwise cubin) into a
        # temporary file, tracks it and returns its name; returns nothing if the kernel already
        # has a cubin path.
        if hasattr(kernel, "_cubin_path"):
            return
        binary_key = "hsaco" if torch.version.hip else "cubin"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(kernel.asm[binary_key])
            self.tmp_files.append(tmp_file)
            return tmp_file.name

    def _make_launcher(
        self,
        compiled_kernel: CompiledKernel,
    ) -> StaticallyLaunchedCudaKernel:
        # [FUNCTION DOC] HELPER: Builds a static launcher from a compiled kernel: writes the
        # binary, sets its path, creates the launcher, reloads the cubin from raw data (raising an
        # error if no path was set), loads the kernel on the current device and returns it.
        cubin_file = self.write_cubin_to_tmp(compiled_kernel)
        compiled_kernel._cubin_path = cubin_file
        result = statically_launched_kernel_by_device(compiled_kernel, GPU_TYPE)
        old_cubin_path = result.cubin_path
        if old_cubin_path is None:
            raise AssertionError
        result.cubin_path = None
        result.reload_cubin_from_raw(old_cubin_path)
        device_interface = get_interface_for_device(GPU_TYPE)
        result.load_kernel(device_interface.current_device())
        return result

    def _make_fast_launcher(self, kernel):
        # [FUNCTION DOC] HELPER: Builds a fast CUDA launcher object from a static kernel: uses its
        # function handle, warps, shared memory and argument type string, counting one scratch
        # slot each for global and profile scratch when present.
        from torch._C import _FastCudaLauncher

        n_scratch = 0
        if getattr(kernel, "has_global_scratch", False):
            n_scratch += 1
        if getattr(kernel, "has_profile_scratch", False):
            n_scratch += 1
        return _FastCudaLauncher(
            kernel.function, kernel.num_warps, kernel.shared, kernel.arg_tys, n_scratch
        )

    def test_basic(self):
        # [FUNCTION DOC] Triton kernel: loads a value, adds a scalar and stores it back.
        # [FUNCTION DOC] Verifies the fast launcher runs a kernel with a tensor and a scalar
        # argument and writes the expected value.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        new_arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(1, 1, 1, stream, new_arg0, 5)
        self.assertEqual(
            new_arg0, torch.tensor([5], dtype=torch.int32, device=GPU_TYPE)
        )

    def test_multiple_tensor_args(self):
        # [FUNCTION DOC] Verifies the fast launcher handles two input tensor pointers plus an
        # output pointer; the output of adding 3 and 7 must be 10.
        """Verify _FastCudaLauncher handles multiple tensor pointer args correctly."""

        # [FUNCTION DOC] Triton kernel adding two loaded values and storing the sum to an output
        # pointer.
        @triton.jit
        def add_kernel(a, b, out):
            x = tl.load(a)
            y = tl.load(b)
            tl.store(out, x + y)

        a = torch.tensor([3], dtype=torch.int32, device=GPU_TYPE)
        b = torch.tensor([7], dtype=torch.int32, device=GPU_TYPE)
        out = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = add_kernel[(1,)](a, b, out)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        out2 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(1, 1, 1, stream, a, b, out2)
        self.assertEqual(out2, torch.tensor([10], dtype=torch.int32, device=GPU_TYPE))

    def test_zero_grid(self):
        # [FUNCTION DOC] Verifies that launching with a zero-sized grid does nothing, leaving the
        # target tensor unchanged.
        """Verify zero-grid launch is a no-op (kernel does not execute)."""

        # [FUNCTION DOC] Triton kernel adding a scalar to a loaded value and storing it back.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            tl.store(arg0, x + arg1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        target = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        fast(0, 1, 1, stream, target, 99)
        self.assertEqual(target, torch.tensor([0], dtype=torch.int32, device=GPU_TYPE))

    def test_wrong_arg_count(self):
        # [FUNCTION DOC] Verifies that calling the fast launcher with fewer arguments than
        # expected raises a RuntimeError.
        """Verify _FastCudaLauncher raises RuntimeError on argument count mismatch."""

        # [FUNCTION DOC] Triton kernel adding a scalar to a loaded value and storing it back.
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            tl.store(arg0, x + arg1)

        arg0 = torch.zeros(1, dtype=torch.int32, device=GPU_TYPE)
        compiled_kernel = simple_kernel[(1,)](arg0, 5)
        launcher = self._make_launcher(compiled_kernel)
        fast = self._make_fast_launcher(launcher)

        device_interface = get_interface_for_device(GPU_TYPE)
        stream = device_interface.get_raw_stream(device_interface.current_device())
        with self.assertRaises(RuntimeError):
            fast(1, 1, 1, stream, arg0)  # missing arg1

    def test_too_many_args_raises_value_error(self):
        # [FUNCTION DOC] Checks that constructing the fast launcher with 130 pointer arguments,
        # above the maximum of 121 arguments, raises a ValueError.
        """Verify _FastCudaLauncher raises ValueError when nArgs > MAX_ARGS (121)."""
        from torch._C import _FastCudaLauncher

        # Use a dummy CUfunction (0) — construction should fail before launch.
        # 'O' is the pointer type code; 130 of them exceeds MAX_ARGS=121.
        arg_tys = "O" * 130
        with self.assertRaises(ValueError):
            _FastCudaLauncher(0, 1, 0, arg_tys, 0)


@requires_gpu_and_triton
@torch._inductor.config.patch(
    {
        "use_static_triton_launcher": True,
        "strict_static_triton_launcher": True,
        "use_fast_triton_launcher": True,
    }
)
@skipIfRocm  # see TestFastCudaLauncher
class TestFastCudaLauncherCompileResult(TestCase):
    # [CLASS DOC] End to end tests that torch.compile builds the fast CUDA launcher when
    # configured and does not when it is disabled or on XPU. Skipped on ROCm.
    """E2E tests verifying _FastCudaLauncher handling in torch.compile.

    CUDA tests assert both correctness (output matches eager) and that the
    _FastCudaLauncher C extension was constructed, not silently skipped. XPU
    uses its own static launcher and must fall back without the CUDA-only fast
    launcher.
    """

    def _patch_build_fast_launcher(self):
        # [FUNCTION DOC] HELPER: Returns a pair: a patcher replacing the autotuner method that
        # builds the fast launcher with a wrapper, and a list that grows by one boolean per call
        # saying whether a launcher was returned.
        """Context manager that tracks _build_fast_launcher calls.

        Returns a list that accumulates True/False for each call,
        indicating whether a _FastCudaLauncher was successfully built.
        """
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        results = []
        original = CachingAutotuner._build_fast_launcher

        def tracking_build(autotuner_self, launcher):
            # [FUNCTION DOC] HELPER: Calls the original build method, records whether it returned
            # a launcher, and returns that result.
            result = original(autotuner_self, launcher)
            results.append(result is not None)
            return result

        return mock.patch.object(
            CachingAutotuner, "_build_fast_launcher", tracking_build
        ), results

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/181491")
    def test_basic_compile(self):
        # [FUNCTION DOC] Compiles an addition with the build call tracked and checks the result.
        # On XPU the build must be reached but never produce a launcher; on other devices at least
        # one launcher must be built. Skipped on XPU for a known issue.
        """Verify torch.compile uses _FastCudaLauncher and produces correct output."""
        patcher, results = self._patch_build_fast_launcher()
        with patcher:

            @torch.compile
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Adds two tensors.
                return x + y

            x = torch.randn(10, device=GPU_TYPE)
            y = torch.randn(10, device=GPU_TYPE)
            self.assertEqual(foo(x, y), x + y)
            if GPU_TYPE == "xpu":
                self.assertTrue(results, "_build_fast_launcher was not reached on XPU")
                self.assertFalse(
                    any(results), "_FastCudaLauncher should not be built on XPU"
                )
            else:
                self.assertTrue(
                    any(results),
                    "_FastCudaLauncher was not built by any CachingAutotuner",
                )

    def test_disable_fast_launcher(self):
        # [FUNCTION DOC] Disables the fast launcher config and checks the compiled result is
        # correct and no fast launcher was built.
        """Verify disabling the config falls back to the regular launcher."""
        patcher, results = self._patch_build_fast_launcher()
        with patcher, torch._inductor.config.patch("use_fast_triton_launcher", False):

            @torch.compile
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Adds two tensors.
                return x + y

            x = torch.randn(10, device=GPU_TYPE)
            y = torch.randn(10, device=GPU_TYPE)
            result = foo(x, y)
            self.assertEqual(result, x + y)
            self.assertFalse(
                any(results),
                "_FastCudaLauncher should not be built when config is disabled",
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # TODO: Enable test on XPU windows once supported.
    if not (HAS_XPU_AND_TRITON and IS_WINDOWS):
        run_tests()
