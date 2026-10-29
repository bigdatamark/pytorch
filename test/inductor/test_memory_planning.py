# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor memory planning, which packs intermediate buffers into a
# shared pool. One class unit-tests buffer-group output detection in MemoryPlanner using fake
# graph objects; another (GPU only, memory planning config enabled) compiles a small module and
# checks generated Python wrapper, C++ wrapper and AOTI code for pool allocation patterns, plus
# a case with unbacked symbolic sizes. Import-time side effect: on Windows CI it writes a
# message to stderr and exits (main) or raises a skip, before the heavy imports. The AOTI helper
# is imported relatively with a fallback to a plain import.

import re
import sys
import unittest
from types import SimpleNamespace

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_memory_planning yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.export import Dim
from torch.testing._internal.common_utils import (
    IS_LINUX,
    skipIfRocm,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
)


try:
    from .test_aot_inductor import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor import (  # @manual=fbcode//caffe2/test/inductor:test_aot_inductor-library
        AOTIRunnerUtil,
    )


class TestMemoryPlanningOutputGroups(TestCase):
    # [CLASS DOC] Unit tests for MemoryPlanner buffer-group computation, checking when a group
    # of buffers that share storage through reuse is flagged as an output. Uses lightweight fake
    # buffer and graph objects, no compilation.
    class FakeBuffer:
        # [CLASS DOC] HELPER: Minimal buffer stand-in that stores a name and returns it from its
        # name getter.
        def __init__(self, name):
            # [FUNCTION DOC] HELPER: Stores the buffer name.
            self._name = name

        def get_name(self):
            # [FUNCTION DOC] HELPER: Returns the stored name.
            return self._name

    class FakeOutputNode:
        # [CLASS DOC] HELPER: Stand-in for a scheduler node representing graph output; its name
        # getter always returns the output marker string.
        def get_name(self):
            # [FUNCTION DOC] HELPER: Returns the fixed output marker name.
            return "OUTPUT"

    @staticmethod
    def make_line(cls, *args):
        # [FUNCTION DOC] HELPER: Static factory that builds an allocate or reuse wrapper line of
        # the given class without running its constructor. Sets the wrapper to None and
        # comm_buffer to false. For an allocate line the single argument is the node; otherwise
        # the two arguments are the node and the buffer it is reused as, and delete_old is set
        # true. Returns the line.
        line = object.__new__(cls)
        line.wrapper = None
        if cls.__name__ == "AllocateLine":
            (line.node,) = args
            line.comm_buffer = False
        else:
            line.node, line.reused_as = args
            line.delete_old = True
            line.comm_buffer = False
        return line

    def compute_single_group(self, *, graph_outputs=(), scheduler_buffers=None):
        # [FUNCTION DOC] HELPER: Runs MemoryPlanner buffer-group computation on one allocate
        # line for buffer buf16 followed by a reuse line aliasing it as buf48, inside a fake
        # graph context whose output names come from graph_outputs and whose scheduler buffer
        # map comes from scheduler_buffers (default empty). Asserts exactly one group was
        # produced and returns it.
        from torch._inductor.codegen.memory_planning import MemoryPlanner
        from torch._inductor.codegen.wrapper import AllocateLine, ReuseLine
        from torch._inductor.virtualized import V

        old = self.FakeBuffer("buf16")
        alias = self.FakeBuffer("buf48")
        fake_graph = SimpleNamespace(
            get_output_names=lambda: list(graph_outputs),
            scheduler=SimpleNamespace(name_to_buf=scheduler_buffers or {}),
        )

        planner = MemoryPlanner(wrapper=None)
        lines = [
            self.make_line(AllocateLine, old),
            self.make_line(ReuseLine, old, alias),
        ]
        with V.set_graph_handler(fake_graph):
            planner.compute_buffer_groups(lines)

        self.assertEqual(len(planner.buffer_groups), 1)
        return planner.buffer_groups[0]

    def test_reused_graph_output_group_is_output(self):
        # [FUNCTION DOC] Checks that when the reused buffer name is among the graph output
        # names, the single group lists both buffer names in order and is marked as output.
        group = self.compute_single_group(graph_outputs=("buf48",))
        self.assertEqual(group.names, ["buf16", "buf48"])
        self.assertTrue(group.is_output)

    def test_reused_scheduler_output_user_group_is_output(self):
        # [FUNCTION DOC] Checks that when no graph output names exist but the scheduler buffer
        # for the reused name has a user that is the output node, the group (both names) is
        # marked as output.
        scheduler_buffers = {
            "buf48": SimpleNamespace(
                users=[SimpleNamespace(node=self.FakeOutputNode())],
            )
        }
        group = self.compute_single_group(scheduler_buffers=scheduler_buffers)
        self.assertEqual(group.names, ["buf16", "buf48"])
        self.assertTrue(group.is_output)


@requires_gpu()
@config.patch(memory_planning=True)
class TestMemoryPlanning(TestCase):
    # [CLASS DOC] GPU tests with memory planning enabled that compile a small module with
    # several simultaneously live intermediates and check the pool allocation code emitted by
    # the Python wrapper, C++ wrapper, and AOTI, using the GPU device type.
    device = GPU_TYPE

    def _generate(self, *, device):
        # [FUNCTION DOC] HELPER: Builds the test case: a module whose forward does several
        # matmuls of x with y, z and intermediate results and returns the sum of two
        # intermediates' sums, plus random inputs of shapes 3x2, 2x4 and 2x3 on the given
        # device. Returns the module and the tuple of three inputs.
        """
        Generate a simple test case that has multiple simultaneously-live intermediate tensors.
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing matmuls among x, y and z, with a transposed
            # matmul, and returning the sum of two resulting tensors' element sums.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Takes x, y, z; computes four matmul products reusing
                # intermediates and returns the sum of the elements of the last two.
                t0 = x.matmul(y)
                t1 = x.matmul(z)
                t0 = x.transpose(0, 1).matmul(t1)
                t1 = x.matmul(t0)
                return t0.sum() + t1.sum()

        x = torch.randn((3, 2), device=device)
        y = torch.randn((2, 4), device=device)
        z = torch.randn((2, 3), device=device)
        return (Foo(), (x, y, z))

    def test_python_wrapper(self):
        # [FUNCTION DOC] Compiles the module with dynamic shapes, captures generated wrapper
        # code, and uses FileCheck to confirm a pool is created with a symbolic size followed by
        # pool-based allocation of the first two buffers; also checks the compiled result
        # matches eager.
        f, args = self._generate(device=GPU_TYPE)
        compiled = torch.compile(f, dynamic=True)
        result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "pool1 = empty_strided_"
            + GPU_TYPE
            + "((4*s27*s77 + align(4*s77*s77), ), (1, )"
        ).check_next(
            "buf0 = alloc_from_pool(pool1, 0, torch.float32, (s77, s77), (s77, 1))"
        ).check("buf1 = alloc_from_pool(pool1, align(4*s77*s77),").run(code)
        self.assertTrue(same(f(*args), result))

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180122")
    def test_cpp_wrapper(self):
        # [FUNCTION DOC] Same as the Python wrapper test but with the C++ wrapper config enabled
        # during the run: checks the generated code allocates from the pool via the AOTI call
        # and wraps tensor handles in RAII handles, and results match eager. Skipped on ROCm
        # because of a known issue.
        f, args = self._generate(device=GPU_TYPE)
        compiled = torch.compile(f, dynamic=True)
        with config.patch({"cpp_wrapper": True}):
            result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "aoti_torch__alloc_from_pool(pool1, 0, cached_torch_dtype_float32, 2, int_array_2, int_array_3, &tmp_tensor_handle_0)"
        ).check_next("auto buf0 = RAIIAtenTensorHandle(tmp_tensor_handle_0);").check(
            "auto buf1 = RAIIAtenTensorHandle(tmp_tensor_handle_1);"
        ).run(code)
        self.assertTrue(same(f(*args), result))

    def test_aoti(self):
        # [FUNCTION DOC] Compiles and runs the module through the AOTI runner with the first
        # dimension of x dynamic, and uses FileCheck to confirm a consecutive sequence of pool
        # size and stride arrays, pool tensor creation, RAII wrapping and the first pool
        # allocation; checks the result matches eager.
        f, args = self._generate(device=GPU_TYPE)
        dim0_x = Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = ({0: dim0_x}, None, None)
        result, code = run_and_get_cpp_code(
            lambda: AOTIRunnerUtil.run(f, args, dynamic_shapes=dynamic_shapes)
        )

        FileCheck().check(
            "int64_t int_array_0[] = {24L + align(12L*s6), };"
        ).check_next("int64_t int_array_1[] = {1L, };").check_next(
            "AtenTensorHandle pool1_handle;"
        ).check_next(
            "aoti_torch_empty_strided(1, int_array_0, int_array_1,"
        ).check_next("RAIIAtenTensorHandle pool1(pool1_handle);").check_next(
            "int64_t int_array_2[] = {s6, 3L};"
        ).check_next("int64_t int_array_3[] = {3L, 1L};").check_next(
            "AtenTensorHandle tmp_tensor_handle_0;"
        ).check_next("aoti_torch__alloc_from_pool(pool1, 0").run(code)
        self.assertTrue(same(f(*args), result))

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/168171",
    )
    @config.patch({"triton.autotune_at_compile_time": False})
    def test_unbacked_symint(self):
        # [FUNCTION DOC] Runs a module with an unbacked symbolic size (from reading a scalar
        # tensor) through AOTI and checks the generated code: the pool is allocated only after
        # the unbacked symbol is defined, shape and stride arrays used by the pool and the
        # allocation are declared before use with the expected symbolic expressions, each pool
        # allocation is followed by a RAII handle (counts checked), and results match eager.
        # Skipped on Linux, ROCm and slow runs per the decorator (a known issue link), and also
        # skips when the device is not the GPU.
        # KNOWN ISSUE: the skip condition uses a logical or including Linux, so the test is
        # skipped on every Linux machine, which is likely unintended.
        # when allocation's size has unbacked symints
        # the unbacked symints are only available after computed
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds one to x, reads it as a scalar, builds a tensor
            # whose first-dimension-dependent size is the product of that scalar and the length of
            # y, fills it with sevens, and returns it times three.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the filled empty-strided tensor, sized by the
                # unbacked scalar times the length of y, multiplied by three; asserts the scalar is
                # at least one.
                x = x + 1
                u0 = x.item()
                torch._check(u0 >= 1)
                s0 = y.size(0)
                expr = u0 * s0
                sevens = torch.empty_strided(
                    size=(10, expr, 32), stride=(expr * 32, 32, 1), device=x.device
                ).fill_(7)
                return sevens * 3

        example_inputs = (
            torch.scalar_tensor(2, dtype=torch.int, device=self.device),
            torch.ones(8, device=self.device),
        )
        model = Repro().to(self.device)
        result, code = run_and_get_cpp_code(
            lambda: AOTIRunnerUtil.run(model, example_inputs)
        )
        self.assertTrue(same(model(*example_inputs), result))

        def find_code(pattern, pos=0):
            # [FUNCTION DOC] HELPER: Searches the generated code from a start position for
            # a regular expression; fails the test with a message if absent, otherwise
            # returns the match object.
            match = re.compile(pattern).search(code, pos)
            if match is None:
                self.fail(f"Expected to find {pattern!r} in generated code")
            return match

        # Check allocation is done after the unbacked symint is computed. The
        # exact int_array_N names are not stable, so capture the emitted names
        # and verify the calls use the matching shape/stride arrays.
        u0_match = find_code(r"auto u0 = u0_raw;")
        pool_shape = find_code(
            r"const int64_t (?P<name>int_array_\d+)\[\] = \{10L, 8L\*u0, 32L\};",
            u0_match.end(),
        )
        pool_stride = find_code(
            r"const int64_t (?P<name>int_array_\d+)\[\] = \{256L\*u0, 32L, 1L\};",
            pool_shape.end(),
        )
        pool_handle = find_code(r"AtenTensorHandle pool0_handle;", pool_stride.end())
        empty_strided = find_code(
            rf"aoti_torch_empty_strided\(3, {pool_shape.group('name')}, {pool_stride.group('name')},",
            pool_handle.end(),
        )
        pool0_raii = find_code(
            r"RAIIAtenTensorHandle pool0\(pool0_handle\);", empty_strided.end()
        )

        # all AtenTensorHandle allocated using aoti_torch__alloc_from_pool are wrapped with RAIIAtenTensorHandle
        # otherwise we'll have memory leak
        FileCheck().check_count(
            "aoti_torch__alloc_from_pool(pool1", 1, exactly=True
        ).check_count("aoti_torch__alloc_from_pool(pool0", 1, exactly=True).run(code)

        array_decls = {
            match.group("name"): match
            for match in re.finditer(
                r"const int64_t (?P<name>int_array_\d+)\[\] = \{(?P<value>[^}]+)\};",
                code,
            )
        }
        pool1_alloc = find_code(
            r"AOTI_TORCH_ERROR_CODE_CHECK\(aoti_torch__alloc_from_pool\(pool1, 0, cached_torch_dtype_int32, 0, int_array_\d+, int_array_\d+, &(?P<handle>tmp_tensor_handle_\d+)\)\);"
        )
        pool1_raii = find_code(
            rf"RAIIAtenTensorHandle\({pool1_alloc.group('handle')}\);",
            pool1_alloc.end(),
        )
        pool0_alloc = find_code(
            r"AOTI_TORCH_ERROR_CODE_CHECK\(aoti_torch__alloc_from_pool\(pool0, 0, cached_torch_dtype_float32, 3, (?P<shape>int_array_\d+), (?P<stride>int_array_\d+), &(?P<handle>tmp_tensor_handle_\d+)\)\);",
            max(pool0_raii.end(), pool1_raii.end()),
        )
        alloc_shape = pool0_alloc.group("shape")
        alloc_stride = pool0_alloc.group("stride")
        self.assertIn(alloc_shape, array_decls)
        self.assertIn(alloc_stride, array_decls)
        self.assertLess(array_decls[alloc_shape].end(), pool0_alloc.start())
        self.assertLess(array_decls[alloc_stride].end(), pool0_alloc.start())
        self.assertEqual(array_decls[alloc_shape].group("value"), "10L, 8L*u0, 32L")
        self.assertEqual(array_decls[alloc_stride].group("value"), "256L*u0, 32L, 1L")
        find_code(
            rf"RAIIAtenTensorHandle\({pool0_alloc.group('handle')}\);",
            pool0_alloc.end(),
        )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
