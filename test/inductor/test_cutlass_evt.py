# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the CUTLASS epilogue visitor tree (EVT) support in Inductor. Uses
# mock IR buffers and scheduler nodes to check the Python EVT code generated from pointwise
# epilogue IR (reads, writes, renames and source text), the creation of CUTLASS example
# tensors, and the rendered C++ EVT argument/type code. Optional cutlass imports and module-
# level fixtures (example tensors, epilogue source, mock tile description) are only defined
# when cutlass is importable.
import unittest

import sympy

import torch
from torch._dynamo.test_case import TestCase
from torch._inductor.codegen.cutlass.utils import (
    cutlass_arch,
    torch_dtype_to_cutlass_type,
    try_import_cutlass,
)
from torch._inductor.ir import ComputedBuffer, FixedLayout, PermuteView, Pointwise
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import OrderedSet
from torch.testing._internal.common_cuda import IS_SM10X, IS_SM90, SM90OrLater
from torch.testing._internal.common_device_type import skipCUDAIf, skipXPUIf
from torch.testing._internal.common_xpu import Xe2_Or_Later
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU_AND_TRITON,
    MockGraphHandler,
)


def _is_cuda_sm90_or_sm10x():
    # [FUNCTION DOC] HELPER: Returns True when the CUDA device is SM90 or an SM10x part, based
    # on the imported capability flags.
    return bool(IS_SM90) or bool(IS_SM10X)


def _cuda_evt_arch_supported():
    # [FUNCTION DOC] HELPER: Reports whether the CUTLASS EVT tracer supports the current GPU
    # arch. Always True for non-CUDA devices; for CUDA, imports the cutlass_cppgen compute-
    # capability map and checks that the detected arch number is a key in it.
    if GPU_TYPE != "cuda":
        return True

    from cutlass_cppgen.backend.evt.passes.util import cc_map

    return int(cutlass_arch(GPU_TYPE)) in cc_map


if try_import_cutlass():
    import cutlass_library as cutlass_lib
    from cutlass_library import EpilogueScheduleType

    LayoutType = cutlass_lib.LayoutType
    DataType = cutlass_lib.DataType
    from cutlass_cppgen.backend.evt.ir.tensor import Tensor as CutlassTensor

    from torch._inductor.codegen.cutlass.lib_extensions.evt_extensions import (
        _render_argument_type,
        _trace,
        trace,
    )

    BIAS_CODE = """def example_epilogue(accum, C, aux, bias):
        F = accum + C + aux
        E = relu(F) + bias
        D = E + F
        return D, F"""

    TYPE_C = DataType.f32
    M = 4224
    N = 2048
    BIAS = CutlassTensor(shape=(M, 1), element=TYPE_C, layout_tag=LayoutType.RowMajor)

    EXAMPLE_TENSORS = {
        "accum": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "bias": BIAS,
        # "beta": 0.5, TODO: mlazos support scalars
        # "alpha": 0.5, TODO: mlazos support scalars
        "D": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "C": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "F": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "aux": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
    }

    # For SM100
    class MockMathInstruction:
        # [CLASS DOC] HELPER: Minimal stand-in for a CUTLASS math instruction; only carries an
        # opcode class.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Sets the opcode class attribute to the tensor-op opcode
            # class of the cutlass library.
            self.opcode_class = cutlass_lib.library.OpcodeClass.TensorOp

    class MockTileDescription:
        # [CLASS DOC] HELPER: Minimal stand-in for a CUTLASS tile description used by the SM100
        # path: fixed threadblock shape, unit cluster shape and a mock math instruction (class
        # attributes only).
        threadblock_shape = (128, 128, 8)
        # SM100 path
        cluster_shape = (1, 1, 1)
        math_instruction = MockMathInstruction()

    def _create_mock_buffer_name_map(example_tensors):
        # [FUNCTION DOC] HELPER: Builds a mapping from tensor name to a mock computed buffer,
        # one for each entry of the input mapping whose value is a CUTLASS tensor. Each buffer
        # is float32 and copies the tensor shape and stride; other entries are skipped. Returns
        # the mapping.
        name_to_buffer = {}
        for name, tensor in example_tensors.items():
            if isinstance(tensor, CutlassTensor):
                name_to_buffer[name] = MockComputedBuffer(
                    name, None, torch.float32, tensor.shape, tensor.stride
                )

        return name_to_buffer


class MockSchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] HELPER: Minimal scheduler node wrapping a single IR node plus the set of buffer
    # names whose last use is this node. Bypasses the base-class constructor.
    def __init__(self, node, last_usage=None):
        # [FUNCTION DOC] HELPER: Stores the wrapped node and its last-usage set (an empty
        # ordered set when none is given). Does not call the parent initializer.
        self.node = node
        self.last_usage = last_usage or OrderedSet()


class MockComputedBuffer(ComputedBuffer):
    # [CLASS DOC] HELPER: Lightweight computed buffer for tests. Holds a name, a pointwise data
    # node built from the given inner function and sizes (no device), and a fixed layout with
    # the given strides. Reports one read so it is never inlined into consumers.
    def __init__(self, name, inner_fn, dtype, size, strides=None):
        # [FUNCTION DOC] HELPER: Initializes name, a pointwise data node (integer ranges from
        # size, given dtype and inner function) and a fixed layout (no device, optional
        # strides). No parent init.
        self.name = name
        ranges = [sympy.Integer(x) for x in size]
        self.data = Pointwise(
            device=None, dtype=dtype, inner_fn=inner_fn, ranges=ranges
        )
        self.layout = FixedLayout(None, dtype, ranges, strides)

    def get_name(self):
        # [FUNCTION DOC] HELPER: Returns the buffer name given at construction.
        return self.name

    def num_reads(self):
        # [FUNCTION DOC] HELPER: Always returns one, which keeps the buffer from being inlined.
        # Needed to not inline in ComputedBuffer
        return 1


class TestCutlassEVT(TestCase):
    # [CLASS DOC] Test case for CUTLASS EVT codegen. Tests are guarded by skip decorators for
    # XPU platform support, CUDA SM90 (or SM90/SM10x) capability and cutlass availability.
    # device_type is the GPU type of the test environment.
    device_type = GPU_TYPE

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_accumulator_return(self):
        # [FUNCTION DOC] Checks Python EVT codegen when a consumer reads the accumulator buffer
        # and the accumulator itself is also returned. Builds two chained pointwise mock buffers
        # over a mock graph handler, then compares reads, writes, renames and generated source
        # to expected text. The accumulator appears among the writes and a temp alias is renamed
        # to it.
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the third buffer: loads the accumulator
            # and two external buffers at the index and returns the product of the first two
            # plus the third.
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the fourth buffer: returns the sum of
            # the accumulator load and the load of the third buffer at the index.
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4, last_usage=OrderedSet(["buf3"])),
                ],
                OrderedSet([]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf0', 'buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'tmp_0': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_2': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum
    tmp_1 = tmp_0 * buf1
    tmp_2 = tmp_1 + buf2
    D = tmp_0 + tmp_2

return tmp_0, tmp_2, D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_disjoint_read_indexing(self):
        # [FUNCTION DOC] Checks that EVT codegen rejects a read of the accumulator through a
        # permuted view. The first buffer is read through a permutation, then codegen is run and
        # the raised not-implemented error is captured; its message (unsupported indexing with
        # index strides and layout stride) is compared to expected text.
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        permuted_buf_0 = PermuteView.create(buf0, [1, 0, 2])
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the third buffer: loads the permuted
            # accumulator view and two external buffers, returning the product of the first two
            # plus the third.
            tmp0 = permuted_buf_0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the fourth buffer: returns the sum of
            # the unpermuted accumulator load and the third buffer load.
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)

        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            result = None
            try:
                CutlassEVTCodegen.ir_to_evt_python_code(
                    "buf0",
                    [MockSchedulerNode(buf3), MockSchedulerNode(buf4)],
                    OrderedSet([]),
                )
            except NotImplementedError as e:
                result = e

            self.assertExpectedInline(
                str(result),
                """Unsupported indexing for buf0 with index 200*i0 + 60000*i1 + i2, \
index strides [200, 60000, 1], and layout stride [60000, 200, 1]""",
            )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_broadcasting(self):
        # [FUNCTION DOC] Checks Python EVT codegen with a broadcast output: the last buffer has
        # a trailing dimension of one and squares the third buffer. Compares reads, writes,
        # renames and the generated source (a multiply used twice, then the add with the
        # accumulator) to expected text.
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the third buffer: product of accumulator
            # and first external load plus the second external load.
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the fourth buffer: accumulator load plus
            # the square of the third buffer load.
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3 * tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer(
            "buf4", inner_fn_buf4, torch.float32, (100, 300, 1)
        )  # broadcast
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4, last_usage=OrderedSet(["buf0"])),
                ],
                OrderedSet([]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf0', 'buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'tmp_0': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_2': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum
    tmp_1 = tmp_0 * buf1
    tmp_2 = tmp_1 + buf2
    tmp_3 = tmp_2 * tmp_2
    D = tmp_0 + tmp_3

return tmp_0, tmp_2, D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen(self):
        # [FUNCTION DOC] Baseline Python EVT codegen test where the accumulator is not an output
        # (it is passed in the removed-buffers set). Verifies reads, writes (only third and
        # fourth buffers), renames and the generated function text, where the accumulator is
        # used directly.
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the third buffer: accumulator times
            # first external load, plus the second external load.
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            # [FUNCTION DOC] HELPER: Pointwise body for the fourth buffer: accumulator load plus
            # third buffer load.
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4),
                ],
                OrderedSet(["buf0"]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_1': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum * buf1
    tmp_1 = tmp_0 + buf2
    D = accum + tmp_1

return tmp_1, D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_neg_constant(self):
        # [FUNCTION DOC] Checks EVT codegen for unary negation and constant ops as produced by
        # decomposed SiLU: one plus the exponential of the negated accumulator. Verifies no
        # external reads, a single write, and that negation is emitted as subtraction from zero
        # in the generated source.
        """Test EVT codegen for neg and constant ops (used in decomposed SiLU)."""
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import ops, V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body: loads the accumulator and returns a float
            # constant of one plus the exponential of the negated load.
            x = buf0.make_loader()(index)
            c = ops.constant(1, torch.float32)
            return c + ops.exp(ops.neg(x))

        buf1 = MockComputedBuffer("buf1", inner_fn, torch.float32, size)
        with V.set_graph_handler(MockGraphHandler({"buf0": buf0, "buf1": buf1})):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [MockSchedulerNode(buf1)],
                OrderedSet(["buf0"]),
            )
        self.assertExpectedInline(reads, """[]""")
        self.assertExpectedInline(writes, """['buf1']""")
        self.assertExpectedInline(
            code,
            """\
def fn(accum):
    tmp_0 = 1.0
    tmp_1 = (0.0 - accum)
    tmp_2 = exp(tmp_1)
    D = tmp_0 + tmp_2

return D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_sigmoid_decomposed(self):
        # [FUNCTION DOC] Checks that a decomposed sigmoid (one divided by one plus exp of
        # negation) is not folded into an activation and stays as separate negate, exp, add and
        # divide steps in the generated source; also verifies empty reads and a single write.
        """Test EVT codegen for decomposed sigmoid: 1/(1+exp(-x))."""
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import ops, V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body computing one divided by the sum of one and
            # the exponential of the negated accumulator load.
            x = buf0.make_loader()(index)
            neg_x = ops.neg(x)
            exp_neg = ops.exp(neg_x)
            one = ops.constant(1, torch.float32)
            denom = one + exp_neg
            return ops.truediv(one, denom)

        buf1 = MockComputedBuffer("buf1", inner_fn, torch.float32, size)
        with V.set_graph_handler(MockGraphHandler({"buf0": buf0, "buf1": buf1})):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [MockSchedulerNode(buf1)],
                OrderedSet(["buf0"]),
            )
        self.assertExpectedInline(reads, """[]""")
        self.assertExpectedInline(writes, """['buf1']""")
        # Sigmoid doesn't match _fuse_activations (not x/denom form with x in num)
        # so it stays decomposed
        self.assertExpectedInline(
            code,
            """\
def fn(accum):
    tmp_0 = (0.0 - accum)
    tmp_1 = exp(tmp_0)
    tmp_2 = 1.0
    tmp_3 = tmp_2 + tmp_1
    D = tmp_2 / tmp_3

return D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_silu_fused(self):
        # [FUNCTION DOC] Checks that the SiLU pattern (x divided by one plus exp of negated x)
        # is fused by the activation-fusing pass into a single silu call; the generated source
        # is only that call plus the return.
        """Test EVT codegen for SiLU: x/(1+exp(-x)) is folded to silu(x)."""
        from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import ops, V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body computing the accumulator load divided by
            # the sum of one and the exponential of its negation.
            x = buf0.make_loader()(index)
            neg_x = ops.neg(x)
            exp_neg = ops.exp(neg_x)
            one = ops.constant(1, torch.float32)
            denom = one + exp_neg
            return ops.truediv(x, denom)

        buf1 = MockComputedBuffer("buf1", inner_fn, torch.float32, size)
        with V.set_graph_handler(MockGraphHandler({"buf0": buf0, "buf1": buf1})):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [MockSchedulerNode(buf1)],
                OrderedSet(["buf0"]),
            )
        self.assertExpectedInline(reads, """[]""")
        self.assertExpectedInline(writes, """['buf1']""")
        # _fuse_activations folds x/(1+exp(0.0-x)) into silu(x)
        self.assertExpectedInline(
            code,
            """\
def fn(accum):
    D = silu(accum)
return D""",
        )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_example_tensor_creation(self):
        # [FUNCTION DOC] Checks create_example_tensors: with two mock buffers (row-major and
        # column-major style strides) and a rename map aliasing the accumulator to the first
        # buffer, the returned tensors carry the buffer shapes, strides and the float32 CUTLASS
        # element type.
        from torch._inductor.codegen.cutlass.lib_extensions.evt_extensions import (
            create_example_tensors,
        )
        from torch._inductor.virtualized import V

        with V.set_graph_handler(MockGraphHandler({})):
            row_major_buf0 = MockComputedBuffer(
                "buf0", None, torch.float32, (3, 4, 1), (4, 1, 0)
            )
            col_major_buf1 = MockComputedBuffer(
                "buf1", None, torch.float32, (3, 2, 1), (1, 3, 0)
            )
            buffer_renames = {"buf0": "buf0", "buf1": "buf1", "acc": "buf0"}
            name_to_buffer = {"buf0": row_major_buf0, "buf1": col_major_buf1}
            result = create_example_tensors(
                buffer_renames, name_to_buffer, lambda x: int(x)
            )
            self.assertEqual(result["acc"].shape, (3, 4, 1))
            self.assertEqual(result["acc"].stride, (4, 1, 0))
            self.assertEqual(
                result["acc"].element, torch_dtype_to_cutlass_type(torch.float32)
            )

            self.assertEqual(result["buf1"].shape, (3, 2, 1))
            self.assertEqual(result["buf1"].stride, (1, 3, 0))
            self.assertEqual(
                result["buf1"].element, torch_dtype_to_cutlass_type(torch.float32)
            )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not _is_cuda_sm90_or_sm10x(), "need sm_90 or sm_10x")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_argument_codegen(self):
        # [FUNCTION DOC] Checks the rendered C++ EVT argument struct for the module-level bias
        # epilogue source. Traces the epilogue for the detected arch, renders the argument type
        # with the mock name map and compares to expected text, which differs on XPU (extra null
        # default for the stored aux). Skips when the arch is unsupported by the EVT tracer.
        arch = int(cutlass_arch(GPU_TYPE))

        def render_code():
            # [FUNCTION DOC] HELPER: Traces the shared epilogue code for the detected arch and
            # returns the first element of the rendered argument type result (the argument
            # source string).
            epilogue_functor = _trace(BIAS_CODE, EXAMPLE_TENSORS, arch)
            return _render_argument_type(
                epilogue_functor,
                _create_mock_buffer_name_map(EXAMPLE_TENSORS),
                lambda x: int(x),
            )[0]

        if not _cuda_evt_arch_supported():
            self.skipTest(f"CUTLASS EVT does not support arch {arch}")

        code = render_code()
        if GPU_TYPE == "xpu":
            self.assertExpectedInline(
                code,
                """\
{ /* thread */
        { /* F */
          { /* compute_1 */
            { /* compute_0 */
              {}, /* accum */
              {}, /* C */
              {}, /* compute_0 */
            },
            {/* ptr_aux */ (float*) (ptr_0 + ptr_0_offset), /* null_default */ float(0), /* dAux */ {2048, _1{}, _0{}}}, /* aux */
            {}, /* compute_1 */
          },
          {/* ptr_aux */ (float*) (ptr_1 + ptr_1_offset), /* null_default */ float(0), /* dAux */ {2048, _1{}, _0{}}}, /* F */
        },
        {/* ptr_col */ (float*) (ptr_2 + ptr_2_offset), /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_2 */
        {}, /* compute_3 */
        {}, /* compute_4 */
      }
""",
            )
        else:
            self.assertExpectedInline(
                code,
                """\
{ /* thread */
        { /* F */
          { /* compute_1 */
            { /* compute_0 */
              {}, /* accum */
              {}, /* C */
              {}, /* compute_0 */
            },
            {/* ptr_aux */ (float*) (ptr_0 + ptr_0_offset), /* null_default */ float(0), /* dAux */ {2048, _1{}, _0{}}}, /* aux */
            {}, /* compute_1 */
          },
          {/* ptr_aux */ (float*) (ptr_1 + ptr_1_offset), /* dAux */ {2048, _1{}, _0{}}}, /* F */
        },
        {/* ptr_col */ (float*) (ptr_2 + ptr_2_offset), /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_2 */
        {}, /* compute_3 */
        {}, /* compute_4 */
      }
""",
            )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not _is_cuda_sm90_or_sm10x(), "need sm_90 or sm_10x")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_argument_codegen_return_accumulator(self):
        # [FUNCTION DOC] Same as the argument-codegen test but for an epilogue that returns the
        # accumulator-derived value as well as the sum with bias. Builds its own example
        # tensors, renders the argument type and compares to expected text (differs for XPU).
        # Skips on unsupported arch.
        arch = int(cutlass_arch(GPU_TYPE))

        code = """
def fn(accum, bias):
    E = accum
    D = E + bias
    return D, E
"""
        example_tensors = {
            "accum": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
            "bias": BIAS,
            # "beta": 0.5, TODO: mlazos support scalars
            # "alpha": 0.5, TODO: mlazos support scalars
            "D": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
            "E": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
        }

        def render_code():
            # [FUNCTION DOC] HELPER: Traces the local epilogue source with its own example
            # tensors for the detected arch and returns the first element of the rendered
            # argument type result.
            epilogue_functor = _trace(code, example_tensors, arch)
            return _render_argument_type(
                epilogue_functor,
                _create_mock_buffer_name_map(example_tensors),
                lambda x: int(x),
            )[0]

        if not _cuda_evt_arch_supported():
            self.skipTest(f"CUTLASS EVT does not support arch {arch}")

        code = render_code()

        if GPU_TYPE == "xpu":
            self.assertExpectedInline(
                code,
                """\
{ /* thread */
        { /* E */
          {}, /* accum */
          {/* ptr_aux */ (float*) (ptr_0 + ptr_0_offset), /* null_default */ float(0), /* dAux */ {2048, _1{}, _0{}}}, /* E */
        },
        {/* ptr_col */ (float*) (ptr_1 + ptr_1_offset), /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_0 */
      }
""",
            )
        else:
            self.assertExpectedInline(
                code,
                """\
{ /* thread */
        { /* E */
          {}, /* accum */
          {/* ptr_aux */ (float*) (ptr_0 + ptr_0_offset), /* dAux */ {2048, _1{}, _0{}}}, /* E */
        },
        {/* ptr_col */ (float*) (ptr_1 + ptr_1_offset), /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_0 */
      }
""",
            )

    @skipXPUIf(not Xe2_Or_Later, "Unsupported platform")
    @skipCUDAIf(not _is_cuda_sm90_or_sm10x(), "need sm_90 or sm_10x")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_codegen(self):
        # [FUNCTION DOC] Checks the full EVT type code from the trace entry point for the bias
        # epilogue with float32 types, the mock tile description and automatic epilogue
        # schedule. Takes the third result element and compares to expected C++ text for SM90,
        # SM10x and XPU (each branch applies only on its platform). Skips on unsupported arch.
        def render_code():
            # [FUNCTION DOC] HELPER: Runs the trace entry point with the bias epilogue, example
            # tensors, float32 types, mock tile, auto schedule, mock name map, an identity size
            # hint function and the device type, and returns the third element of the result
            # (the C++ code).
            return trace(
                BIAS_CODE,
                EXAMPLE_TENSORS,
                DataType.f32,
                DataType.f32,
                MockTileDescription(),
                EpilogueScheduleType.ScheduleAuto,
                _create_mock_buffer_name_map(EXAMPLE_TENSORS),
                lambda x: x,  # static shapes
                device_type=GPU_TYPE,
            )[2]

        if not _cuda_evt_arch_supported():
            self.skipTest(f"CUTLASS EVT does not support arch {cutlass_arch(GPU_TYPE)}")

        code = render_code()
        if IS_SM90:
            self.assertExpectedInline(
                code,
                """\

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_128, _128, _8>, cutlass::epilogue::collective::EpilogueTileAuto,
  float, float,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using ElementC = float;
using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using TensorC = cutlass::epilogue::fusion::Sm90SrcFetch<float>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using AuxDescriptor = cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, \
cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float>;

using Aux = cutlass::epilogue::fusion::Sm90AuxLoad<
    AuxDescriptor::Stages, typename AuxDescriptor::EpilogueTile, float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename AuxDescriptor::SmemLayoutAtom, \
typename AuxDescriptor::CopyOpS2R
>;

using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    Accum,
    TensorC>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    EVTCompute0,
    Aux>;

using FDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float
>;

using F = cutlass::epilogue::fusion::Sm90AuxStore<
    FDescriptor::Stages, typename FDescriptor::EpilogueTile, float,
    cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::Int<1>, \
cute::Int<0>>, typename FDescriptor::SmemLayoutAtom,
    typename FDescriptor::CopyOpR2S
>;

using EVTF = cutlass::epilogue::fusion::Sm90EVT<
    F,
    EVTCompute1>;

using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute3 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute4 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute4 = cutlass::epilogue::fusion::Sm90TopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<0>,
        cute::seq<2, 1>,
        cute::seq<3, 0>,
    >,
    EVTF,
    Bias,
    Compute2,
    Compute3,
    Compute4
>;

using ElementD = float;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

""",
            )

        if IS_SM10X:
            self.assertExpectedInline(
                code,
                """\

using EpilogueDescriptor = cutlass::epilogue::collective::detail::Sm100EpilogueDescriptor<
  cutlass::arch::OpClassTensorOp,
  cute::Shape<_128, _128, _8>, cutlass::epilogue::collective::EpilogueTileAuto,
  float, float, float,
  cutlass::epilogue::collective::EpilogueScheduleAuto, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>,
  false /* IsPerColScaleSupported */,
  false /* IsBlockScaleSupported */
>;

using ElementC = float;
using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using TensorC = cutlass::epilogue::fusion::Sm90SrcFetch<float>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using AuxDescriptor = cutlass::epilogue::collective::detail::Sm100AuxLoadDescriptor<EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float>;

using Aux = cutlass::epilogue::fusion::Sm90AuxLoad<
    AuxDescriptor::Stages, typename AuxDescriptor::EpilogueTile, float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename AuxDescriptor::SmemLayoutAtom, typename AuxDescriptor::CopyOpS2R
>;

using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    Accum,
    TensorC>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    EVTCompute0,
    Aux>;

using FDescriptor = cutlass::epilogue::collective::detail::Sm100AuxStoreDescriptor<
    EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float
>;

using F = cutlass::epilogue::fusion::Sm90AuxStore<
    FDescriptor::Stages, typename FDescriptor::EpilogueTile, float,
    cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename FDescriptor::SmemLayoutAtom,
    typename FDescriptor::CopyOpR2S
>;

using EVTF = cutlass::epilogue::fusion::Sm90EVT<
    F,
    EVTCompute1>;

using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute3 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute4 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute4 = cutlass::epilogue::fusion::Sm90TopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<0>,
        cute::seq<2, 1>,
        cute::seq<3, 0>,
    >,
    EVTF,
    Bias,
    Compute2,
    Compute3,
    Compute4
>;

using ElementD = float;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

""",
            )
        if GPU_TYPE == "xpu":
            self.assertExpectedInline(
                code,
                """\

using TileShape_MNK = cute::Shape<_128, _128, _8>;

using ElementC = float;
using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using TensorC = cutlass::epilogue::fusion::XeSrcFetch<float>;

using Accum = cutlass::epilogue::fusion::XeAccFetch;

using Aux = cutlass::epilogue::fusion::XeAuxLoad<
    float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
>;

using Bias = cutlass::epilogue::fusion::XeColBroadcast<
    0 /*Stages*/, TileShape_MNK, float, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::XeEVT<
    Compute0,
    Accum,
    TensorC>;

using Compute1 = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::XeEVT<
    Compute1,
    EVTCompute0,
    Aux>;

using F = cutlass::epilogue::fusion::XeAuxStore<
    float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
>;

using EVTF = cutlass::epilogue::fusion::XeEVT<
    F,
    EVTCompute1>;

using Compute2 = cutlass::epilogue::fusion::XeCompute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute3 = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute4 = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute4 = cutlass::epilogue::fusion::XeTopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<0>,
        cute::seq<2, 1>,
        cute::seq<3, 0>
    >,
    EVTF,
    Bias,
    Compute2,
    Compute3,
    Compute4
>;

using ElementD = float;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

""",
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
