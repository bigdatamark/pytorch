# Owner(s): ["oncall: export"]
    # [MODULE DOC] Tests for torch._export.converter.TS2EPConverter, which converts a
    # TorchScript module (from jit trace or jit script) into an ExportedProgram. Defines the
    # skip decorators for CUDA and for prepacked linear (needs XNNPACK ops) and a single test
    # class whose tests build small modules exercising one TorchScript construct each, convert
    # them, and compare the converted program with the original TorchScript module. Import
    # time evaluates the availability checks for CUDA and the prepacked ops; running the file
    # starts the runner.

import unittest
from collections import OrderedDict
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter
from torch.export import ExportedProgram
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, xfailIfS390X
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")

# prepacked linear requires XNNPACK support.
requires_prepacked_linear = unittest.skipIf(
    not hasattr(torch.ops.prepacked, "linear_clamp_prepack"),
    "requires XNNPACK (prepacked linear ops)",
)


class TestConverter(TestCase):
    # [CLASS DOC] Test case for TorchScript to ExportedProgram conversion. setUp initializes
    # the torchbind test implementations and stores the queue operator handles in an
    # attribute; tearDown does nothing. Most tests delegate to the shared checker method that
    # traces and/or scripts a module, converts it, and compares state dict keys and outputs
    # against the TorchScript original, optionally over extra runtime inputs.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, initializes torchbind test implementations and
        # stores the queue pop, push and size operator handles in an attribute list.
        super().setUp()
        init_torchbind_implementations()

        self.torch_bind_ops = [
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_size,
        ]

    def tearDown(self):
        # [FUNCTION DOC] Overrides tearDown to do nothing.
        return

    def _check_equal_ts_ep_converter(
        self,
        M,
        tracing_inputs,
        option: list[str] | None = None,
        check_persistent=False,
        lifted_tensor_constants=None,
        runtime_inputs: list[Any] | None = None,
    ) -> list[ExportedProgram]:
        # [FUNCTION DOC] HELPER: Shared checker. Converts the given module with each requested
        # mode (default both trace and script; any other mode raises RuntimeError) into an
        # ExportedProgram using the converter. When the persistence flag is set, fresh module
        # instances are used for the original, the converted and the eager copies, and every
        # input is run ten times to catch stateful side effects. For every input (tracing
        # inputs plus optional runtime inputs) it flattens outputs of the original TorchScript
        # model and the converted module, asserts the program state dict keys equal the
        # TorchScript state dict keys plus any lifted tensor constants (only when the model is
        # a module), and compares outputs with the tensor list helper. Returns the list of
        # converted programs, one per mode.
        # By default, it tests both jit.trace and jit.script.
        if option is None:
            option = ["trace", "script"]

        if check_persistent:
            num_iterations = 10
        else:
            num_iterations = 1

        ep_list = []
        for opt in option:
            if opt == "script":
                # Separate two models for testing non-functional effects
                if check_persistent:
                    original_ts_model = torch.jit.script(M())
                    ts_model = torch.jit.script(M())
                    eager_model = M()
                else:
                    original_ts_model = torch.jit.script(M)
                    ts_model = torch.jit.script(M)
                    eager_model = M
            elif opt == "trace":
                if check_persistent:
                    original_ts_model = torch.jit.trace(M(), tracing_inputs)
                    ts_model = torch.jit.trace(M(), tracing_inputs)
                    eager_model = M()
                else:
                    original_ts_model = torch.jit.trace(M, tracing_inputs)
                    ts_model = torch.jit.trace(M, tracing_inputs)
                    eager_model = M
            else:
                raise RuntimeError(f"Unrecognized mode for torch.jit: {opt}")

            converter = TS2EPConverter(ts_model, tracing_inputs)
            ep = converter.convert()
            ep_list.append(ep)

            if runtime_inputs is None:
                runtime_inputs = []

            for inp in [tracing_inputs] + runtime_inputs:
                for _ in range(num_iterations):
                    orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))
                    ep_out, _ = pytree.tree_flatten(ep.module()(*inp))

                    # Check module.
                    if isinstance(eager_model, torch.nn.Module):
                        expected_state_dict = OrderedDict()
                        expected_state_dict.update(ts_model.state_dict())
                        if lifted_tensor_constants:
                            expected_state_dict.update(lifted_tensor_constants)
                        self.assertEqual(
                            ep.state_dict.keys(),
                            expected_state_dict.keys(),
                        )

                    # Check results
                    self._check_tensor_list_equal(ep_out, orig_out)
        return ep_list

    def _check_tensor_list_equal(self, xs: list[torch.Tensor], ys: list[torch.Tensor]):
        # [FUNCTION DOC] HELPER: Asserts two flat lists have equal length. For each pair of
        # items, tensors must have equal shape and be close in value; non-tensors must have
        # equal type and value.
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(torch.allclose(x, y))
            else:
                self.assertEqual(type(x), type(y))
                self.assertEqual(x, y)

    def test_ts2ep_converter_basic(self):
        # [FUNCTION DOC] Converts a module with a single output and a module with two outputs
        # (cos and sin shifted) in both modes, checking against extra runtime inputs of other
        # widths.
        class MSingle(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        class MMulti(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two tensors from its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the cosine of the first input plus one and
                # the sine of the second minus one, as a tuple.
                x = x.cos() + 1
                y = y.sin() - 1
                return x, y

        inp = (torch.ones(1, 3), torch.ones(1, 3))
        runtime_inps = [
            (torch.ones(1, 4), torch.ones(1, 4)),
            (torch.ones(1, 5), torch.ones(1, 5)),
        ]
        self._check_equal_ts_ep_converter(MSingle(), inp, runtime_inputs=runtime_inps)
        self._check_equal_ts_ep_converter(MMulti(), inp, runtime_inputs=runtime_inps)

    def test_ts2ep_converter_container_output(self):
        # [FUNCTION DOC] Checks conversion of modules returning a list, a tuple, and a nested
        # dict. List and dict outputs are only tested with scripting since traced functions
        # need immutable output structures; the tuple module uses the default modes.
        # Output is a List.
        class MOutputList(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of the square and the doubled input.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns a list holding x squared and y doubled.
                a = x * x
                b = y + y
                return [a, b]

        # Output is a Tuple.
        class MOutputTuple(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of the square and the doubled
            # input.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns a tuple holding x squared and y doubled.
                a = x * x
                b = y + y
                return (a, b)

        # Output is a Dict.
        class MOutputDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a nested dict with the same two values.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns a nested dictionary with a mul entry and an
                # add entry.
                a = x * x
                b = y + y
                return {"data": {"mul": a, "add": b}}

        inp = (torch.tensor(4), torch.tensor(4))
        runtime_inputs = [
            (torch.tensor(5), torch.tensor(5)),
            (torch.tensor(1), torch.tensor(1)),
        ]
        # Traced function must use immutable structure as output.
        self._check_equal_ts_ep_converter(
            MOutputList(), inp, ["script"], runtime_inputs=runtime_inputs
        )
        self._check_equal_ts_ep_converter(
            MOutputTuple(), inp, runtime_inputs=runtime_inputs
        )
        self._check_equal_ts_ep_converter(
            MOutputDict(), inp, ["script"], runtime_inputs=runtime_inputs
        )

    def test_aten_dim(self):
        # [FUNCTION DOC] Checks conversion of the tensor dim query, whose result is used as a
        # size, with an extra runtime input of a different rank-compatible shape.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a ones tensor with length equal to input
            # rank.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones of size equal to the number of dims of
                # x.
                num_dim = x.dim()
                return torch.ones(num_dim)

        inp = (torch.ones(1, 3),)
        self._check_equal_ts_ep_converter(
            Module(), inp, runtime_inputs=[(torch.ones(1, 5),)]
        )

    def test_aten_len(self):
        # [FUNCTION DOC] Checks conversion of the length builtin for different argument kinds:
        # tensor, list of ints, and dicts keyed by int, bool, float and tensor. Each case
        # defines a module of the same name returning ones of that length. Only the tensor
        # case also runs in trace mode; the rest are scripted. String keyed cases are left as
        # comments because jit flatten does not support strings.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of a tensor input.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns ones with length of the first dim of x.
                length = len(x)
                return torch.ones(length)

        # aten::len.Tensor
        inp = (torch.ones(2, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of a list of ints.
            def forward(self, x: list[int]):
                # [FUNCTION DOC] HELPER: Returns ones with length equal to the list length.
                length = len(x)
                return torch.ones(length)

        # aten::len.t
        inp = ([1, 2, 3],)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of an int keyed
            # dict.
            def forward(self, x: dict[int, str]):
                # [FUNCTION DOC] HELPER: Returns ones with length equal to the number of dict
                # entries.
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_int
        inp = ({1: "a", 2: "b", 3: "c"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of a bool keyed
            # dict.
            def forward(self, x: dict[bool, str]):
                # [FUNCTION DOC] HELPER: Returns ones with length equal to the number of dict
                # entries.
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_bool
        inp = ({True: "a", False: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of a float keyed
            # dict.
            def forward(self, x: dict[float, str]):
                # [FUNCTION DOC] HELPER: Returns ones with length equal to the number of dict
                # entries.
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_float
        inp = ({1.2: "a", 3.4: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning ones sized by the length of a tensor keyed
            # dict.
            def forward(self, x: dict[torch.Tensor, str]):
                # [FUNCTION DOC] HELPER: Returns ones with length equal to the number of dict
                # entries.
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_Tensor
        inp = ({torch.zeros(2, 3): "a", torch.ones(2, 3): "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # aten::len.str and aten::len.Dict_str are not supported
        # since torch._C._jit_flatten does not support str
        # inp = ("abcdefg",)
        # self._check_equal_ts_ep_converter(Module(), inp)
        # inp = ({"a": 1, "b": 2},)
        # self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten_add_t(self):
        # [FUNCTION DOC] Checks conversion of list concatenation by addition (scripted only):
        # a list input is added to an empty list twice, concatenating each time. Uses runtime
        # inputs with other tensor shapes.
        # python list append
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that concatenates the list input onto an empty list
            # twice and returns the tensor concatenations after each step.
            def forward(self, x: list[torch.Tensor]):
                # [FUNCTION DOC] HELPER: Returns the concatenation of the list once and of the
                # list repeated twice as two tensors.
                out = []
                out = out + x
                a = torch.cat(out)
                out = out + x
                b = torch.cat(out)
                return a, b

        inp = ([torch.ones(2, 3), torch.ones(2, 3)],)
        runtime_inputs = [
            ([torch.ones(4, 6), torch.ones(8, 6)],),
            ([torch.ones(4, 4), torch.ones(4, 4)],),
        ]
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], runtime_inputs=runtime_inputs
        )

    def test_aten_to_dtype_with_mutating_storage(self):
        # [FUNCTION DOC] Checks conversion when a tensor is cast to another dtype and then
        # mutated in place by an index put, in both modes with a tensor of different dtype as
        # second input.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module casting the first input to the dtype of the second
            # and writing the second into it through an in-place index put.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the cast tensor after the in-place index put.
                x = x.to(y.dtype)
                torch.ops.aten.index_put_(x, [torch.tensor([0])], y)
                return x

        inp = (torch.ones(2, 3), torch.tensor([0, 0, 0]))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_min(self):
        # [FUNCTION DOC] Checks conversion of the min builtin variants (int, float, list of
        # ints, list of floats, mixed float and int, mixed int and float) via lengths of two
        # inputs.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing six minimum variants of two input lengths
            # and returning ones with a length equal to their sum.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns ones with length the sum of the six computed
                # minimums.
                x_len = len(x)
                y_len = len(y)

                # prim::min.int
                len_int = min(x_len, y_len)

                # prim::min.float
                len_float = int(min(x_len * 2.0, y_len * 2.0))

                # prim::min.self_int
                len_self_int = min([x_len, y_len])

                # prim::min.self_float
                len_self_float = int(min([x_len * 2.0, y_len * 2.0]))

                # prim::min.float_int
                len_float_int = int(min(x_len * 2.0, y_len))

                # prim::min.int_float
                len_int_float = int(min(x_len, y_len * 2.0))

                return torch.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        inp = (torch.randn(10, 2), torch.randn(5))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_max(self):
        # [FUNCTION DOC] Checks conversion of the max builtin variants (int, float, list of
        # ints, list of floats, mixed float and int, mixed int and float) via lengths of two
        # inputs.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing six maximum variants of two input lengths
            # and returning ones with a length equal to their sum.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns ones with length the sum of the six computed
                # maximums.
                x_len = len(x)
                y_len = len(y)

                # prim::max.int
                len_int = max(x_len, y_len)

                # prim::max.float
                len_float = int(max(x_len * 2.0, y_len * 2.0))

                # prim::max.self_int
                len_self_int = max([x_len, y_len])

                # prim::max.self_float
                len_self_float = int(max([x_len * 2.0, y_len * 2.0]))

                # prim::max.float_int
                len_float_int = int(max(x_len * 2.0, y_len))

                # prim::max.int_float
                len_int_float = int(max(x_len, y_len * 2.0))

                return torch.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        inp = (torch.randn(10, 2), torch.randn(5))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten___getitem___list(self):
        # [FUNCTION DOC] Checks list indexing: converts a module that splits a tensor and
        # returns the first piece, with an extra runtime input of a wider shape.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the first chunk of splitting its input by
            # size two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the first piece of the split of x.
                y = torch.split(x, 2)
                return y[0]

        inp = (torch.rand((3, 2)),)
        runtime_inps = [(torch.rand((3, 8)),)]
        self._check_equal_ts_ep_converter(Module(), inp, runtime_inputs=runtime_inps)

    def test_aten___getitem___dict(self):
        # [FUNCTION DOC] Checks dict indexing with int, string, bool and float keys using
        # values from a split tensor.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building four dicts with different key types and
            # looking up one entry from each.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the looked up entries from the int, str, bool
                # and float keyed dicts as a tuple.
                y = torch.split(x, 2)
                d_int = {0: y[0], 1: y[1]}
                d_str = {"0": y[0], "1": y[1]}
                d_bool = {True: y[0], False: y[1]}
                d_float = {0.1: y[0], 2.3: y[1]}
                return d_int[0], d_str["0"], d_bool[True], d_float[0.1]

        inp = (torch.rand((3, 2)),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_device(self):
        # [FUNCTION DOC] Checks conversion of reading the device of a tensor and using it to
        # create a new tensor, on CPU.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a ones tensor on the same device as the
            # input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones of shape two by three on the device of
                # x.
                device = x.device
                return torch.ones(2, 3, device=device)

        inp = (torch.rand(3, 4),)
        self._check_equal_ts_ep_converter(Module(), inp)

    @requires_cuda
    def test_prim_device_cuda(self):
        # [FUNCTION DOC] Same as the device test but with a CUDA input; skipped when CUDA is
        # not available.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a ones tensor on the same device as the
            # input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones of shape two by three on the device of
                # x.
                device = x.device
                return torch.ones(2, 3, device=device)

        inp = (torch.rand((3, 4), device="cuda:0"),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_dtype(self):
        # [FUNCTION DOC] Checks conversion of reading the dtype of a tensor and using it to
        # create a new tensor, once each for float32 and float64 random inputs and for uint8,
        # int8 and int32 random integer inputs (values below 128).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a ones tensor with the same dtype as the
            # input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones of shape two by three with the dtype of
                # x.
                dtype = x.dtype
                return torch.ones(2, 3, dtype=dtype)

        for dtype in [
            torch.float32,
            torch.double,
        ]:
            inp = (torch.rand((3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)

        for dtype in [
            torch.uint8,
            torch.int8,
            torch.int32,
        ]:
            inp = (torch.randint(high=128, size=(3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)

    def test_convert_if_basic(self):
        # [FUNCTION DOC] Checks conversion of a data dependent if on a boolean scalar tensor.
        # The default check runs with the true branch; for every converted program after the
        # first (the scripted one) it also compares the false branch result against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the square of y when x is true, else y
            # doubled.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Branches on x and returns y times y or y plus y.
                if x:
                    return y * y
                else:
                    return y + y

        inp = (torch.tensor(True), torch.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(torch.tensor(False), torch.tensor(4)),
                M()(torch.tensor(False), torch.tensor(4)),
            )

    def test_convert_if_tuple_out(self):
        # [FUNCTION DOC] Checks conversion of an if whose branches call helper methods
        # returning tuples. Runs the shared check with the true branch, then for the scripted
        # program also compares the false branch against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose branches call methods returning a pair of
            # tensors; the forward sums the two entries of the chosen pair.
            def true_fn(self, y, z):
                # [FUNCTION DOC] HELPER: Returns the square and the double of z as a tuple (y
                # is unused).
                return (z * z, z + z)

            def false_fn(self, y, z):
                # [FUNCTION DOC] HELPER: Returns the cube and the double of y as a tuple (z is
                # unused).
                return (y * y * y, y + y)

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Computes the square of y, picks a pair using the true
                # or false method according to x, and returns the sum of the pair entries.
                z = y * y

                if x:
                    res = self.true_fn(y, z)
                else:
                    res = self.false_fn(y, z)

                return res[0] + res[1]

        inp = (torch.tensor(True), torch.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(torch.tensor(False), torch.tensor(4)),
                M()(torch.tensor(False), torch.tensor(4)),
            )

    def test_convert_if_multiple_out(self):
        # [FUNCTION DOC] Checks conversion of an if that assigns two variables in each branch.
        # Runs the shared check with the true branch, then compares the false branch against
        # eager for the scripted program.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose two branches assign two intermediate results
            # each and return their sum.
            def true_fn(self, y, z):
                # [FUNCTION DOC] HELPER: Returns the square of z (y is unused).
                return z * z

            def false_fn(self, y, z):
                # [FUNCTION DOC] HELPER: Returns the cube of y (z is unused).
                return y * y * y

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: On true uses the true method result and y; on false
                # uses z and the false method result; returns the sum of the two values.
                z = y * y

                if x:
                    res1 = self.true_fn(y, z)
                    res2 = y
                else:
                    res1 = z
                    res2 = self.false_fn(y, z)

                return res1 + res2

        inp = (torch.tensor(True), torch.tensor(4))
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(torch.tensor(False), torch.tensor(4)),
                M()(torch.tensor(False), torch.tensor(4)),
            )

    def test_profiler__record_function(self):
        # [FUNCTION DOC] Checks conversion of the profiler record function enter and exit ops
        # that surround a multiply and add.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an arithmetic expression in profiler record
            # calls.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Enters a profiler record range, computes x times two
                # plus four, exits the range and returns the value.
                handle = torch.ops.profiler._record_function_enter_new("foo", None)
                y = x * 2 + 4
                torch.ops.profiler._record_function_exit(handle)
                return y

        x = torch.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten_floordiv(self):
        # [FUNCTION DOC] Checks conversion of floor division of a tensor by a constant.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input floor-divided by two.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x floor-divided by two.
                return x // 2

        x = torch.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten___is__(self):
        # [FUNCTION DOC] Checks conversion of the identity comparison operator on two tensors
        # (scripted only, with an extra runtime input), since the traced output must hold
        # tensors.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning whether its two inputs are the same object
            # along with the input plus one.
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> tuple[bool, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns a pair: the identity test of x and y, and x
                # plus one.
                z = x + 1
                return x is y, z

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        runtime_inps = [(torch.randn(20, 2), torch.rand(20, 2))]
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], runtime_inputs=runtime_inps
        )

    def test_aten___isnot__(self):
        # [FUNCTION DOC] Checks conversion of the identity-not comparison on two tensors,
        # scripted only.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning whether its two inputs are different
            # objects along with the input plus one.
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> tuple[bool, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns a pair: the negated identity test of x and y,
                # and x plus one.
                z = x + 1
                return x is not y, z

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_aten___not__(self):
        # [FUNCTION DOC] Checks conversion of logical negation applied to an identity-not
        # comparison, scripted only.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the logical not of an identity-not test and
            # a sum.
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> tuple[bool, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns a pair: not of the identity-not test of x and
                # y, and x plus one.
                z = x + 1
                return not (x is not y), z

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_ts2ep_converter_unpack(self):
        # [FUNCTION DOC] Checks unpacking: a split result unpacked into two tensors that are
        # added, and a tuple input unpacked into two tensors where the first has cosine
        # applied before the addition. Both converted with default modes.
        class MUnpackList(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unpacking the two chunks of a split of its input and
            # adding them.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Splits x into pieces of two, unpacks two pieces,
                # returns their sum.
                x, y = torch.split(x, 2)
                return x + y

        class MUnpackTuple(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unpacking a tuple input into two tensors and
            # combining them.
            def forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]):
                # [FUNCTION DOC] HELPER: Returns the cosine of the first tuple element plus
                # the second.
                x, y = x_tuple
                x = x.cos()
                return x + y

        inp = (torch.ones(4),)
        self._check_equal_ts_ep_converter(MUnpackList(), inp)
        inp = ((torch.zeros(1, 4), torch.ones(1, 4)),)
        self._check_equal_ts_ep_converter(MUnpackTuple(), inp)

    @unittest.skipIf(
        IS_WINDOWS,
        "torch.cond doesn't go through torch.compile on windows"
        "causing output not normalized as list",
    )
    def test_convert_retrace_nested_scripted_modules(self):
        # [FUNCTION DOC] Skipped on Windows. Checks conversion of a doubly nested module where
        # inner modules are already scripted and also wrapped by a plain module, with data
        # dependent branches at both levels; the converter must retrace the nested scripted
        # modules.
        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain module that stores a submodule and delegates forward
            # to it.
            def __init__(self, mod) -> None:
                # [FUNCTION DOC] HELPER: Stores the given module as a submodule.
                super().__init__()
                self.mod = mod

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the wrapped module applied to x and y.
                return self.mod(x, y)

        class LinearM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer of the given size, applied to
            # the second input only.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer of equal input and output
                # size.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to y (x is unused).
                return self.linear(y)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a scripted linear module and a wrapper around
            # the same scripted module; forward branches on x to subtract or add the two
            # results.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Builds the linear module, scripts it, and stores it
                # directly and inside the wrapper.
                super().__init__()
                m = LinearM(dim)
                m = torch.jit.script(m)
                self.mod1 = m
                self.mod2 = Wrapper(m)

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: If x is true returns minus first result minus second
                # result, otherwise minus first plus second.
                if x:
                    return -self.mod1(x, y) - self.mod2(x, y)
                else:
                    return -self.mod1(x, y) + self.mod2(x, y)

        class NestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module holding the scripted version of the previous
            # module and a wrapper around it; forward branches on x to add or subtract the two
            # results.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Builds the inner module, scripts it, and stores it
                # directly and inside the wrapper.
                super().__init__()
                m = M(dim)
                m = torch.jit.script(m)
                self.mod1 = m
                self.mod2 = Wrapper(m)

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: If x is true returns the sum of the two submodule
                # results, otherwise their difference.
                if x:
                    return self.mod1(x, y) + self.mod2(x, y)
                else:
                    return self.mod1(x, y) - self.mod2(x, y)

        inp = (
            torch.tensor(True),
            torch.randn([3, 3]),
        )
        self._check_equal_ts_ep_converter(NestedM(3), inp)

    def test_convert_nn_module_with_nested_param(self):
        # [FUNCTION DOC] Checks conversion of modules with parameters at several nesting
        # depths (linear layers inside one and two levels of submodules); the shared checker
        # also verifies the program state dict keys equal those of the TorchScript module.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single linear layer.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer of equal input and output
                # size.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x.
                return self.linear(x)

        class NestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own linear layer and a nested single-layer
            # module.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates its own linear layer and the nested module.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = M(dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Applies the nested module then its own linear layer.
                return self.linear(self.m(x))

        class SuperNestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own linear layer and a nested two-level
            # module.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates its own linear layer and the nested module.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = NestedM(dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Applies the nested module then its own linear layer.
                return self.linear(self.m(x))

        inp = (torch.ones(3),)
        orig_m = NestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_buffer(self):
        # [FUNCTION DOC] Checks conversion of modules with buffers at several nesting depths
        # (one and two levels of submodules), comparing state dict keys and outputs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single random buffer added to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a random one element buffer.
                super().__init__()
                self.w = torch.nn.Buffer(torch.randn(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the buffer plus x.
                return self.w + x

        class NestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own buffer and a nested single-buffer
            # module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested module and its own random buffer.
                super().__init__()
                self.m = M()
                self.w = torch.nn.Buffer(torch.randn(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns its buffer plus the nested module output.
                return self.w + self.m(x)

        class SuperNestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own buffer and a nested two-level module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested module and its own random buffer.
                super().__init__()
                self.m = NestedM()
                self.w = torch.nn.Buffer(torch.randn(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns its buffer plus the nested module output.
                return self.w + self.m(x)

        inp = (torch.ones(1),)
        orig_m = NestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_if_and_buffer(self):
        # [FUNCTION DOC] Checks conversion of nested modules with buffers and data dependent
        # ifs at two levels, where buffers must be lifted multiple times. After the shared
        # check, the input is modified in place (decremented by one so the other branches are
        # taken) and every converted program is compared with the original module on it.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module with a random buffer and an integer attribute;
            # forward adds the buffer, input and the integer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a random one element buffer and sets an
                # integer attribute to one.
                super().__init__()
                self.w = torch.nn.Buffer(torch.randn(1))
                self.count = 1

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the buffer plus x plus the integer attribute.
                return self.w + x + self.count

        class NestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two leaf submodules and a buffer; forward
            # chooses a submodule depending on the sum of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two leaf submodules and a random buffer.
                super().__init__()
                self.m1 = M()
                self.m2 = M()
                self.w = torch.nn.Buffer(torch.randn(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: If the sum of x exceeds one uses the first submodule
                # else the second; returns the buffer plus that submodule output.
                if torch.sum(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested, parameters need to be lifted
        # multiple times.
        class SuperNestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two nested modules and a buffer; forward chooses
            # one depending on the maximum of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two nested modules and a random buffer.
                super().__init__()
                self.m1 = NestedM()
                self.m2 = NestedM()
                self.w = torch.nn.Buffer(torch.randn(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: If the max of x exceeds one uses the first submodule
                # else the second; returns the buffer plus that submodule output.
                if torch.max(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested module testing.
        inp = (torch.ones(1),)
        orig_m = SuperNestedM()
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 1
        for ep in ep_list:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

    @unittest.skipIf(
        IS_WINDOWS,
        "torch.cond doesn't go through torch.compile on windows"
        "causing output not normalized as list",
    )
    def test_convert_nn_module_with_nested_if_and_param(self):
        # [FUNCTION DOC] Skipped on Windows. Checks conversion of nested modules with linear
        # layer parameters and data dependent ifs at several depths: a plain linear module, a
        # module of two such with an if, and two deeper modules (one branching on max, one on
        # sum, where the input must be lifted recursively). For each, the shared check runs,
        # the input is then decreased by 0.8 to take other branches, and the converted
        # programs after the traced one (traced specializes on a single path) are compared
        # with the original module.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer of equal input and output
                # size.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x.
                return self.linear(x)

        class NestedM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two single-layer submodules and its own linear
            # layer.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates two submodules and a linear layer.
                super().__init__()
                self.m1 = M(dim)
                self.m2 = M(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Branches on the sum of x and applies one of the
                # submodules.
                if torch.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, parameters need to be lifted
        # multiple times.
        class SuperNestedM1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two nested modules and a linear layer; forward
            # branches on the max of the input to choose one nested module, then applies its
            # linear layer.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates two nested modules and a linear layer.
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: If the max of x exceeds one routes through the first
                # nested module else the second, then applies the linear layer.
                if torch.max(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, even the input needs to be
        # lifted recursively due to value propagation optimization.
        class SuperNestedM2(torch.nn.Module):
            # [CLASS DOC] HELPER: Same structure as the previous super nested module but
            # branching on the sum of the input.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates two nested modules and a linear layer.
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: If the sum of x exceeds one routes through the first
                # nested module else the second, then applies the linear layer.
                if torch.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Basic module testing.
        inp = (torch.ones(3),)
        orig_m = M(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Nested module testing.
        inp = (torch.ones(3),)
        orig_m = NestedM(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Super nested module testing.
        inp = (torch.ones(3),)
        orig_m = SuperNestedM1(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Super nested module testing.
        inp = (torch.ones(3),)
        orig_m = SuperNestedM2(3)
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        # Skip jit.traced because it specializes on one path.
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

    def test_convert_if_duplicate_attr_names(self):
        # [FUNCTION DOC] Scripted-only check of a module whose forward mutates plain integer
        # attributes and then uses them in two separate ifs on an integer argument, so
        # converted if blocks may reference attributes with duplicate names.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two integer attributes that are multiplied in
            # each call and added to the input in branches chosen by an integer argument.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the two integer attributes to one and two.
                super().__init__()
                self.w = 1
                self.h = 2

            def forward(self, x: torch.Tensor, y: int):
                # [FUNCTION DOC] HELPER: Scales both attributes by ten and twenty (a side
                # effect on the module), adds one of them to x depending on whether y exceeds
                # ten, then adds one of them again depending on whether y is below ten;
                # returns the result.
                self.w = self.w * 10
                self.h = self.h * 20

                if y > 10:
                    res = self.w + x
                else:
                    res = self.h + x

                if y < 10:
                    res = self.w + res
                else:
                    res = self.h + res

                return res

        inp = (torch.ones(3), 5)
        self._check_equal_ts_ep_converter(M(), inp, option=["script"])

    def test_ts2ep_converter_contains(self):
        # [FUNCTION DOC] Scripted-only check of membership tests: a dtype in a list of dtypes,
        # a dtype comparison (despite the class name, an equality rather than a not-in test),
        # and a tensor key in a dict with both a present and an absent key.
        class MIn(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning whether the dtype of the input is in a list
            # of floats.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns whether x has dtype float32 or float64.
                return x.dtype in [torch.float32, torch.float64]

        class MNotIn(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning whether the input dtype equals int8.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns whether x has dtype int8.
                return x.dtype == torch.int8

        class MTensorIn(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning whether the tensor input is a key of the
            # dict input.
            def forward(self, x: torch.Tensor, x_dict: dict[torch.Tensor, str]):
                # [FUNCTION DOC] HELPER: Returns whether x is a key of x_dict.
                return x in x_dict

        # Traced function must return output that has tensors.
        inp = (torch.tensor(4),)
        self._check_equal_ts_ep_converter(MIn(), inp, ["script"])
        self._check_equal_ts_ep_converter(MNotIn(), inp, ["script"])

        # TODO: update test to use reference for in.
        inp = (torch.tensor(4), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])
        inp = (torch.tensor(1), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])

    def test_ts2ep_converter_custom_op(self):
        # [FUNCTION DOC] Registers a custom op inside a scoped library with a composite
        # explicit autograd implementation and a fake (meta) implementation, both adding the
        # input to itself, then checks conversion of a module calling it. SIDE EFFECT: sets
        # two dynamo config flags (capture scalar outputs, capture dynamic output shape ops)
        # globally and never restores them.
        # KNOWN ISSUE: the config flags are assigned directly rather than through a scoped
        # patch, so they leak into later tests.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            torch.library.define(
                "mylib::foo",
                "(Tensor x) -> Tensor",
                lib=lib,
            )

            # PyTorch custorm op implementation
            @torch.library.impl(
                "mylib::foo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def foo_impl(x):
                # [FUNCTION DOC] HELPER: Real implementation of the op; returns x plus x.
                return x + x

            # Meta function of the custom op.
            @torch.library.register_fake(
                "mylib::foo",
                lib=lib,
            )
            def foo_meta(x):
                # [FUNCTION DOC] HELPER: Fake kernel of the op; returns x plus x.
                return x + x

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module calling the custom op on its input.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the custom op applied to x.
                    return torch.ops.mylib.foo(x)

            inp = (torch.randn(3, 3),)
            m = M()
            self._check_equal_ts_ep_converter(m, inp)

    def test_convert_func_without_param(self):
        # [FUNCTION DOC] Checks conversion of plain functions (not modules): a sum, and a
        # function with a data dependent if. For the branching one, the input is then
        # decreased by one to take the other branch and the converted programs after the
        # traced one are compared with eager.
        def func1(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        def func2(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y if the sum of x is positive, otherwise x
            # minus y.
            if x.sum() > 0:
                return x + y
            else:
                return x - y

        inp = (
            torch.tensor(1),
            torch.tensor(1),
        )
        self._check_equal_ts_ep_converter(func1, inp)

        ep_list = self._check_equal_ts_ep_converter(func2, inp)

        t = inp[0]
        t -= 1
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                func2(*inp),
            )

    def test_implicit_constant_to_tensor_handling(self):
        # [FUNCTION DOC] Checks that constants that are implicitly turned into tensors convert
        # correctly: adding a scalar, arithmetic mixing inputs and scalars, adding a tensor
        # literal, creating a full tensor from an infinite value, a module attribute tensor
        # cloned, a tensor literal added in a module, constants with ones and zeros, and numel
        # and size queries over int8, int32, float32 and float64 inputs. A trailing commented
        # TODO about dynamic shapes is unused.
        def func1(x):
            # [FUNCTION DOC] HELPER: Returns x plus two.
            return x + 2

        def func2(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y divided by (x minus two times y), plus
            # y.
            return x * y / (x - 2 * y) + y

        def func3(x):
            # [FUNCTION DOC] HELPER: Returns x plus a one element constant tensor holding
            # three.
            return x + torch.tensor([3])

        def func4():
            # [FUNCTION DOC] HELPER: Returns a ten by ten tensor filled with a scalar tensor
            # holding infinity; takes no arguments.
            val = torch.tensor(float("inf"))
            return torch.full((10, 10), val)

        def func5():
            # [FUNCTION DOC] HELPER: Returns a pair: minus one times a float ones tensor of
            # length one, and a float zeros tensor of length one; takes no arguments.
            x = -1
            return x * torch.ones(1, dtype=torch.float), torch.zeros(
                1, dtype=torch.float
            )

        def func6(x1, x2, x3, x4):
            # [FUNCTION DOC] HELPER: Returns numel and size of each of four tensors in order,
            # followed by two ones tensors built from the element count and from the size of
            # the first input.
            return (
                x1.numel(),
                x1.size(),
                x2.numel(),
                x2.size(),
                x3.numel(),
                x3.size(),
                x4.numel(),
                x4.size(),
                torch.ones(x1.numel()),  # Just make sure downstream ops still work.
                torch.ones(x1.size()),  # Just make sure downstream ops still work.
            )

        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a tensor made from a given scalar and
            # returning a clone of it; takes no forward arguments.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores a tensor built from value as an attribute.
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a clone of the stored tensor.
                return self.x.clone()

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a tensor literal holding four to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a scalar tensor four plus x.
                return torch.tensor(4) + x

        inp = (torch.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func1, inp)
        inp = (torch.randn([2, 2]), torch.randn([2, 2]))
        self._check_equal_ts_ep_converter(func2, inp)

        inp = (torch.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func3, inp)

        self._check_equal_ts_ep_converter(func4, ())
        self._check_equal_ts_ep_converter(M1(5), ())

        inp = (torch.randn(2),)
        self._check_equal_ts_ep_converter(M2(), inp)

        self._check_equal_ts_ep_converter(func5, ())
        inp = (
            torch.randn([2, 3, 4]).to(torch.int8),
            torch.randn([2, 3, 4]).to(torch.int32),
            torch.randn([2, 3, 4]).to(torch.float32),
            torch.randn([2, 3, 4]).to(torch.float64),
        )
        self._check_equal_ts_ep_converter(func6, inp)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     self.assertEqual(
        #         ep.module()(
        #             torch.randn([1, 1, 1]).to(torch.int8),
        #             torch.randn([1, 1, 1]).to(torch.int32),
        #             torch.randn([1, 1, 1]).to(torch.float32),
        #             torch.randn([1, 1, 1]).to(torch.float64),
        #         )[0], 1
        #     )

    def test_aten_tensor_dtype_int(self):
        # [FUNCTION DOC] Checks that creating a tensor from a constant integer with an
        # explicit int32 dtype converts, and that every converted program records exactly one
        # constant.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a freshly created int32 scalar tensor to the
            # input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns an int32 tensor holding one plus x.
                y = torch.tensor(1, dtype=torch.int32)
                return y + x

        ep_list = self._check_equal_ts_ep_converter(M(), (torch.tensor(1),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 1)

    def test_aten_tensor_prim_dtype(self):
        # [FUNCTION DOC] Same as the previous test but the new tensor takes its dtype from the
        # input; every converted program must again record exactly one constant.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a one tensor created with the input dtype to
            # the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tensor holding one with the dtype of x,
                # plus x.
                y = torch.tensor(1, dtype=x.dtype)
                return y + x

        ep_list = self._check_equal_ts_ep_converter(M(), (torch.tensor(1),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 1)

    def test_aten_tensor_dynamic(self):
        # [FUNCTION DOC] Checks creating tensors from values derived from a shape. For a
        # scalar made from a dimension size every program has zero constants; for a three
        # element list built from the size, only the programs after the traced one must have
        # zero constants (a traced program inlines a constant). Commented extra checks for
        # dynamic shapes are unused.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a scalar tensor built from the size of the
            # first dim.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tensor made from the size of the first dim
                # of x.
                s = x.shape[0]
                y = torch.tensor(s)
                return y

        ep_list = self._check_equal_ts_ep_converter(M(), (torch.ones(3),))
        for ep in ep_list:
            self.assertEqual(len(ep.constants), 0)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     torch.testing.assert_close(
        #         ep.module()(torch.ones(4)),
        #         M()(torch.ones(4)),
        #     )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tensor built from a list of expressions
            # over the size of the first dim.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a three element tensor made from the size,
                # twice the size and one.
                s = x.shape[0]
                y = torch.tensor([s, s * 2, 1])
                return y

        ep_list = self._check_equal_ts_ep_converter(M(), (torch.ones(3),))
        # Trace directly inline a tensor constant.
        for ep in ep_list[1:]:
            self.assertEqual(len(ep.constants), 0)

        # TODO: Additional check once dynamic shape is supported.
        # for ep in ep_list:
        #     torch.testing.assert_close(
        #         ep.module()(torch.ones(4)),
        #         M()(torch.ones(4)),
        #     )

    def test_prim_tolist(self):
        # [FUNCTION DOC] Scripted-only check of the tolist conversion for a one dimensional
        # tensor returning a list of ints and for a two dimensional tensor returning a nested
        # list of ints.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the flat list form of its input.
            def forward(self, x: torch.Tensor) -> list[int]:
                # [FUNCTION DOC] HELPER: Returns the input as a Python list of ints.
                return x.tolist()

        inp = (torch.tensor([1, 2, 3]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the nested list form of its input.
            def forward(self, x: torch.Tensor) -> list[list[int]]:
                # [FUNCTION DOC] HELPER: Returns the input as a nested Python list of ints.
                return x.tolist()

        inp = (torch.tensor([[1, 2, 3], [4, 5, 6]]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_get_tensor_constants(self):
        # [FUNCTION DOC] Checks conversion of modules that hold plain tensor attributes which
        # are only read, so they are lifted as constant tensors rather than parameters or
        # buffers.
        # Since self.data is only read but not written, it is lifted as
        # constant tensors.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random tensor attribute added to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random three by two tensor attribute.
                super().__init__()
                self.data = torch.randn(3, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x plus the tensor attribute.
                return x + self.data

        class Goo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with its own random tensor attribute and a nested Foo
            # module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random tensor attribute and the nested
                # module.
                super().__init__()
                self.data = torch.randn(3, 2)
                self.foo = Foo()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x plus its own tensor attribute, the nested
                # module's attribute, and the nested module output.
                return x + self.data + self.foo.data + self.foo(x)

        inp = (torch.randn(3, 2),)
        goo = Goo()
        self._check_equal_ts_ep_converter(goo, inp)

    def test_prim_SetAttr(self):
        # [FUNCTION DOC] Checks conversion of attribute assignment in forward, scripted only
        # with fresh module instances per run and repeated calls to catch persistence. Cases:
        # a buffer reassigned while the output ignores it; a buffer reassigned and then used;
        # a plain tensor attribute reassigned (the converter turns it into a buffer; the
        # expected state dict also includes it as a lifted constant); an integer attribute
        # incremented (not persistent since export specializes on non-tensor constants); the
        # same with several reads between increments; and a buffer updated by in-place add.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer reassigned to itself plus the input;
            # the output does not depend on the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a ones buffer of shape three by two.
                super().__init__()
                self.data = torch.nn.Buffer(torch.ones(3, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Reassigns the buffer to buffer plus x and returns x
                # plus x.
                self.data = self.data + x
                return x + x

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer reassigned to itself plus the input;
            # the output uses the new buffer value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a ones buffer of shape three by two.
                super().__init__()
                self.data = torch.nn.Buffer(torch.ones(3, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Reassigns the buffer to buffer plus x and returns x
                # plus the buffer.
                self.data = self.data + x
                return x + self.data

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        # export lifts a tensor constant (self.data) as an input if it is not assigned.
        # If it is assigned, export will error and ask users to register it as a buffer.
        # In converter, we change tensor constants that are assigned as a buffer automatically,
        # since it might be hard to manually register them as buffers.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain (non-buffer) tensor attribute reassigned
            # in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ones tensor attribute (not registered as a
                # buffer).
                super().__init__()
                self.data = torch.ones(3, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Reassigns the attribute to attribute plus x and
                # returns x plus it.
                self.data = self.data + x
                return x + self.data

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module,
            inp,
            ["script"],
            check_persistent=True,
            lifted_tensor_constants=OrderedDict([("data", torch.ones(3, 2))]),
        )

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer counter attribute incremented each
            # call and added to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the counter to zero.
                super().__init__()
                self.count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Increments the counter and returns x plus the new
                # counter value.
                self.count += 1
                return x + self.count

        # check_persistent is False since export specializes on non-tensor constants
        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], check_persistent=False
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer counter that is incremented twice per
            # call, with the value read before and after each increment.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the counter to zero.
                super().__init__()
                self.count = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the counter, increments, reads, increments,
                # reads; returns x plus the three read values.
                count1 = self.count
                self.count += 1
                count2 = self.count
                self.count += 1
                count3 = self.count
                return x + count1 + count2 + count3

        inp = (torch.ones(1),)
        self._check_equal_ts_ep_converter(M(), inp, ["script"], check_persistent=False)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer updated in place on every call.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a ones buffer of length one.
                super().__init__()
                self.w2 = torch.nn.Buffer(torch.ones(1))

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Adds one to the buffer in place and returns the
                # buffer.
                self.w2 += 1
                return self.w2

        inp = (torch.ones(1),)
        self._check_equal_ts_ep_converter(M, inp, ["script"], check_persistent=True)

    def test_raise_exception(self):
        # [FUNCTION DOC] Checks conversion of modules that raise an exception inside an if,
        # scripted only. When the input takes the raising branch the shared check must fail
        # with a TorchScript error mentioning the built-in runtime error (matching non-strict
        # export, which also errors); when the non-raising branch is taken the check passes.
        # Done for two modules: one with an early raise and one with an else branch computing
        # a value.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising RuntimeError when the integer argument is
            # positive, otherwise returning x plus y.
            def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Raises RuntimeError if y is positive; else returns x
                # plus y.
                if y > 0:
                    raise RuntimeError("test")
                return x + y

        # match non-strict export behavior that errors when the given input leads to
        # RaiseException.
        with self.assertRaisesRegex(torch.jit.Error, "builtins.RuntimeError"):
            inp = (torch.randn(3, 2), 1)
            self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # Matching non-strict export behavior that only executes 1 if-branch according
        # to the given input.
        inp = (torch.randn(3, 2), 0)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising RuntimeError in the positive branch and
            # computing a value in the else branch.
            def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Raises RuntimeError if y is positive; else computes z
                # as x plus y and returns x plus y plus z.
                z = x
                if y > 0:
                    raise RuntimeError("test")
                    # z = x
                else:
                    z = x + y
                return x + y + z

        # match non-strict export behavior that errors when the given input leads to
        # RaiseException.
        with self.assertRaisesRegex(torch.jit.Error, "builtins.RuntimeError"):
            inp = (torch.randn(3, 2), 1)
            self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # Matching non-strict export behavior that only executes 1 if-branch according
        # to the given input.
        inp = (torch.randn(3, 2), 0)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_context_manager(self):
        # [FUNCTION DOC] Checks conversion of a module whose forward enters a user defined
        # context manager around an addition (default modes).
        class ContextManager:
            # [CLASS DOC] HELPER: Plain Python context manager that counts nesting depth in an
            # integer attribute; incremented on entry and decremented on exit.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the count to zero.
                self.count = 0
                return

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Increments the count on entry; returns nothing.
                self.count += 1
                return

            def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
                # [FUNCTION DOC] HELPER: Decrements the count on exit; returns nothing and
                # does not suppress exceptions.
                self.count -= 1
                return

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds its inputs inside the context manager.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y computed within the context manager.
                with ContextManager():
                    res = x + y
                return res

        inp = (torch.ones(3, 3), torch.ones(3, 3))
        self._check_equal_ts_ep_converter(M(), inp)

    def test_hidden_input_name(self):
        # [FUNCTION DOC] Checks conversion when the function name or input names are hidden or
        # unusual: an already-scripted function, the relu operator object (traced only, since
        # it cannot be scripted again), and a function with variable length arguments that
        # concatenates them and squares (traced only; one input is an empty double tensor).
        @torch.jit.script
        def func1(x):
            # [FUNCTION DOC] HELPER: Scripted function returning x plus one.
            return x + 1

        def func2(*args):
            # [FUNCTION DOC] HELPER: Concatenates all given tensors along dim one and returns
            # the elementwise square.
            v = torch.cat(args, dim=1)
            return v * v

        inp = (torch.randn([1, 1]),)
        self._check_equal_ts_ep_converter(func1, inp)

        inp = (torch.ones(5, 5),)
        # Cannot script again.
        self._check_equal_ts_ep_converter(torch.ops.aten.relu, inp, ["trace"])

        M = 2
        Ns = [4, 2, 1]
        empty = torch.tensor([], dtype=torch.double)
        values = [empty] + [torch.randn(M, N) for N in Ns]
        # Cannot script variable length inputs.
        self._check_equal_ts_ep_converter(func2, tuple(values), ["trace"])

    def test_ts2ep_multi_outputs_on_call_ops(self):
        # [FUNCTION DOC] Checks conversion of ops with several outputs: max over a dim, topk,
        # sort and an adaptive max pool that also returns indices.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the results of multi-output ops (max, topk,
            # sort, pooling with indices).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates an adaptive max pooling layer with a two by
                # two output that returns indices.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns a tuple of the max over dim zero and top-
                # three of x, sort along dim zero of x, and the pooled output of y.
                return (
                    torch.max(x, dim=0),
                    torch.topk(x, 3),
                    torch.sort(x, dim=0),
                    self.pool(y),
                )

        inp = (torch.randn([4, 4]), torch.randn([1, 1, 10, 10]))
        self._check_equal_ts_ep_converter(M(), inp)

    def test_aten_append_t(self):
        # [FUNCTION DOC] Scripted-only check of list append: a list is built by appending
        # results of arithmetic on two input list elements, concatenated, appended to again
        # and concatenated.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building a list by appending sum, difference and
            # product of the first two list elements.
            def forward(self, x: list[torch.Tensor]):
                # [FUNCTION DOC] HELPER: Returns the list and the concatenations made after
                # two and after three appends.
                out = []
                out.append(x[0] + x[1])
                out.append(x[0] - x[1])
                out1 = torch.cat(out)
                out.append(x[0] * x[1])
                out2 = torch.cat(out)
                return out, out1, out2

        inp = ([torch.ones(2, 3), torch.ones(2, 3)],)
        # Trace already unrolls the list.
        self._check_equal_ts_ep_converter(M(), inp, ["script"])

    def test_convert_script_object(self):
        # [FUNCTION DOC] Scripted-only check of conversion for a module holding a torchbind
        # tensor queue: pushes through the method and through the operator, then pops through
        # the operator and the method.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an empty torchbind tensor queue.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the empty queue as an attribute.
                super().__init__()
                self.tq = _empty_tensor_queue()

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Pushes x, pushes the cosine of x via the operator,
                # then returns the result of popping with the operator and popping with the
                # method.
                self.tq.push(x)
                torch.ops._TorchScriptTesting.queue_push(self.tq, x.cos())
                return torch.ops._TorchScriptTesting.queue_pop(self.tq), self.tq.pop()

        inp = (torch.randn(2, 3),)
        self._check_equal_ts_ep_converter(M1(), inp, ["script"])

    def test_ts2ep_with_loop(self):
        # [FUNCTION DOC] Scripted-only check of loop conversion on a function with nested for
        # loops that append to an input list (checked with runtime input of larger tensors).
        # Two other loop functions (a for loop over a dim and a while loop) are defined but
        # their checks are commented out as TODO and never run.
        def func1(x, x_list: list[torch.Tensor]):
            # [FUNCTION DOC] HELPER: Runs nested loops that update three local values and
            # append sums of list entries to the list; returns the original x and the list.
            a, b, c = x, x, x
            for _ in range(1, 5, 2):
                for k in range(5):
                    a = a + a + k
                    b = b + b - k
                    x_list.append(x_list[k] + x_list[k + 1])
                for k in range(5):
                    b = b + b - k
                    c = c + c * k
                    x_list.append(x_list[k] + x_list[k + 1] - x_list[k + 2])
            return x, x_list

        def func2(x):
            # [FUNCTION DOC] HELPER: Unused. Multiplies x by itself and the loop index for
            # each size of dim zero and returns it.
            for i in range(x.size(0)):
                x = x * x * i
            return x

        def func3(x):
            # [FUNCTION DOC] HELPER: Unused. Adds the sine of x to x while the sum is below
            # ten and returns x.
            while x.sum() < 10:
                x += x.sin()
            return x

        inp = (
            torch.tensor(1),
            [torch.ones([2, 2]), torch.ones([2, 2]) * 2],
        )
        runtime_inps = [
            (
                torch.tensor(1),
                [torch.ones([8, 8]), torch.ones([8, 8]) * 2],
            )
        ]
        # Trace unrolls the loop.
        self._check_equal_ts_ep_converter(
            func1, inp, ["script"], runtime_inputs=runtime_inps
        )

        # TODO: (2/N)
        # Trace unrolls the loop.
        # self._check_equal_ts_ep_converter(func2, inp, ["script"])

        # TODO: (3/N)
        # Trace unrolls the loop.
        # self._check_equal_ts_ep_converter(func3, inp, ["script"])

    @unittest.skipIf(
        IS_WINDOWS,
        "Windows does not support qnnpack",
    )
    # qnnpack not supported on s390x
    @xfailIfS390X
    def test_ts2ep_convert_quantized_model1(self):
        # [FUNCTION DOC] Skipped on Windows and expected to fail on s390x. Quantizes a small
        # model (quant stub, two convolutions with the second fused with relu, dequant stub)
        # with qnnpack: fuse, prepare, calibrate, convert. Because quantization changes the
        # state dict, it does not use the shared checker: it scripts the model twice, converts
        # one with the converter and compares outputs of the converted program against the
        # other scripted copy.
        class Standalone(torch.nn.Module):
            # [CLASS DOC] HELPER: Float model with quant stub, two 1x1 convolutions, relu and
            # dequant stub, with a method to fuse the second convolution and relu.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the stubs, two convolutions and relu.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the layers in order and returns the result.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

            def fuse_model(self):
                # [FUNCTION DOC] HELPER: Fuses the second convolution with relu in place.
                torch.ao.quantization.fuse_modules(
                    self, [["conv2", "relu"]], inplace=True
                )

        with override_quantized_engine("qnnpack"):
            model = Standalone()
            model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            model.fuse_model()
            torch.ao.quantization.prepare(model, inplace=True)
            model(torch.randn(4, 1, 4, 4))
            torch.ao.quantization.convert(model, inplace=True)

            # Use customized checking here, because state_dict of quantization will be
            # modified by the quantization pass.
            inp = (torch.randn(4, 1, 4, 4),)
            original_ts_model = torch.jit.script(model)
            ts_model = torch.jit.script(model)
            converter = TS2EPConverter(ts_model, inp)
            ep = converter.convert()

            orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))
            ep_out, _ = pytree.tree_flatten(ep.module()(*inp))
            self._check_tensor_list_equal(orig_out, ep_out)

    # qnnpack/xnnpack not supported on s390x.
    # it is required by
    # torch.ops.prepacked.linear_clamp_prepack
    # and
    # torch.ops.prepacked.linear_clamp_run
    @xfailIfS390X
    @requires_prepacked_linear
    def test_ts2ep_convert_quantized_model_with_opcontext(self):
        # [FUNCTION DOC] Expected to fail on s390x and skipped without prepacked linear ops.
        # Converts a scripted module that runs a prepacked linear op stored as an attribute
        # (an op context object) on the input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a prepacked linear op context.
            def __init__(self, linear_op):
                # [FUNCTION DOC] HELPER: Stores the given op context.
                super().__init__()
                self.linear_op = linear_op

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the prepacked linear run op applied to x with
                # the stored context.
                x = torch.ops.prepacked.linear_clamp_run(x, self.linear_op)
                return x

        linear_op = torch.ops.prepacked.linear_clamp_prepack(
            torch.randn(10, 10), torch.randn(10)
        )
        m = M(linear_op)
        inp = (torch.randn(1, 10),)
        self._check_equal_ts_ep_converter(m, inp, ["script"])

    # qnnpack/xnnpack not supported on s390x.
    # it is required by
    # torch.ops.prepacked.linear_clamp_prepack
    # and
    # torch.ops.prepacked.linear_clamp_run
    @xfailIfS390X
    @requires_prepacked_linear
    def test_ts2ep_convert_quantized_model_with_opcontext_and_constant(self):
        # [FUNCTION DOC] Same as the previous test but the module adds a constant ones tensor
        # to the input before the prepacked linear run, so a tensor constant is also present.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a prepacked linear op context.
            def __init__(self, linear_op):
                # [FUNCTION DOC] HELPER: Stores the given op context.
                super().__init__()
                self.linear_op = linear_op

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the prepacked linear run op applied to x plus
                # ones with the stored context.
                x = torch.ops.prepacked.linear_clamp_run(
                    x + torch.ones(1), self.linear_op
                )
                return x

        linear_op = torch.ops.prepacked.linear_clamp_prepack(
            torch.randn(10, 10), torch.randn(10)
        )

        m = M(linear_op)
        inp = (torch.randn(1, 10),)
        self._check_equal_ts_ep_converter(m, inp, ["script"])


if __name__ == "__main__":
    run_tests()
