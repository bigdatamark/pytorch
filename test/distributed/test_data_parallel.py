# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for torch.nn.DataParallel and the functional helpers in torch.nn.parallel
# (data_parallel, parallel_apply, replicate, scatter, gather). Most tests need at least two CUDA
# devices and are skipped (passing in sandcastle) otherwise. Import time: defines NO_NCCL and
# rebinds gradcheck and _assertGradAndGradgradChecks via partial so batched grad checking is
# disabled (data parallel does not support it). Device-type tests are instantiated at the bottom
# of the file; running the file as a script enables the default dtype check and runs the tests.

import contextlib
import functools
import io
from collections import OrderedDict
from copy import deepcopy
from itertools import product

import torch
import torch.nn.functional as F
import torch.nn.parallel as dp
from torch import nn
from torch.cuda.amp import autocast
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_utils import (
    _assertGradAndGradgradChecks,
    dtype2prec_DONTUSE,
    gradcheck,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


NO_NCCL = not hasattr(torch.distributed, "ProcessGroupNCCL")

# batched grad doesn't support data parallel
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
_assertGradAndGradgradChecks = functools.partial(
    _assertGradAndGradgradChecks, check_batched_grad=False
)


class TestDataParallel(TestCase):
    # [CLASS DOC] Multi-GPU tests of DataParallel and the parallel helper functions:
    # replication, scatter and gather of tensors and nested structures, forward/backward
    # equivalence with single device execution, device placement checks, exceptions, autocast,
    # complex parameters and gradient memory layouts.
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_buffers_requiring_grad(self):
        # [FUNCTION DOC] Wraps a module with a double buffer that requires grad and a detached
        # copy in DataParallel on devices 0 and 1, then runs gradcheck on the output as a
        # function of the grad-requiring buffer.
        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module holding two buffers (one requiring grad, one a detached
            # clone) whose forward multiplies the input by the first and adds the second.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Registers the given tensor as buffer t_rg and a
                # detached clone of it as buffer t_not_rg.
                super().__init__()
                self.t_rg = nn.Buffer(t)
                self.t_not_rg = nn.Buffer(t.detach().clone())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the grad-requiring buffer plus
                # the plain buffer.
                return x * self.t_rg + self.t_not_rg

        m = TestModule(
            torch.randn(100, device="cuda", requires_grad=True, dtype=torch.double)
        )
        self.assertTrue(m.t_rg.requires_grad)

        dpm = nn.DataParallel(m, [0, 1])
        inp = torch.randn(2, 100, device="cuda", dtype=torch.double)

        def fn(t):
            # [FUNCTION DOC] HELPER: Function for gradcheck that ignores its argument and runs
            # the DataParallel module on a fixed input.
            return dpm(inp)

        gradcheck(fn, (m.t_rg,))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_rnn(self):
        # [FUNCTION DOC] Trains an LSTM module with flatten_parameters in forward on one device
        # and its DataParallel deep copy with the same SGD step, first checking DataParallel
        # does not crash under no_grad, and asserts the resulting parameters match.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bidirectional single-layer LSTM; forward
            # flattens the RNN parameters before running it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM layer (input 300, hidden 1024, batch
                # first, bidirectional).
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    300, 1024, 1, batch_first=True, bidirectional=True
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Flattens the RNN parameters and returns the LSTM
                # output.
                self.rnn.flatten_parameters()
                return self.rnn(x)

        def step(model):
            # [FUNCTION DOC] HELPER: One SGD step: runs the model on a ones input on device 0,
            # mean squared error loss against zeros on the first output, backward, optimizer
            # step.
            opt = torch.optim.SGD(model.parameters(), lr=10)
            input = torch.ones(4, 4, 300).to(0)
            output = model(input)
            loss = F.mse_loss(output[0], torch.zeros_like(output[0]))
            loss.backward()
            opt.step()

        with torch.no_grad():
            model = TestModule().to(0)
            model_dp = torch.nn.DataParallel(deepcopy(model))

            # make sure DP does not crash when grad is disabled.
            # See #21108
            model_dp(torch.rand(2, 4, 300).to(0))

        step(model)
        step(model_dp)

        for p1, p2 in zip(model.parameters(), model_dp.parameters()):
            self.assertEqual(p1, p2)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_lazy_linear(self):
        # [FUNCTION DOC] Asserts that wrapping a LazyLinear in DataParallel and running it on an
        # input raises ValueError mentioning an uninitialized parameter.
        with self.assertRaisesRegex(
            ValueError, "Attempted to use an uninitialized parameter"
        ):
            model_dp = torch.nn.DataParallel(torch.nn.LazyLinear(10).to(0))
            model_dp(torch.rand(10, 10).to(0))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply(self):
        # [FUNCTION DOC] Runs parallel_apply on two linear modules on cuda:0 and cuda:1, with
        # inputs given either as tuples of positional args or as single objects, and asserts
        # outputs equal the direct module calls.
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        expected1 = l1(i1)
        expected2 = l2(i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply_autocast(self):
        # [FUNCTION DOC] Like test_parallel_apply but expected outputs are computed under
        # autocast and parallel_apply is also called under autocast; asserts the outputs match.
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        with autocast():
            expected1 = l1(i1)
            expected2 = l2(i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            with autocast():
                outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA unavailable")
    def test_parallel_apply_passes_exception(self):
        # [FUNCTION DOC] Runs parallel_apply on a module whose forward raises KeyError (same
        # module and device twice) and asserts the error propagates as a KeyError whose message
        # names the replica, device 0 and the original traceback.
        # we define and instantiate a module that will throw a KeyError
        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward always raises KeyError by looking up a
            # missing dict key.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Raises KeyError by indexing an empty dict.
                return {}["wonderful"]

        l1 = TestModule().to("cuda", torch.float)
        # and check that parallel_apply passes on the exception
        # (we can use a single device twice for this test)
        with self.assertRaisesRegex(
            KeyError,
            "Caught KeyError in replica \\d "
            "on device 0.\nOriginal Traceback"
            "[\\s\\S]+wonderful",
        ):
            dp.parallel_apply(modules=(l1, l1), inputs=(None, None))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_multiple_input(self):
        # [FUNCTION DOC] Checks data_parallel and DataParallel with multiple positional inputs,
        # a float argument, optional keyword argument and module_kwargs, on various device id
        # orders and single device; compares outputs and input gradients with the single device
        # result.
        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward multiplies two tensors and scales by a
            # float, optionally adding a third tensor.
            def forward(self, var1, var2, float1, var3=None):
                # [FUNCTION DOC] HELPER: Returns the scaled product, adding var3 first if it was
                # given.
                if var3 is None:
                    return float1 * (var1 * var2)
                else:
                    return float1 * (var1 * var2 + var3)

        m = TestModule()
        var1 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        var2 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        var3 = torch.randn(5, 5, dtype=torch.float, requires_grad=False)

        float1 = torch.randn(1).item()

        expected = m(var1, var2, float1)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        def local_test(out):
            # [FUNCTION DOC] HELPER: Zeroes input gradients, backpropagates the sum of the given
            # output and asserts the output and both input gradients equal the expected ones.
            with torch.no_grad():
                var1.grad.fill_(0.0)
                var2.grad.fill_(0.0)
            loss = out.sum()
            loss.backward()
            self.assertEqual(out, expected)
            self.assertEqual(gvar1_exp, var1.grad)
            self.assertEqual(gvar2_exp, var2.grad)

        out = dp.data_parallel(m, (var1, var2, float1), (0, 1))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (1, 0))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (0,))
        local_test(out)

        with torch.no_grad():
            var1.grad.fill_(0.0)
            var2.grad.fill_(0.0)
        expected = m(var1, var2, float1, var3=var3)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        dpm = nn.DataParallel(TestModule())
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        dpm = nn.DataParallel(TestModule(), device_ids=[0])
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        kwarg_wrap = {"var3": var3}
        out = dp.data_parallel(
            m, (var1, var2, float1), (0, 1), module_kwargs=kwarg_wrap
        )
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (0,), module_kwargs=kwarg_wrap)
        local_test(out)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_small_back(self):
        # [FUNCTION DOC] Runs data_parallel on a small batch over devices 0 and 1 and asserts
        # the result equals running the linear layer directly.
        l = nn.Linear(10, 5).float().cuda()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out, l(i))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_model_device(self):
        # [FUNCTION DOC] Tests the check at forward time that the module's parameters and
        # buffers are on the first device id, through both DataParallel and the functional
        # data_parallel, over many placements of modules (single layer and sequential) and
        # device id lists, expecting RuntimeError when mismatched and success otherwise.
        r"""Test device[0] check at forward time."""
        l = nn.Linear(2, 2)
        inp = torch.randn(2, 2)
        inp_cuda0 = inp.cuda(0)
        inp_cuda1 = inp.cuda(1)

        error_msg = "module must have its parameters and buffers on device {}"

        @contextlib.contextmanager
        def dummy_ctx_manager():
        # [FUNCTION DOC] HELPER: Context manager that does nothing, used where no error is
        # expected.
            yield

        def test(inner_m, dp_device, inp, device_ids, should_fail):
        # [FUNCTION DOC] HELPER: Runs one placement case: derives the expected device from the
        # first device id (all visible devices if none), builds DataParallel (optionally moved
        # to a device), runs it and the functional version inside a context that expects
        # RuntimeError or does nothing.
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))

            if isinstance(device_ids[0], torch.device):
                expect_device = device_ids[0]
            else:
                expect_device = torch.device(f"cuda:{device_ids[0]}")

            if should_fail:

                def assert_correct():
                # [FUNCTION DOC] HELPER: Returns an assertRaisesRegex context for a RuntimeError
                # with the message about the module needing its parameters and buffers on the
                # expected device.
                    return self.assertRaisesRegex(
                        RuntimeError, error_msg.format(expect_device)
                    )

            else:
                assert_correct = dummy_ctx_manager

            # test DataParallel module
            dpm = nn.DataParallel(inner_m, device_ids)
            if dp_device is not None:
                dpm = dpm.to(dp_device)

            with assert_correct():
                dpm(inp)

            # test functional
            with assert_correct():
                nn.parallel.data_parallel(inner_m.to(dp_device), inp, device_ids)

        test(l.to("cpu"), None, inp, None, should_fail=True)
        test(l.cuda(1), None, inp_cuda0, None, should_fail=True)
        test(l.cuda(), None, inp_cuda0, [1, 0], should_fail=True)

        test(l.cuda(), None, inp_cuda0, None, should_fail=False)
        test(l.cpu(), "cuda", inp_cuda0, None, should_fail=False)
        test(l.cuda(1), None, inp_cuda1, [1, 0], should_fail=False)
        test(l.cpu(), "cuda:1", inp_cuda1, [1, 0], should_fail=False)

        s = nn.Sequential(l.cpu())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(deepcopy(l).cpu(), l.cuda())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda(1))
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda())
        test(s, None, inp, None, should_fail=False)
        test(s, None, inp, [0, 1], should_fail=False)
        test(s, None, inp, [1, 0], should_fail=True)
        test(s.cpu(), None, inp, [1, 0], should_fail=True)
        test(s.cuda(1), None, inp, [1, 0], should_fail=False)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_model_no_refcycles(self):
        # [FUNCTION DOC] Asserts that running a DataParallel model on a GPU after a garbage
        # collection leaves no reference cycles (a second collection finds zero objects).
        # Python 2.7 will create reference cycles with the following
        # Module on multiple GPUs, but Python 3 shouldn't unless
        # there are refcycles on the PyTorch side (or the defined module)
        import gc

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a single Linear layer from one input to one
            # output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        gc.collect()
        model = nn.DataParallel(Model().cuda())
        data = torch.randn(1, device="cuda")
        model(data)

        refcycles = gc.collect()
        self.assertEqual(refcycles, 0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_no_grad(self):
        # [FUNCTION DOC] Checks that DataParallel replicas run with gradient mode disabled when
        # called under no_grad (the layer asserts grad is off), and that the same call outside
        # no_grad raises AssertionError.
        test = self

        class Layer(nn.Module):
            # [CLASS DOC] HELPER: Module asserting through the test case that grad mode is
            # disabled, then returning its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Asserts grad mode is off and returns the input.
                test.assertFalse(torch.is_grad_enabled())
                return x

        l = Layer()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")
        with torch.no_grad():
            dp.data_parallel(l, i, (0, 1))
        self.assertRaises(AssertionError, lambda: dp.data_parallel(l, i, (0, 1)))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel(self):
        # [FUNCTION DOC] Runs data_parallel with device lists (0,1) and (1,0) on an input on
        # cuda:1 and compares outputs, output device (the first id) and parameter gradients to
        # single device results; also calls it with no device ids.
        l = nn.Linear(10, 5).float().cuda()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda:1")
        l.cuda(1)
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out, expected_out)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad, expected)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_sparse(self):
        # [FUNCTION DOC] Same as test_data_parallel but with a sparse Embedding module; sparse
        # gradients are coalesced before comparison.
        l = nn.Embedding(10, 5, sparse=True).to("cuda:1")
        i = torch.randint(10, (20, 5), device="cuda:1", dtype=torch.long)
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out, expected_out)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.coalesce(), expected.coalesce())

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_output(self):
        # [FUNCTION DOC] Runs data_parallel on a module returning nested lists, tuples, an
        # ordered dict and tensors, and asserts values equal the plain function and the
        # container types and structure are preserved.
        def fn(input):
            # [FUNCTION DOC] HELPER: Builds the nested output structure (list, tuple, list,
            # ordered dict) from the input.
            return [
                input,
                (input.sin(), input.cos(), [input.add(1)]),
                input,
                OrderedDict(a=input, b=[input.sin()]),
            ]

        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the nested structure from fn.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns fn applied to the input.
                return fn(input)

        i = torch.randn(2, 2).float().cuda(1)
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), i, gpus)
        self.assertEqual(output, fn(i))
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], tuple)
        self.assertIsInstance(output[1][0], torch.Tensor)
        self.assertIsInstance(output[1][1], torch.Tensor)
        self.assertIsInstance(output[1][2], list)
        self.assertIsInstance(output[1][2][0], torch.Tensor)
        self.assertIsInstance(output[2], torch.Tensor)
        self.assertIsInstance(output[3], dict)
        self.assertEqual(len(output[3]), 2)
        self.assertIn("a", output[3])
        self.assertIn("b", output[3])
        self.assertIsInstance(output[3]["a"], torch.Tensor)
        self.assertIsInstance(output[3]["b"], list)
        self.assertIsInstance(output[3]["b"][0], torch.Tensor)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_input(self):
        # [FUNCTION DOC] Runs data_parallel with a nested tuple input and asserts the result
        # equals the plain function applied to the same input.
        def fn(input):
            # [FUNCTION DOC] HELPER: Returns the first element of the second element of the
            # input tuple.
            return input[1][0]

        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module taking variadic inputs and returning the nested element
            # selected by fn.
            def forward(self, *input):
                # [FUNCTION DOC] HELPER: Returns fn applied to the argument tuple.
                return fn(input)

        i = torch.randn(20, 3, dtype=torch.float, device="cuda:1")
        input = (i.cos(), (i.sin(), i), i.sin())
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), input, gpus)
        self.assertEqual(output, fn(input))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_module_zero_inputs(self):
        # [FUNCTION DOC] Checks DataParallel and data_parallel on a module with no inputs for
        # one and two devices: output is on device 0 and equals a ones tensor.
        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module without inputs whose forward builds an eye tensor on
            # cuda:0 and returns it plus its complement (all ones).
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a 2 by 3 ones tensor on cuda:0 computed from an
                # identity matrix.
                t = torch.eye(2, 3, device="cuda:0")
                return t + (1 - t)

        def test_helper(output, expected):
            # [FUNCTION DOC] HELPER: Asserts the output is on device 0 and equals the expected
            # tensor.
            self.assertEqual(output.get_device(), 0)
            self.assertEqual(output, expected)

        expected = torch.ones(2, 3, device="cuda:0")
        model = TestModule()

        test_helper(nn.DataParallel(model, [0])(), expected)
        test_helper(nn.DataParallel(model, [0, 1])(), expected)
        test_helper(dp.data_parallel(model, None, [0]), expected)
        test_helper(dp.data_parallel(model, (), [0, 1]), expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_device_args(self):
        # [FUNCTION DOC] Checks data_parallel accepts torch.device objects for output_device and
        # device_ids, comparing against the direct module output.
        cuda0 = torch.device("cuda:0")
        cuda1 = torch.device("cuda:1")

        # test output_device
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(0, 1), output_device=cuda0)
        self.assertEqual(out, l(i))

        # test device_ids
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(cuda0, cuda1), output_device=cuda0)
        self.assertEqual(out, l(i))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_function_deletion(self):
        # [FUNCTION DOC] Regression test for issue 16532: computes a gradient penalty through
        # DataParallel with create_graph, backpropagates and asserts the weight gradient is 0.25
        # per element and the bias gradient is zero.
        # this test case is originated from #16532
        def gradient_penalty(net, x):
            # [FUNCTION DOC] HELPER: Returns the mean of the gradient of the network output with
            # respect to its input, computed with create_graph and retain_graph.
            output = net(x)
            loss = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=x.new_ones(output.size()),
                create_graph=True,
                retain_graph=True,
            )[0].mean()
            return loss

        net = nn.Linear(4, 1).cuda()
        dpn = nn.DataParallel(net, [0, 1])
        x = torch.ones(2, 4, requires_grad=True).cuda()

        dpn.zero_grad()
        loss = gradient_penalty(dpn, x)
        loss.backward()
        grads = [p.grad for p in net.parameters()]
        self.assertEqual(2, len(grads))
        self.assertEqual(
            torch.tensor([[0.25, 0.25, 0.25, 0.25]], device="cuda:0"), grads[0]
        )
        self.assertEqual(torch.tensor([0.0], device="cuda:0"), grads[1])

    def _test_scatter(self, tensor):
        # [FUNCTION DOC] HELPER: Scatters a tensor over devices 0 and 1 and asserts two chunks
        # (first two rows and the rest) on the right devices, that backward through a chunk gives
        # matching input gradient with zeros elsewhere, and runs gradient and gradgrad checks.
        x = tensor.detach().requires_grad_()
        result = dp.scatter(x, (0, 1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x[:2])
        self.assertEqual(result[0].get_device(), 0)
        self.assertEqual(result[1], x[2:])
        self.assertEqual(result[1].get_device(), 1)
        grad = result[0].detach().clone().fill_(2)
        result[0].backward(grad)
        self.assertEqual(x.grad[:2], grad)
        self.assertEqual(x.grad[2:], grad.clone().zero_())
        _assertGradAndGradgradChecks(self, lambda y: dp.scatter(y, (0, 1)), (x,))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_cpu(self):
        # [FUNCTION DOC] Runs the scatter helper check on a CPU double tensor.
        self._test_scatter(torch.randn((4, 4), dtype=torch.double))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_gpu(self):
        # [FUNCTION DOC] Runs the scatter helper check on a CUDA double tensor.
        self._test_scatter(torch.randn((4, 4), dtype=torch.double).cuda())

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_complex_parameters(self):
        # [FUNCTION DOC] Trains a complex-valued two layer model on one device and a
        # DataParallel copy for five epochs with the same seeds and SGD, asserting outputs,
        # gradients, and weights agree within a tolerance that grows with the epoch.
        # test that complex parameters are handled correctly by DataParallel
        class ComplexModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Two linear layers with complex float weights and a complex
            # nonlinearity between them.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two complex float linear layers.
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 16, dtype=torch.cfloat)
                self.fc2 = torch.nn.Linear(16, 8, dtype=torch.cfloat)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer, multiplies by a complex
                # exponential of the real part, applies the second layer.
                x = self.fc1(x)
                x = x * torch.exp(1j * x.real)  # complex nonlinearity
                return self.fc2(x)

        torch.manual_seed(42)
        model_single = ComplexModel().cuda()
        opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)

        torch.manual_seed(42)
        model_dp_base = ComplexModel().cuda()
        model_dp = torch.nn.DataParallel(model_dp_base)
        opt_dp = torch.optim.SGD(model_dp_base.parameters(), lr=0.01)

        num_epochs = 5
        batch_size = 8

        for epoch in range(num_epochs):
            torch.manual_seed(epoch * 100)
            x = torch.randn(batch_size, 8, dtype=torch.cfloat, device="cuda")

            # tolerance grows with epochs due to accumulated numerical differences
            atol = 1e-5 * (10**epoch)

            out_single = model_single(x)
            out_dp = model_dp(x)

            self.assertEqual(out_single.shape, out_dp.shape)
            self.assertEqual(out_single.dtype, out_dp.dtype)
            self.assertTrue(
                torch.allclose(out_single, out_dp, atol=atol),
                lambda msg: f"{msg}\nEpoch {epoch}: outputs differ",
            )

            loss_single = out_single.abs().sum()
            loss_dp = out_dp.abs().sum()

            opt_single.zero_grad()
            opt_dp.zero_grad()

            loss_single.backward()
            loss_dp.backward()

            opt_single.step()
            opt_dp.step()

            for (n1, p1), (n2, p2) in zip(
                model_single.named_parameters(), model_dp_base.named_parameters()
            ):
                self.assertEqual(p1.grad.shape, p2.grad.shape)
                self.assertEqual(p1.grad.dtype, p2.grad.dtype)
                self.assertTrue(
                    torch.allclose(p1.grad, p2.grad, atol=atol),
                    lambda msg: f"{msg}\nEpoch {epoch}: gradients differ for {n1}",
                )

            for (n1, p1), (n2, p2) in zip(
                model_single.named_parameters(), model_dp_base.named_parameters()
            ):
                self.assertTrue(
                    torch.allclose(p1.data, p2.data, atol=atol),
                    lambda msg: f"{msg}\nEpoch {epoch}: weights differ for {n1} after optimizer step",
                )

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_complex_mixed_parameters(self):
        # [FUNCTION DOC] Same comparison as the complex parameters test but with a model that
        # has both real and complex branches returning two outputs; checks dtypes, outputs,
        # gradients and weights over five epochs.
        # test that mix complex and real parameters are handled correctly by DataParallel
        class MixedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a real two layer branch (with relu) and a complex
            # two layer branch that returns both outputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two real and two complex float linear layers.
                super().__init__()
                self.real_fc1 = torch.nn.Linear(8, 16)
                self.real_fc2 = torch.nn.Linear(16, 8)
                self.complex_fc1 = torch.nn.Linear(8, 16, dtype=torch.cfloat)
                self.complex_fc2 = torch.nn.Linear(16, 8, dtype=torch.cfloat)

            def forward(self, x_real, x_complex):
                # [FUNCTION DOC] HELPER: Runs the real input through the real branch and the
                # complex input through the complex branch; returns both results.
                r = torch.relu(self.real_fc1(x_real))
                r = self.real_fc2(r)
                c = self.complex_fc1(x_complex)
                c = c * torch.exp(1j * c.real)
                c = self.complex_fc2(c)
                return r, c

        torch.manual_seed(42)
        model_single = MixedModel().cuda()
        opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)

        torch.manual_seed(42)
        model_dp_base = MixedModel().cuda()
        model_dp = torch.nn.DataParallel(model_dp_base)
        opt_dp = torch.optim.SGD(model_dp_base.parameters(), lr=0.01)

        num_epochs = 5
        batch_size = 8

        for epoch in range(num_epochs):
            torch.manual_seed(epoch * 100)
            x_real = torch.randn(batch_size, 8, device="cuda")
            x_complex = torch.randn(batch_size, 8, dtype=torch.cfloat, device="cuda")

            # tolerance grows with epochs due to accumulated numerical differences
            atol = 1e-5 * (10**epoch)

            out_r_single, out_c_single = model_single(x_real, x_complex)
            out_r_dp, out_c_dp = model_dp(x_real, x_complex)

            self.assertEqual(out_r_single.dtype, torch.float32)
            self.assertEqual(out_c_single.dtype, torch.cfloat)
            self.assertTrue(torch.allclose(out_r_single, out_r_dp, atol=atol))
            self.assertTrue(torch.allclose(out_c_single, out_c_dp, atol=atol))

            loss_single = out_r_single.sum() + out_c_single.abs().sum()
            loss_dp = out_r_dp.sum() + out_c_dp.abs().sum()

            opt_single.zero_grad()
            opt_dp.zero_grad()

            loss_single.backward()
            loss_dp.backward()

            opt_single.step()
            opt_dp.step()

            for (n1, p1), (n2, p2) in zip(
                model_single.named_parameters(), model_dp_base.named_parameters()
            ):
                self.assertEqual(p1.grad.shape, p2.grad.shape)
                self.assertEqual(p1.grad.dtype, p2.grad.dtype)
                self.assertTrue(
                    torch.allclose(p1.grad, p2.grad, atol=atol),
                    lambda msg: f"{msg}\nEpoch {epoch}: gradients differ for {n1}",
                )

            for (n1, p1), (n2, p2) in zip(
                model_single.named_parameters(), model_dp_base.named_parameters()
            ):
                self.assertTrue(
                    torch.allclose(p1.data, p2.data, atol=atol),
                    lambda msg: f"{msg}\nEpoch {epoch}: weights differ for {n1} after optimizer step",
                )

    def _test_gather(self, output_device):
        # [FUNCTION DOC] HELPER: Gathers two double tensors from cuda:0 and cuda:1 (output device
        # or CPU if minus one) and checks size, content, device, backward gradients and gradient
        # checks; repeats with scalar inputs, which are stacked into a vector.
        inputs = (
            torch.randn(2, 4, device="cuda:0", requires_grad=True, dtype=torch.double),
            torch.randn(2, 4, device="cuda:1", requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([4, 4]))
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn((4, 4), dtype=torch.double)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[:2])
        self.assertEqual(inputs[1].grad, grad[2:])
        _assertGradAndGradgradChecks(
            self, lambda x, y: dp.gather((x, y), output_device), inputs
        )

        # test scalar inputs, should stack into a vector in this case
        inputs = (
            torch.randn((), device="cuda:0", requires_grad=True, dtype=torch.double),
            torch.randn((), device="cuda:1", requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([2]))
        self.assertEqual(result[0], inputs[0])
        self.assertEqual(result[1], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn(2, dtype=torch.double)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[0])
        self.assertEqual(inputs[1].grad, grad[1])
        _assertGradAndGradgradChecks(
            self, lambda x, y: dp.gather((x, y), output_device), inputs
        )

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        # [FUNCTION DOC] Runs the gather helper check with CPU output (device minus one).
        self._test_gather(-1)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        # [FUNCTION DOC] Runs the gather helper check with output device 0.
        self._test_gather(0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_different_len_dicts(self):
        # [FUNCTION DOC] Asserts that gathering dicts with different key sets raises ValueError.
        inputs = (
            {"a": torch.randn(1, 2, requires_grad=True, device="cuda:0")},
            {
                "b": torch.randn(1, 2, requires_grad=True, device="cuda:1"),
                "a": torch.randn(1, 2, requires_grad=True, device="cuda:1"),
            },
        )
        with self.assertRaises(ValueError):
            _ = dp.gather(inputs, target_device=0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate(self):
        # [FUNCTION DOC] Replicates a linear module over a tuple and a list of devices and
        # asserts each replica's parameters live on its device and produce the same output.
        module = nn.Linear(10, 5).float().cuda()
        input = torch.randn(2, 10, dtype=torch.float, device="cuda")
        expected_output = module(input)
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(module, devices)
            for i, replica in enumerate(replicas):
                for p in replica.parameters():
                    self.assertEqual(p.get_device(), i)
                replica_input = input.cuda(i)
                self.assertEqual(replica(replica_input), expected_output)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate_buffers(self):
        # [FUNCTION DOC] Replicates a module with batch norm and asserts each replica's running
        # mean, running variance and batch counter buffers are on the replica's device.
        net = nn.Module()
        net.bn = nn.BatchNorm2d(10)
        net.cuda()
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(net, devices)
            for i, replica in enumerate(replicas):
                self.assertEqual(
                    replica.bn.running_mean.get_device(),
                    i,
                    msg="buffer on wrong device",
                )
                self.assertEqual(
                    replica.bn.running_var.get_device(), i, msg="buffer on wrong device"
                )
                self.assertEqual(
                    replica.bn.num_batches_tracked.get_device(),
                    i,
                    msg="buffer on wrong device",
                )

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_zero_grad(self):
        # [FUNCTION DOC] Runs a DataParallel module whose forward calls zero_grad and asserts
        # that a UserWarning says zero_grad on a DataParallel-created module has no effect.
        # zero_grad should warn about using gradients inside forward

        class Net(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that, in forward, expects a UserWarning from calling
            # zero_grad on itself; keeps the test case reference.
            def __init__(self, testcase):
                # [FUNCTION DOC] HELPER: Stores the test case used for assertions.
                super().__init__()
                self._testcase = testcase

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Asserts through the test case that zero_grad emits the
                # warning, then returns the input.
                with self._testcase.assertWarnsRegex(
                    UserWarning,
                    r"Calling \.zero_grad\(\) from a module created with nn\.DataParallel\(\) has no effect.",
                ):
                    self.zero_grad()
                return x

        module = Net(self).cuda()
        dpm = dp.DataParallel(module)
        dpm(torch.rand(4, 3, 6, 5))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_autocast(self):
        # [FUNCTION DOC] Checks that a Linear module with autocast on forward, run through
        # DataParallel, yields float16 output.
        class Model(torch.nn.Linear):
            # [CLASS DOC] HELPER: Linear layer 8 by 8 whose forward runs under autocast for
            # CUDA.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes an 8 by 8 linear layer.
                super().__init__(8, 8)

            @torch.autocast(device_type="cuda")
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the parent forward under CUDA autocast.
                return super().forward(input)

        model = dp.DataParallel(Model().cuda().to(dtype=torch.float32))
        input = torch.randn((8, 8), dtype=torch.float32, device="cuda")
        self.assertTrue(model(input).dtype is torch.float16)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_save_replica_module(self):
        # [FUNCTION DOC] Regression test (gh-37182): replicates a module over two devices with
        # and without detach and saves the replicas to an in-memory buffer; passes if no error
        # is raised.
        # DataParallel replicas can be saved (gh-37182)
        module = torch.nn.Linear(8, 8).cuda()
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=False)
        data = io.BytesIO()
        torch.save(dpm, data)
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=True)
        torch.save(dpm, data)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_strided_grad_layout(self):
        # [FUNCTION DOC] For combinations of memory formats and dtypes of four conv layers,
        # compares gradients of a single-device model and its DataParallel copy over two
        # iterations, checking weight gradients keep the layer's memory format and values match
        # within a tolerance (looser for half). On failure prints case information and
        # re-raises.
        # KNOWN ISSUE: the message variable model_msg formats the name of the imported dtypes
        # decorator instead of the per-case dtype_list, so the printed dtype information is
        # wrong.
        class ConvNet(nn.Module):
            def __init__(self, layouts, dtype_list):
                # [FUNCTION DOC] HELPER: Builds four conv layers with the given memory formats
                # and dtypes and records the dtype list.
                super().__init__()
                self.dtypes = dtype_list
                self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(
                    memory_format=layouts[0], dtype=dtype_list[0]
                )
                self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(
                    memory_format=layouts[1], dtype=dtype_list[1]
                )
                self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(
                    memory_format=layouts[2], dtype=dtype_list[2]
                )
                self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(
                    memory_format=layouts[3], dtype=dtype_list[3]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Casts between the layers' dtypes through the four
                # convolutions and returns the result.
                x = x.to(self.dtypes[0])
                x = self.conv0(x).to(self.dtypes[1])
                x = self.conv1(x).to(self.dtypes[2])
                x = self.conv2(x).to(self.dtypes[3])
                x = self.conv3(x)
                return x

        layer_formats = (
            [torch.contiguous_format] * 4,
            [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
            [torch.channels_last] * 4,
        )
        layer_dtypes = (
            [torch.float] * 4,
            [torch.float] * 2 + [torch.half] * 2,
            [torch.half] * 4,
        )

        ndevs = torch.cuda.device_count()
        input = torch.randn(ndevs * 8, 8, 8, 8, device="cuda:0", dtype=torch.float)
        target = torch.randn(ndevs * 8, 8, 4, 4, device="cuda:0", dtype=torch.float)
        device_ids = list(range(ndevs))

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=False
        ):
            for formats, dtype_list in product(layer_formats, layer_dtypes):
                model_msg = f"formats = {formats} dtypes = {dtypes}"
                try:
                    m = ConvNet(formats, dtype_list).cuda(device="cuda:0")
                    m_dp = dp.DataParallel(deepcopy(m), device_ids=device_ids)
                    opt = torch.optim.SGD(m.parameters(), lr=0.1)
                    opt_dp = torch.optim.SGD(m_dp.parameters(), lr=0.1)
                    has_half = any(p.dtype is torch.half for p in m.parameters())
                    tol = 3.0e-3 if has_half else 1.0e-5
                except BaseException:
                    # Prints case-specific debugging info to narrow down failing case.
                    print(
                        "Caught exception during model creation for " + model_msg,
                        flush=True,
                    )
                    raise
                # 2 iters:  First iter creates grads, second iter tries zeroed grads.
                for it in range(2):
                    iter_msg = f"iter = {it} " + model_msg
                    named_msg = iter_msg
                    try:
                        F.mse_loss(m(input).float(), target).backward()
                        F.mse_loss(m_dp(input).float(), target).backward()
                        for i, ((layer_name, m_child), m_dp_child) in enumerate(
                            zip(m.named_children(), m_dp.module.children())
                        ):
                            named_msg = layer_name + ".weight " + iter_msg
                            self.assertTrue(
                                m_child.weight.grad.is_contiguous(
                                    memory_format=formats[i]
                                ),
                                named_msg,
                            )
                            self.assertTrue(
                                m_dp_child.weight.grad.is_contiguous(
                                    memory_format=formats[i]
                                ),
                                named_msg,
                            )
                            for (param_name, p), p_dp in zip(
                                m_child.named_parameters(), m_dp_child.parameters()
                            ):
                                named_msg = (
                                    layer_name + "." + param_name + " " + iter_msg
                                )
                                self.assertEqual(p.grad, p_dp.grad, rtol=tol, atol=tol)
                        opt.step()
                        opt_dp.step()
                        opt.zero_grad()
                        opt_dp.zero_grad()
                    except BaseException:
                        # Makes sure we still get info if an error occurred somewhere other than the asserts.
                        print(
                            "Caught exception during iterations at " + named_msg,
                            flush=True,
                        )
                        raise

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parameter_list_dict_replica(self):
        # [FUNCTION DOC] Runs a DataParallel module holding a ParameterList and then a
        # ParameterDict; inside forward a check function asserts the replica's parameters equal
        # the originals, require grad and have a grad function.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a data container and a check function that
            # forward runs before returning its input.
            def __init__(self, data, check_fn):
                # [FUNCTION DOC] HELPER: Stores the data container and check function.
                super().__init__()
                self.data = data
                self.check_fn = check_fn

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Calls the check function with the module and returns
                # the input.
                self.check_fn(self)
                return inp

        p1 = torch.nn.Parameter(torch.rand(10))
        p2 = torch.nn.Parameter(torch.rand(10))
        key0 = 0
        key1 = 1

        def check_fn(self_):
            # [FUNCTION DOC] HELPER: Asserts replica parameters at the current keys equal the
            # originals, require grad and have a grad_fn (the keys are rebound later in the
            # test).
            self.assertEqual(p1, self_.data[key0])
            self.assertEqual(p2, self_.data[key1])
            self.assertTrue(self_.data[key0].requires_grad)
            self.assertTrue(self_.data[key1].requires_grad)
            self.assertIsNotNone(self_.data[key0].grad_fn)
            self.assertIsNotNone(self_.data[key1].grad_fn)

        module = MyMod(torch.nn.ParameterList([p1, p2]), check_fn).cuda()
        model = dp.DataParallel(module)
        input = torch.randn((8, 8), device="cuda")

        # Runs the check_fn
        model(input)

        key0 = "0"
        key1 = "1"
        module = MyMod(torch.nn.ParameterDict({"0": p1, "1": p2}), check_fn).cuda()
        model = dp.DataParallel(module)
        input = torch.randn((8, 8), device="cuda")

        # Runs the check_fn
        model(input)


class TestDataParallelDeviceType(TestCase):
    # [CLASS DOC] Device-generic DataParallel tests (CUDA only, not meta), run for float, double
    # and half dtypes and instantiated for device types at the bottom of the file.
    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module(self, device, dtype):
        # [FUNCTION DOC] Runs a Linear layer through DataParallel and asserts the output is on
        # device 0 and equals the direct output within a dtype tolerance.
        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only(self, device, dtype):
        # [FUNCTION DOC] Calls a DataParallel module with only a keyword argument and asserts
        # the result equals the direct output on device 0.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module delegating to a Linear layer captured from the
            # enclosing scope.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the enclosing linear layer.
                super().__init__()
                self.l = l

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the stored linear layer.
                return self.l(input)

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input=i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_list(self, device, dtype):
        # [FUNCTION DOC] Calls DataParallel with a keyword-only dict containing the data and an
        # empty list entry, and asserts the output matches.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module delegating to a captured Linear layer using the data
            # entry of a dict input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the enclosing linear layer.
                super().__init__()
                self.l = l

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the data entry of the input
                # dict.
                return self.l(input["data"])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={"data": i, "unused": []})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_dict(self, device, dtype):
        # [FUNCTION DOC] Calls DataParallel with a keyword-only dict containing the data and an
        # empty dict entry, and asserts the output matches.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module delegating to a captured Linear layer using the data
            # entry of a dict input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the enclosing linear layer.
                super().__init__()
                self.l = l

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the data entry of the input
                # dict.
                return self.l(input["data"])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={"data": i, "unused": {}})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_tuple(self, device, dtype):
        # [FUNCTION DOC] Calls DataParallel with a keyword-only dict containing the data and an
        # empty tuple entry, and asserts the output matches.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module delegating to a captured Linear layer using the data
            # entry of a dict input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the enclosing linear layer.
                super().__init__()
                self.l = l

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the data entry of the input
                # dict.
                return self.l(input["data"])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={"data": i, "unused": ()})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)


instantiate_device_type_tests(TestDataParallelDeviceType, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
