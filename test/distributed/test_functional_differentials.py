# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests of the autograd formulas of the functional collectives (all_reduce,
# all_gather, reduce_scatter, all_to_all_single and their coalesced forms): forward results,
# backward results (all_reduce backward is all_reduce, all_gather backward is reduce_scatter,
# reduce_scatter backward is all_gather, all_to_all backward is all_to_all),
# torch.library.opcheck checks for the c10d functional ops and their _autograd variants, and
# torch.compile integration. Import-time side effects: exits with status 0 if distributed is
# unavailable; builds devices as cpu plus the current accelerator type, if any; DEVICE stays
# cuda. Running as a script calls run_tests.

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fcols
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    MultiThreadedTestCase,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# Determine available devices
DEVICE = "cuda"
devices = ["cpu"]
if acc := torch.accelerator.current_accelerator(True):
    devices += [acc.type]


def with_comms(func=None):
    # [FUNCTION DOC] Decorator for test methods. When called with no function it returns a
    # partial that decorates later; otherwise wraps func so that if CUDA is available with fewer
    # accelerators than world size it exits with the skip exit code, then creates a process
    # group via create_pg for DEVICE, stores it as pg and the device name as device on the test,
    # runs func and always destroys the process group.
    # KNOWN ISSUE: the skip check only looks at CUDA while devices may include other
    # accelerators, and DEVICE is hard-coded to cuda regardless of the accelerator found.
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that applies the skip check, process group setup and
        # cleanup described above, forwarding all arguments to the test function.
        if (
            torch.cuda.is_available()
            and torch.accelerator.device_count() < self.world_size
        ):
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=DEVICE)
        self.device = DEVICE
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


@instantiate_parametrized_tests
class TestFunctionalDifferentials(MultiThreadedTestCase):
    # [CLASS DOC] Forward and backward correctness tests, opcheck registration tests and
    # autograd-op compatibility tests of the functional collectives on 4 multi-threaded fake
    # ranks. Most tests are parametrized over devices; test_utils and autograd_test_utils are
    # class attribute lists of opcheck test names used by parametrize.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 4, the number of threads the multithreaded test
        # case spawns.
        return 4

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and then spawns the rank threads.
        super().setUp()
        self._spawn_threads()

    # ============================================================
    # Forward Correctness Tests
    # ============================================================

    @parametrize("device", devices)
    def test_all_reduce_forward(self, device):
        # [FUNCTION DOC] Forward of all_reduce with sum on a tensor filled with the rank value;
        # asserts every rank gets the sum of ranks. Tensors are different on each rank and the
        # expected values are computed from the rank index. Parametrized over devices.
        """Test all_reduce does all_reduce in forward.

        Tensor is VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value (tensor is varying)
        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = fcols.all_reduce(input_tensor, "sum", group=group_name)

        # Forward does all_reduce: sum of 0+1+2+3 = 6
        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_tensor_forward(self, device, gather_dim):
        # [FUNCTION DOC] Forward of all_gather_single along gather_dim 0, 1 or 2 on a 3x3x3
        # tensor filled with the rank value; asserts the output shape has gather_dim multiplied
        # by world size and each 3-wide chunk along that dim holds the corresponding rank value.
        # Parametrized over devices and gather_dim.
        """Test all_gather_tensor produces correct output shape.

        Tensor is VARYING (different across ranks).
        Forward gathers tensors from all ranks along gather_dim.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank has tensor with its rank value
        input_tensor = torch.full((3, 3, 3), fill_value=float(rank), device=device)
        output = fcols.all_gather_single(
            input_tensor, gather_dim=gather_dim, group=group_name
        )

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[gather_dim] *= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

        # Verify output contains all ranks' data
        # Check each chunk along gather_dim contains the correct rank value
        for r in range(self.world_size):
            chunk = output.narrow(gather_dim, r * 3, 3)
            expected_chunk = torch.full((3, 3, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_tensor_forward(self, device, scatter_dim):
        # [FUNCTION DOC] Forward of reduce_scatter_single along scatter_dim 0 or 1, with
        # each rank contributing its rank value; asserts the output shape divides that dim
        # by world size and every element equals the sum of ranks. Parametrized over devices
        # and scatter_dim.
        """Test reduce_scatter_tensor produces correct output shape.

        Tensor is VARYING (different across ranks).
        Forward reduces and scatters chunks to ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Create input with appropriate size
        if scatter_dim == 0:
            input_tensor = torch.full(
                (4 * self.world_size, 3), fill_value=float(rank), device=device
            )
        else:  # scatter_dim == 1
            input_tensor = torch.full(
                (3, 4 * self.world_size), fill_value=float(rank), device=device
            )

        output = fcols.reduce_scatter_single(
            input_tensor, "sum", scatter_dim=scatter_dim, group=group_name
        )

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[scatter_dim] //= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

        # Each rank should receive sum of all ranks' values: 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        expected = torch.full_like(output, fill_value=expected_value)
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_to_all_single_forward(self, device):
        # [FUNCTION DOC] Forward of all_to_all_single with no split sizes (uniform) on a tensor
        # filled with the rank value; asserts the output has the input shape and the chunk of
        # rows from position r holds value r. Parametrized over devices.
        """Test all_to_all_single with uniform splits.

        Tensor is VARYING (different across ranks).
        Forward exchanges tensor chunks between ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensor = torch.full(
            (2 * self.world_size, 3), fill_value=float(rank), device=device
        )

        # Uniform split
        output = fcols.all_to_all_single(
            input_tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=group_name,
        )

        # Output should have same shape as input for uniform splits
        self.assertEqual(output.shape, input_tensor.shape)

        # Verify each rank receives data from all other ranks
        for r in range(self.world_size):
            chunk = output[r * 2 : (r + 1) * 2, :]
            expected_chunk = torch.full((2, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    @parametrize("device", devices)
    def test_all_reduce_coalesced_forward(self, device):
        # [FUNCTION DOC] Forward of all_reduce_coalesced on two tensors filled with the rank
        # value; asserts each output equals the sum of ranks. Parametrized over devices.
        """Test all_reduce_coalesced does all_reduce on each tensor.

        Tensors are VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensors = [
            torch.full((3, 3), fill_value=float(rank), device=device),
            torch.full((2, 2), fill_value=float(rank), device=device),
        ]
        outputs = fcols.all_reduce_coalesced(input_tensors, "sum", group=group_name)

        # Forward does all_reduce: sum of 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        for output, input_tensor in zip(outputs, input_tensors):
            expected = torch.full_like(input_tensor, fill_value=expected_value)
            self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced_forward(self, device):
        # [FUNCTION DOC] Forward of all_gather_single_coalesced on two tensors; asserts each
        # output shape has the first dim multiplied by world size. Values are not checked.
        # Parametrized over devices.
        """Test all_gather_into_tensor_coalesced gathers each tensor.

        Tensors are VARYING (different across ranks).
        Forward gathers tensors from all ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank has tensors with its rank value
        input_tensors = [
            torch.full((3, 3), fill_value=float(rank), device=device),
            torch.full((2, 2), fill_value=float(rank), device=device),
        ]
        outputs = fcols.all_gather_single_coalesced(input_tensors, group=group_name)

        # Verify output shapes
        for output, input_tensor in zip(outputs, input_tensors):
            expected_shape = list(input_tensor.shape)
            expected_shape[0] *= self.world_size
            self.assertEqual(list(output.shape), expected_shape)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_coalesced_forward(self, device):
        # [FUNCTION DOC] Forward of reduce_scatter_single_coalesced on two tensors with scatter
        # dims 0; asserts output shapes divide the first dim by world size and values equal the
        # sum of ranks. Parametrized over devices.
        """Test reduce_scatter_tensor_coalesced reduces and scatters each tensor.

        Tensors are VARYING (different across ranks).
        Forward reduces and scatters chunks to ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Create inputs with appropriate size (divisible by world_size)
        input_tensors = [
            torch.full((4 * self.world_size, 3), fill_value=float(rank), device=device),
            torch.full((2 * self.world_size, 2), fill_value=float(rank), device=device),
        ]
        scatter_dims = [0, 0]

        outputs = fcols.reduce_scatter_single_coalesced(
            input_tensors, "sum", scatter_dims, group=group_name
        )

        # Each rank should receive sum of all ranks' values: 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        for output, input_tensor in zip(outputs, input_tensors):
            expected_shape = list(input_tensor.shape)
            expected_shape[0] //= self.world_size
            self.assertEqual(list(output.shape), expected_shape)
            expected = torch.full_like(output, fill_value=expected_value)
            self.assertEqual(output, expected)

    # ============================================================
    # Backward Correctness Tests
    # ============================================================

    @parametrize("device", devices)
    def test_all_reduce_backward(self, device):
        # [FUNCTION DOC] Backward of all_reduce with sum: asserts the gradient after a sum
        # backward equals world size, and that the gradient from autograd.grad with random grad
        # outputs equals an all_reduce of those grad outputs. Parametrized over devices.
        """Test all_reduce backward does all_reduce.

        Both tensor AND gradients are VARYING (different across ranks).
        Backward aggregates gradients via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = fcols.all_reduce(input_tensor, "sum", group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be aggregated (backward is all_reduce)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_reduce (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_reduce(grad_outputs, "sum", group=group_name)
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_reduce_avg_backward(self, device):
        # [FUNCTION DOC] Backward of all_reduce with avg: asserts the gradient of a sum backward
        # equals ones, and that autograd.grad with random grad outputs equals an avg all_reduce
        # of those grad outputs. Parametrized over devices.
        """Test all_reduce backward with the 'avg' reduction.

        Both tensor AND gradients are VARYING (different across ranks).
        Backward aggregates gradients via all_reduce(avg).
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = fcols.all_reduce(input_tensor, "avg", group=group_name)

        # Backward with ones: avg of ones across ranks is ones.
        output.sum().backward()
        expected_grad = torch.ones(3, 3, device=device)
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_reduce (avg) of the incoming gradients.
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_reduce(grad_outputs, "avg", group=group_name)
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_tensor_backward(self, device, gather_dim):
        # [FUNCTION DOC] Backward of all_gather_single along gather_dim: asserts the sum
        # backward gradient equals world size and that autograd.grad with random grad outputs
        # equals reduce_scatter_single of them along gather_dim. Parametrized over devices
        # and gather_dim.
        """Test all_gather_tensor backward does reduce_scatter.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward gathers tensors, backward reduces and scatters gradients.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, 3, requires_grad=True, device=device)
        output = fcols.all_gather_single(
            input_tensor, gather_dim=gather_dim, group=group_name
        )

        # Backward with ones
        output.sum().backward()

        # Gradient should be reduce_scatter of ones
        self.assertIsNotNone(input_tensor.grad)
        # Gradient should be all world_size (sum from all ranks)
        expected_grad = torch.full(
            (3, 3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is reduce_scatter (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.reduce_scatter_single(
            grad_outputs, "sum", gather_dim, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_tensor_backward(self, device, scatter_dim):
        # [FUNCTION DOC] Backward of reduce_scatter_single along scatter_dim: asserts the
        # gradient has the input shape and equals ones, and that autograd.grad with random grad
        # outputs equals all_gather_single of them along scatter_dim. Parametrized over devices
        # and scatter_dim.
        """Test reduce_scatter_tensor backward does all_gather.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward reduces and scatters, backward gathers gradients.
        """
        group_name = dist.group.WORLD.group_name

        # Create input with appropriate size
        if scatter_dim == 0:
            input_tensor = torch.randn(
                4 * self.world_size, 3, requires_grad=True, device=device
            )
        else:
            input_tensor = torch.randn(
                3, 4 * self.world_size, requires_grad=True, device=device
            )

        output = fcols.reduce_scatter_single(
            input_tensor, "sum", scatter_dim=scatter_dim, group=group_name
        )

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

        # All gradients should be 1 (gathered from all ranks)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_gather (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_gather_single(
            grad_outputs, scatter_dim, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_to_all_single_backward(self, device):
        # [FUNCTION DOC] Backward of all_to_all_single with uniform splits: asserts the gradient
        # has the input shape and equals ones, and that autograd.grad with random grad outputs
        # equals all_to_all_single of them with no split sizes. Parametrized over devices.
        # KNOWN ISSUE: the docstring says backward reverses split sizes and a comment says
        # backward is all_gather, but only uniform splits are tested and the backward is
        # an all_to_all.
        """Test all_to_all_single backward reverses split sizes.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward does all_to_all, backward does all_to_all with reversed splits.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output = fcols.all_to_all_single(
            input_tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=group_name,
        )

        # Backward
        output.sum().backward()

        # Gradient should have same shape as input
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_gather (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_to_all_single(
            grad_outputs, None, None, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_reduce_coalesced_backward(self, device):
        # [FUNCTION DOC] Backward of all_reduce_coalesced on two tensors: sums all output
        # elements, backpropagates and asserts each gradient equals world size. Parametrized
        # over devices.
        """Test all_reduce_coalesced backward does all_reduce on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Backward aggregates each gradient via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(3, 3, requires_grad=True, device=device),
            torch.randn(2, 2, requires_grad=True, device=device),
        ]
        outputs = fcols.all_reduce_coalesced(input_tensors, "sum", group=group_name)

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be aggregated (backward is all_reduce)
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.full_like(
                input_tensor, fill_value=float(self.world_size)
            )
            self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced_backward(self, device):
        # [FUNCTION DOC] Backward of all_gather_single_coalesced on two tensors: asserts each
        # gradient equals world size after a sum backward. Parametrized over devices.
        """Test all_gather_into_tensor_coalesced backward does reduce_scatter on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Forward gathers each tensor, backward reduce_scatters each gradient.
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(3, 3, requires_grad=True, device=device),
            torch.randn(2, 2, requires_grad=True, device=device),
        ]
        outputs = fcols.all_gather_single_coalesced(input_tensors, group=group_name)

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be reduce_scatter of ones
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.full_like(
                input_tensor, fill_value=float(self.world_size)
            )
            self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_coalesced_backward(self, device):
        # [FUNCTION DOC] Backward of reduce_scatter_single_coalesced on two tensors with
        # scatter dims 0: asserts each gradient equals ones after a sum backward. Parametrized
        # over devices.
        """Test reduce_scatter_tensor_coalesced backward does all_gather on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Forward reduces and scatters each tensor, backward gathers each gradient.
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(4 * self.world_size, 3, requires_grad=True, device=device),
            torch.randn(2 * self.world_size, 2, requires_grad=True, device=device),
        ]
        scatter_dims = [0, 0]

        outputs = fcols.reduce_scatter_single_coalesced(
            input_tensors, "sum", scatter_dims, group=group_name
        )

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be all_gather of ones
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.ones_like(input_tensor)
            self.assertEqual(input_tensor.grad, expected_grad)

    # ============================================================
    # torch.library.opcheck Tests
    # ============================================================

    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        # "test_aot_dispatch_dynamic" - Open issue with check: TBD
    ]

    @parametrize("test_utils", test_utils)
    def test_all_reduce_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs torch.library.opcheck on the c10d functional all_reduce op with an
        # input requiring grad, for the selected test utility (schema, autograd registration or
        # fake tensor). Parametrized over test_utils.
        """Test all_reduce op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does all_reduce)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_reduce,
            (input_tensor, "sum", group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_all_gather_into_tensor_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs torch.library.opcheck on the functional all_gather_into_tensor
        # op with group size equal to world size, for the selected test utility. Parametrized
        # over test_utils.
        """Test all_gather_into_tensor op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does reduce_scatter)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_gather_into_tensor,
            (input_tensor, self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_reduce_scatter_tensor_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs torch.library.opcheck on the functional reduce_scatter_tensor op
        # with sum, a first dim divisible by world size, for the selected test utility.
        # Parametrized over test_utils.
        """Test reduce_scatter_tensor op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does all_gather)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        # Input should be divisible by world_size
        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.reduce_scatter_tensor,
            (input_tensor, "sum", self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_all_to_all_single_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs torch.library.opcheck on the functional all_to_all_single op with
        # explicit equal output and input split sizes derived from the group size, for the
        # selected test utility. Parametrized over test_utils.
        """Test all_to_all_single op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward reverses split sizes)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        # Input should be divisible by world_size
        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_to_all_single,
            (input_tensor, output_split_sizes, input_split_sizes, group_name),
            test_utils=test_utils,
        )

    # ============================================================
    # _c10d_functional_autograd Backward Compatibility Tests
    # ============================================================

    @parametrize("device", devices)
    def test_all_gather_into_tensor_autograd_backward(self, device):
        # [FUNCTION DOC] Calls the autograd variant of all_gather_into_tensor, waits on the
        # result, backpropagates a sum and asserts the gradient equals world size. Parametrized
        # over devices.
        """Test _c10d_functional_autograd.all_gather_into_tensor backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
            input_tensor, self.world_size, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward with ones
        output.sum().backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_autograd_backward(self, device):
        # [FUNCTION DOC] Calls the autograd variant of reduce_scatter_tensor with sum, waits,
        # backpropagates a sum and asserts the gradient equals ones. Parametrized over devices.
        """Test _c10d_functional_autograd.reduce_scatter_tensor backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
            input_tensor, "sum", self.world_size, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_to_all_single_autograd_backward(self, device):
        # [FUNCTION DOC] Calls the autograd variant of all_to_all_single with explicit equal
        # splits, waits, backpropagates a sum and asserts the gradient has input shape and
        # equals ones. Parametrized over devices.
        """Test _c10d_functional_autograd.all_to_all_single backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        output = torch.ops._c10d_functional_autograd.all_to_all_single(
            input_tensor, output_split_sizes, input_split_sizes, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward
        output.sum().backward()

        # Gradient should have same shape as input
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    # ============================================================
    # _c10d_functional_autograd opcheck Tests
    # ============================================================

    # Skip test_faketensor for autograd ops - they share impl with _c10d_functional
    autograd_test_utils = [
        "test_schema",
        "test_autograd_registration",
    ]

    @parametrize("test_utils", autograd_test_utils)
    def test_all_gather_into_tensor_autograd_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs opcheck on the autograd variant of all_gather_into_tensor for the
        # schema and autograd registration utilities (fake tensor is skipped as it shares its
        # implementation). Parametrized over autograd_test_utils.
        """Test _c10d_functional_autograd.all_gather_into_tensor op registration."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.all_gather_into_tensor,
            (input_tensor, self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", autograd_test_utils)
    def test_reduce_scatter_tensor_autograd_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs opcheck on the autograd variant of reduce_scatter_tensor
        # with sum for the schema and autograd registration utilities. Parametrized
        # over autograd_test_utils.
        """Test _c10d_functional_autograd.reduce_scatter_tensor op registration."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.reduce_scatter_tensor,
            (input_tensor, "sum", self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", autograd_test_utils)
    def test_all_to_all_single_autograd_opcheck(self, test_utils):
        # [FUNCTION DOC] Runs opcheck on the autograd variant of all_to_all_single with explicit
        # equal split sizes for the schema and autograd registration utilities. Parametrized
        # over autograd_test_utils.
        """Test _c10d_functional_autograd.all_to_all_single op registration."""
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.all_to_all_single,
            (input_tensor, output_split_sizes, input_split_sizes, group_name),
            test_utils=test_utils,
        )


@instantiate_parametrized_tests
class TestFunctionalDifferentialsWithCompile(DistributedTestBase):
    # [CLASS DOC] torch.compile integration tests of the functional collectives and their
    # autograd variants on a 2-rank DistributedTestBase, using with_comms to create the process
    # group for DEVICE. Instantiated per device for DEVICE only.
    # ============================================================
    # torch.compile Integration Tests
    # ============================================================

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property returning 2 as the world size.
        return 2

    @with_comms
    def test_all_reduce_compile(self):
        # [FUNCTION DOC] Compiles with fullgraph a function that returns the sum of all_reduce
        # of its input, then runs it, backpropagates and asserts the gradient equals world
        # size everywhere.
        """Test that all_reduce backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the sum all_reduce
            # of the tensor over the captured group name.
            output = fcols.all_reduce(tensor, "sum", group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be aggregated
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_gather_tensor_compile(self):
        # [FUNCTION DOC] Compiles a function returning the sum of all_gather_single along dim 0,
        # backpropagates and asserts the gradient equals world size.
        """Test that all_gather_tensor backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the gathered tensor
            # along dim 0.
            output = fcols.all_gather_single(tensor, gather_dim=0, group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_reduce_scatter_tensor_compile(self):
        # [FUNCTION DOC] Compiles a function returning the sum of reduce_scatter_single along
        # dim 0 and asserts the backward gradient equals ones.
        """Test that reduce_scatter_tensor backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the sum reduce-
            # scatter of the tensor along dim 0.
            output = fcols.reduce_scatter_single(
                tensor, "sum", scatter_dim=0, group=group_name
            )
            return output.sum()

        # Input should be divisible by world_size
        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_to_all_single_compile(self):
        # [FUNCTION DOC] Compiles a function returning the sum of all_to_all_single with no
        # split sizes and asserts the backward gradient equals ones.
        """Test that all_to_all_single backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the
            # all_to_all_single of the tensor with default splits.
            output = fcols.all_to_all_single(
                tensor,
                output_split_sizes=None,
                input_split_sizes=None,
                group=group_name,
            )
            return output.sum()

        # Input should be divisible by world_size
        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_to_all with reversed splits (ones)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    # ============================================================
    # _c10d_functional_autograd torch.compile Integration Tests
    # ============================================================

    @with_comms
    def test_all_gather_into_tensor_autograd_compile(self):
        # [FUNCTION DOC] Compiles a function using the autograd variant of
        # all_gather_into_tensor followed by wait_tensor and asserts the backward gradient
        # equals world size.
        """Test that _c10d_functional_autograd.all_gather_into_tensor works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function applying the autograd
            # all_gather_into_tensor, waiting on the result and returning its sum.
            output = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
                tensor, self.world_size, group_name
            )
            output = fcols.wait_tensor(output)
            return output.sum()

        input_tensor = torch.randn(3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_reduce_scatter_tensor_autograd_compile(self):
        # [FUNCTION DOC] Compiles a function using the autograd variant of reduce_scatter_tensor
        # with wait_tensor and asserts the backward gradient equals ones.
        """Test that _c10d_functional_autograd.reduce_scatter_tensor works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function applying the autograd
            # reduce_scatter_tensor, waiting and returning the sum.
            output = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
                tensor, "sum", self.world_size, group_name
            )
            output = fcols.wait_tensor(output)
            return output.sum()

        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_to_all_single_autograd_compile(self):
        # [FUNCTION DOC] Compiles a function using the autograd variant of all_to_all_single
        # with explicit equal split sizes computed outside the compiled function (the raw op
        # needs explicit sizes) and asserts the backward gradient equals ones.
        """Test that _c10d_functional_autograd.all_to_all_single works with torch.compile."""
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )
        # The raw op requires explicit split sizes (SymInt[], not optional); the
        # None default is only resolved by the all_to_all_single Python wrapper.
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            # [FUNCTION DOC] HELPER: Compiled function applying the autograd all_to_all_single
            # with the captured split sizes, waiting and returning the sum.
            output = torch.ops._c10d_functional_autograd.all_to_all_single(
                tensor,
                output_split_sizes,
                input_split_sizes,
                group_name,
            )
            output = fcols.wait_tensor(output)
            return output.sum()

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_to_all with reversed splits (ones)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)


instantiate_device_type_tests(
    TestFunctionalDifferentialsWithCompile, globals(), only_for=DEVICE
)

if __name__ == "__main__":
    run_tests()
