# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for control-flow higher-order operators (cond, while_loop,
# associative_scan, scan, map, switch) compiled through the inductor backend on CPU and GPU.
# Defines small model classes per operator, helpers that prepend predicate or counter inputs, a
# generic loss function, and test classes that compare eager against compiled results, plus
# gradient, dynamic-shape, unbacked-symint, graph-partition and error-case checks. Import has no
# side effects beyond imports and test-class definitions.

import itertools
import unittest
import uuid

import torch
import torch._dynamo.testing
import torch.utils._pytree as pytree
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.map import _fake_map
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._higher_order_ops.switch import switch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


def _prepend_product_of_values(inputs, possible_values, num_to_prepend=1, device=None):
    # [FUNCTION DOC] HELPER: Builds input tuples by prepending scalar tensors to the given inputs.
    # Iterates over the cartesian product of possible_values taken num_to_prepend at a time,
    # creating one scalar tensor per value on the device (taken from the first input when inputs
    # is non-empty). Returns a list of tuples. Raises AssertionError if no device can be
    # determined.
    result = []
    if len(inputs) != 0:
        device = inputs[0].device
    if not device:
        raise AssertionError
    # iterate over the cartesian product of predicate values
    for values in itertools.product(*([possible_values] * num_to_prepend)):
        prepended = [torch.tensor(v, device=device) for v in values]
        result.append((*prepended, *inputs))
    return result


def prepend_predicates(inputs, num_predicates=1, device=None):
    # [FUNCTION DOC] HELPER: Returns input tuples with num_predicates boolean scalar tensors (all
    # False/True combinations) prepended to inputs, on the given device or the first input's
    # device.
    return _prepend_product_of_values(inputs, [False, True], num_predicates, device)


def prepend_counters(inputs, num_counters=1, counter_values=(0, 1, 5)):
    # [FUNCTION DOC] HELPER: Returns input tuples with num_counters integer scalar tensors,
    # covering every combination of the counter values, prepended to inputs. Uses the first
    # input's device.
    return _prepend_product_of_values(inputs, counter_values, num_counters)


# a testing loss_fn
def loss_fn(result) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Testing loss function. Flattens the result pytree, converts
    # non-floating tensors to float, and returns a scalar tensor summing absolute value divided by
    # one plus absolute value over all leaves (a bounded, nan-safe loss). Starts from zero on the
    # first leaf's device, or cpu when there are no leaves.
    flat_results, _ = pytree.tree_flatten(result)
    total_loss = torch.tensor(
        0.0, device=flat_results[0].device if flat_results else torch.device("cpu")
    )

    for res in flat_results:
        # Convert to float if integer tensor to avoid numerical issues
        if not res.dtype.is_floating_point:
            res = res.float()

        # Simple robust loss: abs values + small constant to avoid inf/nan
        total_loss = total_loss + (torch.abs(res) / (1.0 + torch.abs(res))).sum()

    return total_loss


class CondModels:
    # [CLASS DOC] HELPER: Namespace class grouping small torch.nn.Module models that use
    # torch.cond, each exercising one feature (nesting, closures, parameters, aliasing, unbacked
    # symints, and so on). Holds no state itself.
    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with one cond whose branches add or subtract the two tensor
        # operands.
        def forward(self, p, a, b):
            # [FUNCTION DOC] HELPER: Takes predicate p and tensors a, b; returns cond of add
            # (true) or subtract (false) on a and b.
            def true_fn(x, y):
                # [FUNCTION DOC] HELPER: True branch, returns the sum of the two inputs.
                return x + y

            def false_fn(x, y):
                # [FUNCTION DOC] HELPER: False branch, returns the difference of the two inputs.
                return x - y

            return torch.cond(p, true_fn, false_fn, [a, b])

    class SimpleWithIntClosure(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose cond branches close over an integer attribute, using
        # lambdas that return a one-element list.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Initializes the module and stores the integer closure value 3
            # as an attribute.
            super().__init__()
            self.num = 3

        def forward(self, p, a, b):
            # [FUNCTION DOC] HELPER: Returns cond on p with branches returning a list of a plus b
            # plus the stored integer, or a minus b minus it.
            return torch.cond(
                pred=p,
                true_fn=lambda a, b: [a + b + self.num],
                false_fn=lambda a, b: [a - b - self.num],
                operands=(a, b),
            )

    class Nested(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with three levels of nested cond built from nested function
        # definitions; predicates p0, p1, p2 select among eight arithmetic variants.
        def forward(self, p0, p1, p2, a, b, c):
            # [FUNCTION DOC] HELPER: Runs the outer cond on p0 over three tensors, whose branches
            # contain further conds on p1 and p2. Returns the selected result.
            def true_fn(x0, y0, z0):
                # [FUNCTION DOC] HELPER: Outer true branch; dispatches to a nested cond on p1.
                def true_true_fn(x1, y1, z1):
                    # [FUNCTION DOC] HELPER: Innermost branch under true/true; returns the first
                    # minus the product of the other two, times a constant.
                    return (x1 - y1 * z1) * 3.14

                def true_false_fn(x1, y1, z1):
                    # [FUNCTION DOC] HELPER: Branch under true/false; dispatches to a further cond
                    # on p2.
                    def true_false_true_fn(x2, y2, z2):
                        # [FUNCTION DOC] HELPER: Returns the product of the three inputs divided
                        # by a constant.
                        return (x2 * y2 * z2) / 2.71

                    def true_false_false_fn(x2, y2, z2):
                        # [FUNCTION DOC] HELPER: Returns the sum of the three inputs times a
                        # constant.
                        return (x2 + y2 + z2) * 1.23

                    return torch.cond(
                        p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                    )

                return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

            def false_fn(x0, y0, z0):
                # [FUNCTION DOC] HELPER: Outer false branch; dispatches to a nested cond on p1.
                def false_true_fn(x1, y1, z1):
                    # [FUNCTION DOC] HELPER: Branch under false/true; dispatches to a further cond
                    # on p2.
                    def false_true_true_fn(x2, y2, z2):
                        # [FUNCTION DOC] HELPER: Returns the first minus the other two plus a
                        # constant.
                        return (x2 - y2 - z2) + 1.23

                    def false_true_false_fn(x2, y2, z2):
                        # [FUNCTION DOC] HELPER: Returns the first divided by the second and
                        # third, minus a constant.
                        return (x2 / y2 / z2) - 3.14

                    return torch.cond(
                        p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                    )

                def false_false_fn(x1, y1, z1):
                    # [FUNCTION DOC] HELPER: Branch under false/false; returns the first minus the
                    # product of the other two, divided by a constant.
                    return (x1 - y1 * z1) / 2.71

                return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

            return torch.cond(p0, true_fn, false_fn, [a, b, c])

    class Parameters(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose cond branches are themselves modules with parameters
        # (linear layers) rather than functions.
        class InnerModel1(torch.nn.Module):
            # [CLASS DOC] HELPER: Branch module for the true case holding one linear layer from 20
            # to 30 features on the given device.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the linear layer on
                # the device.
                super().__init__()
                self.layer = torch.nn.Linear(20, 30, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the layer applied to the input plus one, scaled
                # by a constant.
                return self.layer(x + 1) * 3.14

        class InnerModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Branch module for the false case holding two linear layers (20
            # to 10, 10 to 30) on the given device.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the two linear layers
                # on the device.
                super().__init__()
                self.layer1 = torch.nn.Linear(20, 10, device=device)
                self.layer2 = torch.nn.Linear(10, 30, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the two layers applied in sequence to the input
                # minus two, scaled by a constant.
                return self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Initializes the module and stores the two branch modules as
            # attributes true_fn and false_fn.
            super().__init__()
            self.true_fn = self.InnerModel1(device)
            self.false_fn = self.InnerModel2(device)

        def forward(self, p, a):
            # [FUNCTION DOC] HELPER: Returns cond with the stored branch modules on a single
            # operand.
            return torch.cond(p, self.true_fn, self.false_fn, [a])

    class ReinterpretView(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches return sliced views and contiguous copies of
        # slices, with inputs also sliced before the cond, to exercise reinterpret-view handling.
        def forward(self, p, a, b):
            # [FUNCTION DOC] HELPER: Calls cond on views of a and b with the last row dropped.
            def true_fn(x, y):
                # [FUNCTION DOC] HELPER: True branch; returns a row slice of the sum and a
                # contiguous column slice of the difference.
                z1 = x + y
                z2 = x - y
                return z1[2:], z2[:, 4:].contiguous()

            def false_fn(x, y):
                # [FUNCTION DOC] HELPER: False branch; returns a row slice of the difference and a
                # contiguous column slice of the sum.
                z1 = x - y
                z2 = x + y
                return z1[2:], z2[:, 4:].contiguous()

            return torch.cond(p, true_fn, false_fn, [a[:-1], b[:-1]])

    class MultipleOutputs(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches return three outputs of different shapes.
        def forward(self, p, a, b, c):
            # [FUNCTION DOC] HELPER: Returns cond on p over three tensors with differently shaped
            # outputs.
            def true_fn(x, y, z):
                # [FUNCTION DOC] HELPER: True branch; returns product of x and y, z over a
                # constant, and the row sum of the difference.
                return x * y, z / 2.71, (y - x).sum(dim=1)

            def false_fn(x, y, z):
                # [FUNCTION DOC] HELPER: False branch; returns y over x, z times a constant, and
                # the row mean of the sum.
                return y / x, z * 3.14, (x + y).mean(dim=1)

            return torch.cond(p, true_fn, false_fn, [a, b, c])

    class OuterCode(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with computation before and after the cond.
        def forward(self, p, a, b):
            # [FUNCTION DOC] HELPER: Computes two intermediates from a and b, runs cond on them
            # (add or subtract), and returns the result squared and divided by a constant.
            c = a * b + 3.14
            d = a / b - 2.71

            def true_fn(x, y):
                # [FUNCTION DOC] HELPER: True branch, returns the sum of its inputs.
                return x + y

            def false_fn(x, y):
                # [FUNCTION DOC] HELPER: False branch, returns the difference of its inputs.
                return x - y

            e = torch.cond(p, true_fn, false_fn, [c, d])

            return e * e / 1.41

    class OuterBuffers(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches use tensors computed outside the cond (closure
        # over outer buffers).
        def forward(self, p, a, b, c):
            # [FUNCTION DOC] HELPER: Computes d and e from a and b, then cond on p over c with
            # branches adding d or subtracting e.
            d = a * 2
            e = b / 2

            def true_fn(x):
                # [FUNCTION DOC] HELPER: True branch; returns the operand plus the captured outer
                # tensor d.
                return x + d

            def false_fn(x):
                # [FUNCTION DOC] HELPER: False branch; returns the operand minus the captured
                # outer tensor e.
                return x - e

            return torch.cond(p, true_fn, false_fn, [c])

    class BranchTensorConstant(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose true branch builds a constant index tensor referenced
        # only inside the subgraph.
        # A branch that materializes a tensor constant (e.g. an index tensor)
        # which is only referenced from within the subgraph.
        def forward(self, p, x):
            # [FUNCTION DOC] HELPER: Runs cond over x with the two branches below.
            def true_fn(t):
                # [FUNCTION DOC] HELPER: True branch; slices every other column and scatter-adds a
                # ones tensor using a constant index tensor created on the operand's device.
                v = t[:, ::2]
                index = torch.tensor(
                    [[0, 0, 1], [1, 1, 2], [2, 2, 0], [0, 0, 2]],
                    dtype=torch.long,
                    device=t.device,
                )
                return v.scatter_add(1, index, torch.ones(4, 3, device=t.device))

            def false_fn(t):
                # [FUNCTION DOC] HELPER: False branch; returns alternate columns starting at one,
                # minus three.
                return t[:, 1::2] - 3.0

            return torch.cond(p, true_fn, false_fn, (x,))

    class WithNonTensorPredicate(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose predicate is a plain Python boolean derived from
        # comparing tensor sizes.
        def forward(self, a, b):
            # [FUNCTION DOC] HELPER: Returns cond with the predicate that the first dim of a
            # exceeds that of b.
            def true_fn(x, y):
                # [FUNCTION DOC] HELPER: True branch; returns the dim-0 sum of x divided by a
                # constant.
                return x.sum(0) / 3.14

            def false_fn(x, y):
                # [FUNCTION DOC] HELPER: False branch; returns the dim-0 sum of y times a
                # constant.
                return y.sum(0) * 2.71

            return torch.cond(a.size(0) > b.size(0), true_fn, false_fn, [a, b])

    class UnbackedSymIntClosure(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches close over an int taken from a shape and an
        # unbacked symint obtained from a tensor value.
        def forward(self, p, x, y, z):
            # [FUNCTION DOC] HELPER: Computes the shape-derived int and an unbacked int via item
            # on a sum, then conds on a comparison of x's first dim to 5.
            a = y.shape[0]
            b = z.sum().to(torch.int64).item()

            def true_fn(x):
                # [FUNCTION DOC] HELPER: True branch; adds the captured shape int to x.
                return x + a

            def false_fn(x):
                # [FUNCTION DOC] HELPER: False branch; adds the captured unbacked int times z to
                # x.
                return x + b * z

            return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

    class MismatchedOutputSize(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose two branches return tensors with different output sizes.
        def forward(self, p, x, y, z):
            # [FUNCTION DOC] HELPER: Computes y and z first-dim sizes, conds on whether x sums
            # above zero, and subtracts the result from y's sum.
            a = y.shape[0]
            b = z.shape[0]

            def true_fn(x):
                # [FUNCTION DOC] HELPER: True branch; adds a captured int to x, drops the first
                # two rows and applies sine.
                return (x + a)[2:].sin()

            def false_fn(x):
                # [FUNCTION DOC] HELPER: False branch; adds b times z to x, keeps the first two
                # rows and applies cosine.
                return (x + b * z)[:2].cos()

            return y.sum() - torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    class MismatchedOutputSizeInnerDim(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches return values with different inner-dimension
        # sizes (a doubled x versus a clone of y).
        def forward(self, p, x, y):
            # [FUNCTION DOC] HELPER: Returns cond on p over x and y.
            def true_fn(x, y):
                # [FUNCTION DOC] HELPER: True branch; returns x doubled.
                return x * 2

            def false_fn(x, y):
                # [FUNCTION DOC] HELPER: False branch; returns a clone of y.
                return y.clone()

            return torch.cond(p, true_fn, false_fn, (x, y))

    class FunctionalCall(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches call a linear layer with externally supplied
        # weight and bias through functional call (ones for true, zeros for false).
        def __init__(self):
            # [FUNCTION DOC] HELPER: Initializes the module and creates a 4 by 4 linear layer.
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, p, x):
            # [FUNCTION DOC] HELPER: Builds ones or zeros weights and biases sized from x,
            # reshapes x, and conds on p over x with branches that apply the linear layer with
            # those parameters.
            true_new_weight = torch.ones(x.size(0), x.size(0), device=x.device)
            false_new_weight = torch.zeros(x.size(0), x.size(0), device=x.device)
            true_new_bias = torch.ones(x.size(0), device=x.device)
            false_new_bias = torch.zeros(x.size(0), device=x.device)
            x = x.reshape(-1, x.size(0))

            def true_fn(x):
                # [FUNCTION DOC] HELPER: True branch; functional call of the linear layer with the
                # ones weight and bias.
                return torch.func.functional_call(
                    self.linear,
                    {
                        "weight": true_new_weight,
                        "bias": true_new_bias,
                    },
                    x,
                )

            def false_fn(x):
                # [FUNCTION DOC] HELPER: False branch; functional call of the linear layer with
                # the zeros weight and bias.
                return torch.func.functional_call(
                    self.linear,
                    {
                        "weight": false_new_weight,
                        "bias": false_new_bias,
                    },
                    x,
                )

            return torch.cond(p, true_fn, false_fn, (x,))

    class SelectWithInputIdx(torch.nn.Module):
        # [CLASS DOC] HELPER: Model that selects a row using an index from a tensor value (an
        # unbacked int) and conds with the same function for both branches.
        def forward(self, p, x, idx):
            # [FUNCTION DOC] HELPER: Converts idx to an int, selects that row of x, and conds on
            # whether its sum is positive with a shared branch.
            u0 = idx.item()
            x0 = x.select(0, u0)

            def fn():
                # [FUNCTION DOC] HELPER: Shared branch with no operands; returns the sine of the
                # captured selected row.
                return x0.sin()

            return torch.cond(x0.sum() > 0, fn, fn)

    class StridePadding(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches return contiguous clones, the false one shifted
        # by one, to test output stride padding.
        def forward(self, p, x):
            # [FUNCTION DOC] HELPER: Returns cond on p over x.
            def true_fn(t):
                # [FUNCTION DOC] HELPER: True branch; returns a contiguous clone of the input.
                return t.clone().contiguous()

            def false_fn(t):
                # [FUNCTION DOC] HELPER: False branch; returns a contiguous clone of the input
                # plus one.
                return t.clone().contiguous() + 1

            return torch.cond(p, true_fn, false_fn, [x])


class CondTests(TestCase):
    # [CLASS DOC] Tests for torch.cond compiled with inductor on cpu and GPU. Uses a shared
    # harness that compiles a model with a counting inductor backend and compares against eager,
    # optionally with dynamic first dims, various predicate combinations and nested or
    # closure-using branches.
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_predicates=1,
    ):
        # [FUNCTION DOC] HELPER: Compiles the model with fullgraph through a compile-counting
        # inductor backend. Moves inputs to the device; with dynamic, adds a set of inputs tiled
        # five times along the first dim and marks every first dim dynamic. For each input set and
        # each combination of prepended predicates, runs eager and compiled, asserts inputs are
        # not mutated and results match, and finally asserts exactly one compilation happened.
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        inputs = [inp.to(device=device) for inp in inputs]
        input_sets = [inputs]
        if dynamic:
            larger_inputs = []
            for inp in inputs:
                # only tile non-scalar tensor inputs
                if inp.ndim > 0:
                    # tile every first dim 5x
                    tiling = [5] + [1] * (inp.ndim - 1)
                    larger_inputs.append(torch.tile(inp, tiling))
                else:
                    larger_inputs.append(inp)
            input_sets.append(larger_inputs)
            for inputs in input_sets:
                for inp in inputs:
                    # mark every first dim as dynamic
                    torch._dynamo.mark_dynamic(inp, 0)

        for inputs in input_sets:
            for inputs_with_predicates in prepend_predicates(
                inputs, num_predicates, device=device
            ):
                cloned_inputs = [inp.clone() for inp in inputs_with_predicates]
                result = model(*inputs_with_predicates)
                result_compiled = compiled_model(*inputs_with_predicates)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_predicates)
                torch.testing.assert_close(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_simple_control_flow(self, device, dynamic):
        # [FUNCTION DOC] Tests cond without nesting, parametrized over device and dynamic shapes.
        # cond control flow without nesting
        self._run_test(
            model=CondModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    def test_cond_subgraph_output_stride_padding(self):
        # [FUNCTION DOC] Tests that a branch output with a large inner dim (19500) which triggers
        # stride padding compiles and matches eager on GPU.
        self._run_test(
            model=CondModels.StridePadding(),
            # inner dim 19500 triggers stride padding
            inputs=(torch.randn(15, 19500),),
            device=GPU_TYPE,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_simple_with_int_closure(self, device):
        # [FUNCTION DOC] Tests a cond whose branches close over an integer attribute, with the
        # model itself pre-compiled with dynamic shapes, on cpu and GPU.
        self._run_test(
            model=torch.compile(CondModels.SimpleWithIntClosure(), dynamic=True),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_cond_unbacked_symint_closure(self, device, dynamic):
        # [FUNCTION DOC] Tests a cond whose false branch closes over an unbacked symint (obtained
        # from a tensor via item with scalar capture enabled) across devices and dynamic modes.
        self._run_test(
            model=CondModels.UnbackedSymIntClosure(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    def test_cond_control_flow_with_precomputed_size(self):
        # [FUNCTION DOC] Tests that a cond with a Python-bool predicate involving an index and a
        # threshold, and a convolution in both branches, matches eager across two different image
        # shapes (GPU only). Compares outputs with a small absolute tolerance.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 512-channel 3 by 3 convolution and an integer
            # threshold of 20; forward conds on a predicate comparing an index to the threshold
            # and applies the convolution in both branches.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Initializes the convolution layer and the threshold
                # attribute.
                super().__init__()
                self.conv2d = torch.nn.Conv2d(
                    512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                )
                self.threshold = 20

            def forward(self, x: torch.Tensor, index) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns cond on the index predicate with convolution
                # branches.
                def true_fn(x: torch.Tensor):
                    # [FUNCTION DOC] HELPER: True branch; returns the convolution of the input.
                    return self.conv2d(x)

                def false_fn(x: torch.Tensor):
                    # [FUNCTION DOC] HELPER: False branch; returns the convolution of the input.
                    return self.conv2d(x)

                return torch.cond(
                    index < self.threshold and index >= 0, true_fn, false_fn, (x,)
                )

        main_model = TestModel().to(GPU_TYPE)
        x1 = torch.rand(2, 512, 128, 72).to(GPU_TYPE)
        x2 = torch.rand(2, 512, 96, 96).to(GPU_TYPE)

        opt_model = torch.compile(main_model)
        out1 = main_model(x1, 1)
        opt_out1 = opt_model(x1, 1)
        self.assertTrue(torch.allclose(out1, opt_out1, atol=1e-5))

        out2 = main_model(x2, 30)
        opt_out2 = opt_model(x2, 30)
        self.assertTrue(torch.allclose(out2, opt_out2, atol=1e-5))

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_nested_control_flow(self, device, dynamic):
        # [FUNCTION DOC] Tests nested cond (three predicates) across devices and dynamic modes.
        # cond control flow with nesting
        self._run_test(
            model=CondModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_predicates=3,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_outer_code_before_after(self, device, dynamic):
        # [FUNCTION DOC] Tests cond with computation before and after it across devices and
        # dynamic modes.
        # some code before and after the conditional
        self._run_test(
            model=CondModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_multiple_outputs(self, device, dynamic):
        # [FUNCTION DOC] Tests cond with multiple outputs of different shapes across devices and
        # dynamic modes.
        # multiple outputs with different shapes
        self._run_test(
            model=CondModels.MultipleOutputs(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(30, 40),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_advanced_dynamic_shapes(self, device):
        # [FUNCTION DOC] Tests that cond subgraph inputs with symbolic-expression shapes (from
        # concatenation) compile with dynamic shapes on each device.
        # subgraphs input shapes include symbolic expressions
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that concatenates a and b, derives two tensors, and conds
            # on p with branches concatenating along dim 1.
            def forward(self, p, a, b):
                # [FUNCTION DOC] HELPER: Builds the derived tensors from the concatenation of a
                # and b and returns the cond result.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: True branch; concatenates x minus three and y times
                    # three along dim 1.
                    return torch.cat([x - 3, y * 3], dim=1)

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: False branch; concatenates x over three and y minus
                    # three along dim 1.
                    return torch.cat([x / 3, y - 3], dim=1)

                c = torch.cat([a, b], dim=0)
                d = c * 2
                e = c / 2

                return torch.cond(p, true_fn, false_fn, [d, e])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(2, 3, 3),
                torch.randn(4, 3, 3),
            ),
            device=device,
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_outer_to_inner(self, device):
        # [FUNCTION DOC] Tests an unbacked symint (from a nonzero count) created outside the cond
        # and used as an input shape, with dynamic output shape capture enabled, on each device.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that builds a ones tensor sized from the nonzero count of
            # a and conds on p using cosine or sine.
            def forward(self, p, a):
                # [FUNCTION DOC] HELPER: Creates the tensor from the nonzero count and returns the
                # cond result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; returns cosine of the input.
                    return torch.cos(x)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; returns sine of the input.
                    return torch.sin(x)

                nz = torch.nonzero(a)
                b = torch.ones([nz.size(0), 8], device=nz.device)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @torch._inductor.config.patch(size_asserts=False)
    # TODO: graph partition does not support creating tensor
    # with dynamic shape in conditional subgraph yet
    @torch._inductor.config.patch(graph_partition=False)
    def test_cond_unbacked_symint_inner(self, device):
        # [FUNCTION DOC] Tests unbacked symints created inside the cond branches (nonzero count),
        # with size asserts and graph partition disabled and dynamic output shapes captured, on
        # each device.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that conds on sine of a, with each branch creating a ones
            # tensor sized from the nonzero count of its operand.
            def forward(self, p, a):
                # [FUNCTION DOC] HELPER: Applies sine to a and returns the cond result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; returns cosine of a ones tensor sized
                    # from the nonzero count of the input.
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; returns sine of a ones tensor sized
                    # from the nonzero count of the input.
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_inner_to_outer(self, device):
        # [FUNCTION DOC] Tests unbacked symints produced inside branches and consumed outside by a
        # following sine, with dynamic output shape capture enabled, on each device.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model like the inner-unbacked one, but applies sine to the cond
            # result.
            def forward(self, p, a):
                # [FUNCTION DOC] HELPER: Applies sine to a, runs the cond, and returns sine of its
                # result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; returns cosine of a ones tensor sized
                    # from the nonzero count of the input.
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; returns sine of a ones tensor sized
                    # from the nonzero count of the input.
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                y = torch.cond(p, true_fn, false_fn, [b])
                return torch.sin(y)

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    def test_cond_use_buffers_from_outer_scope(self):
        # [FUNCTION DOC] Tests a cond whose branches use buffers computed in the outer scope (GPU
        # only, static shape).
        # subgraphs input shapes include symbolic expressions
        self._run_test(
            model=CondModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=False,
        )

    @requires_gpu
    def test_cond_reintepret_view_inputs_outputs(self):
        # [FUNCTION DOC] Tests reinterpret views (slices) as inputs and outputs of cond subgraphs
        # with dynamic shapes on GPU.
        # ReinterpretView in inputs and outputs of the subgraphs
        self._run_test(
            model=CondModels.ReinterpretView(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_subgraphs_with_parameters(self, device, dynamic):
        # [FUNCTION DOC] Tests cond whose branches are nested modules with parameters across
        # devices and dynamic modes.
        # nested Modules with parameters
        self._run_test(
            model=CondModels.Parameters(device),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_branch_tensor_constant(self, device):
        # [FUNCTION DOC] Tests a branch that creates a tensor constant used only inside the
        # subgraph, on each device.
        # branch references a tensor constant only used inside the subgraph
        self._run_test(
            model=CondModels.BranchTensorConstant(),
            inputs=(torch.arange(24, dtype=torch.float32).reshape(4, 6),),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, device, dynamic):
        # [FUNCTION DOC] Tests a Python-bool predicate from tensor sizes with zero prepended
        # predicates, resetting dynamo and trying two second-input sizes, across devices and
        # dynamic modes.
        # model with a boolean predicate
        for b_size_0 in [5, 15]:
            torch._dynamo.reset()
            self._run_test(
                model=CondModels.WithNonTensorPredicate(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(b_size_0, 20),
                ),
                device=device,
                dynamic=dynamic,
                num_predicates=0,
            )

    @requires_gpu
    def test_cond_aliasing_outputs(self):
        # [FUNCTION DOC] Tests that a cond whose branches return an output aliasing another output
        # (a tensor and a slice of it) is rejected: compiling raises the uncaptured
        # higher-order-op error from dynamo.
        # output aliasing in subgraphs: not supported
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model conds on p over a and b with branches that return a sum or
            # difference together with a slice of it.
            def forward(self, p, a, b):
                # [FUNCTION DOC] HELPER: Returns cond of the two aliasing-output branches.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: True branch; returns the sum and a row slice (a view)
                    # of the sum.
                    z = x + y
                    return z, z[1:]

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: False branch; returns the difference and a row slice
                    # (a view) of the difference.
                    z = x - y
                    return z, z[1:]

                return torch.cond(p, true_fn, false_fn, [a, b])

        # AssertionError: Output aliasing is currently not supported...
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            torch.compile(Model())(
                torch.tensor(True),
                torch.randn(10, 20),
                torch.randn(10, 20),
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph(self, device):
        # [FUNCTION DOC] Tests that factory-like ops (zeros_like, ones_like) inside cond branches
        # compile and match eager, on each device.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model conds on p with branches returning zeros or ones shaped
            # like the operand, which is a ones tensor.
            def forward(self, p, a):
                # [FUNCTION DOC] HELPER: Builds a ones tensor like a and returns the cond result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; returns zeros like the input.
                    return torch.zeros_like(x)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; returns ones like the input.
                    return torch.ones_like(x)

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph_recursive(self, device):
        # [FUNCTION DOC] Tests the same decomposition with cond nested in cond branches that
        # reference module-level helper functions, on each device.
        def inner_fn1(x):
            # [FUNCTION DOC] HELPER: Returns zeros like the input.
            return torch.zeros_like(x)

        def inner_fn2(x):
            # [FUNCTION DOC] HELPER: Returns ones like the input.
            return torch.ones_like(x)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose outer branches each run an inner cond on the same
            # predicate with the two helpers in swapped orders.
            def forward(self, p, a):
                # [FUNCTION DOC] HELPER: Builds a ones tensor like a and returns the outer cond
                # result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: Outer true branch; inner cond choosing ones or zeros
                    # helper.
                    return torch.cond(p, inner_fn2, inner_fn1, [x])

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: Outer false branch; inner cond choosing zeros or ones
                    # helper.
                    return torch.cond(p, inner_fn1, inner_fn2, [x])

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    def test_cond_inductor_fx_passes_recursively_applied(self):
        # [FUNCTION DOC] Tests that custom inductor graph passes (pre-grad and post-grad) are
        # applied recursively to cond subgraphs: compiles nested cond with dynamic shapes on GPU
        # and asserts each pass counter equals 11 (a value tied to the number of graphs, including
        # subgraphs, compiled in the test).
        counters = {"pre_grad": 0, "post_grad": 0}

        class PreGradPassCounter(CustomGraphPass):
            # [CLASS DOC] HELPER: Custom graph pass that increments the pre-grad entry of the
            # enclosing counters mapping each time it runs; holds a random uuid string.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a fresh random uuid string.
                self._uuid = str(uuid.uuid4())

            def __call__(self, graph):
                # [FUNCTION DOC] HELPER: Increments the pre-grad counter; the graph is not
                # modified.
                counters["pre_grad"] += 1

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns the stored uuid string, used as the pass
                # identifier.
                return self._uuid

        class PostGradPassCounter(CustomGraphPass):
            # [CLASS DOC] HELPER: Custom graph pass that increments the post-grad entry of the
            # enclosing counters mapping each time it runs; holds a random uuid string.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a fresh random uuid string.
                self._uuid = str(uuid.uuid4())

            def __call__(self, graph):
                # [FUNCTION DOC] HELPER: Increments the post-grad counter; the graph is not
                # modified.
                counters["post_grad"] += 1

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns the stored uuid string, used as the pass
                # identifier.
                return self._uuid

        with torch._inductor.config.patch(
            {
                "pre_grad_custom_pass": PreGradPassCounter(),
                "post_grad_custom_pre_pass": PostGradPassCounter(),
            }
        ):
            self._run_test(
                model=CondModels.Nested(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=GPU_TYPE,
                dynamic=True,
                num_predicates=3,
            )

        self.assertEqual(counters["pre_grad"], 11)
        self.assertEqual(counters["post_grad"], 11)

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_cond_mismatched_branch_output_size(self, device, dynamic):
        # [FUNCTION DOC] Tests cond whose branches return different-sized outputs, across devices
        # and dynamic modes. Note the inputs are passed as a set literal rather than a tuple, so
        # the order is arbitrary.
        self._run_test(
            model=CondModels.MismatchedOutputSize(),
            inputs={
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            },
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_cond_mismatched_branch_output_size_inner_dim(self, device, dynamic):
        # [FUNCTION DOC] Tests branches whose outputs differ in the inner dimension (the symbolic
        # merged strides must not leak into subgraph codegen), across devices and dynamic modes.
        # inner dim mismatch puts an unbacked symbol in the merged
        # output strides, which must not leak into subgraph codegen
        self._run_test(
            model=CondModels.MismatchedOutputSizeInnerDim(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 21),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_cond_functional_call(self, device, dynamic):
        # [FUNCTION DOC] Tests branches that use functional calls of a linear layer with different
        # weights, across devices and dynamic modes.
        self._run_test(
            model=CondModels.FunctionalCall(),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_cond_select_with_input_idx(self, device, dynamic):
        # [FUNCTION DOC] Tests a branch that selects a row with an index obtained from a tensor
        # value (scalar capture enabled), across devices and dynamic modes.
        self._run_test(
            model=CondModels.SelectWithInputIdx(),
            inputs=(torch.randn(10, 20), torch.tensor(0, dtype=torch.int64)),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    def test_output_on_different_device(self):
        # [FUNCTION DOC] Tests a cond whose branches build tensors on cpu and move them to the GPU
        # while the predicate lives on cpu (no inputs, dynamic compile), checking the output
        # device is handled.
        class FactoryBranches(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose cond branches create a float tensor on cpu and
            # transfer it to the GPU; returns the result plus one.
            def forward(self, pred):
                # [FUNCTION DOC] HELPER: Runs the cond with zero-operand lambda branches (a
                # counting tensor or zeros) and adds one.
                tensor = torch.cond(
                    pred,
                    lambda: torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(
                        GPU_TYPE
                    ),
                    lambda: torch.zeros(5, dtype=torch.float32).to(GPU_TYPE),
                )
                return tensor + 1

        self._run_test(
            model=FactoryBranches(),
            inputs=(),
            device="cpu",  # device for predicate
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_buffer_reuse_with_large_subgraph(self, device):
        # [FUNCTION DOC] Regression test: a cond whose subgraph has more buffers than the main
        # graph's scheduler nodes (ten linear layers with relu in the true branch) used to cause
        # an out-of-bounds in memory planning. Compiles it dynamically on each device.
        # Regression test: torch.cond subgraph with more buffers than main
        # graph scheduler nodes caused segmented_tree OOB in memory planning.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a list of ten 64 by 64 linear layers; the true
            # branch applies them with relu and clones the result, the false branch returns zeros
            # of matching shape.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the list of linear
                # layers.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(64, 64) for _ in range(10)]
                )

            def forward(self, p, x):
                # [FUNCTION DOC] HELPER: Returns cond on p over x.
                def true_fn(inp):
                    # [FUNCTION DOC] HELPER: True branch; applies each layer followed by relu in
                    # sequence and returns a clone.
                    h = inp
                    for layer in self.layers:
                        h = torch.relu(layer(h))
                    return h.clone()

                def false_fn(inp):
                    # [FUNCTION DOC] HELPER: False branch; returns a zeros tensor with the input's
                    # first dim and 64 columns.
                    return inp.new_zeros(inp.shape[0], 64)

                return torch.cond(p, true_fn, false_fn, (x,))

        model = Model().to(device)
        inputs = (torch.randn(8, 64),)
        self._run_test(
            model=model,
            inputs=inputs,
            device=device,
            dynamic=True,
        )


class WhileLoopModels:
    # [CLASS DOC] HELPER: Namespace class grouping torch.nn.Module models built on while_loop,
    # each exercising one feature. Holds no state itself.
    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Model running a loop while the counter is positive, decrementing it
        # and updating two tensors.
        def forward(self, ci, a, b):
            # [FUNCTION DOC] HELPER: Runs the loop on a counter and two tensors; returns the final
            # carried values.
            def cond_fn(i, x, y):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return i > 0

            def body_fn(i, x, y):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter, sets x to x plus y and
                # y to y minus x.
                return i - 1, x + y, y - x

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

    class Nested(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with a while_loop whose body contains an inner while_loop.
        def forward(self, ci, cj, a, b):
            # [FUNCTION DOC] HELPER: Runs the outer loop on two counters and two tensors.
            def cond_fn(i1, j1, x1, y1):
                # [FUNCTION DOC] HELPER: Outer condition; true while the first counter is
                # positive.
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                # [FUNCTION DOC] HELPER: Outer body; runs an inner loop then decrements the first
                # counter, clones the second, doubles x and halves y.
                def cond_fn_nested(i2, j2, x2, y2):
                    # [FUNCTION DOC] HELPER: Inner condition; true while the second counter is
                    # positive.
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    # [FUNCTION DOC] HELPER: Inner body; clones the first counter, decrements the
                    # second, adds a constant to x and subtracts one from y.
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = torch._higher_order_ops.while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )

                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class Parameters(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop body is a module with parameters and whose
        # condition is a lambda.
        class InnerModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Body module with two float64 linear layers (20 to 30, 30 to 20)
            # created on the device.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Initializes the two linear layers.
                super().__init__()
                self.layer1 = torch.nn.Linear(
                    20, 30, device=device, dtype=torch.float64
                )
                self.layer2 = torch.nn.Linear(
                    30, 20, device=device, dtype=torch.float64
                )

            def forward(self, c, x):
                # [FUNCTION DOC] HELPER: Returns the decremented counter and the two layers
                # applied to the input minus two, scaled by a constant.
                return c - 1, self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Initializes the module, stores the body module and a
            # condition returning whether the counter is positive.
            super().__init__()
            self.body_fn = self.InnerModel(device)
            self.cond_fn = lambda c, x: c > 0

        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Runs the while_loop with the stored condition and body on a
            # counter and tensor.
            return torch._higher_order_ops.while_loop(
                self.cond_fn, self.body_fn, [c, a]
            )

    class OuterCode(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with computation before and after the loop; the loop swaps and
        # mixes two tensors and the result is the product of the final tensors over a constant.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Computes two intermediates, runs the loop, and returns the
            # product of the two loop tensors divided by a constant.
            d = a * b + 3.14
            e = a / b - 2.71

            def cond_fn(c, x, y):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return c > 0

            def body_fn(c, x, y):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and returns y minus x
                # and x plus y.
                return c - 1, y - x, x + y

            _, f, g = torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, d, e])

            return f * g / 1.41

    # TODO(aakhundov): add while_loop test with outer buffers
    # with dynamic=True once dynamo / export allows while_loop
    # closure capture with mark_dynamic:
    # https://github.com/pytorch/pytorch/issues/123596
    class OuterBuffers(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop body adds and subtracts tensors computed outside
        # the loop (closure over outer buffers).
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Computes d and e from a and b, then runs the loop.
            d = a * 2
            e = b / 2

            def cond_fn(c, x, y):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return c > 0

            def body_fn(c, x, y):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter, adds d to x and
                # subtracts e from y.
                return c - 1, x + d, y - e

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, a, b])

    class PytreeCarry(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop carries a nested structure (list holding a tensor,
        # and a mapping of two tensors) and rotates its elements with simple ops each iteration.
        def forward(self, it, pytree_input):
            # [FUNCTION DOC] HELPER: Runs the loop on an iteration counter and the nested input.
            def cond_fn(it, pytree_input):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return it > 0

            def body_fn(it, pytree_input):
                # [FUNCTION DOC] HELPER: Loop body; unpacks the nested structure, produces new
                # values (sine, cosine, plus one) and returns the decremented counter with a
                # rebuilt structure.
                x = pytree_input[0][0]
                y = pytree_input[1]["x"]
                z = pytree_input[1]["y"]
                new_x = y.sin()
                new_y = z.cos()
                new_z = x + 1
                return it - 1, ([new_x], {"x": new_y, "y": new_z})

            return torch._higher_order_ops.while_loop(
                cond_fn, body_fn, (it, pytree_input)
            )

    class DataDependentOpInSubgraph(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop body uses a data-dependent op (masked select over
        # outer tensors, then a minimum) accumulating into a one-element int64 carry.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Runs the loop with a zero-initialized one-element int64
            # carry.
            def cond_fn(c, reduced_carry):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return c > 0

            def body_fn(c, reduced_carry):
                # [FUNCTION DOC] HELPER: Loop body; masked-selects a using b, concatenates it with
                # its double, and adds the minimum to the carry.
                k = torch.masked_select(a, b)
                d = torch.concat([k, k * 2])
                return c - 1, torch.min(d).unsqueeze(0) + reduced_carry

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, torch.zeros([1], dtype=torch.int64, device=c.device)],
            )

    class DataDependentInOut(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop-carried input has a data-dependent size (rows taken
        # from a sum item).
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Creates a zero int64 tensor with rows given by the sum of a,
            # then runs the loop on it.
            inp = torch.zeros(
                a.sum().to(torch.int64).item(), 3, device=a.device, dtype=torch.int64
            )

            def cond_fn(c, inp):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return c > 0

            def body_fn(c, inp):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and replaces the carry
                # with sine plus one cast to int64.
                return c - 1, (inp.sin() + 1).to(torch.int64)

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, inp],
            )

    class DataDependentInOutMismatch(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop body returns data-dependent shaped outputs (nonzero
        # of the inputs) that differ from the input shapes.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Runs the loop on a counter and two tensors.
            def cond_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter is positive.
                return c > 0

            def body_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and returns the nonzero
                # indices of both tensors.
                return c - 1, a.nonzero(), b.nonzero()

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class InfiniteLoop(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop condition depends on the last dimension of a view,
        # which never changes, so the loop would not terminate.
        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Views a as a column and runs the loop.
            a_view = a.view(-1, 1)

            def cond_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop condition; true while the last dim size is positive.
                return a_view.size(-1) > 0

            def body_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and adds one to the
                # view.
                return c - 1, a_view + 1

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )

    class ZeroLoop(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop condition is false from the start (last dim equals
        # zero is false for a column view), so the body is never run; adds constants to the
        # outputs.
        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Takes sine of a column view, runs the loop, returns the
            # outputs plus one and plus two.
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop condition; true when the last dim size is zero.
                return a_view.size(-1) == 0

            def body_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and adds one to the
                # view.
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop2(torch.nn.Module):
        # [CLASS DOC] HELPER: Variant of the zero-iteration model whose condition returns a Python
        # False constant.
        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Takes sine of a column view, runs the loop, returns the
            # outputs plus one and plus two.
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop condition; always returns the constant False.
                return False

            def body_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and adds one to the
                # view.
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop3(torch.nn.Module):
        # [CLASS DOC] HELPER: Variant of the zero-iteration model whose condition returns the
        # integer constant zero.
        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Takes sine of a column view, runs the loop, returns the
            # outputs plus one and plus two.
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop condition; always returns the integer zero.
                return 0

            def body_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and adds one to the
                # view.
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop4(torch.nn.Module):
        # [CLASS DOC] HELPER: Variant whose condition is a tensor expression (clipped sum below
        # zero, never true) and whose outputs are modified in place afterwards.
        def forward(self, c, a):
            # [FUNCTION DOC] HELPER: Takes sine of a column view, runs the loop, then applies
            # in-place sine to the second output and in-place cosine to the view; returns both.
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop condition; true when the clipped sum of the view is
                # below zero.
                return torch.clip(a_view.sum(), 0, 1) < 0

            def body_fn(c, a_view):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter and adds one to the
                # view.
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out2.sin_(), a_view.cos_()

    class UnbackedSymIntClosure(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose condition closes over unbacked ints (a sum item and a
        # nonzero count) and shape differences.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Computes the two unbacked ints and runs the loop.
            d = a.sum().to(torch.int64).item()
            e = torch.nonzero(b).size(0)

            def cond_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop condition; counter greater than the sum of the
                # unbacked ints plus the first-dim size difference.
                return c > d + e + a.shape[0] - b.shape[0]

            def body_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop body; decrements the counter, adds the nonzero count
                # to a and the sum item to b.
                return c - 1, a + e, b + d

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class SymExprCond(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose condition is a symbolic expression of the counter,
        # unbacked ints and shape sizes, with an incrementing counter.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Computes the two unbacked ints and runs the loop.
            d = a.sum().to(torch.int64).item()
            e = torch.nonzero(b).size(0)

            def cond_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop condition; true while the counter plus the symbolic
                # terms is below ten.
                return c + d + e + a.shape[0] - b.shape[0] < 10

            def body_fn(c, a, b):
                # [FUNCTION DOC] HELPER: Loop body; increments the counter, adds the nonzero count
                # to a and the sum item to b.
                return c + 1, a + e, b + d

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class MixedDevice(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop index lives on cpu while the tensors may be on the
        # GPU.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Moves the counter to cpu and runs the loop.
            # Force the loop idx on cpu
            c = c.to(torch.device("cpu"))

            def cond_fn(loop_idx, a, b):
                # [FUNCTION DOC] HELPER: Loop condition; index below the first dim of a.
                return loop_idx < a.shape[0]

            def body_fn(loop_idx, a, b):
                # [FUNCTION DOC] HELPER: Loop body; increments the index and returns a plus b and
                # a minus b.
                return loop_idx + 1, a + b, a - b

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (c, a, b))

    class MixedDevice2(torch.nn.Module):
        # [CLASS DOC] HELPER: Variant of the mixed-device model where the cpu move result is
        # discarded, so the counter keeps its device, and the index advances by the sum of a.
        def forward(self, c, a, b):
            # [FUNCTION DOC] HELPER: Calls the cpu move without using it and runs the loop.
            # Force the loop idx on cpu
            c.to(torch.device("cpu"))

            def cond_fn(loop_idx, a, b):
                # [FUNCTION DOC] HELPER: Loop condition; index below the first dim of a.
                return loop_idx < a.shape[0]

            def body_fn(loop_idx, a, b):
                # [FUNCTION DOC] HELPER: Loop body; adds the sum of a to the index and returns a
                # plus b and a minus b.
                return loop_idx + a.sum(), a + b, a - b

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (c, a, b))

    class Conv(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose loop body applies a float64 3 by 3 convolution plus one,
        # running while the index is below the batch size.
        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Initializes the convolution layer on the device.
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                4,
                4,
                (3, 3),
                stride=(1, 1),
                padding=(1, 1),
                device=device,
                dtype=torch.float64,
            )

        def forward(self, c, x):
            # [FUNCTION DOC] HELPER: Runs the loop on a counter and the input.
            def cond_fn(loop_idx, x):
                # [FUNCTION DOC] HELPER: Loop condition; index below the first dim of x.
                return loop_idx < x.size(0)

            def body_fn(loop_idx, x):
                # [FUNCTION DOC] HELPER: Loop body; increments the index and returns the
                # convolution of x plus one.
                return loop_idx + 1, self.conv2d(x) + 1

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                (c, x),
            )

    class WhileLoopStackOutputSimple(torch.nn.Module):
        # [CLASS DOC] HELPER: Model using the stack-output variant of while_loop with a linear
        # layer body.
        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Initializes a 3 by 3 linear layer on the device.
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, device=device)

        def forward(self, c, x):
            # [FUNCTION DOC] HELPER: Runs the stack-output loop and returns the stacked counters
            # and stacked tensors.
            def cond_fn(c, x):
                # [FUNCTION DOC] HELPER: Loop condition; counter below the first dim of x.
                return c < x.size(0)

            def body_fn(c, x):
                # [FUNCTION DOC] HELPER: Loop body; increments the counter and applies the linear
                # layer.
                return c + 1, self.linear(x)

            stacked_c, stacked_x = torch.ops.higher_order.while_loop_stack_output(
                cond_fn, body_fn, (c, x), tuple()
            )
            return stacked_c, stacked_x

    class BackwardSumExpandedGrad(torch.nn.Module):
        # [CLASS DOC] HELPER: Model for backward through a loop whose output is reduced by a sum,
        # so the incoming gradient is an expanded tensor.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Runs five iterations of scaling x by 0.9 and adding 0.1,
            # returns the sum of the result.
            def cond_fn(i, x):
                # [FUNCTION DOC] HELPER: Loop condition; counter below five.
                return i < 5

            def body_fn(i, x):
                # [FUNCTION DOC] HELPER: Loop body; increments the counter and returns x scaled
                # and shifted.
                return i + 1, x * 0.9 + 0.1

            init = torch.tensor(0, device=x.device)
            _, result = torch.while_loop(cond_fn, body_fn, (init, x))
            return result.sum()


class WhileLoopTests(TestCase):
    # [CLASS DOC] Tests for while_loop compiled with inductor, via a harness supporting counters,
    # dynamic shapes and optionally gradients.
    def _run_test(
        self, model, inputs, device, dynamic=False, num_counters=1, autograd=False
    ):
        # [FUNCTION DOC] HELPER: Harness. Deep-copies the model as the compiled one (parameters
        # frozen unless autograd), moves pytree inputs to the device, and optionally adds inputs
        # tiled five times along the first dim. For each input set and each combination of
        # prepended counter values, clones the inputs, runs eager and compiled, and asserts inputs
        # unchanged and results close. With autograd and any grad-requiring input, also compares
        # losses, backward results, parameter gradients and input gradients. Finally asserts one
        # compilation.
        import torch.utils._pytree as pytree

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        import copy

        if not autograd:
            for p in model.parameters():
                p.requires_grad_(False)

        compiled_model = copy.deepcopy(model)
        compiled_fn = torch.compile(backend=cnt, fullgraph=True)(compiled_model)

        inputs = pytree.tree_map(lambda t: t.to(device=device), inputs)
        input_sets = [inputs]

        def mark_first_dim_dyn(inp):
            # [FUNCTION DOC] HELPER: Marks the first dimension of the tensor as dynamic.
            torch._dynamo.mark_dynamic(inp, 0)

        if dynamic:

            def tile_fn(inp):
                # [FUNCTION DOC] HELPER: Tiles the tensor five times along the first dimension and
                # returns the result.
                # tile every first dim 5x
                tiling = [5] + [1] * (inp.ndim - 1)
                t = torch.tile(inp, tiling)
                return t

            larger_inputs = pytree.tree_map(tile_fn, inputs)
            input_sets.append(larger_inputs)

        for inputs in input_sets:
            flat_inputs, inp_spec = pytree.tree_flatten(inputs)
            for flat_inputs_with_counters in prepend_counters(
                flat_inputs, num_counters
            ):
                counters, flat = (
                    flat_inputs_with_counters[:num_counters],
                    flat_inputs_with_counters[num_counters:],
                )
                unflat_inputs = pytree.tree_unflatten(flat, inp_spec)
                inputs_with_counters = counters + unflat_inputs

                def process_inputs(inp):
                    # [FUNCTION DOC] HELPER: Clones the tensor, marks its first dim dynamic when
                    # dynamic is requested, and enables requires-grad on it when autograd is on
                    # and it is floating point; returns the clone.
                    inp = inp.clone()
                    if dynamic:
                        mark_first_dim_dyn(inp)

                    if autograd and inp.dtype.is_floating_point:
                        inp.requires_grad_(True)
                    return inp

                cloned_inputs = pytree.tree_map(process_inputs, inputs_with_counters)
                cloned_inputs2 = pytree.tree_map(process_inputs, inputs_with_counters)

                result = model(*cloned_inputs)
                result_compiled = compiled_fn(*cloned_inputs2)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_counters)
                torch.testing.assert_close(
                    result, result_compiled, atol=1e-4, rtol=1e-4
                )

                if autograd and any(
                    pytree.tree_map_only(
                        torch.Tensor, lambda t: t.requires_grad, cloned_inputs
                    )
                ):
                    result_loss = loss_fn(pytree.tree_flatten(result)[0])
                    compiled_loss = loss_fn(pytree.tree_flatten(result_compiled)[0])
                    self.assertTrue(
                        not torch.isnan(result_loss) and not torch.isinf(compiled_loss)
                    )
                    self.assertTrue(
                        not torch.isnan(compiled_loss)
                        and not torch.isinf(compiled_loss)
                    )

                    self.assertEqual(result_loss, compiled_loss)

                    result_loss.backward()
                    compiled_loss.backward()

                    model_parameters = dict(model.named_parameters())
                    compiled_parameters = dict(compiled_model.named_parameters())
                    for name, param in model_parameters.items():
                        self.assertEqual(param, compiled_parameters[name])
                        self.assertEqual(
                            param.grad,
                            compiled_parameters[name].grad,
                            atol=1e-4,
                            rtol=1e-4,
                        )

                    for inp1, inp2 in zip(
                        pytree.tree_flatten(cloned_inputs)[0],
                        pytree.tree_flatten(cloned_inputs2)[0],
                    ):
                        if inp1.requires_grad:
                            self.assertEqual(
                                inp1.grad,
                                inp2.grad,
                                atol=1e-4,
                                rtol=1e-4,
                            )

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_simple_control_flow(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests a simple while_loop across devices, dynamic modes and autograd
        # settings.
        # while_loop control flow without nesting
        self._run_test(
            model=WhileLoopModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_nested_control_flow(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop nested in while_loop with two counters, across devices,
        # dynamic modes and autograd.
        # while_loop control flow with nesting
        self._run_test(
            model=WhileLoopModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_counters=2,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_outer_code(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop with computation before and after it, across devices,
        # dynamic modes and autograd.
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_parameters(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop whose body is a module with parameters (float64 inputs),
        # across devices, dynamic modes and autograd.
        # while_loop control flow with parameters
        self._run_test(
            model=WhileLoopModels.Parameters(device),
            inputs=(torch.randn(10, 20, dtype=torch.float64),),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    # dynamic=True doesn't work now due to
    # https://github.com/pytorch/pytorch/issues/123596
    @parametrize("dynamic", [False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_outer_buffers(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop whose body closes over outer buffers; only static shapes
        # are parametrized because dynamic shapes are not supported yet (referenced upstream
        # issue).
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_pytree_inputs(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop carrying a nested list and mapping structure as input,
        # across devices, dynamic modes and autograd.
        self._run_test(
            model=WhileLoopModels.PytreeCarry(),
            inputs=(
                (
                    [torch.randn(10, 20)],
                    {"x": torch.randn(10, 20), "y": torch.randn(10, 20)},
                ),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_data_dependent_ops(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop with a data-dependent op (masked select) in the body,
        # with dynamic output shape capture enabled, across devices, dynamic modes and autograd.
        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=WhileLoopModels.DataDependentOpInSubgraph(),
                inputs=(
                    torch.tensor([1, 2, 3, 4, 5]),
                    torch.tensor(
                        [True, True, True, True, True],
                    ),
                ),
                device=device,
                dynamic=dynamic,
                autograd=autograd,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_data_dependent_in_out(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop whose carried input has data-dependent size (from a sum
        # item), with output shape and scalar capture enabled, across devices, dynamic modes and
        # autograd.
        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
                "capture_scalar_outputs": True,
            }
        ):
            self._run_test(
                model=WhileLoopModels.DataDependentInOut(),
                inputs=(
                    torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                    torch.tensor(
                        [True, True, True, True, True],
                    ),
                ),
                device=device,
                dynamic=dynamic,
                autograd=autograd,
            )

    @parametrize("dynamic", [True, False])
    def test_while_loop_with_data_dependent_in_out_mismatch(self, dynamic):
        # [FUNCTION DOC] Tests that a body returning outputs whose metadata differs from the
        # carried inputs (nonzero outputs) raises the uncaptured higher-order-op error mentioning
        # the mismatch of body outputs and carried inputs. Runs on cpu with both dynamic settings.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected body_fn_output and carried_inputs to have same metadata but found",
        ):
            with torch._dynamo.config.patch(
                {
                    "capture_dynamic_output_shape_ops": True,
                }
            ):
                self._run_test(
                    model=WhileLoopModels.DataDependentInOutMismatch(),
                    inputs=(
                        torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                        torch.tensor(
                            [True, True, True, True, True],
                        ),
                    ),
                    device="cpu",
                    dynamic=dynamic,
                )

    def test_while_loop_infinite_loop_error(self):
        # [FUNCTION DOC] Tests that a loop whose condition never becomes false is rejected:
        # running raises the uncaptured higher-order-op error mentioning the while_loop op, on cpu
        # with static shapes.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "torch.while_loop",
        ):
            self._run_test(
                model=WhileLoopModels.InfiniteLoop(),
                inputs=(torch.tensor([1, 2, 3, 4, 5]),),
                device="cpu",
                dynamic=False,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_while_loop_zero_loop(self, device, dynamic):
        # [FUNCTION DOC] Tests while_loop variants whose condition is false on the first check
        # (four zero-iteration models) across devices and dynamic modes, looping over each model.
        for model in [
            WhileLoopModels.ZeroLoop(),
            WhileLoopModels.ZeroLoop2(),
            WhileLoopModels.ZeroLoop3(),
            WhileLoopModels.ZeroLoop4(),
        ]:
            self._run_test(
                model=model,
                inputs=(torch.tensor([1, 2, 3, 4, 5]),),
                device=device,
                dynamic=dynamic,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch(
        {"capture_scalar_outputs": True, "capture_dynamic_output_shape_ops": True}
    )
    @parametrize("autograd", [False, True])
    def test_while_loop_with_unbacked_symint_closure(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop whose condition uses unbacked ints from scalar and
        # nonzero capture, across devices, dynamic modes and autograd.
        self._run_test(
            model=WhileLoopModels.UnbackedSymIntClosure(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", [GPU_TYPE])
    def test_while_loop_models_with_mixed_device(self, device):
        # [FUNCTION DOC] Tests two mixed-device loops on the GPU: the loop index on cpu works with
        # dynamic shapes, while the second model (index becoming a different device after the
        # first iteration) must raise the uncaptured higher-order-op error about mismatched
        # metadata at the front end.
        self._run_test(
            model=WhileLoopModels.MixedDevice(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=True,
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected body_fn_output and carried_inputs to have same metadata but found",
        ):
            # Error at front end because device are promoted to a different one
            # after the first iteration
            self._run_test(
                model=WhileLoopModels.MixedDevice2(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch(
        {"capture_scalar_outputs": True, "capture_dynamic_output_shape_ops": True}
    )
    def test_while_loop_with_sym_expr_cond(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop with a condition that is a symbolic expression of
        # unbacked ints and sizes, across devices, dynamic modes and autograd.
        self._run_test(
            model=WhileLoopModels.SymExprCond(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_conv(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests while_loop with a convolution body on float64 4D inputs, across
        # devices, dynamic modes and autograd.
        self._run_test(
            model=WhileLoopModels.Conv(device),
            inputs=(torch.randn(2, 4, 4, 4, dtype=torch.float64),),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_stack_output_simple(self, device, dynamic):
        # [FUNCTION DOC] Tests the stack-output variant of while_loop with a linear body, across
        # devices and dynamic modes.
        self._run_test(
            model=WhileLoopModels.WhileLoopStackOutputSimple(device),
            inputs=(torch.randn(3, 3, dtype=torch.float32),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_backward_sum_expanded_grad(self, device, dynamic):
        # [FUNCTION DOC] Tests backward through a loop followed by a sum (expanded gradient), with
        # zero prepended counters and autograd enabled, across devices and dynamic modes.
        self._run_test(
            model=WhileLoopModels.BackwardSumExpandedGrad(),
            inputs=(torch.randn(4, 64, device=device),),
            device=device,
            dynamic=dynamic,
            num_counters=0,
            autograd=True,
        )


class AssociativeScanTests(TestCase):
    # [CLASS DOC] Tests for associative_scan compiled with inductor.
    @requires_gpu
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("backend", ["inductor"])
    @parametrize("device", [torch.device("cpu"), GPU_TYPE])
    # This test will fail as flip in combination with particular input lengths
    # produces weird results.
    # This is under investigations in
    # https://github.com/pytorch/pytorch/issues/131805
    @decorateIf(unittest.skip, lambda params: params["device"] == GPU_TYPE)
    def test_associative_scan_CUDA_flip(self, combine_mode, backend, device):
        # [FUNCTION DOC] Checks associative_scan with addition against its eager form and against
        # cumulative sum, for forward, reversed, and flipped-input variants, using input length
        # nine. On cpu with pointwise combine mode, the compiled call is expected to raise
        # (pointwise mode supports GPU tensors only) and the test returns early. The test is
        # skipped for the GPU device because of a referenced issue with flip and certain lengths.
        def fct(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Combine function returning the sum of its two inputs.
            return x + y

        # for n in range(10):
        for n in [9]:
            x = torch.arange(n, device=device)
            torch.compiler.reset()
            associative_scan1 = torch.compile(
                associative_scan, backend=backend, fullgraph=True
            )
            associative_scan2 = associative_scan

            if combine_mode == "pointwise" and device == torch.device("cpu"):
                with self.assertRaisesRegex(Exception, r"."):
                    associative_scan1(
                        fct, x, 0, reverse=False, combine_mode=combine_mode
                    )

                # Skipping test because combine_mode currently only supports CUDA tensors
                return

            result1 = associative_scan1(
                fct, x, 0, reverse=False, combine_mode=combine_mode
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=False, combine_mode=combine_mode
            )
            result3 = torch.cumsum(x, 0)

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Flip only non-compiled and compare with compiled reverse=True
            result1 = associative_scan1(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result2 = torch.flip(
                associative_scan2(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Flip only compiled and compare with non-compiled reverse=True
            result1 = torch.flip(
                associative_scan1(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Use reverse=False, but flip both results before and after
            result1 = torch.flip(
                associative_scan1(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result2 = torch.flip(
                associative_scan2(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Reverse=True
            result1 = associative_scan1(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)


class ScanModels:
    # [CLASS DOC] HELPER: Namespace class grouping scan test models. Holds no state itself.
    class SimpleScan(torch.nn.Module):
        # [CLASS DOC] HELPER: Model scanning with a combine function that updates a mapping carry
        # (matrix product plus bias, sine of bias) and returns cloned carry plus a sine output.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Stores the reverse flag and the scan dimension.
            super().__init__()
            self.reverse = reverse
            self.dim = dim

        def forward(self, _input, weight, bias):
            # [FUNCTION DOC] HELPER: Runs scan over the input with a mapping carry of weight and
            # bias, using the stored dim and direction.
            def combine_fn(carry, x):
                # [FUNCTION DOC] HELPER: Scan combine function; computes the new carry mapping,
                # returns it along with a pair of the cloned carry and a mapping holding the sine
                # of the element.
                from torch.utils import _pytree as pytree

                new_carry = {
                    "param": carry["param"] @ x + carry["bias"],
                    "bias": carry["bias"].sin(),
                }
                return new_carry, (
                    pytree.tree_map(lambda x: x.clone(), new_carry),
                    {"dummy": x.sin()},
                )

            return scan(
                combine_fn,
                {"param": weight, "bias": bias},
                _input,
                reverse=self.reverse,
                dim=self.dim,
            )

    class ScanLinearWithView(torch.nn.Module):
        # [CLASS DOC] HELPER: Model scanning a linear layer over views, with reverse flag, dim and
        # a float64 4 by 4 linear layer.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Stores reverse, dim and creates the linear layer.
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.linear = torch.nn.Linear(4, 4, dtype=torch.float64)

        def forward(self, scan_op, init, xs):
            # [FUNCTION DOC] HELPER: Runs the supplied scan operation with the stored dim and
            # direction.
            def combine_fn(carry, x):
                # [FUNCTION DOC] HELPER: Combine function; flattens the leading dims, applies the
                # linear layer, restores the shape, and returns the view and a clone of it as
                # carry and output.
                prev_sz = x.size()
                x = self.linear(x.view(-1, x.size(-1)))
                x_view = x.view(*prev_sz)
                return x_view, x_view.clone()

            return scan_op(combine_fn, init, xs, dim=self.dim, reverse=self.reverse)

    class ScanConv(torch.nn.Module):
        # [CLASS DOC] HELPER: Model scanning a float64 3 by 3 convolution, with reverse flag and
        # dim.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Stores reverse, dim and creates the convolution layer.
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.conv2d = torch.nn.Conv2d(
                4, 4, (3, 3), stride=(1, 1), padding=(1, 1), dtype=torch.float64
            )

        # init = torch.randn(2, 4, 4, 4)
        # xs = torch.randn(scan_dim, 2, 4, 4, 4)
        def forward(self, scan_op, init, xs):
            # [FUNCTION DOC] HELPER: Runs the supplied scan operation with the stored dim and
            # direction.
            def combine_fn(carry, x):
                # [FUNCTION DOC] HELPER: Combine function; applies the convolution and returns the
                # result as carry and a clone as output.
                x = self.conv2d(x)
                return x, x.clone()

            return scan_op(combine_fn, init, xs, dim=self.dim, reverse=self.reverse)

    class ScanInCond(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with the scan placed inside the branches of a cond, one branch
        # scanning in the stored direction and the other in the reversed direction.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Creates two linear-view scan modules, one with reverse
            # flipped.
            super().__init__()
            self.true_scan_linear = ScanModels.ScanLinearWithView(reverse, dim)
            self.false_scan_linear = ScanModels.ScanLinearWithView(not reverse, dim)

        def forward(self, scan_op, pred, init, xs):
            # [FUNCTION DOC] HELPER: Runs cond on pred with zero-operand branches that each run a
            # scan module.
            def true_fn():
                # [FUNCTION DOC] HELPER: True branch; scans and returns the sum of the last carry
                # and the sine of the outputs.
                last_carry, y = self.true_scan_linear(scan_op, init, xs)
                return last_carry.sum(), y.sin()

            def false_fn():
                # [FUNCTION DOC] HELPER: False branch; scans and returns the negated sum of the
                # last carry and the cosine of the outputs.
                last_carry, y = self.false_scan_linear(scan_op, init, xs)
                return -last_carry.sum(), y.cos()

            return torch.cond(pred, true_fn, false_fn, tuple())

    class CondInScan(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with a cond inside the scan combine function, choosing between
        # two linear layers by a predicate on the carry.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Stores reverse, dim and creates two 4 by 4 linear layers.
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.true_linear = torch.nn.Linear(4, 4)
            self.false_linear = torch.nn.Linear(4, 4)

        def forward(self, scan_op, init, xs):
            # [FUNCTION DOC] HELPER: Runs the supplied scan operation with the stored dim and
            # direction.
            def combine_fn(carry, x):
                # [FUNCTION DOC] HELPER: Combine function; flattens the carry, conds on whether
                # all values exceed one to apply sine of one linear layer or cosine of the other,
                # and returns the updated carry and the cond output.
                old_sizes = carry.size()
                carry_view = carry.view(-1, carry.size()[-1])
                new_carry_out = torch.cond(
                    torch.all(carry_view > 1),
                    lambda: self.true_linear(carry_view).sin(),
                    lambda: self.false_linear(carry_view).cos(),
                    tuple(),
                )
                return carry + new_carry_out.view(*old_sizes), new_carry_out

            return scan_op(
                combine_fn,
                init,
                xs,
                dim=self.dim,
                reverse=self.reverse,
            )

    class SimpleWithPytreeInOuts(torch.nn.Module):
        # [CLASS DOC] HELPER: Model like the first scan model, taking the scan operation as an
        # argument, with pytree carry and outputs.
        def __init__(self, reverse, dim):
            # [FUNCTION DOC] HELPER: Stores the reverse flag and scan dimension.
            super().__init__()
            self.reverse = reverse
            self.dim = dim

        def forward(self, scan_op, _input, weight, bias):
            # [FUNCTION DOC] HELPER: Runs the supplied scan operation with a mapping carry.
            def combine_fn(carry, x):
                # [FUNCTION DOC] HELPER: Combine function; computes the new carry mapping, returns
                # it along with a pair of its clone and a mapping holding the sine of the element.
                new_carry = {
                    "param": carry["param"] @ x + carry["bias"],
                    "bias": carry["bias"].sin(),
                }
                return new_carry, (
                    pytree.tree_map(lambda x: x.clone(), new_carry),
                    {"dummy": x.sin()},
                )

            return scan_op(
                combine_fn,
                {"param": weight, "bias": bias},
                _input,
                reverse=self.reverse,
                dim=self.dim,
            )

    class ChunkedCE(torch.nn.Module):
        # [CLASS DOC] HELPER: Model computing chunked loss and gradients with a scan over chunks,
        # using functional gradient transforms; the loss is an absolute-difference sum. Stores
        # chunk size and a loss lambda.
        def __init__(self, chunk_size):
            # [FUNCTION DOC] HELPER: Stores the chunk size and defines the loss as the sum of
            # absolute differences.
            super().__init__()
            self.chunk_size = chunk_size
            self.ce = lambda logits, target: torch.abs(target - logits).sum()

        def forward(self, scan_op, _input, weight, target, bias):
            # [FUNCTION DOC] HELPER: Splits input and target into chunks, scans over them
            # accumulating gradients of weight and bias and loss, and returns averages of weight
            # grad, bias grad, loss and the flattened input grads over the number of chunks.
            CHUNK_SIZE = self.chunk_size

            def compute_loss(input_chunk, weight, bias, target):
                # [FUNCTION DOC] HELPER: Computes logits via an affine map of the chunk, casts to
                # float and returns the loss.
                logits = torch.addmm(bias, input_chunk, weight.t())
                logits = logits.float()
                loss = self.ce(logits, target)
                return loss

            grad_weight = torch.zeros_like(weight)
            grad_bias = torch.zeros_like(bias)
            loss_acc = torch.zeros((), device=_input.device)

            chunks = _input.shape[0] // CHUNK_SIZE

            _input_chunks = _input.view(CHUNK_SIZE, chunks, *_input.shape[1:])
            target_chunks = target.view(CHUNK_SIZE, chunks, *target.shape[1:])

            def combine_fn(carry, xs):
                # [FUNCTION DOC] HELPER: Combine function; computes loss and gradients for one
                # chunk, adds them to the accumulated carry, and returns the chunk's input
                # gradient as the output.
                grad_weight, grad_bias, loss_acc = carry
                input_chunk, target_chunk = xs
                (
                    (
                        chunk_grad_input,
                        chunk_grad_weight,
                        chunk_grad_bias,
                    ),
                    chunk_loss,
                ) = torch.func.grad_and_value(compute_loss, argnums=(0, 1, 2))(
                    input_chunk, weight, bias, target_chunk
                )
                return (
                    (
                        grad_weight + chunk_grad_weight,
                        grad_bias + chunk_grad_bias,
                        loss_acc + chunk_loss,
                    ),
                    chunk_grad_input,
                )

            (grad_weight, grad_bias, loss_acc), grad_inputs = scan_op(
                combine_fn,
                (grad_weight, grad_bias, loss_acc),
                (_input_chunks, target_chunks),
            )
            return (
                grad_weight / chunks,
                grad_bias / chunks,
                loss_acc / chunks,
                grad_inputs.view(-1, *_input.shape[1:]) / chunks,
            )

    class ChunkedCENoScan(torch.nn.Module):
        # [CLASS DOC] HELPER: Same chunked loss model as the scan version but without scan:
        # gradients are accumulated in place by a compiled per-chunk function called from a Python
        # loop. Stores chunk size and loss lambda.
        def __init__(self, chunk_size):
            # [FUNCTION DOC] HELPER: Stores the chunk size and defines the loss as the sum of
            # absolute differences.
            super().__init__()
            self.chunk_size = chunk_size
            self.ce = lambda logits, target: torch.abs(target - logits).sum()

        def forward(self, scan_op, _input, weight, target, bias):
            # [FUNCTION DOC] HELPER: Splits input and target into chunks with a chunk call, calls
            # the compiled per-chunk function on each (accumulating gradients and loss in place),
            # and returns averages of weight grad, bias grad, loss and the concatenated input
            # grads over the number of chunks. The scan operation argument is unused.
            CHUNK_SIZE = self.chunk_size

            def compute_loss(input_chunk, weight, bias, target):
                # [FUNCTION DOC] HELPER: Computes logits via an affine map of the chunk, casts to
                # float and returns the loss.
                logits = torch.addmm(bias, input_chunk, weight.t())
                logits = logits.float()
                loss = self.ce(logits, target)
                return loss

            grad_weight = torch.zeros_like(weight)
            grad_inputs = []
            grad_bias = torch.zeros_like(bias)
            loss_acc = torch.zeros((), device=_input.device)

            chunks = _input.shape[0] // CHUNK_SIZE

            def accumulate_chunk(input_chunk, target_chunk):
                # [FUNCTION DOC] HELPER: Computes loss and gradients for one chunk, adds weight,
                # bias and loss into the outer accumulators in place, and returns the chunk's
                # input gradient.
                (
                    (
                        chunk_grad_input,
                        chunk_grad_weight,
                        chunk_grad_bias,
                    ),
                    chunk_loss,
                ) = torch.func.grad_and_value(compute_loss, argnums=(0, 1, 2))(
                    input_chunk, weight, bias, target_chunk
                )
                grad_weight.add_(chunk_grad_weight)
                grad_bias.add_(chunk_grad_bias)
                loss_acc.add_(chunk_loss)
                return chunk_grad_input

            accumulate_chunk = torch.compile(accumulate_chunk)

            input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
            target_chunks = torch.chunk(target, chunks=chunks, dim=0)
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))
            return (
                grad_weight / chunks,
                grad_bias / chunks,
                loss_acc / chunks,
                torch.cat(grad_inputs, dim=0) / chunks,
            )

    class ScanWithClamp(torch.nn.Module):
        # [CLASS DOC] HELPER: Model scanning a step that adds the element to the carry and clamps
        # the minimum at 0.1.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Calls the base module initializer only.
            super().__init__()

        def forward(self, scan_op, initial, xs):
            # [FUNCTION DOC] HELPER: Runs the supplied scan operation with the step and returns
            # the final carry and stacked outputs.
            def step(h_prev, x_t):
                # [FUNCTION DOC] HELPER: Scan step; returns the clamped sum as carry and a clone
                # of it as output.
                h_next = (h_prev + x_t).clamp(min=0.1)
                return h_next, h_next.clone()

            final, ys = scan_op(step, initial, xs)
            return final, ys


class ScanTests(TestCase):
    # [CLASS DOC] Tests for scan: compares the supplied scan operation, a reference fake scan, and
    # compiled versions, in eager and compiled models.
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic,
        autograd=False,
    ):
        # [FUNCTION DOC] HELPER: Harness. Casts floating inputs to require grad as requested,
        # moves inputs and model to the device, makes four model copies (reference with fake scan,
        # eager scan, compiled scan, compiled with fake scan), runs each, and asserts all results
        # (forward values, loss and, with autograd, input and parameter gradients) match the
        # reference.
        import copy

        inputs = [
            inp.requires_grad_(autograd) if inp.dtype.is_floating_point else inp
            for inp in inputs
        ]
        inputs = [inp.to(device=device) for inp in inputs]
        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(autograd)

        model1 = copy.deepcopy(model)
        model2 = copy.deepcopy(model)
        model3 = copy.deepcopy(model)
        model4 = copy.deepcopy(model)
        model3.compile(fullgraph=True, dynamic=dynamic)
        model4.compile(fullgraph=True, dynamic=dynamic)

        def _run_model(model, inputs):
            # [FUNCTION DOC] HELPER: Clones tensor inputs, runs the model, computes the loss, and
            # with autograd runs backward and returns the results, loss, input grads and parameter
            # grads; otherwise returns results and loss.
            cloned_inputs = [
                inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
            ]
            fw_result = model(*cloned_inputs)
            loss = loss_fn(fw_result)
            if autograd:
                loss.backward()
                return (
                    fw_result,
                    loss,
                    [
                        inp.grad
                        for inp in cloned_inputs
                        if isinstance(inp, torch.Tensor)
                    ],
                    {n: p.grad for n, p in model.named_parameters()},
                )
            else:
                return fw_result, loss

        result_exp = _run_model(model1, [_fake_scan] + inputs)
        result_eager = _run_model(model2, [scan] + inputs)
        result_compiled = _run_model(model3, [scan] + inputs)
        result_compiled_exp = _run_model(
            model4,
            [_fake_scan] + inputs,
        )

        self.assertEqual(result_exp, result_eager)
        self.assertEqual(result_exp, result_compiled)
        self.assertEqual(result_exp, result_compiled_exp)

    def _compare_result(
        self,
        model1,
        model2,
        inputs,
        device,
    ):
        # [FUNCTION DOC] HELPER: Moves inputs to the device, clones them, runs both models with
        # the real scan operation and asserts the outputs are equal.
        inp_on_device = [elem.to(device=device) for elem in inputs]
        cloned_inputs = [arg.clone() for arg in inp_on_device]
        model1_out = model1(scan, *cloned_inputs)
        model2_out = model2(scan, *cloned_inputs)
        self.assertEqual(model1_out, model2_out)

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2])
    @parametrize("autograd", [True, False])
    @torch._inductor.config.patch(shape_padding=False)
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_pytree_in_out(self, device, dynamic, reverse, dim, autograd):
        # [FUNCTION DOC] Tests scan with pytree (mapping) carry and outputs across devices,
        # dynamic modes, direction, scan dimension and autograd; shape padding is disabled.
        self._run_test(
            model=ScanModels.SimpleWithPytreeInOuts(reverse=reverse, dim=dim),
            inputs=(
                torch.ones(2, 2, 2),
                torch.ones(2, 2),
                torch.ones(2),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 3])
    @parametrize("scan_length", [1, 5])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_nn_modules(
        self, device, dynamic, reverse, dim, scan_length, autograd
    ):
        # [FUNCTION DOC] Tests scan whose combine function applies a linear layer over views,
        # across devices, dynamic modes, direction, dims, scan length and autograd. Scan axis is
        # created by moving the first dim to dim.
        init = torch.randn(20, 16, 4, 4, dtype=torch.float64)
        xs = torch.randn(scan_length, 20, 16, 4, 4, dtype=torch.float64)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanLinearWithView(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 3])
    @parametrize("scan_length", [1, 5])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_conv(self, device, dynamic, reverse, dim, scan_length, autograd):
        # [FUNCTION DOC] Tests scan whose combine function is a convolution, across the same
        # parameter grid.
        init = torch.randn(2, 4, 4, 4, dtype=torch.float64)
        xs = torch.randn(scan_length, 2, 4, 4, 4, dtype=torch.float64)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanConv(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 3])
    @parametrize("pred", [True, False])
    @parametrize("scan_length", [1, 5])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_in_cond(
        self, device, dynamic, reverse, dim, pred, scan_length, autograd
    ):
        # [FUNCTION DOC] Tests scan inside cond branches over the same parameter grid with a
        # predicate. Skipped whenever autograd is true because of a referenced backward-pass
        # issue.
        # TODO: remove when https://github.com/pytorch/pytorch/issues/182381 is resolved.
        if autograd:
            raise unittest.SkipTest(
                "Fails due to issues with backward pass when compiled."
            )

        init = torch.randn(4, 4, 4, dtype=torch.float64)
        xs = torch.randn(scan_length, 4, 4, 4, dtype=torch.float64)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanInCond(reverse=reverse, dim=dim),
            inputs=(
                torch.tensor(pred),
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 3])
    @parametrize("scan_length", [1, 5])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_cond_in_scan(self, device, dynamic, reverse, dim, scan_length, autograd):
        # [FUNCTION DOC] Tests cond inside the scan combine function over the usual parameter
        # grid.
        init = torch.randn(2, 4, 4, 4)
        xs = torch.randn(scan_length, 4, 4, 4)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.CondInScan(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_chunked_ce(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests the chunked cross-entropy-style model computed with scan across
        # devices, dynamic modes and autograd.
        self._run_test(
            model=ScanModels.ChunkedCE(10),
            inputs=(
                torch.randn(100, 20),
                torch.randn(20, 20),
                torch.randn(100, 20),
                torch.randn(20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_compare_chunked_ce_with_no_scan(self, device, dynamic):
        # [FUNCTION DOC] Tests that a compiled scan-based chunked loss matches the no-scan version
        # for two chunk-size and shape settings, across devices and dynamic modes.
        for trunk_size, B, T in zip([10, 20], [10, 100], [20, 40]):
            self._compare_result(
                model1=torch.compile(ScanModels.ChunkedCE(trunk_size), dynamic=dynamic),
                model2=ScanModels.ChunkedCENoScan(trunk_size),
                inputs=(
                    torch.randn(B, T),
                    torch.randn(T, T),
                    torch.randn(B, T),
                    torch.randn(T),
                ),
                device=device,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_scan_with_clamp(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests scan whose step clamps its carry, across devices, dynamic modes and
        # autograd.
        B = 4
        T = 8
        H = 16
        self._run_test(
            model=ScanModels.ScanWithClamp(),
            inputs=(
                torch.randn((B, H)),
                torch.randn((T, B, H)),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )


class MapModels:
    # [CLASS DOC] HELPER: Namespace class grouping map test models. Holds no state itself.
    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Model mapping a function that adds a ones tensor, created outside,
        # to the sine of the element.
        def forward(self, map_op, x):
            # [FUNCTION DOC] HELPER: Creates a 3 by 4 ones tensor on the input's device and
            # applies the supplied map operation.
            a = torch.ones(3, 4, device=x.device)

            def f(x):
                # [FUNCTION DOC] HELPER: Mapped function; returns sine of the element plus the
                # captured ones tensor.
                return x.sin() + a

            return map_op(f, x)

    class SimpleWithLinearWithView(torch.nn.Module):
        # [CLASS DOC] HELPER: Model mapping a linear layer plus sine over a reshaped input.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Creates a 3 to 5 linear layer.
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, map_op, x):
            # [FUNCTION DOC] HELPER: Reshapes x to four rows of three and applies the supplied map
            # operation.
            def f(x):
                # [FUNCTION DOC] HELPER: Mapped function; returns sine of the linear layer applied
                # to the element.
                return self.linear(x).sin()

            return map_op(f, x.view(4, 3))

    class PytreeInOut(torch.nn.Module):
        # [CLASS DOC] HELPER: Model mapping over a nested mapping/tuple input and returning a
        # nested output.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Creates a 3 to 5 linear layer.
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, map_op, x, y, z):
            # [FUNCTION DOC] HELPER: Packs inputs into a nested structure and applies the supplied
            # map operation.
            def f(x_y_z):
                # [FUNCTION DOC] HELPER: Mapped function; unpacks the nested element and returns
                # sine of linear x plus a pair of linear y and cosine z.
                x = x_y_z["x"]
                y, (z,) = x_y_z["y_z"]
                return self.linear(x).sin(), (self.linear(y), z.cos())

            return map_op(f, {"x": x, "y_z": (y, (z,))})

    class ReinterpretView(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose mapped function returns slices (views) of its results.
        def forward(self, map_op, x, y, z):
            # [FUNCTION DOC] HELPER: Applies the supplied map operation to the three inputs as a
            # tuple.
            def f(xyz):
                # [FUNCTION DOC] HELPER: Mapped function; returns the first two rows of sine x and
                # the first two rows of cosine y plus a clone of the last two rows of z.
                x, y, z = xyz
                return x.sin()[:2], y.cos()[:2] + z[-2:].clone()

            return map_op(f, (x, y, z))

    class NestedWithCond(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with a map nested inside both branches of a cond on the sign
        # of the sum of x.
        def forward(self, map_op, x, y, z):
            # [FUNCTION DOC] HELPER: Defines the branches and conds on whether the sum of x is
            # positive.
            def true_fn(x, y, z):
                # [FUNCTION DOC] HELPER: True branch; maps an addition of the pair y and z.
                def inner_f(yz):
                    # [FUNCTION DOC] HELPER: Mapped function; returns the sum of the pair.
                    y, z = yz
                    return y + z

                return map_op(inner_f, (y, z))

            def false_fn(x, y, z):
                # [FUNCTION DOC] HELPER: False branch; maps a subtraction of the pair y and z.
                def inner_f(yz):
                    # [FUNCTION DOC] HELPER: Mapped function; returns the difference of the pair.
                    y, z = yz
                    return y - z

                return map_op(inner_f, (y, z))

            return torch._higher_order_ops.cond(
                x.sum() > 0, true_fn, false_fn, (x, y, z)
            )


class MapTests(TestCase):
    # [CLASS DOC] Tests for map compiled with inductor against eager and a fake map reference.
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        autograd=False,
    ):
        # [FUNCTION DOC] HELPER: Harness. Moves inputs and model to the device, makes eager and
        # compiled copies, compiles with a counting inductor backend, optionally marks inputs as
        # requiring grad, and checks that eager, fake-map and compiled outputs agree. With
        # autograd also backpropagates the loss through all three and compares parameters and
        # gradients.
        import copy

        inputs = [inp.to(device=device) for inp in inputs]
        model = model.to(device=device)
        model_eager = copy.deepcopy(model)
        model_compiled = copy.deepcopy(model)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True, dynamic=dynamic)(
            model_compiled
        )

        if autograd:
            pytree.tree_map_only(torch.Tensor, lambda t: t.requires_grad_(True), inputs)

        cloned_inputs = [inp.clone() for inp in inputs]
        result = model(torch._higher_order_ops.map, *cloned_inputs)
        result_exp = model_eager(_fake_map, *cloned_inputs)
        result_compiled = compiled_model(torch._higher_order_ops.map, *cloned_inputs)

        self.assertEqual(result, result_exp)
        self.assertEqual(result, result_compiled)

        if autograd:
            loss_fn(result).backward()
            loss_fn(result_exp).backward()
            loss_fn(result_compiled).backward()

            model_params = dict(model.named_parameters())
            model_eager_params = dict(model_eager.named_parameters())
            model_compiled_params = dict(model_compiled.named_parameters())
            for name, param in model_eager_params.items():
                self.assertEqual(param, model_params[name])
                self.assertEqual(param, model_compiled_params[name])
                self.assertEqual(param.grad, model_params[name].grad)
                self.assertEqual(param.grad, model_compiled_params[name].grad)

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_map_simple(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests a simple map across devices, dynamic modes and autograd.
        self._run_test(
            model=MapModels.Simple(),
            inputs=(torch.randn(3, 4),),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_map_simple_linear_with_view(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests map with a linear layer applied to a reshaped input across devices,
        # dynamic modes and autograd.
        self._run_test(
            model=MapModels.SimpleWithLinearWithView(),
            inputs=(torch.randn(3, 4),),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_map_pytree_in_out(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests map with pytree inputs and outputs across devices, dynamic modes
        # and autograd.
        self._run_test(
            model=MapModels.PytreeInOut(),
            inputs=(
                torch.randn(2, 5, 3),
                torch.randn(2, 5, 3),
                torch.randn(2, 4, 3),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_map_nested_with_cond(self, device, dynamic, autograd):
        # [FUNCTION DOC] Tests map nested inside cond across devices, dynamic modes and autograd.
        self._run_test(
            model=MapModels.NestedWithCond(),
            inputs=(
                torch.randn(3, 2),
                torch.randn(3, 10, 5),
                torch.randn(3, 10, 5),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )


class SwitchModels:
    # [CLASS DOC] HELPER: Namespace class grouping switch test models. Holds no state itself.
    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Model that dispatches on an index among three branches adding one,
        # doubling or negating.
        def forward(self, idx, x):
            # [FUNCTION DOC] HELPER: Calls the switch operation with the index and the three
            # lambda branches on x.
            return switch(
                idx,
                [
                    lambda a: a + 1.0,
                    lambda a: a * 2.0,
                    lambda a: -a,
                ],
                (x,),
            )

    class MultipleOutputs(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose three branches each return a pair of outputs.
        def forward(self, idx, x, y):
            # [FUNCTION DOC] HELPER: Calls the switch operation with the index and the three
            # branches on x and y.
            return switch(
                idx,
                [
                    lambda a, b: (a + b, a * b),
                    lambda a, b: (a - b, a / (b + 1e-6)),
                    lambda a, b: (a * 3.0, b - 1.0),
                ],
                (x, y),
            )

    class OuterCode(torch.nn.Module):
        # [CLASS DOC] HELPER: Model with computation before and after the switch (sine, cosine or
        # absolute value branches).
        def forward(self, idx, x):
            # [FUNCTION DOC] HELPER: Doubles x, switches among three unary branches, and adds one
            # to the result.
            c = x * 2.0
            result = switch(
                idx,
                [
                    lambda a: a.sin(),
                    lambda a: a.cos(),
                    lambda a: a.abs(),
                ],
                (c,),
            )
            return result + 1.0

    class OuterBuffers(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose branches close over a scalar computed outside the
        # switch.
        def forward(self, idx, x, y):
            # [FUNCTION DOC] HELPER: Computes the sum of x as a scale and switches among branches
            # multiplying, dividing by scale plus one, or subtracting it, applied to y.
            scale = x.sum()

            return switch(
                idx,
                [
                    lambda a: a * scale,
                    lambda a: a / (scale + 1.0),
                    lambda a: a - scale,
                ],
                (y,),
            )

    class WithNNModuleParams(torch.nn.Module):
        # [CLASS DOC] HELPER: Model whose three branches are linear modules scaled by different
        # constants.
        class Branch(torch.nn.Module):
            # [CLASS DOC] HELPER: Branch module with a 4 by 4 linear layer and a scale value.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Stores the linear layer and scale.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.scale = scale

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x, times the scale.
                return self.linear(x) * self.scale

        def __init__(self, device):
            # [FUNCTION DOC] HELPER: Creates three branch modules with scales one, two and three
            # on the device.
            super().__init__()
            self.b0 = self.Branch(1.0).to(device)
            self.b1 = self.Branch(2.0).to(device)
            self.b2 = self.Branch(3.0).to(device)

        def forward(self, idx, x):
            # [FUNCTION DOC] HELPER: Calls the switch operation with the index and the three
            # modules on x.
            return switch(idx, [self.b0, self.b1, self.b2], (x,))


class SwitchTests(TestCase):
    # [CLASS DOC] Tests for the switch operator compiled with inductor.
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_branches=3,
    ):
        # [FUNCTION DOC] HELPER: Harness. Compiles the model with a counting backend, optionally
        # adds inputs tiled five times along the first dim and marks first dims dynamic, then for
        # every input set and every branch index runs eager and compiled with an index tensor
        # prepended, asserts operands unchanged and results close, and asserts one compilation.
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        inputs = [inp.to(device=device) for inp in inputs]
        input_sets = [inputs]
        if dynamic:
            larger_inputs = []
            for inp in inputs:
                if inp.ndim > 0:
                    tiling = [5] + [1] * (inp.ndim - 1)
                    larger_inputs.append(torch.tile(inp, tiling))
                else:
                    larger_inputs.append(inp)
            input_sets.append(larger_inputs)
            for inputs in input_sets:
                for inp in inputs:
                    torch._dynamo.mark_dynamic(inp, 0)

        # Exercise every branch index (0 .. num_branches-1) so that all
        # generated code paths are verified against the eager reference.
        for inputs in input_sets:
            for branch_idx in range(num_branches):
                idx_tensor = torch.tensor(branch_idx, device=device)
                inputs_with_idx = [idx_tensor, *inputs]
                cloned = [inp.clone() for inp in inputs_with_idx]
                result = model(*inputs_with_idx)
                result_compiled = compiled_model(*inputs_with_idx)
                # operands must not be mutated
                torch.testing.assert_close(cloned, inputs_with_idx)
                torch.testing.assert_close(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_switch_simple(self, device, dynamic):
        # [FUNCTION DOC] Tests a simple switch of three unary branches across devices and dynamic
        # modes.
        self._run_test(
            model=SwitchModels.Simple(),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_switch_multiple_outputs(self, device, dynamic):
        # [FUNCTION DOC] Tests switch where branches return tuples (multiple output nodes) across
        # devices and dynamic modes.
        # each branch returns a tuple — exercises MultiOutput IR node fan-out
        self._run_test(
            model=SwitchModels.MultipleOutputs(),
            inputs=(torch.randn(10, 20), torch.randn(10, 20)),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_switch_outer_code_before_after(self, device, dynamic):
        # [FUNCTION DOC] Tests ops before and after the switch across devices and dynamic modes.
        # ops surrounding the switch fuse into the outer graph
        self._run_test(
            model=SwitchModels.OuterCode(),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_switch_outer_buffers_captured(self, device, dynamic):
        # [FUNCTION DOC] Tests branches that capture tensors computed outside the switch across
        # devices and dynamic modes.
        # branches close over tensors computed outside the switch
        self._run_test(
            model=SwitchModels.OuterBuffers(),
            inputs=(torch.randn(10, 20), torch.randn(10, 20)),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_switch_nn_module_params(self, device):
        # [FUNCTION DOC] Tests switch with branch modules holding parameters across devices
        # (static shapes).
        self._run_test(
            model=SwitchModels.WithNNModuleParams(device),
            inputs=(torch.randn(4, 4),),
            device=device,
        )


instantiate_parametrized_tests(CondTests)
instantiate_parametrized_tests(WhileLoopTests)
instantiate_parametrized_tests(AssociativeScanTests)
instantiate_parametrized_tests(ScanTests)
instantiate_parametrized_tests(MapTests)
instantiate_parametrized_tests(SwitchTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
