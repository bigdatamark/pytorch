# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor decomposition of the DTensor shard-dim all-to-all op into
# the functional c10d all-to-all plus wait collectives. Covers graph tracing of the
# decomposition, numerical equivalence against a pure-tensor reference run on simulated ranks,
# the backward formula (swapped gather and shard dims), the cases where the decomposition
# declines, and the autograd registration of the op. No real process group is used; the group
# size lookup and collectives are mocked.

import math
import unittest
from unittest import mock

import torch
from torch import fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


@unittest.skipIf(not torch.distributed.is_available(), "requires distributed support")
class TestDecomposeShardDimAllToAll(TestCase):
    # [CLASS DOC] Tests of the shard-dim all-to-all decomposition; skipped when torch
    # distributed is unavailable. Helper methods provide a reference implementation (concatenate
    # along the gather dim, then split along the shard dim), a local simulation of the
    # decomposition across ranks, access to the registered decomposition, a group size mock, and
    # tracing/calling helpers.
    def _reference_shard_dim_alltoall(
        self,
        inputs: list[torch.Tensor],
        *,
        gather_dim: int,
        shard_dim: int,
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Reference result of the op for a list of per-rank inputs. The
        # group size is the number of inputs; it concatenates all inputs along the gather dim
        # and splits the result into that many equal chunks along the shard dim (negative dims
        # are wrapped). Returns the list of per-rank chunks.
        group_size = len(inputs)
        ndim = inputs[0].dim()
        gathered = torch.cat(inputs, dim=gather_dim % ndim)
        return list(torch.chunk(gathered, group_size, dim=shard_dim % ndim))

    def _run_decomp_locally(
        self,
        inputs: list[torch.Tensor],
        *,
        gather_dim: int,
        shard_dim: int,
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Runs the registered decomposition once per simulated rank
        # without a real group. Computes, for each rank input, the same pre-collective layout
        # (shard dim split into group-size pieces, moved to the front and made contiguous) and
        # the collective result each rank would receive (the rank-th slice from every rank,
        # concatenated on the first dim). Mocks the functional all-to-all single op to assert
        # the group name is the test one, both split size lists are all ones and the input
        # equals that rank pre-collective tensor, then return the precomputed result; the wait
        # op is mocked as identity. Runs with the decomposition config on and the group size
        # mocked. Returns the list of per-rank outputs.
        group_size = len(inputs)
        ndim = inputs[0].dim()
        gather_dim = gather_dim % ndim
        shard_dim = shard_dim % ndim
        input_shape = list(inputs[0].shape)
        local_shard_dim = input_shape[shard_dim] // group_size

        pre_view_shape = list(input_shape)
        pre_view_shape[shard_dim] = local_shard_dim
        pre_view_shape.insert(shard_dim, group_size)

        post_view_shape = list(input_shape)
        post_view_shape[shard_dim] = local_shard_dim
        post_view_shape[gather_dim] *= group_size

        pre_alltoall = [
            inp.view(pre_view_shape)
            .movedim(shard_dim, 0)
            .clone(memory_format=torch.contiguous_format)
            for inp in inputs
        ]
        collective_outputs = [
            torch.cat([pre.narrow(0, rank, 1) for pre in pre_alltoall], dim=0)
            for rank in range(group_size)
        ]

        current_rank = 0

        def all_to_all_single(input, output_split_sizes, input_split_sizes, group_name):
            # [FUNCTION DOC] HELPER: Mock all-to-all single; asserts the group name, the output
            # and input split sizes (all ones with length of group size) and that the input
            # equals the current rank pre-collective tensor, then returns that rank precomputed
            # collective output. The current rank is read from the enclosing scope and updated
            # by the loop.
            self.assertEqual(group_name, "test_pg")
            self.assertEqual(output_split_sizes, [1] * group_size)
            self.assertEqual(input_split_sizes, [1] * group_size)
            self.assertEqual(input, pre_alltoall[current_rank])
            return collective_outputs[current_rank]

        def wait_tensor(input):
            # [FUNCTION DOC] HELPER: Mock wait op that returns its input unchanged.
            return input

        outputs = []
        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": True}):
            with get_group_size:
                with (
                    mock.patch.object(
                        torch.ops._c10d_functional.all_to_all_single,
                        "default",
                        side_effect=all_to_all_single,
                    ),
                    mock.patch.object(
                        torch.ops._c10d_functional.wait_tensor,
                        "default",
                        side_effect=wait_tensor,
                    ),
                ):
                    for rank, inp in enumerate(inputs):
                        current_rank = rank
                        outputs.append(
                            self._decomp()(inp, gather_dim, shard_dim, "test_pg")
                        )

        return outputs

    def _decomp(self):
        # [FUNCTION DOC] HELPER: Looks up and returns the registered decomposition callable for
        # the shard dim all-to-all op from the Inductor decomposition table.
        return select_decomp_table()[torch.ops._dtensor.shard_dim_alltoall.default]

    def _group_size_mock(self, group_size: int):
        # [FUNCTION DOC] HELPER: Returns an (unstarted) patcher object that replaces the
        # distributed group-size-by-name lookup with a function returning the given fixed group
        # size; callers use it as a context manager.
        return mock.patch(
            "torch.distributed.distributed_c10d._get_group_size_by_name",
            return_value=group_size,
        )

    def _trace_decomp(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
        group_size: int = 4,
    ) -> fx.GraphModule:
        # [FUNCTION DOC] HELPER: Traces the shard dim all-to-all op on an empty tensor of the
        # given shape and dtype with a fake-tensor make_fx trace, using the Inductor
        # decomposition table, the decomposition config enabled and the group size mocked.
        # Returns the resulting graph module.
        inp = torch.empty(shape, dtype=dtype)

        def fn(x):
            # [FUNCTION DOC] HELPER: Traced function that applies the shard dim all-to-all op
            # with the captured gather dim, shard dim and the test group name.
            return torch.ops._dtensor.shard_dim_alltoall.default(
                x, gather_dim, shard_dim, "test_pg"
            )

        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": True}):
            with get_group_size:
                return make_fx(
                    fn,
                    decomposition_table=select_decomp_table(),
                    tracing_mode="fake",
                )(inp)

    def _call_decomp(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
        group_size: int = 4,
        enabled: bool = True,
    ):
        # [FUNCTION DOC] HELPER: Calls the registered decomposition directly on an empty
        # tensor with the given shape, dtype, dims and mocked group size, with the
        # decomposition config set to the enabled flag. Returns whatever it returns (may be a
        # not-implemented sentinel).
        inp = torch.empty(shape, dtype=dtype)
        get_group_size = self._group_size_mock(group_size)
        with config.patch({"decompose_shard_dim_alltoall": enabled}):
            with get_group_size:
                return self._decomp()(inp, gather_dim, shard_dim, "test_pg")

    def test_decomposition_exposes_c10d_collective(self) -> None:
        # [FUNCTION DOC] Traces the decomposition for an eight by five by six input gathering
        # the last dim and sharding the first, with a group of four. Asserts the graph passes
        # linting, the original op is gone, the functional all-to-all and wait ops are present,
        # the all-to-all split sizes are all ones and group name is the test one, the output
        # value shape is two by five by twenty-four, and the decomposition counter equals one.
        counters.clear()
        gm = self._trace_decomp(shape=(8, 5, 6), gather_dim=2, shard_dim=0)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)

        alltoall_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.all_to_all_single.default
        )
        self.assertEqual(alltoall_node.args[1], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[2], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[3], "test_pg")
        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(output_node.args[0].meta["val"].shape, (2, 5, 24))
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 1)

    def test_decomposition_matches_original_api_semantics_locally(self) -> None:
        # [FUNCTION DOC] For several group sizes, shapes and gather/shard dim pairs
        # (including a negative dim), builds per-rank inputs from a range, runs the local
        # simulation and compares it with the reference (subtests per case); also asserts
        # each output is contiguous.
        cases = [
            (4, (5, 8), 0, 1),
            (4, (8, 5, 6), 2, 0),
            (4, (3, 8, 4), -1, 1),
            (2, (6, 4, 5), 1, 0),
        ]

        for group_size, shape, gather_dim, shard_dim in cases:
            with self.subTest(
                group_size=group_size,
                shape=shape,
                gather_dim=gather_dim,
                shard_dim=shard_dim,
            ):
                inputs = torch.arange(
                    group_size * math.prod(shape),
                    dtype=torch.float32,
                ).reshape(group_size, *shape)
                per_rank_inputs = list(inputs.unbind(0))

                expected = self._reference_shard_dim_alltoall(
                    per_rank_inputs,
                    gather_dim=gather_dim,
                    shard_dim=shard_dim,
                )
                actual = self._run_decomp_locally(
                    per_rank_inputs,
                    gather_dim=gather_dim,
                    shard_dim=shard_dim,
                )

                self.assertEqual(actual, expected)
                self.assertTrue(all(out.is_contiguous() for out in actual))

    def test_backward_formula_matches_original_api_semantics_locally(self) -> None:
        # [FUNCTION DOC] Checks the backward formula: forward outputs come from the reference,
        # per-rank gradient outputs are built from ranges offset by rank, and the expected
        # gradients are the reference result with gather and shard dims swapped. Asserts the
        # local decomposition run with swapped dims gives the same gradients.
        group_size = 4
        input_shape = (8, 5, 6)
        gather_dim = 2
        shard_dim = 0
        inputs = torch.arange(
            group_size * math.prod(input_shape),
            dtype=torch.float32,
        ).reshape(group_size, *input_shape)
        forward_outputs = self._reference_shard_dim_alltoall(
            list(inputs.unbind(0)),
            gather_dim=gather_dim,
            shard_dim=shard_dim,
        )

        grad_outputs = [
            torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) + rank
            for rank, out in enumerate(forward_outputs)
        ]

        expected_grads = self._reference_shard_dim_alltoall(
            grad_outputs,
            gather_dim=shard_dim,
            shard_dim=gather_dim,
        )
        actual_grads = self._run_decomp_locally(
            grad_outputs,
            gather_dim=shard_dim,
            shard_dim=gather_dim,
        )
        self.assertEqual(actual_grads, expected_grads)

    def test_falls_back_for_unsupported_cases(self) -> None:
        # [FUNCTION DOC] For each of four cases (config disabled, shard size not divisible by
        # group size, complex dtype, same gather and shard dim) calls the decomposition and
        # asserts it returns the not-implemented sentinel and that the decomposition counter
        # stays zero. Counters cleared per case.
        cases = [
            {"shape": (5, 8), "enabled": False},
            {"shape": (5, 7)},
            {"shape": (5, 8), "dtype": torch.complex64},
            {"shape": (5, 8), "gather_dim": 1, "shard_dim": 1},
        ]

        for case in cases:
            with self.subTest(**case):
                counters.clear()
                result = self._call_decomp(**case)
                self.assertIs(result, NotImplemented)
                self.assertEqual(
                    counters["inductor"]["decompose_shard_dim_alltoall"], 0
                )

    def test_shard_dim_alltoall_registered_autograd_backward(self) -> None:
        # [FUNCTION DOC] Registers a CPU implementation of the op in a scoped library that
        # records its calls (first call returns a detached clone, later calls return a tensor
        # filled with seven), runs the op on an input requiring grad and sums and back-
        # propagates. Asserts two calls happened, the backward call has the gather and shard
        # dims swapped with the same group name and a non-grad input, the output required grad,
        # and the input gradient is all sevens.
        import torch.distributed.tensor._collective_utils

        calls = []

        with torch.library._scoped_library("_dtensor", "IMPL") as lib:

            def impl(input, gather_dim: int, shard_dim: int, group_name):
                # [FUNCTION DOC] HELPER: Recording implementation: appends its arguments to the
                # shared call list, returns a detached clone on the first call and a tensor of
                # sevens shaped like the input after.
                calls.append((input, gather_dim, shard_dim, group_name))
                if len(calls) == 1:
                    return input.detach().clone()
                return torch.full_like(input, 7.0)

            lib.impl("shard_dim_alltoall", impl, "CPU")

            x = torch.randn(2, 3, 4, requires_grad=True)
            y = torch.ops._dtensor.shard_dim_alltoall.default(x, 2, 0, "test_pg")
            self.assertTrue(y.requires_grad)
            y.sum().backward()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:], (2, 0, "test_pg"))
        self.assertEqual(calls[1][1:], (0, 2, "test_pg"))
        self.assertFalse(calls[1][0].requires_grad)
        self.assertEqual(x.grad, torch.full_like(x, 7.0))


if __name__ == "__main__":
    run_tests()
