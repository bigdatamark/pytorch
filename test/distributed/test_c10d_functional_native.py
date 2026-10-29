# Owner(s): ["module: c10d"]
# [MODULE DOC] Tests for the native _c10d_functional custom ops and the Python functional
# collective wrappers. Contains multi-GPU eager tests (TestWithNCCL), tests with a dummy
# process group that discards data (work lifetime, cross thread wait, process group arguments,
# opaque type registration), and inductor compile tests on fake process groups that check
# generated code and AOT Inductor runs. Import time side effects: loads the inductor AOT test
# utility module through a file loader, and exits the process with status zero if
# torch.distributed is unavailable.
import gc
import re
import threading
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C import FileCheck
from torch._inductor.utils import fresh_cache, run_and_get_code, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_single,
    all_gather_single_coalesced,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    AsyncCollectiveTensor,
    reduce_scatter_single,
    reduce_scatter_single_coalesced,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_device_type import e4m3_type
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_LINUX,
    run_tests,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU


def load_test_module(name):
    # [FUNCTION DOC] HELPER: loads a test module by dotted name from the parent of this test
    # directory. Temporarily patches sys.path to include that directory, builds the file path
    # by replacing dots with slashes, loads it with a source file loader and returns the
    # module object.
    import sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from unittest import mock

    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


AOTIRunnerUtil = load_test_module("inductor.test_aot_inductor_utils").AOTIRunnerUtil

import sys


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)


@requires_accelerator_dist_backend()
class TestWithNCCL(DistributedTestBase):
    # [CLASS DOC] Two rank accelerator tests of the eager _c10d_functional ops and their
    # Python wrappers; the whole class requires an accelerator distributed backend and each
    # test needs at least 2 GPUs. Each test initializes the default group registered under the
    # name default. Tests check numerics, whether results alias inputs, and that Python
    # wrappers return AsyncCollectiveTensor that completes on first use, plus work registry
    # bookkeeping and inductor behavior.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property: world size for these tests, always 2.
        return 2

    @property
    def ranks(self) -> list[int]:
        # [FUNCTION DOC] Property: list of all ranks, from zero up to world size minus one.
        return list(range(self.world_size))

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] Property: torch device whose index is this rank.
        return torch.device(self.rank)

    def _init_process_group(self) -> None:
        # [FUNCTION DOC] HELPER: creates the process group for the device type, then registers
        # the world group in the native process group registry under the name default so ops
        # can refer to it by that name.
        self.create_pg(self.device.type)
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_single(self) -> None:
        # [FUNCTION DOC] Test: functional all_reduce with average on a rank filled tensor
        # returns a new tensor (not the input) holding the mean over ranks after wait_tensor.
        # The Python wrapper returns an AsyncCollectiveTensor that is not completed until its
        # values are first used, after which it is completed.
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        if id(output) == id(input):
            raise AssertionError("Expected output to be a different object than input")
        expect = sum(self.ranks) / self.world_size
        if not output.eq(expect).all():
            raise AssertionError(f"Expected output to equal {expect}")

        # Test Python API and AsyncCollectiveTensor
        output = all_reduce(
            input,
            "avg",
            "default",
        )
        if not isinstance(output, AsyncCollectiveTensor):
            raise AssertionError(f"Expected AsyncCollectiveTensor, got {type(output)}")
        if output.completed:
            raise AssertionError("Expected output.completed to be False")
        if not output.eq(expect).all():
            raise AssertionError(f"Expected output to equal {expect}")
        if not output.completed:
            raise AssertionError("Expected output.completed to be True after access")

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_single_(self) -> None:
        # [FUNCTION DOC] Test: in place all_reduce with average returns the same object as the
        # input after wait_tensor, holding the mean over ranks.
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_reduce_(
            input,
            "avg",
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        if id(output) != id(input):
            raise AssertionError("Expected output to be the same object as input")
        expect = sum(self.ranks) / self.world_size
        if not output.eq(expect).all():
            raise AssertionError(f"Expected output to equal {expect}")

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced(self) -> None:
        # [FUNCTION DOC] Test: coalesced all_reduce over ten tensors with average gives new
        # tensors equal to the mean rank times the index; the Python wrapper returns
        # incomplete async tensors that complete when accessed.
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            if id(output) == id(input):
                raise AssertionError("Expected output to be different from input")
            expected = sum(self.ranks) / self.world_size * i
            if not output.eq(expected).all():
                raise AssertionError(f"Expected output to equal {expected}")

        # Test Python API and AsyncCollectiveTensor
        outputs = all_reduce_coalesced(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            if output.completed:
                raise AssertionError("Expected output.completed to be False")
            expected = sum(self.ranks) / self.world_size * i
            if not output.eq(expected).all():
                raise AssertionError(f"Expected output to equal {expected}")
            if not output.completed:
                raise AssertionError(
                    "Expected output.completed to be True after access"
                )

    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_(self) -> None:
        # [FUNCTION DOC] Test: in place coalesced all_reduce with average returns the very
        # same tensor objects after wait_tensor, with the expected mean values.
        self._init_process_group()

        inputs = [
            torch.full((i, i), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_reduce_coalesced_(
            inputs,
            "avg",
            "default",
        )
        for i, (output, input) in enumerate(zip(outputs, inputs)):
            output = torch.ops._c10d_functional.wait_tensor(output)
            if id(output) != id(input):
                raise AssertionError("Expected output to be the same object as input")
            expected = sum(self.ranks) / self.world_size * i
            if not output.eq(expected).all():
                raise AssertionError(f"Expected output to equal {expected}")

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor_single(self) -> None:
        # [FUNCTION DOC] Test: all_gather_into_tensor concatenates rank filled tensors in rank
        # order; checks the plain op, the out variant that writes into a preallocated tensor,
        # and the Python wrapper returning an async tensor that completes when accessed.
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor(
            input,
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((10, 10), float(rank), device=self.device)
                for rank in self.ranks
            ]
        )
        if not torch.allclose(output, expect):
            raise AssertionError("Expected output to be close to expect")
        if not output.eq(expect).all():
            raise AssertionError("Expected output to equal expect")

        # Test out-variant of all_gather_into_tensor
        output = torch.empty(expect.shape, device=self.device)
        output = torch.ops._c10d_functional.all_gather_into_tensor_out(
            input,
            self.world_size,
            "default",
            out=output,
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        if not torch.allclose(output, expect):
            raise AssertionError("Expected output to be close to expect")
        if not output.eq(expect).all():
            raise AssertionError("Expected output to equal expect")

        # Test Python API and AsyncCollectiveTensor
        output = all_gather_single(
            input,
            0,
            "default",
        )
        if not isinstance(output, AsyncCollectiveTensor):
            raise AssertionError(f"Expected AsyncCollectiveTensor, got {type(output)}")
        if output.completed:
            raise AssertionError("Expected output.completed to be False")
        if not output.eq(expect).all():
            raise AssertionError("Expected output to equal expect")
        if not output.completed:
            raise AssertionError("Expected output.completed to be True after access")

    # https://github.com/pytorch/pytorch/issues/133421
    @skip_if_lt_x_gpu(2)
    def test_functional_collectives_inference_mode(self) -> None:
        # [FUNCTION DOC] Test: regression check for issue 133421. Under inference mode an
        # all_gather_single result converted with a dtype change must trigger the wait, give a
        # plain tensor and equal the expected gathered bfloat16 values.
        self._init_process_group()

        with torch.inference_mode():
            input = torch.full((2, 2), float(self.rank), device=self.device)
            out1 = funcol.all_gather_single(
                input, gather_dim=0, group=torch.distributed.group.WORLD
            )
            out2 = out1.to(dtype=torch.bfloat16)
            # this tests that the call to .to() properly triggered a wait() on the AsyncCollectiveTensor
            self.assertTrue(type(out2) is torch.Tensor)
            self.assertEqual(
                out2,
                torch.tensor(
                    [[0, 0], [0, 0], [1, 1], [1, 1]],
                    device=self.device,
                    dtype=torch.bfloat16,
                ),
            )

    @skip_if_lt_x_gpu(2)
    def test_functional_collectives_batched(self) -> None:
        # [FUNCTION DOC] Test: batched point to point ops under inference mode. Rank 0 sends
        # and rank 1 receives via batch_p2p_ops_inplace followed by wait_tensor. Rank 0 is
        # expected to get an empty tensor and rank 1 a tensor equal to rank 0's data, zeros.
        self._init_process_group()

        def f(x):
            # [FUNCTION DOC] HELPER: scales the input by this rank and does a batched isend
            # (rank 0) or irecv (rank 1) with the other rank, returning the waited result.
            # Ranks outside 0 and 1 return nothing.
            if self.rank not in [0, 1]:
                return
            tensor = self.rank * x
            return funcol.wait_tensor(
                funcol.batch_p2p_ops_inplace(
                    op_list=["isend" if self.rank == 0 else "irecv"],
                    peer_list=[1 if self.rank == 0 else 0],
                    tensors=[tensor],
                    tag_list=[0],
                    group_name="",
                )[0]
            )

        with torch.inference_mode():
            input = torch.ones((10), device=self.device)
            output = f(input)
            if self.rank == 0:
                self.assertEqual(
                    output,
                    torch.empty(0, device=self.device, dtype=input.dtype),
                )
            else:
                self.assertEqual(output, 0 * input)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # https://github.com/pytorch/pytorch/issues/126338
    def test_inductor_dtypeview_memory_leak(self):
        # [FUNCTION DOC] Test: regression for issue 126338. Compiles a function doing
        # all_gather_into_tensor followed by a dtype view and wait_tensor, checks the
        # generated code keeps the view dtype call as input of wait_tensor, then runs the
        # compiled function repeatedly and raises an error if peak memory differs between the
        # last two iterations (memory leak).
        self._init_process_group()

        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: all_gathers the argument over the default group, views
            # the result as int32 and waits on it.
            ag0 = torch.ops._c10d_functional.all_gather_into_tensor.default(
                arg,
                self.world_size,
                "default",
            )
            ag0_view = torch.ops.aten.view.dtype(ag0, torch.int32)
            return funcol.wait_tensor(ag0_view)

        arg = torch.full(
            (10, 10),
            float(self.rank),
            device=self.device,
            dtype=torch.float32,
        )
        compiled = torch.compile(func)
        mem_usage = {}
        # check if the aten.view.dtype is compiled to aten.view.dtype
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("torch.ops._c10d_functional.wait_tensor.default(aten.view.dtype")
            .run(code)
        )
        # check memory leak
        for i in range(1, 10):
            mem_usage[i] = torch.accelerator.max_memory_allocated()
            compiled(arg)

        if mem_usage[9] != mem_usage[8]:
            raise AssertionError(
                f"Memory leak detected: {mem_usage[9]} != {mem_usage[8]}"
            )

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor_coalesced(self) -> None:
        # [FUNCTION DOC] Test: coalesced all_gather_into_tensor over ten tensors returns
        # concatenations in rank order; Python wrapper results are incomplete async tensors
        # that complete when accessed.
        self._init_process_group()

        inputs = [
            torch.full((10, 10), float(self.rank * i), device=self.device)
            for i in range(10)
        ]
        outputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
            inputs,
            self.world_size,
            "default",
        )
        expect = [
            torch.cat(
                [
                    torch.full((10, 10), float(rank) * i, device=self.device)
                    for rank in self.ranks
                ]
            )
            for i in range(10)
        ]
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            if not output.eq(expect[i]).all():
                raise AssertionError(f"Expected output to equal expect[{i}]")

        # Test Python API and AsyncCollectiveTensor
        outputs = all_gather_single_coalesced(
            inputs,
            "default",
        )
        for i, output in enumerate(outputs):
            if output.completed:
                raise AssertionError("Expected output.completed to be False")
            if not output.eq(expect[i]).all():
                raise AssertionError(f"Expected output to equal expect[{i}]")
            if not output.completed:
                raise AssertionError(
                    "Expected output.completed to be True after access"
                )

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_single(self) -> None:
        # [FUNCTION DOC] Test: reduce_scatter_tensor with average of a tensor of rank ids
        # gives each rank a value equal to its rank; the Python wrapper returns an async
        # tensor that completes when accessed.
        self._init_process_group()

        input = torch.tensor(self.ranks, device=self.device)
        output = torch.ops._c10d_functional.reduce_scatter_tensor(
            input,
            "avg",
            self.world_size,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        if not output.eq(self.rank).all():
            raise AssertionError(f"Expected output to equal {self.rank}")

        # Test Python API and AsyncCollectiveTensor
        output = reduce_scatter_single(
            input,
            "avg",
            0,
            "default",
        )
        if not isinstance(output, AsyncCollectiveTensor):
            raise AssertionError(f"Expected AsyncCollectiveTensor, got {type(output)}")
        if output.completed:
            raise AssertionError("Expected output.completed to be False")
        if not output.eq(self.rank).all():
            raise AssertionError(f"Expected output to equal {self.rank}")
        if not output.completed:
            raise AssertionError("Expected output.completed to be True after access")

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_out(self) -> None:
        # [FUNCTION DOC] Test: reduce_scatter_tensor_out with average writes the scattered
        # result into the provided out tensor; after waiting, the out tensor equals the rank.
        self._init_process_group()

        input = torch.tensor(self.ranks, device=self.device)
        out = torch.tensor([-1], device=self.device)
        w = torch.ops._c10d_functional.reduce_scatter_tensor_out(
            input,
            "avg",
            self.world_size,
            "default",
            out=out,
        )
        torch.ops._c10d_functional.wait_tensor(w)
        if not out.eq(self.rank).all():
            raise AssertionError(f"Expected out to equal {self.rank}")

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self) -> None:
        # [FUNCTION DOC] Test: coalesced reduce_scatter_tensor over ten tensors with average
        # gives rank times index; the Python wrapper returns incomplete async tensors that
        # complete when accessed.
        self._init_process_group()

        inputs = [torch.tensor(self.ranks, device=self.device) * i for i in range(10)]
        outputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
            inputs,
            "avg",
            self.world_size,
            "default",
        )
        for i, output in enumerate(outputs):
            output = torch.ops._c10d_functional.wait_tensor(output)
            expected = self.rank * i
            if not output.eq(expected).all():
                raise AssertionError(f"Expected output to equal {expected}")

        # Test Python API and AsyncCollectiveTensor
        outputs = reduce_scatter_single_coalesced(
            inputs,
            "avg",
            [0] * 10,
            "default",
        )
        for i, output in enumerate(outputs):
            if output.completed:
                raise AssertionError("Expected output.completed to be False")
            expected = self.rank * i
            if not output.eq(expected).all():
                raise AssertionError(f"Expected output to equal {expected}")
            if not output.completed:
                raise AssertionError(
                    "Expected output.completed to be True after access"
                )

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self) -> None:
        # [FUNCTION DOC] Test: all_to_all_single with random per rank split sizes (seeded)
        # sends rank filled data; the received tensor must equal the concatenation of rank
        # filled chunks of the output split sizes. Also checks the Python wrapper completes
        # lazily. Sets the accelerator device index to the rank first.
        self._init_process_group()
        torch.accelerator.set_device_index(self.rank)

        torch.manual_seed(42)
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        input_split_sizes = send_sz_matrix[self.rank].tolist()
        output_split_sizes = send_sz_matrix[:, self.rank].tolist()
        input = torch.full((sum(input_split_sizes),), float(self.rank)).to(
            self.device.type
        )

        output = torch.ops._c10d_functional.all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        expect = torch.cat(
            [
                torch.full((sz,), float(rank)).to(self.device.type)
                for rank, sz in enumerate(output_split_sizes)
            ]
        )
        if not output.eq(expect).all():
            raise AssertionError("Expected output to equal expect")

        # Test Python API and AsyncCollectiveTensor
        output = all_to_all_single(
            input, output_split_sizes, input_split_sizes, "default"
        )
        if output.completed:
            raise AssertionError("Expected output.completed to be False")
        if not output.eq(expect).all():
            raise AssertionError("Expected output to equal expect")
        if not output.completed:
            raise AssertionError("Expected output.completed to be True after access")

    @skip_if_lt_x_gpu(2)
    def test_broadcast(self) -> None:
        # [FUNCTION DOC] Test: broadcast from rank 1 returns a new tensor (not the input)
        # filled with 1 after wait_tensor; the Python wrapper returns an async tensor that
        # completes when accessed.
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        output = torch.ops._c10d_functional.broadcast(
            input,
            1,
            "default",
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        if id(output) == id(input):
            raise AssertionError("Expected output to be a different object than input")
        expect = 1
        if not output.eq(expect).all():
            raise AssertionError(f"Expected output to equal {expect}")

        # Test Python API and AsyncCollectiveTensor
        output = funcol.broadcast(
            input,
            1,
            "default",
        )
        if not isinstance(output, AsyncCollectiveTensor):
            raise AssertionError(f"Expected AsyncCollectiveTensor, got {type(output)}")
        if output.completed:
            raise AssertionError("Expected output.completed to be False")
        if not output.eq(expect).all():
            raise AssertionError(f"Expected output to equal {expect}")
        if not output.completed:
            raise AssertionError("Expected output.completed to be True after access")

    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # [FUNCTION DOC] Test: the native work registry has size 0, grows to 1 after launching
        # a functional all_reduce and returns to 0 immediately after wait_tensor.
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        output = torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
        torch.ops._c10d_functional.wait_tensor(output)
        # `wait_tensor(output)` will pop the work from the work registry immediately
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # [FUNCTION DOC] Test: launches a functional all_reduce and never waits on it; only
        # checks the work registry size is 1 and that the process can still exit cleanly.
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        self._init_process_group()

        input = torch.full((10, 10), float(self.rank), device=self.device)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        torch.ops._c10d_functional.all_reduce(
            input,
            "avg",
            "default",
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @fresh_cache()
    def test_threading(self):
        # [FUNCTION DOC] Test: compiles a function containing an all_reduce and
        # checks the generated code uses the in place variant on the intermediate
        # buffer, then runs the eager and compiled functions from a second thread
        # and re-raises any exception there, showing the process group registered in
        # the main thread can be resolved from another thread.
        self._init_process_group()
        device = self.device

        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: adds 42 to the argument, all_reduces it with average over
            # group name 0, waits and adds one.
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0 + 1

        arg = torch.rand(4, 4, device=device)
        func(arg)

        compiled = torch.compile(func, fullgraph=True)
        code = run_and_get_triton_code(compiled, arg)
        FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code)

        # Unless explicitly specified (e.g. in a custom runtime), the process
        # group registry is shared among all threads in a process. Here we
        # verify that a process group registered in main thread can be resolved
        # in a different thread.
        class TestThread(threading.Thread):
            # [CLASS DOC] HELPER: thread subclass that runs the eager and compiled function,
            # stores any exception in an attribute exc and re-raises it from join. Since exc
            # is set inside run, join before the thread has started running is not supported.
            def run(self):
                # [FUNCTION DOC] HELPER: thread body; resets exc to none, calls the eager
                # function and the compiled function with the captured argument, and stores
                # any exception.
                self.exc = None
                try:
                    func(arg)
                    compiled(arg)
                except BaseException as exc:
                    self.exc = exc

            def join(self):
                # [FUNCTION DOC] HELPER: joins the thread and then raises the exception stored
                # by run, if any.
                threading.Thread.join(self)
                if self.exc:
                    raise self.exc

        t = TestThread()
        t.start()
        t.join()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "_scaled_mm currently only supports sm>=90 on cuda and gfx94/95 on ROCm",
    )
    @skip_if_lt_x_gpu(2)
    @fresh_cache()
    def test_fixed_striding(self):
        # [FUNCTION DOC] Test: skipped without FP8 matmul support. Compares eager and compiled
        # results of an FP8 rowwise backward pass that uses all_gather_single and a scaled
        # matmul, and checks the compiled weight gradient is close to the eager one (guards
        # against wrong strides for collective outputs).
        self._init_process_group()

        def scale(t):
            # [FUNCTION DOC] HELPER: row wise FP8 quantization; returns the input multiplied
            # by the max representable e4m3 value over the row max abs, cast to e4m3, together
            # with the scale.
            scale = (
                torch.finfo(e4m3_type).max / t.abs().amax(dim=-1, keepdim=True).float()
            )
            t = t.mul(scale).to(e4m3_type)
            return t, scale

        def fp8_rowwise_backward(in_, w, out_grad):
            # [FUNCTION DOC] HELPER: computes the input gradient from FP8 quantized
            # all_gathered output gradient and weight using a scaled matmul, and the weight
            # gradient from the all_gathered transposed output gradient times the input.
            # Returns both.
            out_grad_fp8, scale_out_grad = scale(out_grad)
            w_fp8, scale_w = scale(w.t().contiguous())
            out_grad_fp8 = funcol.all_gather_single(
                out_grad_fp8, gather_dim=0, group=torch.distributed.group.WORLD
            )
            scale_out_grad = funcol.all_gather_single(
                scale_out_grad, gather_dim=0, group=torch.distributed.group.WORLD
            )
            in_grad = torch._scaled_mm(
                out_grad_fp8,
                w_fp8.t(),
                scale_a=scale_out_grad,
                scale_b=scale_w.t(),
                out_dtype=torch.bfloat16,
            )

            out_grad = funcol.all_gather_single(
                out_grad.t().contiguous(),
                gather_dim=0,
                group=torch.distributed.group.WORLD,
            )
            w_grad = out_grad @ in_

            return in_grad, w_grad

        m, n, k = 128, 256, 64
        in_ = torch.randn((m, k), device=self.device.type, dtype=torch.bfloat16)
        w = torch.randn((n, k), device=self.device.type, dtype=torch.bfloat16)
        out_grad = torch.randn((m, n), device=self.device.type, dtype=torch.bfloat16)

        eager_in_grad, eager_w_grad = fp8_rowwise_backward(in_, w, out_grad)
        compile_in_grad, compile_w_grad = torch.compile(fp8_rowwise_backward)(
            in_, w, out_grad
        )

        self.assertTrue(torch.allclose(compile_w_grad, eager_w_grad))


def dummy_init_pg() -> None:
# [FUNCTION DOC] HELPER: initializes a one rank gloo default group over an in memory store if
# no default group exists yet; otherwise does nothing.
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


class _DummyWork(dist.Work):
    # [CLASS DOC] HELPER: work object returned by the dummy process group. Holds a reference
    # to the group; wait counts calls on the group and every destruction is counted on the
    # group, which lets tests observe work lifetime.
    def __init__(self, pg: "ProcessGroupDummy") -> None:
        # [FUNCTION DOC] HELPER: initializes the base work and stores the owning dummy group.
        super().__init__()
        self.pg = pg

    def wait(self, timeout: timedelta | None = None) -> bool:
        # [FUNCTION DOC] HELPER: increments the wait counter of the owning group and returns
        # true; the timeout argument is ignored.
        self.pg.waits += 1
        return True

    def __del__(self):
        # [FUNCTION DOC] HELPER: increments the owning group's destruction counter.
        self.pg.dels += 1


class ProcessGroupDummy(dist.ProcessGroup):
    # [CLASS DOC] HELPER: single rank process group whose collectives (broadcast,
    # coalesced all_gather, all_reduce, coalesced reduce_scatter) do nothing and
    # return a dummy work object. Attributes: waits and dels count waits and
    # destructions of its work objects; the group name defaults to dummy:dummy. The
    # register method makes it usable as a named backend.
    """
    This process group discards all data passed to it and returns success. This
    is intended for rare cases where we want to discard certain operations
    without modifying the underlying library.

    This PG only supports world_size of 1.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: initializes the base group as rank 0 of size 1, sets the
        # group name and zeroes the wait and destruction counters.
        super().__init__(0, 1)

        self._group_name = "dummy:dummy"

        self.waits = 0
        self.dels = 0

    def broadcast(self, tensor_list: list[torch.Tensor], opts: object) -> dist.Work:
        # [FUNCTION DOC] HELPER: discards the tensors and returns a new dummy work.
        return _DummyWork(self)

    def allgather_into_tensor_coalesced(
        self,
        output_lists: list[torch.Tensor],
        input_list: list[torch.Tensor],
        opts: object,
    ) -> dist.Work:
        # [FUNCTION DOC] HELPER: discards the tensors and returns a new dummy work.
        return _DummyWork(self)

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> dist.Work:
        # [FUNCTION DOC] HELPER: discards the tensors and returns a new dummy work.
        return _DummyWork(self)

    def reduce_scatter_tensor_coalesced(
        self,
        outputTensors: list[torch.Tensor],
        inputTensors: list[torch.Tensor],
        opts: object,
    ) -> dist.Work:
        # [FUNCTION DOC] HELPER: discards the tensors and returns a new dummy work.
        return _DummyWork(self)

    @property
    def group_name(self) -> str:
        # [FUNCTION DOC] HELPER: property returning the group name; raises ValueError if the
        # name is unset.
        if self._group_name is None:
            raise ValueError("ProcessGroup name not set")
        return self._group_name

    def _set_group_name(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: replaces the stored group name.
        self._group_name = name

    def register(self) -> dist.ProcessGroup:
        # [FUNCTION DOC] HELPER: registers a backend named after the group name, whose creator
        # always returns this very object, for cpu devices, then creates and returns a new
        # group over rank 0 with that backend and a 60 second timeout (unused). Side effect:
        # global backend registration and a new group.
        def create_pg(
            prefix_store: dist.PrefixStore, rank: int, world_size: int, timeout: float
        ) -> dist.ProcessGroup:
            # [FUNCTION DOC] HELPER: backend creator callback that ignores its arguments and
            # returns the enclosing dummy group.
            return self

        dist.Backend.register_backend(self.group_name, create_pg, devices=["cpu"])

        return dist.new_group(
            ranks=[0],
            backend=self.group_name,
            group_desc=self.group_name,
            timeout=timedelta(seconds=60.0),  # this timeout isn't used
        )


class CrossThreadWaitTest(TestCase):
    # [CLASS DOC] Single process test case for wait_tensor finding work that was registered
    # from a different thread.
    """
    Test that wait_tensor can find work registered on a different thread.
    This validates the cross-thread work registry lookup in wait_tensor
    that handles the case where wait() is called from a different thread
    than where the collective was initiated.
    """

    def test_wait_tensor_cross_thread(self) -> None:
        # [FUNCTION DOC] Test: registers a custom work for a tensor in the main
        # thread, checks the registry size is 1, calls wait_tensor from another
        # thread, re-raises any exception raised there, and verifies the work wait
        # ran and the registry is empty again.
        """
        Test that wait_tensor works when called from a different thread
        than where the collective was registered.
        """
        wait_called = False
        exception_in_thread: BaseException | None = None

        class MyWork(dist.Work):
            # [CLASS DOC] HELPER: work subclass whose wait sets the enclosing flag
            # and returns true.
            def wait(self, _=None):
                # [FUNCTION DOC] HELPER: marks the enclosing flag as set and returns true.
                nonlocal wait_called
                wait_called = True
                return True

        # Create tensor and register work in main thread
        tensor = torch.rand(2, 2)
        work = MyWork()
        torch._C._distributed_c10d._register_work(tensor, work)

        # Verify work is registered
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)

        # Wait for the tensor in a different thread
        def wait_in_different_thread():
            # [FUNCTION DOC] HELPER: thread target; calls wait_tensor on the tensor and stores
            # any exception in the enclosing variable.
            nonlocal exception_in_thread
            try:
                # This should find the work registered in the main thread
                torch.ops._c10d_functional.wait_tensor(tensor)
            except Exception as e:
                exception_in_thread = e

        thread = threading.Thread(target=wait_in_different_thread)
        thread.start()
        thread.join()

        # Check no exception occurred
        if exception_in_thread is not None:
            raise exception_in_thread

        # Verify wait was called
        self.assertTrue(wait_called)

        # Verify work was removed from registry
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)


class PyWorkTest(TestCase):
    # [CLASS DOC] Single process test case checking Python subclasses of work and process
    # groups stay alive while native functional collectives hold them.
    """
    Native functional collectives have some interesting interactions with
    PyProcessGroup due to Python reference counting and pybind trampoline
    classes with C++ types. This validates that PyProcessGroup and PyWork
    aren't getting prematurely freed.
    """

    def test_wait_tensor(self) -> None:
        # [FUNCTION DOC] Test: registers a Python work subclass for a tensor, deletes the
        # Python reference and forces garbage collection, then calls wait_tensor and expects
        # the work wait to run (it would hang or fail if the work were freed).
        wait_called = False

        class MyWork(dist.Work):
            # [CLASS DOC] HELPER: work subclass that records that its wait was called.
            def wait(self, _):
                # [FUNCTION DOC] HELPER: sets the enclosing flag; returns nothing.
                nonlocal wait_called
                wait_called = True

        # check registration and implicit unregistration

        tensor = torch.rand(2, 2)
        work = MyWork()
        torch._C._distributed_c10d._register_work(tensor, work)

        # Force GC collection of the MyWork object, if we're not doing correct
        # reference counting we'll deadlock in wait_tensor.
        del work
        gc.collect()

        torch.ops._c10d_functional.wait_tensor(tensor)
        self.assertTrue(wait_called)

    def test_collectives(self) -> None:
        # [FUNCTION DOC] Test: runs functional all_reduce, broadcast, all_gather_single and
        # reduce_scatter_single on the dummy group. After each, garbage collection must not
        # destroy the work (destruction count unchanged); calling wait on the result
        # increments the wait count and destroys the work exactly once.
        dummy_init_pg()

        pg = ProcessGroupDummy().register()

        x = torch.rand(2, 2)
        x = funcol.all_reduce(x, "sum", group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 0)
        x.wait()
        self.assertEqual(pg.waits, 1)
        self.assertEqual(pg.dels, 1)

        x = torch.rand(2, 2)
        x = funcol.broadcast(x, 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 1)
        x.wait()
        self.assertEqual(pg.waits, 2)
        self.assertEqual(pg.dels, 2)

        x = torch.rand(2, 2)
        x = funcol.all_gather_single(x, 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 2)
        x.wait()
        self.assertEqual(pg.waits, 3)
        self.assertEqual(pg.dels, 3)

        x = torch.rand(2, 2)
        x = funcol.reduce_scatter_single(x, "sum", 0, group=pg)
        gc.collect()
        self.assertEqual(pg.dels, 3)
        x.wait()
        self.assertEqual(pg.waits, 4)
        self.assertEqual(pg.dels, 4)


class ProcessGroupArgTest(TestCase):
    # [CLASS DOC] Single process test case checking _c10d_functional ops accept a process
    # group object directly in place of the group name string, using the dummy group.
    """
    Test that _c10d_functional ops accept ProcessGroup objects directly
    (not just string group names) via the Any-typed group argument.
    """

    def setUp(self):
        # [FUNCTION DOC] Test fixture: calls parent setUp, ensures a default group exists and
        # registers a dummy group stored as self.pg.
        super().setUp()
        dummy_init_pg()
        self.pg = ProcessGroupDummy().register()

    def test_all_reduce(self) -> None:
        # [FUNCTION DOC] Test: the all_reduce op accepts a process group object; result is
        # waited on.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.all_reduce(x, "sum", self.pg)
        torch.ops._c10d_functional.wait_tensor(out)

    def test_all_reduce_(self) -> None:
        # [FUNCTION DOC] Test: the in place all_reduce op accepts a process group object;
        # result is waited on.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.all_reduce_(x, "sum", self.pg)
        torch.ops._c10d_functional.wait_tensor(out)

    def test_all_gather_into_tensor(self) -> None:
        # [FUNCTION DOC] Test: all_gather_into_tensor accepts a process group object with
        # group size 1.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.all_gather_into_tensor(x, 1, self.pg)
        torch.ops._c10d_functional.wait_tensor(out)

    def test_reduce_scatter_tensor(self) -> None:
        # [FUNCTION DOC] Test: reduce_scatter_tensor accepts a process group object with group
        # size 1.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.reduce_scatter_tensor(x, "sum", 1, self.pg)
        torch.ops._c10d_functional.wait_tensor(out)

    def test_broadcast(self) -> None:
        # [FUNCTION DOC] Test: broadcast accepts a process group object.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.broadcast(x, 0, self.pg)
        torch.ops._c10d_functional.wait_tensor(out)

    def test_broadcast_(self) -> None:
        # [FUNCTION DOC] Test: the in place broadcast accepts a process group object.
        x = torch.rand(2, 2)
        out = torch.ops._c10d_functional.broadcast_(x, 0, self.pg)
        torch.ops._c10d_functional.wait_tensor(out)


class ProcessGroupOpaqueTypeRegistrationTest(TestCase):
    # [CLASS DOC] Single process test case for the opaque type registration of ProcessGroup.
    def test_process_group_is_registered_on_distributed_import(self) -> None:
        # [FUNCTION DOC] Test: importing the distributed package registers ProcessGroup as an
        # opaque custom class whose size member uses the real object, and schema inference for
        # a function with a ProcessGroup argument yields the fully qualified class name in the
        # schema string.
        from torch._library.opaque_object import (
            get_member_type,
            is_custom_class,
            MemberType,
        )

        self.assertTrue(is_custom_class(dist.ProcessGroup))
        self.assertEqual(
            get_member_type(dist.ProcessGroup, "size"), MemberType.USE_REAL
        )

        def f(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: identity function on the tensor, only used for
            # schema inference.
            return x

        self.assertEqual(
            torch.library.infer_schema(f, mutates_args=()),
            "(Tensor x, torch.distributed.distributed_c10d.ProcessGroup pg) -> Tensor",
        )


def find_buffer_assignments(code):
# [FUNCTION DOC] HELPER: scans generated code text for assignments of the form buf number =
# empty_strided_ and returns a tuple of the buffer names in order of appearance.
    pattern = r"buf(\d+) = empty_strided_"
    matches = re.finditer(pattern, code)
    return tuple(f"buf{match.group(1)}" for match in matches)


class CompileTestCPU(TestCase):
    # [CLASS DOC] CPU inductor test case on a fake process group of world size 2.
    def setUp(self):
        # [FUNCTION DOC] Test fixture: calls parent setUp; if no default group is initialized,
        # sets rank 0 and world size 2 on self and initializes a fake backend group over a
        # fake store.
        super().setUp()

        if not dist.is_initialized():
            self.rank = 0
            self.world_size = 2

            store = FakeStore()
            dist.init_process_group(
                backend="fake",
                world_size=self.world_size,
                rank=self.rank,
                store=store,
            )

    def tearDown(self):
        # [FUNCTION DOC] Test fixture: destroys the default process group without checking it
        # is initialized.
        # KNOWN ISSUE: setUp sets rank and world size on self only inside the not-
        # initialized branch.
        dist.destroy_process_group()

    @fresh_cache()
    def _test_inductor_all_reduce_cpu(self, cpp_wrapper=False):
        # [FUNCTION DOC] HELPER: compiles a function with all_reduce and wait_tensor on cpu,
        # optionally with the cpp wrapper enabled by config, and asserts the generated code
        # mentions the expected in place all_reduce and wait_tensor calls (C shim names for
        # the cpp wrapper, op names otherwise). Then runs it through the AOT Inductor runner
        # and synchronizes.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: adds 42, all_reduces with average over group
            # name 0 and waits.
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0

        arg = torch.rand(4, 4, device="cpu")
        with torch._inductor.config.patch({"cpp_wrapper": cpp_wrapper}):
            compiled = torch.compile(func)

            _, (code,) = run_and_get_code(compiled, arg)
            include_ops = (
                [
                    "aoti_torch_cpu__c10d_functional_all_reduce_",
                    "aoti_torch_cpu__c10d_functional_wait_tensor",
                ]
                if cpp_wrapper
                else [
                    "torch.ops._c10d_functional.all_reduce_.default",
                    "torch.ops._c10d_functional.wait_tensor.default",
                ]
            )
            for op in include_ops:
                self.assertIn(op, code)

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.cpu.synchronize()

    def test_inductor_all_reduce_cpu(self):
        # [FUNCTION DOC] Test: runs the cpu inductor all_reduce check without and
        # with the cpp wrapper.
        self._test_inductor_all_reduce_cpu(cpp_wrapper=False)
        self._test_inductor_all_reduce_cpu(cpp_wrapper=True)


class CompileTest(TestCase):
    # [CLASS DOC] Inductor tests on the current accelerator with a fake process group of world
    # size 2; most tests compile a function, check the generated code with pattern matching
    # and run it through the AOT Inductor runner.
    def setUp(self):
        # [FUNCTION DOC] Test fixture: sets rank 0 and world size 2, selects accelerator index
        # 0 as the device and initializes a fake backend group if none is initialized.
        super().setUp()

        self.rank = 0
        self.world_size = 2
        torch.accelerator.set_device_index(0)
        self.device = torch.accelerator.current_accelerator()

        if not dist.is_initialized():
            store = FakeStore()
            dist.init_process_group(
                backend="fake",
                world_size=self.world_size,
                rank=self.rank,
                store=store,
            )

    def tearDown(self):
        # [FUNCTION DOC] Test fixture: destroys the default process group if initialized.
        if dist.is_initialized():
            dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.enable_simple_overlap": False}
    )
    def test_inductor_all_reduce_single(self):
        # [FUNCTION DOC] Test: compiled all_reduce on an inductor allocated buffer must use
        # the in place variant on that buffer, while an all_reduce of a graph input uses a
        # clone; wait_tensor results must be unused and buffers returned without copies. Also
        # checks the cpp wrapper deletes four tensor objects and does not call the non in
        # place op, and runs AOT Inductor. Overlap optimization is disabled.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: two waited all_reduce ops, one on an intermediate buffer
            # and one on the graph input; returns both.
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce(arg, "avg", "0")
            ar1 = funcol.wait_tensor(ar1)
            return ar0, ar1

        arg = torch.rand(4, 4, device=self.device)
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"{buf1} = empty")
            # Expect in-place with inductor allocated buf
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            # Expect no in-place with graph input
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            # Expect no extra copy on return
            .check(f"return ({buf0}, {buf1}, )")
            .run(code)
        )
        # Check the return tensor from wait_tensor is not used anywhere
        if "= torch.ops._c10d_functional.wait_tensor.default" in code:
            raise AssertionError(
                "Expected wait_tensor return value to not be used in code"
            )

        with torch._inductor.config.patch({"cpp_wrapper": True}):
            code = run_and_get_triton_code(compiled, arg)
            # Check the return tensors from all_reduce and wait_tensor are not used anywhere by
            # checking if they are explicitly deleted by calling aoti_torch_delete_tensor_object
            FileCheck().check_not(
                # all_reduce must have been rewritten into all_reduce_
                "aoti_torch_cpu__c10d_functional_all_reduce(buf"
            ).check_count("aoti_torch_delete_tensor_object(buf", 4).run(code)

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_coalesced(self):
        # [FUNCTION DOC] Test: coalesced version of the previous test, with two tensors: in
        # place on buffers allocated by inductor, clones for graph inputs, no used wait_tensor
        # results, and AOT Inductor run.
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: coalesced all_reduce of derived buffers and of the raw
            # inputs; returns both lists.
            bufs = [arg + 42 for arg in args]
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce_coalesced(bufs, "avg", "0")
            ar0 = [funcol.wait_tensor(out) for out in ar0]
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce_coalesced(args, "avg", "0")
            ar1 = [funcol.wait_tensor(out) for out in ar1]
            return ar0, ar1

        args = [torch.rand(4, 4, device=self.device.type) for _ in range(2)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        buf0, buf1, buf2, buf3 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"{buf1} = empty")
            .check(f"{buf2} = empty")
            .check(f"{buf3} = empty")
            # Expect in-place with inductor allocated buf
            .check(
                f"torch.ops._c10d_functional.all_reduce_coalesced_.default([{buf0}, {buf2}]"
            )
            # Expect no in-place with graph input ({buf1}, {buf3} are clones)
            .check(
                f"torch.ops._c10d_functional.all_reduce_coalesced_.default([{buf1}, {buf3}]"
            )
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf2}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf3}")
            # Expect no extra copy on return
            .check(f"return ({buf0}, {buf2}, {buf1}, {buf3}, )")
            .run(code)
        )
        if "= torch.ops._c10d_functional.wait_tensor.default" in code:
            raise AssertionError(
                "Expected wait_tensor return value to not be used in code"
            )

        # Test aoti
        out = AOTIRunnerUtil.run(func, (args,))  # noqa: F841
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_inplace_op_on_view(self):
        # [FUNCTION DOC] Test: all_reduce of a view of an intermediate gets a contiguous
        # buffer, so the in place op acts on a fresh buffer that is waited on and returned.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: takes the first two rows of the input plus 10,
            # all_reduces and waits.
            buf0 = (arg + 10)[:2]
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            return ar0

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        (buf0,) = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            # We always call .contiguous() on the input to all_reduce_,
            # so input will not be a view anymore.
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            .check(f"return ({buf0}")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_non_contig_input(self):
        # [FUNCTION DOC] Test: compiles all_reduce on a transposed (non contiguous) input and
        # checks wait_tensor appears in the generated code. Then repeats the same compile of
        # the first function.
        # KNOWN ISSUE: the second function is defined but never compiled; the second half re-
        # compiles the first function, so the in place variant is not tested.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: all_reduces the argument and waits.
            ar0 = funcol.all_reduce(arg, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect allocation
            return ar0

        arg = torch.rand(4, 4, device=self.device.type).T
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        # clone induced by non contig input
        if "torch.ops._c10d_functional.wait_tensor.default" not in code:
            raise AssertionError("Expected wait_tensor.default in code")

        def func2(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: in place all_reduce of the argument, returns it.
            torch.ops._c10d_functional.all_reduce_(arg, "avg", "0")
            return arg

        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        # clone induced by non contig input
        if "torch.ops._c10d_functional.wait_tensor.default" not in code:
            raise AssertionError("Expected wait_tensor.default in code")

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(
        torch._inductor.config.triton.native_matmul, "no extern_kernels.mm"
    )
    @fresh_cache()
    def test_inductor_reuse_buffer_after_inplace_collective(self):
        # [FUNCTION DOC] Test: after an in place all_reduce, inductor must reuse the freed
        # buffer for a later matmul output. Checks allocation, the in place op, the reuse
        # statement and the returned buffers, and that wait results are unused. Skipped when
        # native matmul is enabled.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: all_reduce an intermediate, then two chained matrix
            # multiplications; returns the two products.
            # Expect allocation
            buf0 = arg + 42
            ar0 = funcol.all_reduce(buf0, "avg", "0")
            ar0 = funcol.wait_tensor(ar0)
            # Expect allocation
            buf1 = torch.mm(arg, ar0)
            # Expect buf0 to be reused
            buf2 = torch.mm(arg, buf1)
            return buf1, buf2

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            # Expect allocation
            .check(f"{buf0} = empty")
            .check(f"torch.ops._c10d_functional.all_reduce_.default({buf0}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf0}")
            # Expect allocation
            .check(f"{buf1} = empty")
            .check(f"extern_kernels.mm(arg0_1, {buf0}, out={buf1}")
            # Expect {buf0} to be reused
            .check(f"buf8 = {buf0}; del {buf0}  # reuse")
            .check(f"extern_kernels.mm(arg0_1, {buf1}, out=buf8")
            # Expect no extra copy on return
            .check(f"return ({buf1}, buf8, )")
            .run(code)
        )
        if "= torch.ops._c10d_functional.wait_tensor.default" in code:
            raise AssertionError(
                "Expected wait_tensor return value to not be used in code"
            )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_gather_into_tensor_single(self):
        # [FUNCTION DOC] Test: compiled all_gather_single emits the non in place
        # all_gather_into_tensor op feeding wait_tensor with the result returned directly;
        # also runs AOT Inductor.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: all_gather_single on dim 0 of the argument then wait.
            ag0 = funcol.all_gather_single(arg, 0, "0")
            ag0 = funcol.wait_tensor(ag0)
            return ag0

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(arg0_1"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no extra copy on return
            .check("return (buf0, )")
            .run(code)
        )
        if "= torch.ops._c10d_functional.wait_tensor.default" in code:
            raise AssertionError(
                "Expected wait_tensor return value to not be used in code"
            )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.accelerator.synchronize()

    @unittest.skipIf(
        TEST_WITH_TORCHINDUCTOR or IS_LINUX,
        "https://github.com/pytorch/pytorch/issues/146806",
    )
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_gather_into_tensor_coalesced(self):
        # [FUNCTION DOC] Test: compiled coalesced all_gather of four tensors produces one
        # coalesced op, indexes each output and waits on each; also runs AOT Inductor. Skipped
        # under inductor testing or on Linux (issue 146806).
        # KNOWN ISSUE: the skip condition is true on every Linux machine, so the test is
        # effectively disabled there, which looks broader than the linked issue suggests.
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: coalesced all_gather_single over a list,
            # waiting each output.
            ag0 = funcol.all_gather_single_coalesced(args, "0")
            ag0 = [funcol.wait_tensor(out) for out in ag0]
            return ag0

        args = [torch.rand(4, 4, device=self.device.type) for _ in range(4)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.all_gather_into_tensor_coalesced"
                ".default([arg0_1, arg1_1, arg2_1, arg3_1]"
            )
            .check("buf1 = buf0[0]")
            .check("buf2 = buf0[1]")
            .check("buf3 = buf0[2]")
            .check("buf4 = buf0[3]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf4")
            # Expect no extra copy on return
            .check("return (buf1, buf2, buf3, buf4, )")
            .run(code)
        )

        # Test aoti
        out = AOTIRunnerUtil.run(func, (args,))  # noqa: F841
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_reduce_output_stride_matches_eager(self):
        # [FUNCTION DOC] Test: strides of compiled all_reduce and wait_tensor outputs on a
        # transposed input match those from eager execution.
        def func(x):
            # [FUNCTION DOC] HELPER: transposes the input, calls the raw all_reduce op and the
            # raw wait_tensor op, and returns both outputs.
            x = x.t()
            y = torch.ops._c10d_functional.all_reduce.default(x, "sum", "0")
            z = torch.ops._c10d_functional.wait_tensor.default(y)
            return y, z

        arg = torch.rand(16, 8, device=self.device)

        eager_y, eager_z = func(arg)

        compiled_f = torch.compile(func)
        compiled_y, compiled_z = compiled_f(arg)

        self.assertEqual(compiled_y.stride(), eager_y.stride())
        self.assertEqual(compiled_z.stride(), eager_z.stride())

    @unittest.skipIf(not HAS_GPU, "This is a GPU test!")
    @fresh_cache()
    def test_wait_tensor(self):
        # [FUNCTION DOC] Test: a compiled function using the raw all_reduce op and
        # funcol wait_tensor calls wait_tensor on the first buffer and returns it;
        # also runs AOT Inductor.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: raw all_reduce op then wait_tensor.
            t = torch.ops._c10d_functional.all_reduce(arg, "avg", "0")
            return funcol.wait_tensor(t)

        # Test aoti
        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_reduce_scatter_tensor_single(self):
        # [FUNCTION DOC] Test: compiled reduce_scatter_single emits the reduce_scatter_tensor
        # op feeding wait_tensor with the result returned directly; also runs AOT Inductor.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: reduce_scatter_single with average on dim 0, then wait.
            rs0 = funcol.reduce_scatter_single(arg, "avg", 0, "0")
            rs0 = funcol.wait_tensor(rs0)
            return rs0

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, arg)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.reduce_scatter_tensor.default(arg0_1"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(buf0")
            # Expect no extra copy on return
            .check("return (buf0, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_reduce_scatter_tensor_coalesced(self):
        # [FUNCTION DOC] Test: compiled coalesced reduce_scatter of four tensors
        # produces one coalesced op, indexes each output, waits each and returns
        # them; also runs AOT Inductor.
        def func(args: list[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: coalesced reduce_scatter_single with average over a list,
            # waiting each output.
            rs0 = funcol.reduce_scatter_single_coalesced(
                args, "avg", [0] * len(args), "0"
            )
            rs0 = [funcol.wait_tensor(out) for out in rs0]
            return rs0

        args = [torch.rand(4, 4, device=self.device.type) for _ in range(4)]
        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, args)
        (
            FileCheck()
            .check(
                "buf0 = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced"
                ".default([arg0_1, arg1_1, arg2_1, arg3_1]"
            )
            .check("buf1 = buf0[0]")
            .check("buf2 = buf0[1]")
            .check("buf3 = buf0[2]")
            .check("buf4 = buf0[3]")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf4")
            # Expect no extra copy on return
            .check("return (buf1, buf2, buf3, buf4, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (args,))
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_inductor_all_to_all_single(self):
        # [FUNCTION DOC] Test: compiled all_to_all_single with dynamic shapes where split
        # sizes come from tensors converted to lists (dynamic shape config patched). Checks
        # the generated code passes symbolic split size lists to the op and calls wait_tensor.
        def func(
            input: torch.Tensor,
            output_split_sizes: torch.Tensor,
            input_split_sizes: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: all_to_all_single using split size lists from the
            # tensors, then wait.
            output = funcol.all_to_all_single(
                input,
                output_split_sizes.tolist(),
                input_split_sizes.tolist(),
                "0",
            )
            return funcol.wait_tensor(output)

        torch.manual_seed(42)
        send_sz_matrix = torch.randint(0, 20, (self.world_size, self.world_size))

        input_split_sizes = send_sz_matrix[self.rank]
        output_split_sizes = send_sz_matrix[:, self.rank].contiguous()
        input = torch.full((input_split_sizes.sum().item(),), float(self.rank)).to(
            self.device.type
        )

        with torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            compiled = torch.compile(func, dynamic=True)
            code = run_and_get_triton_code(
                compiled, input, output_split_sizes, input_split_sizes
            )
        (
            FileCheck()
            .check_regex(
                "torch.ops._c10d_functional.all_to_all_single.default\\("
                "arg\\d+_\\d+, \\[u\\d+, u\\d+\\], \\[u\\d+, u\\d+\\]"
            )
            .check("torch.ops._c10d_functional.wait_tensor.default(")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.enable_simple_overlap": False}
    )
    def test_inductor_broadcast(self):
        # [FUNCTION DOC] Test: compiled broadcast on an inductor buffer uses the in place
        # broadcast op and, for a graph input, a clone; checks wait_tensor and return of
        # buffers without copy; runs AOT Inductor. Overlap optimization is disabled.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: broadcast of an intermediate from rank 1 and of the input
            # from rank 0, each waited; returns both.
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            br0 = funcol.broadcast(buf0, 1, "0")
            br0 = funcol.wait_tensor(br0)
            # Expect no in-place with graph input
            br1 = funcol.broadcast(arg, 0, "0")
            br1 = funcol.wait_tensor(br1)
            return br0, br1

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func)

        code = run_and_get_triton_code(compiled, arg)
        buf0, buf1 = find_buffer_assignments(code)
        (
            FileCheck()
            .check(f"{buf0} = empty")
            .check(f"buf1 = {buf0}")
            .check(f"{buf1} = empty")
            # Expect in-place with inductor allocated buf
            .check("torch.ops._c10d_functional.broadcast_.default(buf1")
            .check("torch.ops._c10d_functional.wait_tensor.default(buf1")
            # Expect no in-place with graph input (buf5 is a clone)
            .check(f"torch.ops._c10d_functional.broadcast_.default({buf1}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf1}")
            # Expect no extra copy on return
            .check(f"return (buf1, {buf1}, )")
            .run(code)
        )

        # Test aoti
        AOTIRunnerUtil.run(func, (arg,))
        torch.accelerator.synchronize()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_ranks_and_tag(self):
        # [FUNCTION DOC] Test: all_reduce given an explicit ranks list and empty tag is
        # compiled with fullgraph and the generated code uses the in place all_reduce with
        # group name 0.
        def func(arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: two waited all_reduce ops over ranks 0 and 1, one on an
            # intermediate and one on the input.
            buf0 = arg + 42
            # Expect in-place with inductor allocated buf
            ar0 = funcol.all_reduce(buf0, "avg", [0, 1], "")
            ar0 = funcol.wait_tensor(ar0)
            # Expect no in-place with graph input
            ar1 = funcol.all_reduce(arg, "avg", [0, 1], "")
            ar1 = funcol.wait_tensor(ar1)
            return ar0, ar1

        arg = torch.rand(4, 4, device=self.device.type)
        compiled = torch.compile(func, fullgraph=True)

        code = run_and_get_triton_code(compiled, arg)
        (FileCheck().check("all_reduce_.default(buf0, 'avg', '0')").run(code))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_compile_all_gather_into_tensor_out(self):
        # [FUNCTION DOC] Test: compiling all_gather_into_tensor_out with fullgraph gives an
        # output of world size times the input length and generated code that allocates a
        # buffer, passes it as out and waits on it.
        def func(input, out):
            # [FUNCTION DOC] HELPER: all_gather_into_tensor_out into the given out tensor,
            # wait, then add one.
            y = torch.ops._c10d_functional.all_gather_into_tensor_out(
                input, self.world_size, "0", out=out
            )
            y = torch.ops._c10d_functional.wait_tensor(y)
            return y + 1

        input_t = torch.ones(4, device=self.device)
        out_t = torch.empty(4 * self.world_size, device=self.device)
        compiled = torch.compile(func, fullgraph=True)
        result = compiled(input_t, out_t)
        self.assertEqual(result.shape, (4 * self.world_size,))

        code = run_and_get_triton_code(compiled, input_t, out_t)
        buf = find_buffer_assignments(code)[0]
        (
            FileCheck()
            .check(f"{buf} = empty")
            .check("torch.ops._c10d_functional.all_gather_into_tensor_out.default(")
            .check_same(f"out={buf}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf})")
            .run(code)
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_compile_reduce_scatter_tensor_out(self):
        # [FUNCTION DOC] Test: compiling reduce_scatter_tensor_out with fullgraph gives an
        # output of the input length divided by world size and generated code that allocates a
        # buffer, passes it as out and waits on it.
        def func(input, out):
            # [FUNCTION DOC] HELPER: reduce_scatter_tensor_out into the given out tensor,
            # wait, then add one.
            y = torch.ops._c10d_functional.reduce_scatter_tensor_out(
                input, "sum", self.world_size, "0", out=out
            )
            y = torch.ops._c10d_functional.wait_tensor(y)
            return y + 1

        input_t = torch.ones(4 * self.world_size, device=self.device)
        out_t = torch.empty(4, device=self.device)
        compiled = torch.compile(func, fullgraph=True)
        result = compiled(input_t, out_t)
        self.assertEqual(result.shape, (4,))

        code = run_and_get_triton_code(compiled, input_t, out_t)
        buf = find_buffer_assignments(code)[0]
        (
            FileCheck()
            .check(f"{buf} = empty")
            .check("torch.ops._c10d_functional.reduce_scatter_tensor_out.default(")
            .check_same(f"out={buf}")
            .check(f"torch.ops._c10d_functional.wait_tensor.default({buf})")
            .run(code)
        )


class ACTCompileTest(TestCase):
    # [CLASS DOC] Single process test case for AsyncCollectiveTensor inputs to compiled
    # regions and direct AOT autograd functions.
    """
    Test that AsyncCollectiveTensor inputs to compiled regions are resolved
    before AOT autograd tracing begins. ACTs are transient eager-mode wrappers
    for async collective overlap; if they leak into the traced graph as input
    types, AOT autograd records them in tangent metadata and then hits a type
    mismatch at runtime because autograd produces plain-tensor tangents.
    """

    def test_act_compile_backward_tangent_mismatch(self):
        # [FUNCTION DOC] Test: an AsyncCollectiveTensor passed to a function compiled with
        # aot_eager and a view op must be waited for (trigger_wait is patched to record the
        # call), backward must not fail with a tangent type mismatch, and the output equals
        # the unsqueezed underlying data.
        """
        When a bare AsyncCollectiveTensor (ACT) enters a torch.compiled
        region and passes through a view op, the output remains an ACT.
        If the ACT is not unwrapped before tracing, AOT autograd
        fakifies it as-is and records it in SubclassCreationMeta. At runtime, autograd
        produces a plain tensor tangent, causing:
            RuntimeError: Expected a AsyncCollectiveTensor tangent
            but got a plain Tensor.

        This occurs in practice when TP async collectives produce a
        DTensor(ACT), an eager caller does to_local() to get a bare ACT,
        and that ACT flows into a compiled sub-module whose forward
        contains view ops (unsqueeze, reshape, transpose, etc.).

        We use unsqueeze (a view op that preserves ACT) to ensure the
        compiled function's output carries ACT type, which is what
        triggers the tangent metadata recording.
        """
        from unittest.mock import patch

        elem = torch.randn(4, 4, requires_grad=True)
        act = AsyncCollectiveTensor(elem)

        compiled_fn = torch.compile(lambda x: x.unsqueeze(0), backend="aot_eager")

        original_trigger_wait = AsyncCollectiveTensor.trigger_wait
        wait_called = False

        def tracked_trigger_wait(self):
            # [FUNCTION DOC] HELPER: replacement for trigger_wait that sets the enclosing flag
            # and calls the original implementation, returning its result.
            nonlocal wait_called
            wait_called = True
            return original_trigger_wait(self)

        with patch.object(AsyncCollectiveTensor, "trigger_wait", tracked_trigger_wait):
            out = compiled_fn(act)
            # Without ACT unwrapping, this raises:
            #   RuntimeError: Expected a AsyncCollectiveTensor tangent
            #   but got a plain Tensor.
            out.sum().backward()

        self.assertTrue(wait_called, "trigger_wait() was never called")

        # Verify numerics: trigger_wait() must fire so the correct data
        # flows through. ACT wraps elem, so results must match elem directly.
        ref = elem.detach().unsqueeze(0)
        self.assertEqual(out.detach(), ref)

    def test_act_runtime_unwrap(self):
        # [FUNCTION DOC] Test: a custom AOT autograd backend asserts none of the
        # runtime arguments are AsyncCollectiveTensor, for both the first
        # (compiling) call and a second call that reuses the cached graph; results
        # must equal twice the underlying data.
        """
        Verify that ACT inputs are resolved before the compiled graph
        executes.

        With inductor, triton kernels bypass __torch_dispatch__ entirely,
        so an un-waited ACT would silently feed stale data to the kernel.
        The aot_eager backend masks this because it runs graph ops through
        __torch_dispatch__ (which calls trigger_wait). This test uses a
        custom backend that asserts inputs are plain tensors to catch the
        problem regardless of backend.
        """

        def assert_no_act_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: backend wrapping the graph with aot_autograd, using a
            # forward compiler whose callable checks the arguments.
            from torch._dynamo.backends.common import aot_autograd

            def inner_compiler(gm, example_inputs):
                # [FUNCTION DOC] HELPER: forward compiler returning a boxed callable
                # that fails if any argument is an AsyncCollectiveTensor and
                # otherwise runs the graph.
                def compiled(args):
                    # [FUNCTION DOC] HELPER: boxed callable that checks every argument in the
                    # list is not an AsyncCollectiveTensor (assertion failure message includes
                    # the argument index) and then calls the graph with them.
                    for i, a in enumerate(args):
                        self.assertNotIsInstance(
                            a,
                            AsyncCollectiveTensor,
                            lambda msg: f"{msg}\narg {i} is still an ACT — trigger_wait() "
                            "was not called before the compiled function",
                        )
                    return gm(*args)

                compiled._boxed_call = True
                return compiled

            return aot_autograd(fw_compiler=inner_compiler)(gm, example_inputs)

        compiled_fn = torch.compile(lambda x: x * 2, backend=assert_no_act_backend)

        # First call: triggers compilation.
        elem1 = torch.randn(4, 4)
        act1 = AsyncCollectiveTensor(elem1)
        r1 = compiled_fn(act1)
        self.assertEqual(r1, elem1 * 2)

        # Second call: dynamo reuses cached graph. process_inputs does
        # NOT run again — the runtime wrapper must unwrap ACTs.
        elem2 = torch.randn(4, 4)
        act2 = AsyncCollectiveTensor(elem2)
        r2 = compiled_fn(act2)
        self.assertEqual(r2, elem2 * 2)

    def test_direct_aot_top_level_act_path_accepts_plain_runtime_input(self):
        # [FUNCTION DOC] Test: a function wrapped with aot_function (no dynamo guards) is
        # first called with an AsyncCollectiveTensor and then with a plain tensor; both calls
        # must work and return twice the input.
        """
        Unlike torch.compile, direct AOTAutograd entry points do not have
        Dynamo input-type guards to force a recompile when a top-level ACT input
        is later a plain tensor. The runtime wait path must therefore guard
        before calling trigger_wait() on the recorded ACT input path.
        """
        from torch._functorch.aot_autograd import aot_function

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: forward compiler that returns the graph module unchanged.
            return gm

        compiled_fn = aot_function(lambda x: x * 2, fw_compiler=fw_compiler)

        elem = torch.randn(4, 4)
        r1 = compiled_fn(AsyncCollectiveTensor(elem))
        self.assertEqual(r1, elem * 2)

        plain = torch.randn(4, 4)
        r2 = compiled_fn(plain)
        self.assertEqual(r2, plain * 2)

    def test_act_guard_recompiles(self):
        # [FUNCTION DOC] Test: with a compile counter backend, switching an input between
        # plain tensor and AsyncCollectiveTensor forces a second compilation; returning to a
        # plain tensor reuses the first graph, so frame count goes 1, 2, 2.
        """
        Dynamo must recompile when an input switches between plain tensor
        and AsyncCollectiveTensor (or vice versa).
        """
        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(lambda x: x * 2, backend=cnt)

        elem = torch.randn(4, 4)

        # Call 1: plain tensor — triggers first compilation.
        r1 = compiled_fn(elem)
        self.assertEqual(r1, elem * 2)
        self.assertEqual(cnt.frame_count, 1)

        # Call 2: ACT — should trigger a recompile because the input
        # type changed from Tensor to AsyncCollectiveTensor.
        act = AsyncCollectiveTensor(elem)
        r2 = compiled_fn(act)
        self.assertEqual(r2, elem * 2)
        self.assertEqual(cnt.frame_count, 2)

        # Call 3: plain tensor again — should reuse the first compiled
        # graph, no new compilation.
        r3 = compiled_fn(elem)
        self.assertEqual(r3, elem * 2)
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    run_tests()
