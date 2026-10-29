# mypy: allow-untyped-defs

# [MODULE DOC]
# Test mixins (not unittest cases by themselves) for TorchScript integration with RPC: RRef
# APIs and typing in script, Future typing, local RRefs of script classes and modules, rpc_async,
# rpc_sync and remote called from script, async_execution, callbacks, and profiling /
# record_function with RPC. The mixins are combined with a backend fixture in JitRpcTest and
# the classes preceding it. Many helpers are scripted functions or classes compiled at import time.

import io
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.autograd.profiler_legacy import profile as _profile
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import _build_rpc_profiling_key, RPCExecMode
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def rref_isinstance(rref, cls_to_check):
    # [FUNCTION DOC] HELPER: Returns whether the owner-local value of the RRef is an instance of
    # the given class (only valid on the owner).
    return isinstance(rref.local_value(), cls_to_check)


def sleep(t):
    # [FUNCTION DOC] HELPER: RPC target that blocks the calling thread for t seconds.
    time.sleep(t)


def rpc_return_rref(dst):
    # [FUNCTION DOC] HELPER: Creates and returns an RRef owned by worker dst holding a 2x2 ones
    # tensor plus one (a remote tensor add).
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


@torch.jit.script
def rref_local_value(rref: RRef[Tensor]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function returning the owner-local value of the RRef;
    # raises if called on a non-owner.
    return rref.local_value()


@torch.jit.script
def list_create() -> list[int]:
    # [FUNCTION DOC] HELPER: Scripted function returning a new list of the integers one to three.
    global_list = [1, 2, 3]
    return global_list


@torch.jit.script
def rref_list_mutate(rref: RRef[list[int]]) -> None:
    # [FUNCTION DOC] HELPER: Scripted function run on the owner that appends 4, 5 and 6 to the
    # RRef's list through local_value, to_here and to_here with a timeout, showing all three
    # access the same list object. Mutates the owner's list; returns nothing.
    rref.local_value().append(4)
    rref.to_here().append(5)
    rref.to_here(5.0).append(6)


def return_value(value: int) -> int:
    # [FUNCTION DOC] HELPER: Identity function on an integer, used as an RPC target.
    return value


class RRefAPITest:
    # [CLASS DOC]
    # Mixin of tests for RRef methods used from TorchScript (is_owner, local_value, to_here,
    # confirmed-user checks, list mutation). Uses self.rank, self.world_size and assertion
    # helpers from the combined test class; stateless.
    @dist_init
    def test_rref_is_owner(self):
        # [FUNCTION DOC]
        # Creates an RRef owned by the next worker and checks that the scripted is_owner call
        # returns false on the creator.
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref_var = rpc_return_rref(dst_worker_name)

        @torch.jit.script
        def rref_tensor_is_owner(rref_var: RRef[Tensor]) -> bool:
            # [FUNCTION DOC] HELPER: Scripted wrapper returning whether this worker owns the RRef.
            return rref_var.is_owner()

        res = rref_tensor_is_owner(rref_var)
        self.assertEqual(res, False)

    @dist_init
    def test_rref_local_value(self):
        # [FUNCTION DOC]
        # Rank 0 only. Calling the scripted local_value on a non-owner must raise an error saying
        # it cannot be called on a non-owner RRef; running it on the owner via RPC returns the
        # tensor value.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref = rpc_return_rref(dst_worker_name)

        with self.assertRaisesRegex(
            RuntimeError, r"Can't call RRef.local_value\(\) on a non-owner RRef"
        ):
            rref_local_value(rref)

        ret = rpc.rpc_sync(dst_worker_name, rref_local_value, (rref,))
        self.assertEqual(ret, torch.add(torch.ones(2, 2), 1))

    @dist_init
    def test_local_rref_local_value(self):
        # [FUNCTION DOC]
        # Rank 0 only. Creates a remote call targeting the local worker (so the worker owns the
        # RRef) and checks the scripted local_value returns the integer five.
        if self.rank != 0:
            return

        dst_worker_name = worker_name(self.rank)
        rref = rpc.remote(dst_worker_name, return_value, (5,), {})

        ret = rref_local_value(rref)
        self.assertEqual(ret, 5)

    def _create_rref(self):
        # [FUNCTION DOC] HELPER: Creates and returns an RRef owned by the worker two ranks ahead
        # (modulo world size), holding a 2x2 zeros tensor plus one.
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote(
            worker_name(owner_rank), torch.add, args=(torch.zeros(2, 2), 1)
        )

    @dist_init
    def test_user_rrefs_confirmed(self):
        # [FUNCTION DOC]
        # Passes an RRef owned elsewhere to the next worker via rpc_sync and checks the scripted
        # confirmation query there returns true.
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync(
            worker_name(dst_rank), script_check_rref_confirmed, args=(rref,)
        )
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        # [FUNCTION DOC]
        # Same as the rpc_sync confirmation test but the query is issued through remote and the
        # resulting RRef is fetched; expects true.
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote(
            worker_name(dst_rank), script_check_rref_confirmed, args=(rref,)
        )
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_list_mutate(self):
        # [FUNCTION DOC]
        # Creates a remote list RRef, has the owner append through three accessors, and checks
        # the fetched list holds one through six.
        dst = worker_name((self.rank + 1) % self.world_size)
        list_rref = rpc.remote(dst, list_create)

        rpc.rpc_sync(dst, rref_list_mutate, args=(list_rref,))
        self.assertEqual(list_rref.to_here(), [1, 2, 3, 4, 5, 6])


@torch.jit.script
def no_arg():
    # [FUNCTION DOC] HELPER: Scripted function with no arguments returning zero.
    return 0


@torch.jit.script
def one_arg(value):
    # [FUNCTION DOC] HELPER: Scripted function returning its argument plus one.
    return value + 1


@torch.jit.script
def script_add_ones(x):
    # [FUNCTION DOC] HELPER: Scripted function adding a one-element ones tensor to x.
    return torch.add(x, torch.ones(1))


@torch.jit.script
def script_add_ones_with_record_function(x, block: str):
    # [FUNCTION DOC] HELPER: Scripted add of ones to x executed inside a record_function scope
    # named by block, so the profiler records an event with that name.
    with record_function(block):
        return torch.add(x, torch.ones(1))


@torch.jit.script
def record_function_on_caller_rpc_async(dst_worker_name: str, block: str) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function that, inside a record_function scope named block,
    # issues two async RPC adds to dst_worker_name and returns the sum of both results.
    t: Tensor = torch.ones(1)
    with record_function(block):
        fut1 = rpc.rpc_async(dst_worker_name, script_add_ones, (t,))
        # Extra operator call to avoid de-duplication of the next async call
        # see https://github.com/pytorch/pytorch/pull/62710#discussion_r694680279
        zero = torch.zeros_like(t)
        fut2 = rpc.rpc_async(dst_worker_name, script_add_ones, (t,))
        res = fut1.wait() + fut2.wait() + zero
    return res


@torch.jit.script
def script_fork_wait_udf(tensor):
    # [FUNCTION DOC] HELPER: Scripted function forking the add-ones function, waiting, and
    # returning the result.
    fut = torch.jit._fork(script_add_ones, tensor)
    x = torch.jit._wait(fut)
    return x


@torch.jit.script
def rref_to_here(rref_var: RRef[Tensor]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function fetching the RRef's tensor value to this worker.
    return rref_var.to_here()


@torch.jit.script
def return_rref(rref_var: RRef[Tensor]) -> RRef[Tensor]:
    # [FUNCTION DOC] HELPER: Scripted identity function on an RRef.
    return rref_var


@torch.jit.script
def script_raise_func(value):
    # [FUNCTION DOC] HELPER: Scripted function that raises a value error when the tensor has
    # exactly two elements, otherwise returns the tensor plus one.
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1


@torch.jit.script
def script_fork_wait_throw(invalue):
    # [FUNCTION DOC] HELPER: Scripted function forking the raising function and waiting, so the
    # error propagates at the wait. Returns the forked result when no error occurs.
    fut = torch.jit._fork(script_raise_func, invalue)
    value = torch.jit._wait(fut)
    return value


@torch.jit.script
def call_rpc_with_profiling(
    record: torch.classes.profiler._RecordFunction, dst_worker_name: str
) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function that issues an async RPC of the plus-one function
    # to dst_worker_name, attaches the end callbacks of the given record-function object to
    # the future, waits and returns the result.
    # Call rpc_async from within ScriptFunction and ensure that we can attach
    # profiling callbacks. Note that handle here is a Tensor representation of
    # RecordFunction.
    fut = rpc.rpc_async(dst_worker_name, one_arg, (torch.tensor(1),))
    torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
    ret = fut.wait()
    return ret


@torch.jit.script
def call_rpc_torchscript_with_record_function(
    dst_worker_name: str, block: str
) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function making an async RPC to a remote scripted function
    # that itself runs inside a record_function scope named block; waits and returns the result.
    fut = rpc.rpc_async(
        dst_worker_name, script_add_ones_with_record_function, (torch.tensor(1), block)
    )
    return fut.wait()


@torch.jit.script
def call_fork_with_profiling(record: torch.classes.profiler._RecordFunction) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function that forks the plus-one function, attaches the
    # record-function end callbacks to the future, waits and returns the result.
    # Call fork from within ScriptFunction and ensure that we can attach profiling
    # callbacks to the resulting future. Note that handle here is a Tensor
    # representation of RecordFunction.
    fut = torch.jit._fork(one_arg, torch.tensor(1))
    torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
    ret = fut.wait()
    return ret


class MyScriptModuleWithRRefs(torch.jit.ScriptModule):
    # [CLASS DOC] HELPER:
    # Script module holding a list of four RRefs, each to a remote 2x2 tensor of twos; forward
    # fetches them and adds them onto a 2x2 ones tensor.
    def __init__(self, dst_worker):
        # [FUNCTION DOC] HELPER: Creates four remote RRefs on dst_worker and stores them in a list.
        super().__init__()
        self.rrefs = []
        for _ in range(4):
            self.rrefs.append(rpc_return_rref(dst_worker))

    @torch.jit.script_method
    def forward(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns 2x2 ones plus the fetched value of every stored RRef.
        res_tensor = torch.ones(2, 2)
        for rref in self.rrefs:
            res_tensor += rref.to_here()

        return res_tensor


@torch.jit.ignore
def rref_python_annotation(rref_var: RRef[Tensor]) -> RRef[Tensor]:
    # [FUNCTION DOC] HELPER: Python-only (ignored by the compiler) identity on an RRef whose
    # signature carries an RRef type annotation.
    return rref_var


@torch.jit.script
def rref_script_annotation(rref_var: RRef[Tensor]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function that passes the RRef through the ignored Python
    # function and fetches its value.
    return rref_python_annotation(rref_var).to_here()


class RRefTypingTest:
    # [CLASS DOC]
    # Mixin of tests that RRef typed arguments and return values work with scripted functions,
    # script modules holding RRefs, and Python annotations. Stateless.
    @dist_init
    def test_rref_as_arg_and_return(self):
        # [FUNCTION DOC]
        # Creates a local RRef, then checks passing it to a scripted function and getting it back,
        # both through rpc_sync and remote, yields values equal to the local plus-one result.
        n = self.rank + 1
        dst_rank = n % self.world_size
        local_ret = one_arg(torch.ones(2, 2))

        # create rref on current rank
        rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2),))

        # pass rref to another user in rpc call
        ret = rpc.rpc_sync(worker_name(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(ret, local_ret)

        # return rref in rpc call
        rref1 = rpc.rpc_sync(worker_name(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref1.to_here(), local_ret)

        # pass rref to another user in remote call
        rref2 = rpc.remote(worker_name(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(rref2.to_here(), local_ret)

        # return rref in remote call
        rref3 = rpc.remote(worker_name(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref3.to_here().to_here(), local_ret)

    @dist_init
    def test_my_script_module_with_rrefs(self):
        # [FUNCTION DOC]
        # Builds the script module with four remote RRefs and checks its forward returns a 2x2
        # tensor filled with nine.
        n = self.rank + 1
        dst_rank = n % self.world_size

        module_with_rrefs = MyScriptModuleWithRRefs(worker_name(dst_rank))
        res = module_with_rrefs()
        self.assertEqual(res, torch.ones(2, 2) * 9)

    @dist_init
    def test_rref_python_annotation(self):
        # [FUNCTION DOC]
        # Checks a scripted function can use an ignored Python function with RRef annotations
        # and fetch the value, expecting a 2x2 tensor of twos.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))

        res = rref_script_annotation(rref_var)
        self.assertEqual(res, torch.ones(2, 2) + 1)


class FutureTypingTest:
    # [CLASS DOC]
    # Mixin of tests for passing RPC futures between Python and TorchScript. Stateless.
    @dist_init
    def test_future_passed_between_python_and_jit(self):
        # [FUNCTION DOC]
        # Checks a Python-created RPC future can be waited on inside a scripted function, and a
        # future created inside script can be returned to Python and waited there; both give
        # the expected sum tensor.
        dst_rank = (self.rank + 1) % self.world_size
        inputs = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        ret_fut = rpc.rpc_async(worker_name(dst_rank), two_args_two_kwargs, args=inputs)
        expected_res = torch.tensor([10, 10])

        @torch.jit.script
        def future_wait_in_script(fut: Future[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted function that waits on the given future.
            return fut.wait()

        self.assertEqual(future_wait_in_script(ret_fut), expected_res)

        @torch.jit.script
        def future_return_to_python(
            dst_rank: int, inputs: tuple[Tensor, Tensor]
        ) -> Future[Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function returning an unwaited RPC future.
            return rpc.rpc_async(f"worker{dst_rank}", two_args_two_kwargs, inputs)

        fut_res = future_return_to_python(dst_rank, inputs)
        self.assertEqual(fut_res.wait(), expected_res)

    @dist_init
    def test_future_python_annotation(self):
        # [FUNCTION DOC]
        # Rank 0 only. Checks a scripted function can call an ignored Python function annotated
        # to return a future, wait on it, and get the expected tensor sum.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        input_0 = torch.ones(2, 2)
        input_1 = 1
        expected_res = torch.add(input_0, input_1)

        @torch.jit.ignore
        def python_return_future() -> Future[Tensor]:
            # [FUNCTION DOC] HELPER: Python-only function starting an async RPC tensor add on the
            # next worker and returning its future.
            fut = rpc.rpc_async(dst_worker_name, torch.add, (input_0, input_1), {})
            return fut

        @torch.jit.script
        def script_use_future() -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted function that calls the Python future provider
            # and waits on the future.
            fut = python_return_future()
            return fut.wait()

        res = script_use_future()
        self.assertEqual(res, expected_res)


@torch.jit.script
class MyScriptClass:
    # [CLASS DOC] HELPER:
    # Scripted (TorchScript) class wrapping one integer attribute a, used as RRef content.
    def __init__(self, a: int):
        # [FUNCTION DOC] HELPER: Stores the integer a.
        self.a = a

    def get_value(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the stored integer.
        return self.a


@torch.jit.interface
class MyModuleInterface(torch.nn.Module):
    # [CLASS DOC] HELPER:
    # TorchScript interface declaring a forward method that returns a tensor and takes no
    # inputs; implemented by MyScriptModule and used as the RRef type hint.
    def forward(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Interface method declaration with no body.
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass


class MyScriptModule(torch.jit.ScriptModule):
    # [CLASS DOC] HELPER:
    # Script module holding a tensor a of ones with length equal to the given rank. It
    # implements the module interface through forward; custom_func is an extra method.
    def __init__(self, rank):
        # [FUNCTION DOC] HELPER: Stores a ones tensor of length rank as attribute a.
        super().__init__()
        self.a = torch.ones(rank)

    @torch.jit.script_method
    def forward(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the stored tensor.
        return self.a

    @torch.jit.script_method
    def custom_func(self) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the stored tensor (extra method beyond the interface).
        return self.a


def owner_create_rref_my_script_class(a):
    # [FUNCTION DOC] HELPER: Run on the owner; wraps a new scripted class instance in a local RRef.
    return rpc.RRef(MyScriptClass(a))


def owner_create_rref_my_script_module(a):
    # [FUNCTION DOC] HELPER: Run on the owner; wraps a new script module in a local RRef with the
    # module interface as type hint (required for script modules).
    return rpc.RRef(MyScriptModule(a), type_hint=MyModuleInterface)


@torch.jit.script
def script_rref_get_value_my_script_class(rref: RRef[MyScriptClass]) -> int:
    # [FUNCTION DOC] HELPER: Scripted function fetching the class instance from the RRef and
    # returning its integer value.
    return rref.to_here().get_value()


@torch.jit.script
def script_rref_run_forward_my_script_module(rref: RRef[MyModuleInterface]) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function fetching the module from the RRef and returning
    # the result of its forward.
    return rref.to_here().forward()


class LocalRRefTest:
    # [CLASS DOC]
    # Mixin of tests for local RRefs that hold scripted classes or script modules, created in
    # Python and used from Python or script, locally or across RPC. Stateless.
    @dist_init
    def test_create_local_script_class_rref_in_py(self):
        # [FUNCTION DOC]
        # Rank 0 only. Creates a local RRef of the scripted class and checks the fetched value
        # equals the rank.
        if self.rank != 0:
            return

        # Create a local RRef<MyScriptClass>.
        rref_script_class = rpc.RRef(MyScriptClass(self.rank))
        ret = rref_script_class.to_here().get_value()
        self.assertEqual(ret, self.rank)

    @dist_init
    def test_create_local_script_module_rref_in_py(self):
        # [FUNCTION DOC]
        # Rank 0 only. Creates a local RRef of a script module with the interface type hint and
        # checks forward gives ones of length rank; creating it without a type hint must raise
        # an error requiring the module interface hint.
        if self.rank != 0:
            return

        # Create a local RRef<MyModuleInterface>.
        rref_script_module = rpc.RRef(MyScriptModule(self.rank), MyModuleInterface)
        ret = rref_script_module.to_here().forward()
        self.assertEqual(ret, torch.ones(self.rank))

        # Create a local RRef<MyModuleInterface> without type hint.
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "The RRef being created contains a ScriptModule, "
                "must provide its ModuleInterface type hint."
            ),
        ):
            rref_script_module = rpc.RRef(MyScriptModule(self.rank))

    @dist_init
    def test_return_local_script_class_rref_in_py_and_use_in_script(self):
        # [FUNCTION DOC]
        # Rank 0 only. Has the next worker create a local RRef of the scripted class, then uses
        # it on the owner through a scripted function called via rpc_async, both from plain
        # Python and from a scripted caller; both must return the rank value.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # Create a local RRef<MyScriptClass> remotely in Python.
        rref = rpc.rpc_sync(
            dst_worker_name, owner_create_rref_my_script_class, args=(self.rank,)
        )

        def use_rref_on_owner(rref: RRef[MyScriptClass]) -> int:
            # [FUNCTION DOC] HELPER: Runs the class value getter on the RRef owner through an
            # async RPC (owner given by worker info) and returns the integer.
            args = (rref,)
            kwargs: dict[str, Any] = {}
            fut = rpc.rpc_async(
                rref.owner(), script_rref_get_value_my_script_class, args, kwargs
            )
            ret = fut.wait()
            return ret

        # Use RRef<MyScriptClass> in local Python RPC and remote Script run.
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, self.rank)

        # Use RRef<MyScriptClass> in local Script RPC and remote Script run.
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, self.rank)

    @dist_init
    def test_return_local_script_module_rref_in_py_and_use_in_script(self):
        # [FUNCTION DOC]
        # Rank 0 only. Like the script class variant, but the owner creates a script module RRef
        # with the interface hint and the scripted forward is run on the owner; the result must
        # be ones of length rank, from both Python and scripted callers.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # Create a local RRef<MyModuleInterface> remotely in Python.
        rref = rpc.rpc_sync(
            dst_worker_name, owner_create_rref_my_script_module, args=(self.rank,)
        )

        def use_rref_on_owner(rref: RRef[MyModuleInterface]) -> Tensor:
            # [FUNCTION DOC] HELPER: Runs the module forward on the RRef owner (addressed by owner
            # name) through an async RPC and returns the tensor.
            args = (rref,)
            kwargs: dict[str, Any] = {}
            fut = rpc.rpc_async(
                rref.owner_name(),
                script_rref_run_forward_my_script_module,
                args,
                kwargs,
            )
            ret = fut.wait()
            return ret

        # Use RRef<MyScriptClass> in local Python RPC and remote Script run.
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, torch.ones(self.rank))

        # Use RRef<MyScriptClass> in local Script RPC and remote Script run.
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, torch.ones(self.rank))


def python_function():
    # [FUNCTION DOC] HELPER: Plain Python (not scripted) function returning zero; used to show
    # that scripts cannot call Python functions remotely.
    return 0


@torch.jit.script
def two_args_two_kwargs(
    first_arg,
    second_arg,
    first_kwarg=torch.tensor([3, 3]),
    second_kwarg=torch.tensor([4, 4]),
):
    # [FUNCTION DOC] HELPER: Scripted RPC target returning the sum of two positional and two
    # keyword tensors (keyword defaults are 3s and 4s).
    return first_arg + second_arg + first_kwarg + second_kwarg


@torch.jit.script
def assorted_types_args_kwargs(
    tensor_arg: Tensor,
    str_arg: str,
    int_arg: int,
    tensor_kwarg: Tensor = torch.tensor([2, 2]),
    str_kwarg: str = "str_kwarg",
    int_kwarg: int = 2,
):
    # [FUNCTION DOC] HELPER: Scripted RPC target taking tensor, string and integer positional
    # args and keyword args with defaults. Returns a tuple: tensors added, strings concatenated,
    # integers added.
    return tensor_arg + tensor_kwarg, str_arg + str_kwarg, int_arg + int_kwarg


@torch.jit.script
def raise_script():
    # [FUNCTION DOC] HELPER: Scripted function that always raises a runtime error.
    raise RuntimeError("Expected error")


@torch.jit.script
def script_rpc_async_call(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    # [FUNCTION DOC] HELPER: Scripted async RPC of the two-args-two-kwargs function; waits for
    # and returns the result.
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    ret = fut.wait()
    return ret


@torch.jit.script
def script_rpc_sync_call(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    # [FUNCTION DOC] HELPER: Scripted blocking RPC of the same function; returns the result.
    res = rpc.rpc_sync(dst_worker_name, two_args_two_kwargs, args, kwargs)
    return res


@torch.jit.script
def script_rpc_remote_call(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    # [FUNCTION DOC] HELPER: Scripted remote call of the same function; fetches and returns the
    # RRef value.
    rref_res = rpc.remote(dst_worker_name, two_args_two_kwargs, args, kwargs)
    return rref_res.to_here()


class JitRpcOpTest:
    # [CLASS DOC]
    # Mixin of tests for calling functions remotely from TorchScript with rpc_async, rpc_sync
    # and remote: keyword defaults, mixed argument types, missing or extra arguments, and
    # unsupported or failing targets. Stateless.
    # Call functions remotely from Script.
    @dist_init
    def test_all_kwargs_are_populated_by_defaults(self):
        # [FUNCTION DOC]
        # Rank 0 only. For each scripted call flavor, passing no kwargs must use both defaults,
        # giving all tens.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {}

        for script_op in [
            script_rpc_async_call,
            script_rpc_sync_call,
            script_rpc_remote_call,
        ]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([10, 10]))

    @dist_init
    def test_some_kwargs_are_populated_by_defaults(self):
        # [FUNCTION DOC]
        # Rank 0 only. Passing only the first kwarg must default the second; expects all nines
        # for every call flavor.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {"first_kwarg": torch.tensor([2, 2])}

        for script_op in [
            script_rpc_async_call,
            script_rpc_sync_call,
            script_rpc_remote_call,
        ]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([9, 9]))

    @dist_init
    def test_no_kwargs_are_populated_by_defaults(self):
        # [FUNCTION DOC]
        # Rank 0 only. Passing both kwargs overrides all defaults; expects all eights for every
        # call flavor.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        for script_op in [
            script_rpc_async_call,
            script_rpc_sync_call,
            script_rpc_remote_call,
        ]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([8, 8]))

    @dist_init
    def test_args_and_kwargs_contain_different_types(self):
        # [FUNCTION DOC]
        # Rank 0 only. Scripted async RPC with tensor, string and integer positional and keyword
        # arguments (kwargs typed as a mapping to Any); checks the returned tuple of sums.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_with_assorted_types(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC of the assorted-types function with
            # fixed args and kwargs; waits and returns the tuple result.
            args = (torch.tensor([1, 1]), "str_arg", 1)
            # Must annotate the value type as `Any`, because JIT type inference
            # does not support multiple types when defining a Dict.
            # The error JIT gives is,
            # "Dict values must contain only a single type, "
            # "expected: Tensor but found str instead."
            kwargs: dict[str, Any] = {
                "tensor_kwarg": torch.tensor([3, 3]),
                "str_kwarg": "_str_kwarg",
                "int_kwarg": 3,
            }
            fut = rpc.rpc_async(
                dst_worker_name, assorted_types_args_kwargs, args, kwargs
            )
            ret = fut.wait()
            return ret

        ret = script_rpc_async_call_with_assorted_types(dst_worker_name)
        self.assertEqual(ret, (torch.tensor([4, 4]), "str_arg_str_kwarg", 4))

    @dist_init
    def test_kwargs_not_passed(self):
        # [FUNCTION DOC]
        # Rank 0 only. Scripted async RPC passing an empty args tuple and no kwargs to a
        # no-argument function; expects zero.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_without_kwargs_passed(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC with empty args and no kwargs; returns
            # the result.
            args = ()
            fut = rpc.rpc_async(dst_worker_name, no_arg, args)
            ret = fut.wait()
            return ret

        ret = script_rpc_async_call_without_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)

    @dist_init
    def test_args_kwargs_are_neither_passed(self):
        # [FUNCTION DOC]
        # Rank 0 only. Scripted async RPC passing neither args nor kwargs to a no-argument
        # function; expects zero.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_without_args_kwargs_passed(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC with no args or kwargs; returns result.
            fut = rpc.rpc_async(dst_worker_name, no_arg)
            ret = fut.wait()
            return ret

        ret = script_rpc_async_call_without_args_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)

    @dist_init
    def test_less_than_needed_args_are_specified(self):
        # [FUNCTION DOC]
        # Rank 0 only. Compiling a script that passes too few positional args must fail at
        # scripting time with an error naming the missing argument.
        if self.rank != 0:
            return

        # Notice, args matching happens during scripting.
        with self.assertRaisesRegex(RuntimeError, "Argument second_arg not provided"):

            @torch.jit.script
            def script_rpc_async_call_with_less_args(
                dst_worker_name: str,
            ):
                # [FUNCTION DOC] HELPER: Script expected to fail compilation: passes one
                # positional arg where two are required.
                args = (torch.tensor([1, 1]),)
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret

    @dist_init
    def test_more_than_needed_args_are_specified(self):
        # [FUNCTION DOC]
        # Rank 0 only. Compiling a script that passes five positional args to a function taking
        # at most four must fail at scripting time with a too-many-arguments error.
        if self.rank != 0:
            return

        # Notice, args matching happens during scripting.
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected at most 4 arguments but found 5 positional arguments",
        ):

            @torch.jit.script
            def script_rpc_async_call_with_more_args(
                dst_worker_name: str,
            ):
                # [FUNCTION DOC] HELPER: Script expected to fail compilation: passes five
                # positional args to a function accepting at most four.
                args = (
                    torch.tensor([1, 1]),
                    torch.tensor([2, 2]),
                    torch.tensor([3, 3]),
                    torch.tensor([4, 4]),
                    torch.tensor([5, 5]),
                )
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret

    @dist_init
    def test_unexepected_kwarg_is_specified(self):
        # [FUNCTION DOC]
        # Rank 0 only. A script passing an unknown keyword compiles, but running it must raise
        # an unknown keyword argument error (kwarg matching happens at execution time).
        # KNOWN ISSUE: method name misspells "unexpected".
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # Notice, kwargs matching happens during execution.
        @torch.jit.script
        def script_rpc_async_call_with_unexpected_kwarg(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC passing an unknown keyword argument;
            # raises at execution.
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {"third_kwarg": torch.tensor([1, 1])}
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "Unknown keyword argument 'third_kwarg'"
        ):
            ret = script_rpc_async_call_with_unexpected_kwarg(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_python_function_remotely_from_script_not_supported(self):
        # [FUNCTION DOC]
        # Rank 0 only. A script that targets a plain Python function through rpc_async must
        # raise an error about an undefined function when run.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
            # [FUNCTION DOC] HELPER: Scripted async RPC of a Python-only function; fails at run.
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, python_function, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_raises_remotely_from_script(self):
        # [FUNCTION DOC]
        # Rank 0 only. A script that remotely runs a script function which raises must surface
        # an error containing the expected message on wait.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        # Notice, TorchScript always translates(emits) Python `raise` statement,
        # as the exception message string, "Exception",
        # no matter what exception type and exception message are in the statement,
        @torch.jit.script
        def rpc_async_call_remote_raising_torchscript_in_torchscript(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC of the always-raising script function.
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, raise_script, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            ret = rpc_async_call_remote_raising_torchscript_in_torchscript(
                dst_worker_name
            )
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_not_exists_remotely_from_script(self):
        # [FUNCTION DOC]
        # Rank 0 only. A script function defined only inside the test is not registered on the
        # remote worker, so calling it remotely must raise an undefined function error naming it.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def nonexisting_script():
            # [FUNCTION DOC] HELPER: Scripted function returning zero, local to the test.
            return 0

        @torch.jit.script
        def rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
            dst_worker_name: str,
        ):
            # [FUNCTION DOC] HELPER: Scripted async RPC of the test-local script function.
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, nonexisting_script, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function nonexisting_script"
        ):
            ret = rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
                dst_worker_name
            )
            self.assertEqual(ret, 0)


@torch.jit.ignore
def my_script_module_init(rank: int) -> MyModuleInterface:
    # [FUNCTION DOC] HELPER: Python-only (ignored by the compiler) factory building a script
    # module for the given rank, usable as the module interface type from script.
    return MyScriptModule(rank)


@torch.jit.script
def construct_my_script_module(rank: int) -> MyModuleInterface:
    # [FUNCTION DOC] HELPER: Scripted function that constructs the script module via the
    # ignored Python factory.
    return my_script_module_init(rank)


@torch.jit.script
def run_ref_script_module(
    ref_script_module: RRef[MyModuleInterface], t: Tensor
) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted function that fetches the module from the RRef and returns
    # its forward output plus t.
    module = ref_script_module.to_here()
    return module.forward() + t


@torch.jit.script
def script_check_rref_confirmed(rref: RRef[Tensor]) -> bool:
    # [FUNCTION DOC] HELPER: Scripted function returning whether the owner has confirmed the RRef.
    return rref.confirmed_by_owner()


@torch.jit.script
def save_rref(rref_var: RRef[Tensor], fname: str) -> None:
    # [FUNCTION DOC] HELPER: Scripted function saving the RRef to file fname; writes a file and
    # raises when JIT RRef pickling is not enabled.
    torch.save(rref_var, fname)


@torch.jit.script
def script_add(x: Tensor, y: Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Scripted elementwise sum of two tensors.
    return x + y


@rpc.functions.async_execution
@torch.jit.script
def async_add(to: str, x: Tensor, y: Tensor) -> Future[Tensor]:
    # [FUNCTION DOC] HELPER: Async-execution scripted function returning a future from an async
    # RPC of the add function to worker to.
    return rpc.rpc_async(to, script_add, (x, y))


@rpc.functions.async_execution
@torch.jit.script
def async_wrong_type() -> Tensor:
    # [FUNCTION DOC] HELPER: Async-execution function that wrongly returns a tensor instead of a
    # future; used to trigger the return type error.
    return torch.zeros(2)


def load_script_module_with_pickled_rref(pickled_script_module):
    # [FUNCTION DOC] HELPER: RPC target that loads a script module from the given bytes, calls
    # it and returns its output.
    f = io.BytesIO(pickled_script_module)
    m = torch.jit.load(f)
    return m()


class JitRpcTest(
    RRefAPITest,
    RRefTypingTest,
    LocalRRefTest,
    JitRpcOpTest,
    FutureTypingTest,
    RpcAgentTestFixture,
):
    # [CLASS DOC]
    # Main JIT RPC test case: composes the RRef, local RRef, op and future mixins with the
    # agent fixture and adds tests for script functions and modules over RPC, JIT RRef pickling,
    # callbacks on futures, profiling, and async_execution. Stateless.
    @dist_init
    def test_torchscript_function(self):
        # [FUNCTION DOC]
        # Checks a script function gives the same result locally, via rpc_sync, via remote and via
        # remote to the same worker.
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        local_ret = one_arg(torch.ones(2, 2))
        ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(ret, local_ret)
        rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(rref.to_here(), local_ret)
        # create rref to itself
        local_rref = rpc.remote(
            worker_name(self.rank), one_arg, args=(torch.ones(2, 2),)
        )
        self.assertEqual(local_rref.to_here(), local_ret)

    @dist_init
    def test_torchscript_function_exception(self):
        # [FUNCTION DOC]
        # Passing too many arguments to a script function over rpc_sync or remote must raise an
        # error saying it expected at most a given number of arguments.
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))

        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            rpc.remote(dst_worker_name, one_arg, args=(10, 20))

    @dist_init
    def test_torchscript_functions_not_supported(self):
        # [FUNCTION DOC]
        # Creates a local script module, then initializes a process group and barriers so module
        # creation does not race with server threads. Checks rpc_sync accepts a script class like
        # a Python callable, while sending a script module method raises a pickling type error.
        # Side effect: initializes the default process group.
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        my_local_script_module = MyScriptModule(self.rank)

        # It is not thread safe to instantiate MyScriptModule in multiple threads,
        # wait for local MyScriptModule instantiation to finish,
        # otherwise it could instantiate MyScriptModule in parallel with
        # server thread in the below
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        # rpc_sync still accepts script class and run it in
        # the same code path as python call.
        rpc.rpc_sync(dst_worker_name, MyScriptClass, args=(self.rank,))

        # rpc_sync does not accept script module method.
        # Python 3.5 and Python 3.6 throw different error message, the only
        # common word can be grepped is "pickle".
        with self.assertRaisesRegex(TypeError, "pickle"):
            rpc.rpc_async(dst_worker_name, my_local_script_module.forward, args=())

    @dist_init
    def test_remote_script_module(self):
        # [FUNCTION DOC]
        # Sets the global flag that ignores RRef leaks at shutdown (side effect, never reset),
        # creates a script module remotely, passes its RRef to the owner where forward result
        # plus ones is checked, and checks sending the RRef from the owner to itself raises an
        # error about script module RRefs not sendable from the owner.
        # TODO, need more investigation
        # there is rref leak when shutting down, suspect it is because
        # ref as arg is passed to pybind boundary, and the ref is not garbage
        # collected by python when calling shutdown()
        import torch.distributed.rpc.api as api

        api._ignore_rref_leak = True

        local_ret = torch.ones(self.rank) + torch.ones(self.rank)

        n = self.rank + 1
        dst_rank = n % self.world_size
        remote_ref = rpc.remote(
            worker_name(dst_rank), construct_my_script_module, args=(self.rank,)
        )

        # pass rref arg to owner
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            run_ref_script_module,
            args=(remote_ref, torch.ones(self.rank)),
        )
        self.assertEqual(ret, local_ret)

        # pass rref arg to self/user
        with self.assertRaisesRegex(
            RuntimeError,
            "is an RRef to a ScriptModule. It can't be sent through RPC from owner,",
        ):
            ret = rpc.rpc_sync(
                worker_name(self.rank),
                run_ref_script_module,
                args=(remote_ref, torch.ones(self.rank)),
            )

    @dist_init
    def test_create_script_module_on_remote(self):
        # [FUNCTION DOC]
        # Checks a script module built remotely by rpc_sync comes back as a script module that
        # runs forward; with remote, the owner holds a script module, remote forward and
        # custom method calls work, and fetching it to this worker yields a working module.
        dst_name = worker_name((self.rank + 1) % self.world_size)
        # Construct on remote end with rpc_sync
        created_script_module = rpc.rpc_sync(
            dst_name, MyScriptModule, args=(self.rank,)
        )
        # Forward should output a ones tensor of self.rank.
        self.assertTrue(isinstance(created_script_module, torch.jit.ScriptModule))
        rank_ones_tensor = created_script_module()
        self.assertEqual(torch.ones(self.rank), rank_ones_tensor)

        # Construct ScriptModule with rpc.remote.
        remote_script_module = rpc.remote(dst_name, MyScriptModule, args=(self.rank,))
        # Verify it is an instance of ScriptModule on remote end.
        remote_end_is_script = rpc.rpc_sync(
            remote_script_module.owner(),
            rref_isinstance,
            args=(remote_script_module, torch.jit.ScriptModule),
        )
        self.assertTrue(remote_end_is_script)
        # Run forward pass remotely.
        remote_forward_output = remote_script_module.rpc_sync().forward()
        self.assertEqual(remote_forward_output, torch.ones(self.rank))
        # Run function defined on ScriptModule remotely.
        remote_func_output = remote_script_module.rpc_sync().custom_func()
        self.assertEqual(remote_func_output, torch.ones(self.rank))
        # Ensure we can transfer ScriptModule RRef to this rank and run
        # forward pass.
        local_script_module = remote_script_module.to_here()
        self.assertTrue(isinstance(local_script_module, torch.jit.ScriptModule))
        rank_ones_tensor = local_script_module()
        self.assertEqual(rank_ones_tensor, torch.ones(self.rank))
        local_script_func_output = local_script_module.custom_func()
        self.assertEqual(local_script_func_output, torch.ones(self.rank))

    @dist_init
    def test_load_script_module_with_pickled_rref(self):
        # [FUNCTION DOC]
        # Saves a script module holding RRefs to a buffer with JIT RRef pickling temporarily
        # enabled, loads and runs it on the next worker, and checks the output equals running an
        # equivalent module locally. Toggles the global JIT RRef pickle switch (on, then off).
        dst_name = worker_name((self.rank + 1) % self.world_size)
        m1 = MyScriptModuleWithRRefs(dst_name)
        m2 = MyScriptModuleWithRRefs(dst_name)

        f = io.BytesIO()

        rpc._enable_jit_rref_pickle()
        torch.jit.save(m1, f)
        rpc._disable_jit_rref_pickle()

        out1 = rpc.rpc_sync(
            dst_name, load_script_module_with_pickled_rref, args=(f.getvalue(),)
        )
        out2 = m2()
        self.assertEqual(out1, out2)

    @dist_init
    def test_rref_jit_pickle_not_supported(self):
        # [FUNCTION DOC]
        # Saving an RRef through JIT outside an RPC call must raise an error saying RRef JIT
        # pickling is only allowed inside RPC calls. Uses a temporary file.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))
        with (
            TemporaryFileName() as fname,
            self.assertRaisesRegex(
                RuntimeError, "RRef jit pickling is only allowed inside RPC calls"
            ),
        ):
            save_rref(rref_var, fname)

    @dist_init
    def test_remote_script_throw(self):
        # [FUNCTION DOC]
        # A remote call to a raising script function must make the RRef fetch raise an error
        # with the expected message.
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            script_raise_func,
            args=(torch.ones(2),),
        )
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            rref.to_here()

    @dist_init
    def test_remote_script_udf(self):
        # [FUNCTION DOC]
        # A remote call to a script function that forks and waits must fetch a tensor of twos.
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2) * 2)

    @dist_init
    def test_async_script_udf(self):
        # [FUNCTION DOC]
        # An async RPC to the fork-and-wait script function must resolve to a tensor of twos.
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )
        self.assertEqual(future.wait(), torch.ones(2) * 2)

    @dist_init
    def test_callback_simple(self):
        # [FUNCTION DOC]
        # Chains one then-callback adding one on an async RPC future; expects twos plus one.
        def callback(fut):
            # [FUNCTION DOC] HELPER: Returns the waited future value plus one.
            return fut.wait() + 1

        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        ).then(callback)
        self.assertEqual(future.wait(), torch.ones(2) * 2 + 1)

    @dist_init
    def test_callback_chain(self):
        # [FUNCTION DOC]
        # Chains twenty then-callbacks, each adding one, onto an async RPC future; expects the
        # plus-one result plus twenty.
        n = self.rank + 1

        def callback(fut):
            # [FUNCTION DOC] HELPER: Returns the waited future value plus one.
            return fut.wait() + 1

        fut = rpc.rpc_async(
            worker_name(n % self.world_size), one_arg, args=(torch.ones(n, n),)
        )

        num_cbs = 20
        for _ in range(num_cbs):
            fut = fut.then(callback)

        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_add_done_callback(self):
        # [FUNCTION DOC]
        # Registers a done callback that stores double the result, adds a later then-callback to
        # ensure the first has run, and checks the stored value is four times ones.
        callback_called = None

        def callback(fut):
            # [FUNCTION DOC] HELPER: Stores double the waited future value in the enclosing
            # variable.
            nonlocal callback_called
            callback_called = fut.wait() * 2

        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_udf,
            args=(torch.ones(2),),
        )

        future.add_done_callback(callback)
        future_then = future.then(lambda _: True)

        self.assertEqual(future.wait(), torch.ones(2) * 2)

        # We have no guarantee that the add_done_callback fn will execute before the test finishes.
        # Adding a 'then' callback that runs afterwards to guarantee we wait for the first callback
        future_then.wait()
        self.assertEqual(callback_called, torch.ones(2) * 4)

    @dist_init
    def test_async_script_throw(self):
        # [FUNCTION DOC]
        # An async RPC to a raising script function must raise the expected error on wait.
        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_throw,
            args=(torch.ones(2),),
        )
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            future.wait()

    @dist_init
    def test_callback_with_exception(self):
        # [FUNCTION DOC]
        # A then-callback on a failing RPC future observes the original error, then raises a
        # new error; waiting on the chained future must raise the new error.
        def callback(fut):
            # [FUNCTION DOC] HELPER: Asserts waiting raises the expected error, then raises
            # another runtime error.
            with self.assertRaisesRegex(Exception, ".*Expected error.*"):
                fut.wait()
            raise RuntimeError("Another expected error")

        future = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            script_fork_wait_throw,
            args=(torch.ones(2),),
        ).then(callback)

        with self.assertRaisesRegex(RuntimeError, "Another expected error"):
            future.wait()

    @dist_init
    def test_call_rpc_with_profiling(self):
        # [FUNCTION DOC]
        # Rank 0 only. Under the legacy profiler, runs a script that attaches the end callbacks of
        # a record-function event to an RPC future, then checks the recorded function event name
        # contains the qualified name of the remote script function.
        # Ensures that we can call torch.ops.profiler._call_end_callbacks_on_jit_fut on a jit
        # future from within a script function that calls rpc_async
        if self.rank == 0:
            with _profile() as prof:
                prof_key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(one_arg),
                    "worker0",
                    "worker1",
                )
                with torch.autograd.profiler.record_function(prof_key) as rf:
                    call_rpc_with_profiling(rf.record, "worker1")
            # TODO: Can't get a reliable time for this profiling event since
            # it's hard to estimate the execution time on the remote end for non-UDFs.
            # This can be resolved by https://github.com/pytorch/pytorch/issues/36272.
            # After that, this test should be modified to validate the function time.
            events = prof.function_events
            function_event = get_function_event(events, prof_key)
            self.assertTrue(
                torch._jit_internal._qualified_name(one_arg) in function_event.name
            )

    @dist_init
    def test_rpc_async_jit_profiled(self):
        # [FUNCTION DOC]
        # Rank 0 only. Profiles a scripted rpc_async call and checks exactly one local event for
        # the target function exists with the expected async-JIT profiling key, and inspects
        # the remote events recorded.
        # Tests that rpc_async calls made from within a TorchScript function are
        # profiled.
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {}
            with _profile() as prof:
                script_rpc_async_call(dst_worker_name, args, kwargs)

            # Ensure rpc_async call is profiled
            function_events = prof.function_events
            qual_name = torch._jit_internal._qualified_name(two_args_two_kwargs)
            rpc_async_jit_event = [
                event
                for event in function_events
                if qual_name in event.name and event.node_id == self.rank
            ]
            self.assertEqual(len(rpc_async_jit_event), 1)
            rpc_async_jit_event = rpc_async_jit_event[0]
            profiled_name = _build_rpc_profiling_key(
                RPCExecMode.ASYNC_JIT,
                qual_name,
                worker_name(self.rank),
                dst_worker_name,
            )
            self.assertEqual(profiled_name, rpc_async_jit_event.name)
            remote_events = [event for event in function_events if event.is_remote]
            # All remote events should have taken place on dst_rank
            remote_event_node_ids = {
                remote_event.node_id for remote_event in remote_events
            }
            self.assertEqual(remote_event_node_ids, {dst_rank})
            # script_rpc_async_call invokes add operator
            # so we should see this as a remote event.
            remote_add = next(
                remote_event
                for remote_event in remote_events
                if "aten::add" in remote_event.name
            )
            remote_add_profiled_name = f"{profiled_name}#remote_op: aten::add"
            self.assertEqual(remote_add.name, remote_add_profiled_name)

    @dist_init
    def test_record_function_on_caller_rpc_async(self):
        # [FUNCTION DOC]
        # Rank 0 only. Profiles a script that makes two async RPCs inside a record_function
        # scope; checks one scope event and two RPC events exist, and the scope's total CPU
        # time exceeds each RPC event's time (not the sum, since the calls overlap).
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            block_scope = "foo"
            with _profile() as prof:
                # Runs 2 rpc_async calls within JIT under record_function.
                record_function_on_caller_rpc_async(dst_worker_name, block_scope)

            # Ensure record_function event is profiled.
            function_events = prof.function_events
            record_function_scope_event = [
                event for event in function_events if event.name == block_scope
            ]
            self.assertEqual(1, len(record_function_scope_event))
            record_function_scope_event = record_function_scope_event[0]
            # Ensure RPC future is profiled.
            expected_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC_JIT,
                torch._jit_internal._qualified_name(script_add_ones),
                worker_name(self.rank),
                dst_worker_name,
            )
            jit_rpc_events = [
                event for event in function_events if event.name == expected_key
            ]
            self.assertEqual(2, len(jit_rpc_events))
            # Validate that the record_function scope time is greater than both
            # of the individual RPC async call times. The reason it is not necessarily
            # greater than the sum is because the two can execute in parallel.
            for jit_rpc_event in jit_rpc_events:
                self.assertTrue(
                    record_function_scope_event.cpu_time_total
                    > jit_rpc_event.cpu_time_total
                )

    @dist_init
    def test_rpc_torchscript_record_function(self):
        # [FUNCTION DOC]
        # Rank 0 only. Profiles an RPC to a script function that uses a record_function scope
        # remotely; finds the remote event whose name combines the profiling key, the remote
        # op marker and the scope name and inspects its CPU children.
        # KNOWN ISSUE: the final assertion passes a generator expression to a truth check, so it
        # is always true and does not actually verify that a child event is an add op.
        # tests that torchscript functions can be profiled using with
        # record_function(...) over RPC.
        REMOTE_OP_STR = "#remote_op: "
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            block_scope = "foo"
            with _profile() as prof:
                call_rpc_torchscript_with_record_function(dst_worker_name, block_scope)

            # Need to call below to populate CPU children.
            prof.key_averages()
            function_events = prof.function_events
            expected_key = (
                _build_rpc_profiling_key(
                    RPCExecMode.ASYNC_JIT,
                    torch._jit_internal._qualified_name(
                        script_add_ones_with_record_function
                    ),
                    worker_name(self.rank),
                    dst_worker_name,
                )
                + REMOTE_OP_STR
                + block_scope
            )
            remote_record_function_event = next(
                evt for evt in function_events if evt.name == expected_key
            )
            self.assertTrue(block_scope in remote_record_function_event.name)
            remote_children = remote_record_function_event.cpu_children
            self.assertTrue("aten::add" in child.name for child in remote_children)

    def test_record_function_jit_end_callbacks_with_fork(self):
        # [FUNCTION DOC]
        # Not wrapped by dist_init (no RPC used). In eager mode, forks a sleeping function and
        # attaches the record_function end callbacks to the JIT future; checks the profiled
        # event is named foo and its CPU time is at least the sleep interval.
        # Ensures that we can call rf._call_end_callbacks_on_future on a jit
        # future in python eager mode with torch.jit.fork
        sleep_interval = 1
        with _profile() as prof:
            with torch.autograd.profiler.record_function("foo") as rf:
                fut = torch.jit._fork(sleep, sleep_interval)
                rf._call_end_callbacks_on_future(fut)
            fut.wait()

        function_events = prof.function_events
        sleep_event = get_function_event(function_events, "foo")
        self.assertEqual(sleep_event.name, "foo")
        # Validate that callbacks were fired at the right time by checking the
        # profiling event cpu time
        self.assertGreaterAlmostEqual(sleep_event.cpu_time * 1e-6, sleep_interval)

    def test_call_fork_in_jit_with_profiling(self):
        # [FUNCTION DOC]
        # Not wrapped by dist_init (no RPC used). Runs a script that forks and attaches the end
        # callbacks of a record_function event to the future; checks the profiler recorded an
        # event named foo.
        # Ensures that we can call torch.ops.profiler._call_end_callbacks_on_jit_fut on a jit
        # future from within a script function with torch.jit.fork
        with _profile() as prof, torch.autograd.profiler.record_function("foo") as rf:
            call_fork_with_profiling(rf.record)

        events = prof.function_events
        function_event = get_function_event(events, "foo")
        self.assertEqual(function_event.name, "foo")

    @dist_init
    def test_async_function_simple(self):
        # [FUNCTION DOC]
        # An rpc_sync to an async-execution script function that itself forwards the add to a
        # third worker must return the sum, twos.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(
            dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2))
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_wrong_return_type(self):
        # [FUNCTION DOC]
        # An async-execution function returning a tensor instead of a future must make rpc_sync
        # raise an error saying async functions must return a future.
        with self.assertRaisesRegex(
            RuntimeError,
            "Async functions must return an IValue of Future type, but got Tensor",
        ):
            rpc.rpc_sync(
                worker_name((self.rank + 1) % self.world_size), async_wrong_type
            )

    @dist_init
    def test_async_function_wrong_decorator_order(self):
        # [FUNCTION DOC]
        # Applying the script decorator on top of the async-execution decorator must make
        # compilation raise an error (the exact message is intentionally not checked).
        # @torch.jit.script complains about undefined value rpc. Error is shown
        # below. The reason for not checking error string is to avoid making
        # JIT error handling code depend on RPC tests, as we don't have any
        # restrictions on the error message here.
        #
        # RuntimeError:
        # undefined value rpc:
        # def async_wrong_decorator_order(to, x, y):
        #    # type: (str, Tensor, Tensor) -> Future[Tensor]
        #    return rpc.rpc_async(to, script_add, (x, y))
        #           ~~~ <--- HERE
        with self.assertRaises(RuntimeError):

            @torch.jit.script
            @rpc.functions.async_execution
            def async_wrong_decorator_order(
                to: str, x: Tensor, y: Tensor
            ) -> Future[Tensor]:
                # [FUNCTION DOC] HELPER: Function with the wrong decorator order; compilation
                # fails so this body is never run.
                return rpc.rpc_async(to, script_add, (x, y))

    @dist_init
    def test_async_function_remote(self):
        # [FUNCTION DOC]
        # Same as the simple async-execution test but via remote; the fetched value is twos.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        rref = rpc.remote(
            dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2))
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_remote_multi(self):
        # [FUNCTION DOC]
        # Issues twenty remote async-execution adds with increasing offsets and checks each fetched
        # result equals ones plus the offset.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        num = 20
        rrefs = [
            rpc.remote(
                dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2) * i)
            )
            for i in range(num)
        ]

        for i in range(num):
            self.assertEqual(rrefs[i].to_here(), torch.ones(2, 2) + i)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        # [FUNCTION DOC]
        # Remote variant of the wrong return type test: fetching the RRef must raise the error
        # saying async functions must return a future.
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size), async_wrong_type
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Async functions must return an IValue of Future type, but got Tensor",
        ):
            rref.to_here()
