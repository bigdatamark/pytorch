# mypy: allow-untyped-defs
# [MODULE DOC] Test suite for the torch.distributed.rpc framework. Defines module-level helper
# functions and classes that are shipped to remote workers as RPC payloads (arithmetic helpers,
# RRef helpers, async_execution functions, scripted functions, custom exceptions, pickle test
# classes, parameter-server models), the RpcTestCommon mixin with shared test bodies, and the
# RpcTest class (plus further test classes below) that is mixed with RpcAgentTestFixture so each
# backend can run the same tests. Import-time side effects: creates module-level concurrent futures
# (VALUE_FUTURE, DONE_FUTURE), a global barrier counter and a global RRef slot, and compiles
# several functions with the TorchScript compiler.

import concurrent.futures
import contextlib
import json
import operator
import os
import sys
import threading
import time
import unittest
from collections import namedtuple
from functools import partial
from threading import Event, Lock
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.autograd.profiler_legacy import profile as _profile
from torch.distributed.rpc import (
    _get_debug_info,
    _rref_context_get_debug_info,
    RRef,
    WorkerInfo,
)
from torch.distributed.rpc.api import _thread_local_var, _use_rpc_pickler, _wait_all
from torch.distributed.rpc.internal import (
    _build_rpc_profiling_key,
    _internal_rpc_pickler,
    PythonUDF,
    RPCExecMode,
)
from torch.futures import Future
from torch.testing._internal.common_distributed import (
    captured_output,
    skip_if_lt_x_gpu,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    IS_LINUX,
    IS_MACOS,
    load_tests,
    skip_but_pass_in_sandcastle_if,
    TemporaryFileName,
)
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    wait_until_node_failure,
    wait_until_owners_and_forks_on_rank,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def foo_add():
    # [FUNCTION DOC] HELPER: Remote payload that adds two single-element ones tensors and returns
    # the result tensor.
    return torch.add(torch.ones(1), torch.ones(1))


def udf_with_torch_ops(device=-1, use_record_function=False):
    # [FUNCTION DOC] HELPER: Remote payload that runs a fixed chain of tensor ops (add, mul,
    # relu, sigmoid) for profiler tests. Args: device selects a CUDA device context (-1 means
    # none); use_record_function wraps the ops in a profiler record_function scope named forward.
    # Returns nothing.
    device_ctx = contextlib.nullcontext() if device == -1 else torch.cuda.device(device)
    record_function_ctx = (
        torch.autograd.profiler.record_function("##forward##")
        if use_record_function
        else contextlib.nullcontext()
    )
    with device_ctx, record_function_ctx:
        t1, t2 = torch.ones(1), torch.ones(1)
        t = torch.add(t1, t2)
        t = torch.mul(t, t)
        t = t.relu()
        t = t.sigmoid()


# Events (operator invocations) that are expected to be run as part of the above
# function.
EXPECTED_REMOTE_EVENTS = [
    "aten::ones",
    "aten::ones",
    "aten::add",
    "aten::mul",
    "aten::relu",
    "aten::clamp_min",
    "aten::sigmoid",
]

# Remote operations are prefixed with the following string for RPC profiling.
REMOTE_OP_STR = "#remote_op: "


VALUE_FUTURE = concurrent.futures.Future()
DONE_FUTURE = concurrent.futures.Future()

FIFTY_MIL_CYCLES = 50000000

_rpc_barrier_count = 0


def _increment_count():
    # [FUNCTION DOC] HELPER: Increments the module-level barrier counter; run on the leader worker
    # via RPC. Side effect: mutates global state.
    global _rpc_barrier_count
    _rpc_barrier_count += 1


def _reset_count():
    # [FUNCTION DOC] HELPER: Resets the module-level barrier counter to zero; run on the leader
    # worker via RPC. Side effect: mutates global state.
    global _rpc_barrier_count
    _rpc_barrier_count = 0


class StubRpcAgent:
    # [CLASS DOC] HELPER: Minimal fake RPC agent used with the stub backend; remembers only the
    # world size and reports worker infos.
    def __init__(self, world_size):
        # [FUNCTION DOC] HELPER: Stores world_size on the stub agent.
        self.world_size = world_size

    def get_worker_infos(self):
        # [FUNCTION DOC] HELPER: Returns a set of WorkerInfo objects, one per rank in the world,
        # named with worker_name.
        return {
            WorkerInfo(name=worker_name(rank), id=rank)
            for rank in range(self.world_size)
        }


def _stub_construct_rpc_backend_options_handler(**kwargs):
    # [FUNCTION DOC] HELPER: Stub options-constructor handler for a fake backend; ignores its
    # keyword arguments and returns a mock object.
    return mock.Mock()  # RpcBackendOptions.


def _stub_init_rpc_backend_handler(store, name, rank, world_size, rpc_backend_options):
    # [FUNCTION DOC] HELPER: Stub backend init handler; ignores the store, name, rank and options
    # and returns a StubRpcAgent for the given world size.
    return StubRpcAgent(world_size=world_size)


def set_value(value):
    # [FUNCTION DOC] HELPER: Remote payload that resolves the module-level VALUE_FUTURE with the
    # given value (raises if already set).
    VALUE_FUTURE.set_result(value)


def wait_for_value_future():
    # [FUNCTION DOC] HELPER: Remote payload that blocks until VALUE_FUTURE is resolved and returns
    # its value.
    return VALUE_FUTURE.result()


def set_and_check_done(value):
    # [FUNCTION DOC] HELPER: Remote payload that resolves VALUE_FUTURE with value, then blocks
    # until DONE_FUTURE is resolved and returns its result.
    VALUE_FUTURE.set_result(value)
    return DONE_FUTURE.result()


# it is used to test python user defined function over rpc
# classes and functions are used to test python user defined class and
# methods over rpc
TensorClass = namedtuple("TensorClass", ["tensors"])


class MyPickleClass:
    # [CLASS DOC] HELPER: Picklable class whose pickling hooks nest the RPC internal pickler: state
    # is a serialized PythonUDF that, when restored, computes a tensor sum and stores it in
    # attribute t.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes attribute t to None.
        self.t = None

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickling hook; serializes a PythonUDF for my_tensor_function with
        # two 2x2 ones tensors using the internal RPC pickler and returns the pair (pickled bytes,
        # tensor list).
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(my_tensor_function, (torch.ones(2, 2), torch.ones(2, 2)), None)
        )
        return (pickled_python_udf, tensors)

    def __setstate__(self, obj):
        # [FUNCTION DOC] HELPER: Unpickling hook; deserializes the nested PythonUDF from obj, runs
        # it, and stores the result in t.
        python_udf = _internal_rpc_pickler.deserialize(obj[0], obj[1])
        result = python_udf.func(python_udf.args[0], python_udf.args[1])
        self.t = result

    def set(self, val):
        # [FUNCTION DOC] HELPER: Setter for attribute t.
        self.t = val


class SlowPickleClass:
    # [CLASS DOC] HELPER: Picklable class that sleeps for t seconds in both pickling and
    # unpickling, used to make serialization slow (timeouts, shutdown tests).
    def __init__(self, t):
        # [FUNCTION DOC] HELPER: Stores the sleep duration t.
        self.t = t

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickling hook; sleeps t seconds and returns a one-element tuple
        # holding t.
        time.sleep(self.t)
        return (self.t,)

    def __setstate__(self, obj):
        # [FUNCTION DOC] HELPER: Unpickling hook; restores t from obj and sleeps t seconds.
        self.t = obj[0]
        time.sleep(self.t)


class MyClass:
    # [CLASS DOC] HELPER: Simple stateful class used as an RRef target and for remote method calls;
    # holds a value a and offers instance, class and static methods.
    def __init__(self, a, delay=False):
        # [FUNCTION DOC] HELPER: Stores a; if delay is true, sleeps 2 seconds to simulate a slow or
        # failing construction.
        self.a = a
        # delay initialization to simulate errors if specified
        if delay:
            time.sleep(2)

    def my_instance_method(self, b):
        # [FUNCTION DOC] HELPER: Returns a plus b.
        return self.a + b

    @classmethod
    def my_class_method(cls, d, e):
        # [FUNCTION DOC] HELPER: Class method; returns d plus e.
        return d + e

    @staticmethod
    def my_static_method(f):
        # [FUNCTION DOC] HELPER: Static method; returns whether f is greater than 10.
        return f > 10

    def increment_value(self, increment):
        # [FUNCTION DOC] HELPER: Mutates a by adding increment; returns nothing.
        self.a += increment

    def get_value(self):
        # [FUNCTION DOC] HELPER: Returns the current value of a.
        return self.a

    def my_slow_method(self, my_tensor_arg):
        # [FUNCTION DOC] HELPER: Sleeps 5 seconds, then returns torch.add of a and my_tensor_arg
        # (used for timeout tests).
        time.sleep(5)
        return torch.add(self.a, my_tensor_arg)


def _call_method_on_rref(method, rref, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Applies method to the local value of rref with the extra args;
    # intended to run on the RRef owner.
    return method(rref.local_value(), *args, **kwargs)


def get_rref_list(values):
    # [FUNCTION DOC] HELPER: Returns a list of RRefs, each owning a new MyClass built from one of
    # the given values.
    return [RRef(MyClass(a)) for a in values]


def add_rref_to_value(rref, value):
    # [FUNCTION DOC] HELPER: Fetches the RRef value with to_here and returns it plus value.
    return rref.to_here() + value


def run_nested_pickle(pickle_cls_instance, tensor):
    # [FUNCTION DOC] HELPER: Returns the t attribute of the unpickled object plus tensor.
    return pickle_cls_instance.t + tensor


def build_sparse_tensor(coalesce=False):
    # [FUNCTION DOC] HELPER: Builds a small 2x3 sparse COO tensor with fixed indices and values;
    # coalesces it if coalesce is true.
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3, 4, 5]
    tensor = torch.sparse_coo_tensor(i, v, (2, 3))
    if coalesce:
        tensor = tensor.coalesce()
    return tensor


def build_complex_tensors():
    # [FUNCTION DOC] HELPER: Builds nested containers of the same ones tensor (lists and a dict
    # keyed by the tensor) and returns five of them in a list.
    a = torch.ones(3, 3)
    b = [a, a]
    c = [b, b]
    d = [a, b]
    e = {a: d}
    return [a, b, c, d, e]


def non_cont_test(t_view, t_cont):
    # [FUNCTION DOC] HELPER: Validates inputs that cross RPC: raises a generic Exception if
    # t_view is contiguous, if t_cont is not contiguous, or if the two are not equal; otherwise
    # returns t_view.
    if t_view.is_contiguous():
        raise Exception("t_view is contiguous!")  # noqa: TRY002
    if not t_cont.is_contiguous():
        raise Exception("t_cont is not contiguous!")  # noqa: TRY002
    if not torch.equal(t_view, t_cont):
        raise Exception("t_view is not equal to t_cont!")  # noqa: TRY002
    return t_view


def my_function(a, b, c):
    # [FUNCTION DOC] HELPER: Returns a + b + c.
    return a + b + c


def my_tensor_function(a, b):
    # [FUNCTION DOC] HELPER: Returns a + b.
    return a + b


def my_container_sum(a):
    # [FUNCTION DOC] HELPER: Returns the sum of all tensors in the sequence a, accumulating in
    # place into the first element (mutates it).
    result = a[0]
    for tensor in a[1:]:
        result += tensor
    return result


def my_sleep_func(seconds=1):
    # [FUNCTION DOC] HELPER: Sleeps for the given seconds, then returns a tensor product of two
    # scalar tensors (value 1).
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))


def my_complex_tensor_function(list_input, tensor_class_input, dict_input):
    # [FUNCTION DOC] HELPER: Sums list_input (accumulating in place into its first tensor) and all
    # values of dict_input into one result; returns a tuple of that result and the first three
    # tensors of tensor_class_input.tensors.
    res = list_input[0]
    for t in list_input:
        res += t
    for v in dict_input.values():
        res += v
    complex_tensors = tensor_class_input.tensors
    return (res, complex_tensors[0], complex_tensors[1], complex_tensors[2])


def my_rref_function(rref_a, rref_b):
    # [FUNCTION DOC] HELPER: Fetches both RRefs with to_here and returns the sum.
    return rref_a.to_here() + rref_b.to_here()


def delayed_add(a, b, seconds=0.05):
    # [FUNCTION DOC] HELPER: Sleeps for the given seconds and returns a + b.
    time.sleep(seconds)
    return a + b


def identity(a):
    # [FUNCTION DOC] HELPER: Returns its argument unchanged.
    return a


def no_result():
    # [FUNCTION DOC] HELPER: Prints a message and returns None.
    print("do nothing")


def raise_or_inc(value):
    # [FUNCTION DOC] HELPER: Raises ValueError with the message Expected error if value has exactly
    # two elements; otherwise returns value plus one.
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1


def nested_rpc(dst):
    # [FUNCTION DOC] HELPER: Issues a synchronous RPC to dst computing ones(2, 2) plus 1 and
    # returns the result.
    return rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))


def nested_rpc_sparse(dst):
    # [FUNCTION DOC] HELPER: Issues a synchronous RPC to dst adding two sparse tensors and returns
    # the result.
    return rpc.rpc_sync(
        dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())
    )


def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # [FUNCTION DOC] HELPER: Fire-and-forget chain: while ttl is positive, sends an async RPC to
    # worker dst that recurses on the next rank with ttl reduced by one, and returns 0; returns
    # None when ttl is not positive. Does not wait for the nested call.
    # this method returns immediately without blocking the callee, but will
    # generate additional requests.
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        rpc.rpc_async(
            current_dst,
            multi_layer_nested_async_rpc,
            args=(next_dst, world_size, ttl - 1),
        )
        return 0


def nested_rref(dst):
    # [FUNCTION DOC] HELPER: Creates two RRefs on dst (ones(2, 2) plus 1 and plus 2) and returns
    # them as a tuple.
    return (
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1)),
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 2)),
    )


def nested_rref_sparse(dst):
    # [FUNCTION DOC] HELPER: Creates two RRefs on dst, each holding the sum of two sparse tensors,
    # and returns them as a tuple.
    return (
        rpc.remote(dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())),
        rpc.remote(dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())),
    )


def nested_remote(dst):
    # [FUNCTION DOC] HELPER: Creates an RRef on dst computing ones(2, 2) plus 3 and returns its
    # fetched value.
    rref = rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 3))
    return rref.to_here()


def nested_remote_sparse(dst):
    # [FUNCTION DOC] HELPER: Creates an RRef on dst holding the sum of two sparse tensors and
    # returns its fetched value.
    rref = rpc.remote(
        dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())
    )
    return rref.to_here()


def rref_forward_chain(dst, world_size, rref, ttl):
    # [FUNCTION DOC] HELPER: Recursive RRef chain; while ttl is positive, creates an RRef on worker
    # dst running this function on the next rank with ttl reduced, and returns it in a one-element
    # list; at ttl zero or below returns the fetched value of rref.
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        ret_rref = rpc.remote(
            current_dst, rref_forward_chain, args=(next_dst, world_size, rref, ttl - 1)
        )
        return [ret_rref]
    else:
        return rref.to_here()


def rpc_return_rref(dst):
    # [FUNCTION DOC] HELPER: Creates an RRef on dst computing ones(2, 2) plus 1 and returns the
    # RRef itself (not the value).
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


def light_rpc():
    # [FUNCTION DOC] HELPER: Trivial payload returning 0 with no computation.
    return 0


def heavy_rpc(tensor):
    # [FUNCTION DOC] HELPER: CPU-heavy payload; modifies tensor in place in a loop (multiply and
    # divide), then returns 0.
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


def heavy_rpc_sparse(tensor):
    # [FUNCTION DOC] HELPER: Sparse variant of heavy_rpc; reassigns the tensor on each iteration
    # instead of in-place division, then returns 0.
    for i in range(1, 100):
        tensor *= i
        tensor = tensor / (i + 1)
    return 0


@torch.jit.script
def heavy_rpc_torchscript(tensor):
    # [FUNCTION DOC] HELPER: TorchScript version of heavy_rpc; loops over in-place multiply and
    # divide and returns 0.
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


@torch.jit.script
def my_script_func(tensor):
    # [FUNCTION DOC] HELPER: TorchScript function returning tensor plus tensor.
    return torch.add(tensor, tensor)


expected_err = "Expected error"


# Note that it needs to inherit from Exception, not BaseException. See comment
# in rpc/internal.py
class CustomException(Exception):
    # [CLASS DOC] HELPER: Custom exception type for testing exception propagation over RPC; stores
    # a boolean flag in attribute bool and passes msg to the base Exception. Must derive from
    # Exception rather than BaseException (see rpc/internal.py).
    def __init__(self, bool, msg):
        # [FUNCTION DOC] HELPER: Stores the flag bool and initializes the base exception with msg.
        self.bool = bool
        super().__init__(msg)


def raise_func():
    # [FUNCTION DOC] HELPER: Always raises ValueError with the module-level expected_err message.
    raise ValueError(expected_err)


def custom_raise_func():
    # [FUNCTION DOC] HELPER: Always raises CustomException with flag true and message foo.
    raise CustomException(True, "foo")


@torch.jit.script
def raise_func_script(expected_err: str) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: TorchScript function that always raises ValueError with the given
    # message string.
    raise ValueError(expected_err)


expected_err_escape = (
    "\nFirst line of error \n next line of error \n last line of error"
)


def raise_func_escape():
    # [FUNCTION DOC] HELPER: Always raises ValueError whose message has multiple lines
    # (expected_err_escape), to test message escaping.
    raise ValueError(expected_err_escape)


global_rref = None


def set_global_rref(rref):
    # [FUNCTION DOC] HELPER: Remote payload that stores rref in the module-level global_rref.
    global global_rref
    global_rref = rref


def clear_global_rref():
    # [FUNCTION DOC] HELPER: Remote payload that resets the module-level global_rref to None.
    global global_rref
    global_rref = None


def check_rref_confirmed(rref):
    # [FUNCTION DOC] HELPER: Returns whether the owner has confirmed the given
    # RRef (confirmed_by_owner).
    return rref.confirmed_by_owner()


def get_rref_debug_info():
    # [FUNCTION DOC] HELPER: Returns the RRef context debug info mapping of the current process.
    return _rref_context_get_debug_info()


def add_use_future_cb(to, x, y, z):
    # [FUNCTION DOC] HELPER: Runs an async RPC of torch.add(x, y) to the target, attaches a
    # callback that adds z to the result, and blocks on a standard-library future to return the
    # final value.
    out = concurrent.futures.Future()

    def callback(fut):
        # [FUNCTION DOC] HELPER: Callback; resolves the outer standard-library future with the
        # waited value of fut plus z.
        out.set_result(fut.wait() + z)

    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(callback)
    return out.result()


def get_events_from_profile(profile_rref):
    # [FUNCTION DOC] HELPER: Returns the process-global function events of the profiler held by the
    # local value of profile_rref.
    return profile_rref.local_value().process_global_function_events


def add_use_future_set_result(to, x, y, z):
    # [FUNCTION DOC] HELPER: Runs an async RPC of torch.add(x, y), chains a callback that sets a
    # torch future to the result plus z, and returns that future's waited value.
    out = torch.futures.Future()
    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(lambda fut: out.set_result(fut.wait() + z))
    return out.wait()


def add_use_future_nested_cb(to, x, y, z):
    # [FUNCTION DOC] HELPER: Like add_use_future_cb but nested: the callback issues a second async
    # RPC adding z to the first result; returns the final waited value from a torch future.
    out = torch.futures.Future()

    def callback(fut1):
        # [FUNCTION DOC] HELPER: Callback for the first RPC; launches a second async RPC adding z
        # and chains a callback that resolves the outer future.
        fut2 = rpc.rpc_async(to, torch.add, args=(fut1.wait(), z))
        fut2.then(lambda fut2: out.set_result(fut2.wait()))

    fut1 = rpc.rpc_async(to, torch.add, args=(x, y))
    fut1.then(callback)
    return out.wait()


def fail_on_fut(fut):
    # [FUNCTION DOC] HELPER: No-op callback that ignores its future.
    pass


@rpc.functions.async_execution
def async_raise_func():
    # [FUNCTION DOC] HELPER: async_execution function that raises RuntimeError with message
    # Expected error before returning a future.
    raise RuntimeError("Expected error")


@rpc.functions.async_execution
def async_wrong_type():
    # [FUNCTION DOC] HELPER: async_execution function that wrongly returns a tensor instead of a
    # future, to test the type check.
    return torch.zeros(2, 2)


@rpc.functions.async_execution
def async_add(to, x, y):
    # [FUNCTION DOC] HELPER: async_execution function that returns the future of an async RPC
    # computing torch.add(x, y) on the target.
    return rpc.rpc_async(to, torch.add, args=(x, y))


def slow_add(x, y, device="cpu"):
    # [FUNCTION DOC] HELPER: Sleeps 1 second, moves x and y to the given device, adds them and
    # returns the sum on CPU.
    time.sleep(1)
    x = x.to(device)
    y = y.to(device)
    return torch.add(x, y).cpu()


@rpc.functions.async_execution
def slow_async_add(to, x, y, device="cpu"):
    # [FUNCTION DOC] HELPER: async_execution function that returns the future of an async RPC
    # running slow_add on the target.
    return rpc.rpc_async(to, slow_add, args=(x, y, device))


@rpc.functions.async_execution
def async_add_with_future_ctor(to, x, y, z):
    # [FUNCTION DOC] HELPER: async_execution function that creates a future by constructor, chains
    # an async add RPC whose callback resolves it with the sum plus z, and returns it.
    fut = torch.futures.Future()
    rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut1: fut.set_result(fut1.wait() + z)
    )
    return fut


@rpc.functions.async_execution
def async_add_chained(to, x, y, z):
    # [FUNCTION DOC] HELPER: async_execution function returning a future of an async add RPC
    # chained with a callback adding z.
    return rpc.rpc_async(to, torch.add, args=(x, y)).then(lambda fut: fut.wait() + z)


@rpc.functions.async_execution
def async_add_chained_multi(to, x, num, step):
    # [FUNCTION DOC] HELPER: async_execution function that starts an async add of x and zero, then
    # chains num callbacks each adding step; returns the final future.
    fut = rpc.rpc_async(to, torch.add, args=(x, 0))
    for _ in range(num):
        fut = fut.then(lambda fut: fut.wait() + step)
    return fut


@rpc.functions.async_execution
def async_add_nested(to, x, y, z):
    # [FUNCTION DOC] HELPER: async_execution function that calls async_add on the target through an
    # RPC and chains a callback adding z; returns that future.
    return rpc.rpc_async(to, async_add, args=(to, x, y)).then(
        lambda fut: fut.wait() + z
    )


@rpc.functions.async_execution
def async_add_multi_fanout(to, x, num, step):
    # [FUNCTION DOC] HELPER: async_execution function that fans out num async add RPCs (the first
    # adds x and step, the others add zero and step), accumulates the replies under a lock into a
    # shared state, and returns a future resolved once all replies have arrived.
    futs = []
    for i in range(num):
        if i == 0:
            futs.append(rpc.rpc_async(to, torch.add, args=(x, step)))
        else:
            futs.append(rpc.rpc_async(to, torch.add, args=(0, step)))

    # TODO: use torch.futures.collect_all
    lock = Lock()
    state = {"cnt": 0, "ret": torch.zeros_like(x)}
    ret_future = torch.futures.Future()

    def inc_and_set(fut):
        # [FUNCTION DOC] HELPER: Callback; under the lock, counts the reply and adds its value into
        # the shared total, resolving ret_future when all futures have arrived.
        with lock:
            state["cnt"] += 1
            state["ret"] += fut.wait()
            if state["cnt"] >= len(futs):
                ret_future.set_result(state["ret"])

    for fut in futs:
        fut.then(inc_and_set)

    return ret_future


@rpc.functions.async_execution
def async_cuda_sleep_and_set_to_one(t):
    # [FUNCTION DOC] HELPER: async_execution CUDA function; on a new stream waiting for the current
    # one, sleeps about one second of GPU time, fills t with ones, and returns a future (tracking
    # the device) already resolved with t.
    device = t.device
    original_stream = torch.cuda.current_stream(device)
    new_stream = torch.cuda.Stream(device)
    new_stream.wait_stream(original_stream)
    with torch.cuda.stream(new_stream):
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        t.fill_(1)
        fut = Future(devices=[device])
        fut.set_result(t)
        return fut


@rpc.functions.async_execution
def async_cuda_nested_add(to, x, y, z):
    # [FUNCTION DOC] HELPER: async_execution CUDA function that returns a chained future of an
    # async add RPC; the callback sleeps on the GPU for about one second and returns the value
    # plus z.
    def cb(fut):
        # [FUNCTION DOC] HELPER: Callback; sleeps on the GPU for about one second and returns the
        # future value plus z.
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        return fut.value() + z

    return rpc.rpc_async(to, torch.add, args=(x, y)).then(cb)


# A custom Python class that contains a tensor, needed to see if we correctly
# use the Python pickler to extract tensors from non-IValue-convertible types.
class TensorWrapper:
    # [CLASS DOC] HELPER: Python class wrapping a tensor, used to check that tensors are extracted
    # by the Python pickler from non-IValue-convertible types. Slots: tensor, lock, event (a CUDA
    # event) and thread (a started thread, a non-picklable field that must be skipped).
    __slots__ = ("tensor", "lock", "event", "thread")

    def __init__(self, t):
        # [FUNCTION DOC] HELPER: Stores the tensor and creates the non-picklable lock, CUDA timing
        # event and started thread. Side effect: starts a thread.
        self.tensor = t
        # Add one non-picklable field, to ensure it's ignored/skipped.
        self.lock = Lock()
        self.event = torch.cuda.Event(enable_timing=True)
        self.thread = threading.Thread()
        self.thread.start()

    def increase(self, v):
        # [FUNCTION DOC] HELPER: Adds v to the wrapped tensor in place under the lock.
        with self.lock:
            self.tensor += v

    def sum(self):
        # [FUNCTION DOC] HELPER: Records the CUDA event under the lock and returns the sum of
        # the tensor.
        with self.lock:
            self.event.record()
            return self.tensor.sum()


class AsyncExecutionClass:
    # [CLASS DOC] HELPER: Holder of async_execution functions in static, class and bound form, used
    # to test decorator ordering with those method kinds.
    @staticmethod
    @rpc.functions.async_execution
    def static_async_add(to, x, y, z):
        # [FUNCTION DOC] HELPER: Static async_execution method; returns a future of an async add
        # RPC plus z.
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )

    @classmethod
    @rpc.functions.async_execution
    def class_async_add(cls, to, x, y, z):
        # [FUNCTION DOC] HELPER: Class async_execution method; builds a future that is resolved by
        # a callback with the async add result plus z, and returns it.
        ret_fut = torch.futures.Future()
        rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: ret_fut.set_result(fut.wait() + z)
        )
        return ret_fut

    @rpc.functions.async_execution
    def bound_async_add(self, to, x, y, z):
        # [FUNCTION DOC] HELPER: Bound async_execution method; returns a future of an async add RPC
        # plus z.
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )


def return_future():
    # [FUNCTION DOC] HELPER: Returns a new unresolved torch future.
    return torch.futures.Future()


class FooBackendOptions(rpc.RpcBackendOptions):
    # [CLASS DOC] HELPER: Custom RpcBackendOptions subclass for the fake Foo backend;
    # holds init_method.
    def __init__(self, init_method):
        # [FUNCTION DOC] HELPER: Calls the superclass initializer directly (not through super, due
        # to pybind) and stores init_method.
        # Must call the __init__ of the superclass (and do so directly,
        # without using super()) because... pybind.
        rpc.RpcBackendOptions.__init__(self)
        self.init_method = init_method


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class MyEmbeddingBagModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module holding an EmbeddingBag with 10 embeddings of size 10,
    # optionally sparse, for parameter-server tests.
    def __init__(self, sparse):
        # [FUNCTION DOC] HELPER: Creates the embedding bag with the given sparse flag.
        super().__init__()
        self.eb = torch.nn.EmbeddingBag(10, 10, sparse=sparse)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the embedding bag output for input x.
        return self.eb(x)


class MyParameterServer:
    # [CLASS DOC] HELPER: Simple parameter server used with RRefs; keeps a lock, the number
    # of trainers, current iteration, update count, pending futures, running total and
    # averaged gradient.
    def __init__(self, trainers):
        # [FUNCTION DOC] HELPER: Initializes the lock, trainer count and the iteration, update,
        # futures, total and gradient state.
        self.lock = Lock()
        self.trainers = trainers
        self.iteration = 0
        self.updates = 0
        self.futures = []
        self.total = None
        self.gradient = None

    @staticmethod
    def get_gradient(rref):
        # [FUNCTION DOC] HELPER: Static method returning the latest averaged gradient of the server
        # held by rref.
        return rref.local_value().gradient

    @staticmethod
    @rpc.functions.async_execution
    def average(rref, riteration, tensor):
        # [FUNCTION DOC] HELPER: Static async_execution method; under the lock, resets state when a
        # newer iteration begins, adds the trainer's tensor to the total, and once all trainers
        # have reported computes the average gradient and resolves every pending future with it.
        # Returns the future for this trainer.
        self = rref.local_value()
        fut = torch.futures.Future()
        with self.lock:
            if riteration > self.iteration:
                self.iteration = riteration
                self.updates = 0
                self.futures.clear()
            self.futures.append(fut)
            if self.total is None:
                self.total = tensor
            else:
                self.total += tensor
            self.updates += 1
            if self.trainers == self.updates:
                self.gradient = self.total / float(self.trainers)
                for fut in self.futures:
                    result = self.total / float(self.trainers)
                    fut.set_result(result)
        return fut


class MyConvNetForMNIST(nn.Module):
    # [CLASS DOC] HELPER: Small CNN for MNIST-shaped input used in CUDA RPC tests; its pickled
    # state is deliberately empty so the owner does not inspect model contents.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Builds a sequential convolution network on the given device and
        # remembers the device.
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)
        self.device = device

    def forward(self, x, is_rref=False):
        # [FUNCTION DOC] HELPER: Forward pass; if is_rref is true, fetches x from an RRef; runs the
        # network on the current stream after intentionally sleeping on the GPU. Returns the
        # network output.
        x = x.to_here() if is_rref else x
        with torch.cuda.stream(torch.cuda.current_stream(self.device)):
            # intentionally adding delay to current CUDA stream
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            return self.net(x)

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickling hook returning an empty dict so model contents are not
        # inspected on the owner.
        # return an empty dict to avoid inspecting the model contents on the
        # owner
        return {}


class RpcTestCommon:
    # [CLASS DOC] HELPER: Mixin with shared helper bodies for RPC tests (call dispatch by mode,
    # remote call patterns, workload drivers, parameter-server flow, CUDA future extraction).
    # Expects the host test class to provide rank, world_size, rpc_backend, rpc_backend_options,
    # file_init_method and the unittest assertions.
    def _run_func_in_mode(self, to, fn, mode, args=None, kwargs=None):
        # [FUNCTION DOC] HELPER: Calls fn on worker to with the given args and kwargs using the
        # execution mode (sync, async, or remote) and returns the result value, waiting on the
        # future or RRef as needed. Returns None for an unrecognized mode.
        if mode == RPCExecMode.SYNC:
            return rpc.rpc_sync(to, fn, args=args, kwargs=kwargs)
        elif mode == RPCExecMode.ASYNC:
            return rpc.rpc_async(to, fn, args=args, kwargs=kwargs).wait()
        elif mode == RPCExecMode.REMOTE:
            return rpc.remote(to, fn, args=args, kwargs=kwargs).to_here()

    def _self_py_udf_remote(self, worker_info, x, y, z):
        # [FUNCTION DOC] HELPER: Creates an RRef on worker_info running my_function and asserts
        # that its fetched value equals x plus y plus z.
        rref = rpc.remote(worker_info, my_function, args=(x, y, z))
        self.assertEqual(rref.to_here(), x + y + z)

    def _self_remote_rref_as_rpc_arg(self, dst, x, y, z):
        # [FUNCTION DOC] HELPER: Creates an RRef owned by the current worker and passes it to dst
        # as an argument of sync and async RPCs; asserts both results.
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        fut = rpc.rpc_async(dst, add_rref_to_value, args=(rref, x))
        ret = rpc.rpc_sync(dst, add_rref_to_value, args=(rref, x + y))
        self.assertEqual(ret, x + y + z + x + y)
        self.assertEqual(fut.wait(), x + y + z + x)

    def _self_remote_rref_as_remote_arg(self, dst, x, y, z):
        # [FUNCTION DOC] HELPER: Creates an RRef owned by the current worker, passes it to dst via
        # a remote call, and asserts the fetched result.
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        ret_rref = rpc.remote(dst, add_rref_to_value, args=(rref, x))
        self.assertEqual(ret_rref.to_here(), x + y + z + x)

    def _world_size_one(self, a, b):
        # [FUNCTION DOC] HELPER: On rank 0 only, initializes RPC with a world size of one and
        # checks sync, async and remote calls to itself each return a * 2 (given matching a and b),
        # then shuts RPC down.
        if self.rank == 0:
            rpc.init_rpc(
                name="me",
                backend=self.rpc_backend,
                rank=0,
                world_size=1,
                rpc_backend_options=self.rpc_backend_options,
            )

            def _rpc_sync(x, y):
                # [FUNCTION DOC] HELPER: Checks a sync RPC to the single worker returns twice x.
                expect = x * 2
                result = rpc.rpc_sync("me", my_tensor_function, args=(x, y))
                self.assertEqual(expect, result)

            def _rpc_async(x, y):
                # [FUNCTION DOC] HELPER: Checks an async RPC to the single worker returns twice x.
                expect = x * 2
                result = rpc.rpc_async("me", my_tensor_function, args=(x, y)).wait()
                self.assertEqual(expect, result)

            def _remote(x, y):
                # [FUNCTION DOC] HELPER: Checks a remote call to the single worker returns twice x.
                expect = x * 2
                result = rpc.remote("me", my_tensor_function, args=(x, y)).to_here()
                self.assertEqual(expect, result)

            _rpc_sync(a, b)
            _rpc_async(a, b)
            _remote(a, b)

            rpc.shutdown()

    def _multi_rpc(self, sparse):
        # [FUNCTION DOC] HELPER: Sends 20 sync add RPCs to the next rank, with dense or sparse
        # tensors depending on sparse, and asserts each result is twice x.
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            if sparse:
                x = build_sparse_tensor() * n
                y = build_sparse_tensor() * n
            else:
                x = torch.ones(2, 2)
                y = torch.ones(2, 2)
            ret = rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(x, y),
            )
            self.assertEqual(ret, x * 2)

    def _run_uneven_workload(self, f, x, num_repeat=30):
        # [FUNCTION DOC] HELPER: On rank 0 only, drives uneven load: asserts the world has at least
        # 3 workers, sends num_repeat async calls of f(x) to worker1 and checks they return 0, then
        # does the same for worker2. Other ranks do nothing.
        # worker0 drives and waits for worker1 and worker2
        # throughout the test.
        if self.rank == 0:
            self.assertTrue(self.world_size >= 3)

            # Phase 1: Only worker1 has workload.
            dst = "worker1"
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            for fut in torch.futures.collect_all(futs).wait():
                self.assertEqual(fut.wait(), 0)

            # Phase 2: Only worker2 has workload.
            # If join is not correctly implemented,
            # worker2 should be closed by now.
            dst = "worker2"
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            for val in torch.futures.wait_all(futs):
                self.assertEqual(val, 0)

    def _wait_all_workers(self, f, x):
        # [FUNCTION DOC] HELPER: Initializes a process group and RPC, runs the uneven workload,
        # calls the internal wait-all-workers once, hits a barrier, and shuts down non-gracefully.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload(f, x)

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some work after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()

        # Wait before proceeding to shutdown to ensure worker0 RPCs make
        # it through to other workers.
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _wait_all_workers_twice(self, f, x):
        # [FUNCTION DOC] HELPER: Same as _wait_all_workers but calls the internal wait-all-workers
        # twice to check it is repeatable.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload(f, x)

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some work after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()
        rpc.api._wait_all_workers()

        # Wait before proceeding to shutdown to ensure worker0 RPCs make
        # it through to other workers.
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _nested_rpc(self, f, expected):
        # [FUNCTION DOC] HELPER: Calls f on the next rank with the current worker name as argument
        # and asserts the result equals expected.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            f,
            args=(worker_name(self.rank),),
        )
        self.assertEqual(ret, expected)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        # [FUNCTION DOC] HELPER: Sends repeat async calls of f to the next rank, waits for all,
        # asserts each returns 0, and prints the elapsed time. Side effect: prints to stdout.
        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = rpc.rpc_async(worker_name(dst_rank), f, args=args)
            futs.append(fut)

        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, 0)
        tok = time.time()
        print(
            f"Rank {self.rank} finished testing {repeat} times in {tok - tik} seconds."
        )

    def _builtin_remote_ret(self, x, y, expected):
        # [FUNCTION DOC] HELPER: Calls torch.add remotely on the next rank and asserts the fetched
        # value equals expected.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(x, y),
        )
        self.assertEqual(rref.to_here(), expected)

    def _builtin_remote_self(self, x, y, expected):
        # [FUNCTION DOC] HELPER: Calls torch.add remotely on the current worker and asserts the
        # local value equals expected.
        rref = rpc.remote(
            worker_name(self.rank),
            torch.add,
            args=(x, y),
        )
        self.assertEqual(rref.local_value(), expected)

    def _test_multi_remote_call(
        self, fn, sparse, args_fn=lambda x, y: (), kwargs_fn=lambda x, y: {}
    ):
        # [FUNCTION DOC] HELPER: Creates 10 remote calls of fn on the next rank with args and
        # kwargs produced by args_fn and kwargs_fn, computes the expected values locally, and
        # asserts each fetched value matches.
        m = 10
        n = self.rank + 1
        dst_rank = n % self.world_size
        rrefs = []
        expected = []
        for i in range(m):
            n = n + i
            rrefs.append(
                rpc.remote(
                    worker_name(dst_rank),
                    fn,
                    args=args_fn(n, sparse),
                    kwargs=kwargs_fn(n, sparse),
                )
            )
            expected.append(fn(*args_fn(n, sparse), **kwargs_fn(n, sparse)))

        for i in range(m):
            self.assertEqual(rrefs[i].to_here(), expected[i])

    def _py_rref_args(self, a, b, x, y, expected):
        # [FUNCTION DOC] HELPER: Creates two RRefs (adds) on the next rank, passes them to a third
        # remote call of my_rref_function, and asserts the fetched sum equals expected.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), torch.add, args=(a, b))
        rref_b = rpc.remote(worker_name(dst_rank), torch.add, args=(x, y))
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rref_args_user_share(self, a, b, c, x, y, z, expected):
        # [FUNCTION DOC] HELPER: Like _py_rref_args, but the RRefs are owned by one rank and the
        # combining call runs on another (a user) rank, testing RRef sharing.
        n = self.rank + 1
        owner_rank = n % self.world_size
        user_rank = (n + 1) % self.world_size
        rref_a = rpc.remote(worker_name(owner_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(owner_rank), my_function, args=(x, y, z))
        rref_c = rpc.remote(
            worker_name(user_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rpc_rref_args(self, a, b, c, x, y, z, expected):
        # [FUNCTION DOC] HELPER: Creates two RRefs on the next rank and passes them as arguments to
        # a sync RPC of my_rref_function; asserts the result equals expected.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(dst_rank), my_function, args=(x, y, z))

        c = rpc.rpc_sync(worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(c, expected)

    def _nested_remote(self, f, expected):
        # [FUNCTION DOC] HELPER: Calls f remotely on one rank with a second rank name as argument
        # (nested remote); asserts the fetched value equals expected.
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size

        rref = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), expected)

    def _nested_rref(self, f, expected1, expected2):
        # [FUNCTION DOC] HELPER: Asks one rank to run f, producing two RRefs owned by a second
        # rank; fetches the RRef list and asserts its length and each fetched value against
        # expected1 and expected2.
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref_of_rrefs = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )

        # Say C has 2 OwnerRRefs.
        # B has 2 UserRRefs to those 2 OwnerRRefs, respectively.
        # This call is effectively A asking B to share its 2 UserRRefs.
        rrefs = rref_of_rrefs.to_here()

        self.assertEqual(len(rrefs), 2)
        self.assertEqual(rrefs[0].to_here(), expected1)
        self.assertEqual(rrefs[1].to_here(), expected2)

    def _nested_rref_stress(self, f, expected1, expected2):
        # [FUNCTION DOC] HELPER: Stress version of _nested_rref: creates 20 RRef-of-RRefs and
        # checks the two nested values of each.
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        all_rrefs = [
            rpc.remote(
                worker_name(dst_rank1),
                f,
                args=(worker_name(dst_rank2),),
            )
            for _ in range(20)
        ]

        for i in range(20):
            rref_of_rrefs = all_rrefs[i]
            rrefs = rref_of_rrefs.to_here()
            self.assertEqual(len(rrefs), 2)
            self.assertEqual(rrefs[0].to_here(), expected1)
            self.assertEqual(rrefs[1].to_here(), expected2)

    def _trainer_func(self, rref, sparse):
        # [FUNCTION DOC] HELPER: Trainer body for the parameter server: for 10 iterations computes
        # an embedding bag gradient, submits it to the parameter server through the RRef proxy
        # async average call, and asserts the averaged gradient equals the server's stored gradient
        # (converting sparse results to dense).
        m = MyEmbeddingBagModel(sparse=sparse)
        loss_fn = nn.MSELoss()
        for i in range(10):
            outputs = m(torch.rand(10, 10).long())
            loss_fn(outputs, torch.rand(10, 10)).backward()
            gradient = next(iter(m.parameters())).grad
            fut = rref.rpc_async().average(rref, i, gradient)
            gradient = fut.wait()
            if gradient.is_sparse:
                gradient = gradient.to_dense().double()
            ps_gradient = rref.rpc_sync().get_gradient(rref)
            if ps_gradient.is_sparse:
                ps_gradient = ps_gradient.to_dense().double()
            self.assertTrue(torch.equal(gradient, ps_gradient))

    def _my_parameter_server(self, sparse):
        # [FUNCTION DOC] HELPER: Creates a parameter server RRef and runs the trainer function on
        # every other rank via async RPC; waits for all futures.
        ps_rref = RRef(MyParameterServer(self.world_size - 1))
        futures = [
            rpc.rpc_async(
                worker_name((self.rank + index) % self.world_size),
                self._trainer_func,
                args=(ps_rref, sparse),
            )
            for index in range(1, self.world_size)
        ]
        torch.futures.wait_all(futures)

    def _test_cuda_future_extraction(self, wrapper, unwrapper, sparse_tensor):
        # [FUNCTION DOC] HELPER: Checks CUDA stream synchronization of futures: modifies a
        # tensor on one stream, wraps it with wrapper and sets it as the future result, then, on
        # another stream, unwraps the waited value and asserts it matches the expected tensor
        # (dense or sparse).
        # We check proper CUDA stream synchronization by adding to the tensor
        # in one stream to get the expected value, and reading it from another stream.
        future = Future(devices=["cuda:0"])
        with torch.cuda.device("cuda:0"):
            stream = torch.cuda.Stream()
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                if sparse_tensor:
                    tensor = build_sparse_tensor().to("cuda:0")
                    add_tensor = build_sparse_tensor().to("cuda:0")
                    expected_tensor = (tensor + add_tensor).coalesce()
                else:
                    tensor = torch.zeros((100,), device="cuda:0")
                    add_tensor = torch.ones((100,), device="cuda:0")
                    expected_tensor = tensor + add_tensor
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor += add_tensor
                if sparse_tensor:
                    tensor = tensor.coalesce()
                future.set_result(wrapper(tensor))
            with torch.cuda.stream(another_stream):
                tensor = unwrapper(future.wait())
                if sparse_tensor:
                    self.assertTrue(
                        torch.eq(tensor.indices(), expected_tensor.indices())
                        .all()
                        .item()
                    )
                    self.assertTrue(
                        torch.eq(tensor.values(), expected_tensor.values()).all().item()
                    )
                    self.assertEqual(tensor.size(), expected_tensor.size())
                else:
                    self.assertTrue(torch.eq(tensor, expected_tensor).all().item())


class RpcTest(RpcAgentTestFixture, RpcTestCommon):
    # [CLASS DOC] Core RPC test class mixing RpcAgentTestFixture with RpcTestCommon; covers worker
    # info, init and shutdown, sync/async/remote calls, RRef proxies, barriers, timeouts and
    # profiling over any backend.
    @dist_init
    def test_worker_id(self):
        # [FUNCTION DOC] Checks get_worker_info returns the correct names for self and peer, and
        # raises RuntimeError for an unknown worker name.
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_info = rpc.get_worker_info()
        peer_worker_info = rpc.get_worker_info(worker_name(peer_rank))

        self.assertEqual(self_worker_info.name, worker_name(self.rank))
        self.assertEqual(peer_worker_info.name, worker_name(peer_rank))

        with self.assertRaisesRegex(RuntimeError, "could not find destination"):
            rpc.get_worker_info("WorkerUnknown")

    @dist_init
    def test_get_worker_infos(self):
        # [FUNCTION DOC] Checks that the agent reports worker infos whose names and ids equal the
        # expected sets for all ranks.
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()

        worker_names = {worker_info.name for worker_info in worker_infos}
        expected_worker_names = {worker_name(rank) for rank in range(self.world_size)}
        self.assertEqual(worker_names, expected_worker_names)

        worker_ids = {worker_info.id for worker_info in worker_infos}
        expected_worker_ids = set(range(self.world_size))
        self.assertEqual(worker_ids, expected_worker_ids)

    @dist_init
    def test_self_add(self):
        # [FUNCTION DOC] Calls torch.add on the current worker through both sync and async RPC and
        # checks the results.
        self_worker_info = rpc.get_worker_info()
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_send_to_rank(self):
        # [FUNCTION DOC] Sends add calls to the next rank by integer rank in all modes, and
        # checks that an out-of-range or negative rank raises RuntimeError and a non-integer rank
        # raises ValueError.
        dst_rank = (self.rank + 1) % self.world_size

        # Test dense tensor
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            ret = self._run_func_in_mode(
                dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
            )
            self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test invalid ranks
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(
                    self.world_size + 1,
                    torch.add,
                    exec_mode,
                    args=(torch.ones(2, 2), 1),
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(
                    -1, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(
                    dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(
                    dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

    @dist_init
    def test_self_py_udf_remote(self):
        # [FUNCTION DOC] Runs the self remote UDF helper on the current worker.
        self._self_py_udf_remote(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_rpc_arg(self):
        # [FUNCTION DOC] Runs the RRef-as-RPC-argument helper against the next rank.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg(self):
        # [FUNCTION DOC] Runs the RRef-as-RPC-argument helper with the current worker
        # as destination.
        self._self_remote_rref_as_rpc_arg(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_remote_arg(self):
        # [FUNCTION DOC] Runs the RRef-as-remote-argument helper against the next rank.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_remote_arg(self):
        # [FUNCTION DOC] Runs the RRef-as-remote-argument helper with the current worker
        # as destination.
        self._self_remote_rref_as_remote_arg(
            rpc.get_worker_info(), torch.ones(2, 2), 1, 3
        )

    @dist_init
    def test_rref_proxy_non_exist(self):
        # [FUNCTION DOC] Checks that calling a non-existent method through the sync, async and
        # remote RRef proxies raises AttributeError.
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        msg = "has no attribute 'non_exist'"
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_sync().non_exist()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_async().non_exist().wait()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.remote().non_exist()

    def _test_rref_proxy_tensor(self, dst):
        # [FUNCTION DOC] HELPER: Creates an RRef of a tensor on dst and checks proxy calls (size,
        # add, view) through the sync, async and remote proxies.
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))

        expected = torch.ones(2, 2) + 1 + 3
        self.assertEqual(expected.size(), rref.rpc_sync().size())
        self.assertEqual(expected + 1, rref.rpc_async().add(1).wait())
        self.assertEqual(expected.view(1, 4), rref.remote().view(1, 4).to_here())

    @dist_init
    def test_rref_proxy_tensor(self):
        # [FUNCTION DOC] Runs the RRef tensor proxy check against the next rank.
        self._test_rref_proxy_tensor(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_tensor_self(self):
        # [FUNCTION DOC] Runs the RRef tensor proxy check against the current worker.
        self._test_rref_proxy_tensor(rpc.get_worker_info())

    @dist_init
    def test_rref_proxy_reuse(self):
        # [FUNCTION DOC] Checks that a single RRef can create its sync, async and remote proxies
        # once and reuse them for multiple method calls.
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            my_function,
            args=(torch.ones(2, 2), 1, 3),
        )
        expected = torch.ones(2, 2) + 1 + 3

        proxy_rpc_sync = rref.rpc_sync()
        proxy_rpc_async = rref.rpc_async()
        proxy_remote = rref.remote()

        self.assertEqual(expected.size(), proxy_rpc_sync.size())
        self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
        self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))

        self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
        self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
        self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())

        self.assertEqual(expected.size(), proxy_remote.size().to_here())
        self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
        self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())

    def _test_rref_proxy_class(self, dst):
        # [FUNCTION DOC] HELPER: Creates an RRef of MyClass on dst and checks proxy calls to
        # instance, static, class methods and a mutating method through all three proxies.
        rref = rpc.remote(dst, MyClass, args=(7,))
        expected = MyClass(7)
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        expected.increment_value(3)
        self.assertEqual(None, rref.rpc_sync().increment_value(1))
        self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
        self.assertEqual(None, rref.remote().increment_value(1).to_here())

        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        self.assertEqual(
            expected.my_instance_method(2), rref.rpc_sync().my_instance_method(2)
        )
        self.assertEqual(
            expected.my_instance_method(3),
            rref.rpc_async().my_instance_method(3).wait(),
        )
        self.assertEqual(
            expected.my_instance_method(4),
            rref.remote().my_instance_method(4).to_here(),
        )

        self.assertEqual(
            expected.my_static_method(9), rref.rpc_sync().my_static_method(9)
        )
        self.assertEqual(
            expected.my_static_method(10), rref.rpc_async().my_static_method(10).wait()
        )
        self.assertEqual(
            expected.my_static_method(11), rref.remote().my_static_method(11).to_here()
        )

        self.assertEqual(
            expected.my_class_method(2, torch.zeros(2, 2)),
            rref.rpc_sync().my_class_method(2, torch.zeros(2, 2)),
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(3, 3)),
            rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait(),
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(4, 4)),
            rref.remote().my_class_method(2, torch.ones(4, 4)).to_here(),
        )

    @dist_init
    def test_rref_proxy_class(self):
        # [FUNCTION DOC] Runs the RRef class proxy check against the next rank.
        self._test_rref_proxy_class(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_class_self(self):
        # [FUNCTION DOC] Runs the RRef class proxy check against the current worker.
        self._test_rref_proxy_class(rpc.get_worker_info())

    @mock.patch.object(torch.distributed.autograd, "_init")
    @mock.patch.object(torch.distributed.rpc.api, "_set_and_start_rpc_agent")
    @dist_init(setup_rpc=False)
    def test_register_rpc_backend_and_set_and_start_rpc_backend(
        self, mock_rpc_agent, mock_dist_autograd_init
    ):
        # [FUNCTION DOC] Registers a stub backend with the backend registry (checking that a
        # duplicate registration raises RuntimeError) and initializes RPC with it. The agent start
        # and autograd init are mocked (mock_rpc_agent and mock_dist_autograd_init).
        backend_name = "stub_backend"

        backend = rpc.backend_registry.register_backend(
            backend_name,
            _stub_construct_rpc_backend_options_handler,
            _stub_init_rpc_backend_handler,
        )

        with self.assertRaisesRegex(
            RuntimeError, "^RPC backend .+: already registered$"
        ):
            backend = rpc.backend_registry.register_backend(
                backend_name,
                _stub_construct_rpc_backend_options_handler,
                _stub_init_rpc_backend_handler,
            )

        rpc.init_rpc(
            name="worker1",
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        # [FUNCTION DOC] Checks that initializing the backend with a name already used by another
        # worker raises RuntimeError about non-unique names.
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc._init_rpc_backend(
                backend=self.rpc_backend,
                store=store,
                name="duplicate_name",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_duplicate_name_2(self):
        # [FUNCTION DOC] Checks that init_rpc with a name colliding with another rank raises
        # RuntimeError about non-unique names.
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            rpc.init_rpc(
                name=worker_name(self.rank % (self.world_size - 1)),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_reinit(self):
        # [FUNCTION DOC] Checks that calling init_rpc twice raises RuntimeError (address in use on
        # rank 0 under TCP init, otherwise already initialized), then shuts down.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        # TODO: with TCP init, rank 0 raises Address already in use because
        # rank 0 is the start daemon and the store is created before checking if
        # RPC is already initialized in init_rpc.
        if os.environ.get("RPC_INIT_WITH_TCP", None) == "1" and self.rank == 0:
            expected_reinit_err = "Address already in use"
        else:
            expected_reinit_err = "is already initialized"

        with self.assertRaisesRegex(RuntimeError, expected_reinit_err):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_pg_init_no_rpc_init(self):
        # [FUNCTION DOC] Initializes only a process group (no RPC), wraps a model in
        # DistributedDataParallel, and checks that creating RRefs of its parameters raises
        # RuntimeError because the RPC agent is not set.
        dist.init_process_group(
            backend="gloo",
            init_method=self.file_init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module with one linear layer (3 inputs, 4 outputs).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.lin = torch.nn.Linear(3, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                return self.lin(x)

        model = MyModel()
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "Current RPC agent is not set! Did you initialize the RPC framework",
        ):
            [RRef(param) for param in model.parameters()]

    def test_world_size_one(self):
        # [FUNCTION DOC] Runs the world-size-one helper on two ones tensors.
        self._world_size_one(torch.ones(2, 2), torch.ones(2, 2))

    @dist_init(setup_rpc=False)
    def test_invalid_names(self):
        # [FUNCTION DOC] Checks that WorkerInfo construction raises RuntimeError for invalid names
        # (bad characters, blank, empty, too long).
        worker_id = 0
        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            WorkerInfo("abc*", worker_id)

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            WorkerInfo(" ", worker_id)

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            WorkerInfo("", worker_id)

        # If the number in the message does not match, it is likely that the
        # value of MAX_NAME_LEN in RPC WorkerInfo has changed.
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            WorkerInfo("".join(["a" for i in range(500)]), worker_id)

    # Test that WorkerInfo can be pickled and sent in RPC call
    @dist_init
    def test_worker_info_pickle(self):
        # [FUNCTION DOC] Checks that a WorkerInfo survives pickling when sent through an RPC call.
        dst_rank = (self.rank + 1) % self.world_size
        worker_info = rpc.api.get_worker_info()
        ret = rpc.rpc_sync(worker_name(dst_rank), identity, args=(worker_info,))
        self.assertEqual(ret, worker_info)

    @dist_init
    def test_add(self):
        # [FUNCTION DOC] Checks sync RPC add of two ones tensors on the next rank.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @staticmethod
    def return_callee_id():
        # [FUNCTION DOC] HELPER: Static method returning the id of the current worker.
        return rpc.get_worker_info().id

    @dist_init
    def test_int_callee(self):
        # [FUNCTION DOC] Checks an RPC to a destination given as an integer rank runs on that rank.
        dst_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(dst_rank, RpcTest.return_callee_id)
        self.assertEqual(ret, dst_rank)

    @dist_init
    def test_add_with_id(self):
        # [FUNCTION DOC] Checks sync RPC add with the destination given as a WorkerInfo.
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_info = rpc.get_worker_info(worker_name(dst_rank))

        ret = rpc.rpc_sync(
            workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        # [FUNCTION DOC] Checks sync RPC add of a tensor and a scalar.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), n))
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @dist_init
    def test_async_add(self):
        # [FUNCTION DOC] Checks async RPC add of two ones tensors.
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_nonzero(self):
        # [FUNCTION DOC] Checks RPC of torch.nonzero on a tensor with one zero entry.
        n = self.rank + 1
        dst_rank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        # [FUNCTION DOC] Runs the dense multi-RPC helper.
        self._multi_rpc(False)

    @dist_init
    def test_future_wait_twice(self):
        # [FUNCTION DOC] Checks that waiting on futures that failed with an error raises the same
        # error both from wait_all and again on each individual wait.
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = [rpc.rpc_async(dst, raise_func) for _ in range(20)]

        with self.assertRaisesRegex(ValueError, "Expected error"):
            torch.futures.wait_all(futs)

        for fut in futs:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()

    @dist_init(setup_rpc=False)
    def test_wait_all_workers_timeout(self):
        # [FUNCTION DOC] Checks that a graceful shutdown with a tiny timeout raises RuntimeError
        # when the wait-all-workers step is replaced by a slow version; restores the original
        # function afterward. Side effect: temporarily patches rpc.api.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        og_func = rpc.api._wait_all_workers

        def wait_all_workers_sleep(timeout):
            # [FUNCTION DOC] HELPER: Replacement for wait-all-workers; performs an all-gather of a
            # slow-to-pickle object with the given timeout.
            rpc.api._all_gather(SlowPickleClass(0.5), timeout=timeout)

        rpc.api._wait_all_workers = wait_all_workers_sleep

        try:
            with self.assertRaisesRegex(RuntimeError, ""):
                rpc.shutdown(graceful=True, timeout=0.01)
        finally:
            rpc.api._wait_all_workers = og_func
        dist.barrier()

    def test_wait_all_workers_dense(self):
        # [FUNCTION DOC] Runs the wait-all-workers helper with the dense heavy workload.
        self._wait_all_workers(heavy_rpc, torch.ones(100, 100))

    def test_wait_all_workers_twice_dense(self):
        # [FUNCTION DOC] Runs the wait-all-workers-twice helper with the dense heavy workload.
        self._wait_all_workers_twice(heavy_rpc, torch.ones(100, 100))

    @dist_init
    def test_all_gather(self):
        # [FUNCTION DOC] Checks the internal all-gather returns a mapping from worker name to id
        # for all workers.
        info = rpc.get_worker_info()
        results = rpc.api._all_gather(info.id)
        expected = {}
        for info in rpc._get_current_rpc_agent().get_worker_infos():
            expected[info.name] = info.id

        self.assertEqual(expected, results)

    @dist_init
    def test_all_gather_timeout(self):
        # [FUNCTION DOC] Sets a short RPC timeout and checks that all-gather of a slow-to-pickle
        # object times out: rank 0 expects the timed-out-in-all-gather error, other ranks expect
        # the backend timeout error. Side effect: changes the global RPC timeout.
        rpc._set_rpc_timeout(0.1)

        if self.rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "timed out in _all_gather after 0\\.10 seconds"
            ):
                rpc.api._all_gather(SlowPickleClass(0.5))
        else:
            expected_error = self.get_timeout_error_regex()
            with self.assertRaisesRegex(RuntimeError, expected_error):
                rpc.api._all_gather(SlowPickleClass(0.5))

    def _test_barrier_helper(self, info, names, multi_threaded=False):
        # [FUNCTION DOC] HELPER: Resets the barrier counter on the leader (lowest name), barriers,
        # increments the counter on the leader, barriers again, and on the leader (when single
        # threaded) asserts the counter equals the number of participants.
        names = sorted(names)
        leader = names[0]
        rpc.rpc_sync(leader, _reset_count)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, 0)
        rpc.api._barrier(names)
        rpc.rpc_sync(leader, _increment_count)
        rpc.api._barrier(names)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, len(names))

    @dist_init
    def test_rpc_barrier_all(self):
        # [FUNCTION DOC] Checks the RPC barrier when all workers participate.
        # Test rpc barrier when called with full list of workers
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_subset(self):
        # [FUNCTION DOC] Checks the RPC barrier when workers call it with different subsets (even
        # and odd ids).
        # Test rpc barrier when processes are called with different subsets of the full list
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [worker.name for worker in all_worker_info if not worker.id % 2]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_partial_subset(self):
        # [FUNCTION DOC] Checks the RPC barrier when some workers do not take part (even ids use
        # only themselves).
        # Test rpc barrier when some processes are not involved in the barrier
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [f"worker{info.id}"]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_multithreaded(self):
        # [FUNCTION DOC] Starts three threads that each call the barrier helper concurrently and
        # joins them; only checks that it does not hang.
        # This test validates the implementation of barrier when multiple threads call into it
        # We only need to check that it does not hang in this case
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        threads = []
        for _ in range(3):
            th = threading.Thread(
                target=self._test_barrier_helper, args=(info, names, True)
            )
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

    @dist_init
    def test_graceful_shutdown_with_uneven_workload(self):
        # [FUNCTION DOC] Runs the uneven workload helper with the heavy payload to check
        # graceful shutdown.
        """Test graceful termination."""
        self._run_uneven_workload(heavy_rpc, torch.ones(100, 100))

    @dist_init(setup_rpc=False)
    def test_shutdown_followed_by_rpc(self):
        # [FUNCTION DOC] Initializes RPC manually, performs an RPC, shuts down, and checks that a
        # further RPC raises RuntimeError that RPC has not been initialized.
        # Initialize RPC.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.shutdown()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

    @dist_init
    def test_expected_src(self):
        # [FUNCTION DOC] Checks that the callee observes the expected source rank value set via
        # an RPC.
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        self.assertEqual(value, expected_src_rank)

    @dist_init
    def test_py_built_in(self):
        # [FUNCTION DOC] Checks RPC of the Python builtin min.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        # [FUNCTION DOC] Checks RPC of a user-defined function called with keyword arguments only.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    def test_build_rpc_profiling_key(self):
        # [FUNCTION DOC] Checks that the RPC profiling key built for each exec mode contains the
        # mode value, function name and both worker names.
        # Tests that the name that shows up as an Event in profiling RPCs has all
        # the necessary information.
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            rpc_profiling_key = _build_rpc_profiling_key(
                exec_mode, "foo", "worker0", "worker1"
            )
            self.assertIn(exec_mode.value, rpc_profiling_key)
            self.assertIn("foo", rpc_profiling_key)
            self.assertIn("worker0", rpc_profiling_key)
            self.assertIn("worker1", rpc_profiling_key)

    def check_profiling_info(
        self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode
    ):
        # [FUNCTION DOC] HELPER: Asserts that a profiled RPC event name contains the source and
        # destination worker names, the function name (qualified name for scripted functions) and
        # exec mode value, and that its count is 1.
        self.assertTrue(self_worker_name in rpc_event.name)
        self.assertTrue(dst_worker_name in rpc_event.name)
        if isinstance(func, torch.jit.ScriptFunction):
            self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
        else:
            self.assertTrue(func.__name__ in rpc_event.name)
        self.assertTrue(rpc_exec_mode.value in rpc_event.name)
        self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_rpc_record_shapes(self):
        # [FUNCTION DOC] On rank 1, checks that remote profiling with record_shapes yields the same
        # input shapes for the remote add event as a local profile of the same op.
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        t1, t2 = torch.ones(100), torch.ones(100)
        with _profile(record_shapes=True) as prof:
            rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))

        function_events = prof.function_events
        remote_events = [event for event in function_events if event.is_remote]
        remote_add_event = next(
            event for event in remote_events if "aten::add" in event.name
        )
        remote_add_input_shapes = remote_add_event.input_shapes
        # Run profiler on equivalent local op and validate shapes are the same.
        with _profile(record_shapes=True) as prof:
            torch.add(t1, t2)

        local_function_events = prof.function_events
        local_add_event = next(
            event for event in local_function_events if "aten::add" in event.name
        )
        local_add_input_shapes = local_add_event.input_shapes
        self.assertEqual(remote_add_input_shapes, local_add_input_shapes)

    @dist_init
    def test_profiler_rpc_memory(self):
        # [FUNCTION DOC] On rank 1, checks that profiling with memory enabled propagates remote CPU
        # memory usage over RPC, and that it is all zero when memory profiling is disabled.
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile(profile_memory=True) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        # if cpu_memory_usage was not propagated over the wire, this set would
        # only contain 0 (indicates no memory being profiled)
        self.assertNotEqual({0}, event_cpu_mem_usages)
        # No memory profiled if profile_memory=False
        with _profile(profile_memory=False) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        self.assertEqual({0}, event_cpu_mem_usages)

    @dist_init
    def test_profiler_export_trace(self):
        # [FUNCTION DOC] On rank 1, checks that the exported chrome trace of a profile including a
        # remote call contains the expected remote op names and the async RPC event.
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile() as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        with TemporaryFileName() as fname:
            path = fname
            p.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
                event_names = [event["name"] for event in trace]
                for expected_event_name in EXPECTED_REMOTE_EVENTS + [
                    RPCExecMode.ASYNC.value
                ]:
                    event_exists = any(
                        expected_event_name in event_name for event_name in event_names
                    )
                    self.assertTrue(event_exists)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/125365")
    @dist_init
    def test_profiler_rpc_key_names(self):
        # [FUNCTION DOC] On rank 1, checks that remote events profiled from concurrent RPCs (via a
        # thread pool) are prefixed with the right RPC profiling key, have the right node id and
        # cover the expected operator names. Disabled on Linux by skipIf.
        # tests that remote events are properly prefixed with the RPC profiling key.
        if self.rank != 1:
            return

        # Spawn multiple threads that send RPCs to ensure keys are correctly
        # prefixed when there are multiple RPCs being created/in flight at the
        # same time.
        dst_ranks = [rank for rank in range(self.world_size) if rank != self.rank]

        def rpc_with_profiling(dst_worker):
            # [FUNCTION DOC] HELPER: Runs one profiled async RPC to the destination and validates
            # the keys, node id and operator names of its remote events.
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()

            events = prof.function_events
            remote_event_names = {
                event.name: event for event in events if event.is_remote
            }
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                dst_worker,
            )

            remote_event_name_set = set(EXPECTED_REMOTE_EVENTS)
            for name, event in remote_event_names.items():
                # Ensure that we have the expected key as part of the remote
                # event.
                self.assertTrue(name.startswith(rpc_profiling_key))
                self.assertTrue(event.is_remote)
                self.assertTrue(event.node_id == rpc.get_worker_info(dst_worker).id)
                # Ensure that the remote event name also contains the operator.
                operator_name_substr = name[len(rpc_profiling_key) :]
                # Note: we don't assert that every remote event needs to be
                # in the above set, the set is just a representative set of
                # what we expect to see. The profiler can change and add more
                # events, but we should always expect to see this representative
                # set.
                matching_event = {
                    remote_event_name
                    for remote_event_name in remote_event_name_set
                    if remote_event_name in operator_name_substr
                }
                remote_event_name_set -= matching_event

            # The set should be empty, otherwise its contained elements did
            # not show up in the remote profiler output.
            self.assertTrue(
                remote_event_name_set == set(),
                f"Expected {remote_event_name_set} to be included in remote profiler output.",
            )

        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            num_parallel_rpcs = 2
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_parallel_rpcs
            ) as executor:
                futs = [
                    executor.submit(rpc_with_profiling, dst_worker)
                    for _ in range(num_parallel_rpcs)
                ]
                # Wait for workers to finish test
                for fut in futs:
                    fut.result()

    def _run_test_profiler_remote_events_profiled(self):
        # [FUNCTION DOC] HELPER: On rank 1, for every other worker, profiles an async RPC of
        # udf_with_torch_ops and checks the RPC event info, the remote event keys, node ids and
        # that the order and set of remote events match the expected list.
        # Tests that we can successfully invoke the profiler on a remote node,
        # and collect the remote events back in the local profiler.
        if self.rank != 1:
            return

        dst_ranks = [rank for rank in range(self.world_size) if rank != self.rank]
        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()

            events = prof.function_events

            rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
            self.check_profiling_info(
                worker_name(self.rank),
                dst_worker,
                udf_with_torch_ops,
                rpc_event,
                RPCExecMode.ASYNC,
            )

            remote_events = {event.name: event for event in events if event.is_remote}
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                worker_name(dst),
            )

            for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
                expected_key = (
                    rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
                )
                self.assertTrue(expected_key in remote_events)
                remote_event = remote_events[expected_key]
                # Remote event should have a node ID corresponding to the worker
                # it ran on.
                self.assertEqual(remote_event.node_id, dst)

            # Validate order remote events show up in profiling output.
            def convert_remote_to_local(event_name):
                # [FUNCTION DOC] HELPER: Strips everything up to and including the remote-op marker
                # from an event name.
                remote_op_key = rpc_profiling_key + REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key) :]

            remote_events_list = [
                convert_remote_to_local(event.name)
                for event in events
                if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS
            ]
            self.assertEqual(
                set(remote_events_list),
                set(EXPECTED_REMOTE_EVENTS),
                f"Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}",
            )

    @dist_init
    def test_profiler_remote_events_profiled(self):
        # [FUNCTION DOC] Runs the remote-events profiling check.
        self._run_test_profiler_remote_events_profiled()

    @dist_init
    def test_profiler_remote_events_profiled_single_threaded(self):
        # [FUNCTION DOC] Runs the remote-events profiling check (single-threaded variant).
        # KNOWN ISSUE: body is identical to the non-single-threaded test; any difference must come
        # from a subclass configuration.
        self._run_test_profiler_remote_events_profiled()

    def run_profiling_workload(self, dst):
        # [FUNCTION DOC] HELPER: Sends an async mul RPC with two grad-requiring scalar tensors to
        # dst and waits.
        fut = rpc.rpc_async(
            worker_name(dst),
            torch.mul,
            args=(
                torch.tensor(1.0, requires_grad=True),
                torch.tensor(1.0, requires_grad=True),
            ),
        )
        fut.wait()

    def _run_rpc_profiling_async_function(self, device="cpu"):
        # [FUNCTION DOC] HELPER: On rank 1, profiles an async RPC of slow_async_add (which makes a
        # nested RPC from dst1 to dst2) and checks that the nested RPC and its add are recorded as
        # remote events with the correct keys and node ids.
        if self.rank != 1:
            return

        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        x = torch.ones(2)
        y = torch.ones(2)
        with _profile() as prof:
            ret = rpc.rpc_async(
                dst1, slow_async_add, args=(dst2, x, y, device), timeout=20
            )
            ret.wait()

        function_events = prof.function_events
        # slow_async_add resulted in an RPC from dst1 -> dst2, so this should be
        # recorded.
        key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1
        )

        nested_rpc_key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2
        )
        expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
        remote_events = [event for event in function_events if event.is_remote]
        rpc_remote_event = [
            event for event in remote_events if event.name == expected_key
        ]
        self.assertEqual(1, len(rpc_remote_event))
        rpc_remote_event = rpc_remote_event[0]
        self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
        # slow_async_add's RPC does an add on dst2, which should be reflected as well.
        remote_add_key = (
            expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
        )
        remote_add_event = [
            event for event in remote_events if event.name == remote_add_key
        ]
        self.assertEqual(1, len(remote_add_event))
        remote_add_event = remote_add_event[0]
        # Validate that node_id is dst2.
        self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)

    @dist_init
    def test_rpc_profiling_async_function(self):
        # [FUNCTION DOC] Initializes a process group and profiles an async_execution function on
        # CPU, and on CUDA if available.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    def test_rpc_profiling_async_function_single_threaded(self):
        # [FUNCTION DOC] Same as test_rpc_profiling_async_function (single-threaded variant).
        # KNOWN ISSUE: body is identical to the non-single-threaded test.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    def test_rpc_profiling_remote_record_function(self):
        # [FUNCTION DOC] On rank 1, checks that a function using record_function run over RPC shows
        # the profiled block remotely and that its descendant events match the local profile of the
        # same function.
        # test that functions run over RPC with record_function show the expected
        # profiled block.
        if self.rank != 1:
            return
        dst_ranks = [i for i in range(self.world_size) if i != self.rank]
        for dst_rank in dst_ranks:
            dst_worker = worker_name(dst_rank)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
                fut.wait()

            function_events = prof.function_events
            record_function_remote_event = [
                evt for evt in function_events if "##forward##" in evt.name
            ]
            self.assertEqual(1, len(record_function_remote_event))
            record_function_remote_event = record_function_remote_event[0]
            self.assertEqual(record_function_remote_event.node_id, dst_rank)
            # cpu_children only returns direct children, so here we get all
            # children recursively.

            def get_cpu_children(event):
                # [FUNCTION DOC] HELPER: Recursively collects all CPU children of an event.
                # KNOWN ISSUE: extends the event's own cpu_children list while iterating over it,
                # which mutates profiler data and can grow the list being iterated.
                if not event.cpu_children:
                    return []
                cpu_children = event.cpu_children
                for e in event.cpu_children:
                    cpu_children.extend(get_cpu_children(e))
                return cpu_children

            remote_children = get_cpu_children(record_function_remote_event)
            # Get local children and verify parity.
            with _profile() as prof:
                udf_with_torch_ops(-1, True)

            local_function_events = prof.function_events
            local_record_function_event = next(
                evt for evt in local_function_events if "##forward##" in evt.name
            )
            local_children = get_cpu_children(local_record_function_event)
            local_children_names = [evt.name for evt in local_children]

            REMOTE_OP_STR = "#remote_op: "

            def convert_remote_to_local(event_name):
                # [FUNCTION DOC] HELPER: Strips everything up to and including the remote-op marker
                # from an event name.
                remote_op_key = REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key) :]

            for evt in remote_children:
                local_name = convert_remote_to_local(evt.name)
                self.assertTrue(local_name in local_children_names)

    def validate_profiling_workload(self, dst, prof):
        # [FUNCTION DOC] HELPER: Checks that the profile contains a remote mul event on node dst
        # with correct profiling info.
        def convert_remote_to_local(event_name):
            # [FUNCTION DOC] HELPER: Strips everything up to and including the remote-op marker
            # from an event name.
            return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR) :]

        events = prof.function_events
        remote_events = {
            convert_remote_to_local(event.name): event
            for event in events
            if event.is_remote
        }
        self.assertTrue("aten::mul" in remote_events)
        remote_mul_event = remote_events["aten::mul"]
        self.assertEqual(remote_mul_event.node_id, dst)
        self.check_profiling_info(
            worker_name(self.rank),
            worker_name(dst),
            torch.mul,
            remote_mul_event,
            RPCExecMode.ASYNC,
        )

    def _run_test_profiler_with_autograd_context(self):
        # [FUNCTION DOC] HELPER: On rank 1, profiles a workload inside a distributed autograd
        # context in both context manager orders and validates the profiling results.
        dst = (self.rank + 1) % self.world_size
        if self.rank == 1:
            # Cases where we can double wrap messages with profiling information and autograd info.
            with dist_autograd.context(), _profile() as prof:
                self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

            # Ensure that flipped order of ctx managers results in events being
            # recorded as expected.
            with _profile() as prof, dist_autograd.context():
                self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

    @dist_init
    def test_profiler_with_autograd_context_single_threaded(self):
        # [FUNCTION DOC] Runs the profiler plus autograd context check (single-threaded variant).
        # KNOWN ISSUE: body is identical to the non-single-threaded test.
        self._run_test_profiler_with_autograd_context()

    @dist_init
    def test_profiler_with_autograd_context(self):
        # [FUNCTION DOC] Runs the profiler plus autograd context check.
        self._run_test_profiler_with_autograd_context()

    def _profiler_test_with_rpc(
        self,
        rpc_exec_mode,
        func,
        args,
        use_record_function=False,
        dst=None,
        kineto_profile=False,
    ):
        # [FUNCTION DOC] HELPER: On rank 1, runs func with args over RPC in the given mode,
        # optionally inside a record_function scope and with the kineto profiler. Checks the RPC
        # event exists with correct node id, remote events are present, the scope event encloses
        # the RPC event, and ordering; with kineto, checks that no RPC events are recorded.
        dst = dst if dst is not None else (self.rank + 1) % self.world_size

        # only run profiler on rank 1.
        p = _profile if not kineto_profile else torch.profiler.profile  # kineto
        if self.rank == 1:
            with p() as prof:
                record_function_ctx_mgr = (
                    contextlib.nullcontext()
                    if not use_record_function
                    else torch.autograd.profiler.record_function("foo")
                )
                with record_function_ctx_mgr:
                    if rpc_exec_mode == RPCExecMode.SYNC:
                        rpc.rpc_sync(worker_name(dst), func, args=args)
                    elif rpc_exec_mode == RPCExecMode.ASYNC:
                        fut = rpc.rpc_async(worker_name(dst), func, args=args)
                        if kineto_profile:
                            # Ensure multiple async RPCs don't cause issues.
                            # Would have raised
                            # "RuntimeError: Cannot call
                            # RemoteProfilerManager::setCurrentKey when current
                            # key is already set." error if RPC profiling was
                            # not disabled properly for kineto.
                            fut2 = rpc.rpc_async(worker_name(dst), func, args=args)
                            fut2.wait()
                        fut.wait()
                    else:
                        self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                        rref = rpc.remote(worker_name(dst), func, args=args)
                        rref.to_here()
                        # To avoid flakiness, wait for the RRef to be profiled. This
                        # means that we received the acknowledgement of successful
                        # creation on the owner and ran the callbacks responsible
                        # for recording the profiling event.
                        rref._get_profiling_future().wait()

            events = prof.function_events if not kineto_profile else prof.events()
            if kineto_profile:
                # RPC profiling is disabled so there should be no rpc related
                # events.
                with self.assertRaises(IndexError):
                    get_function_event(events, rpc_exec_mode.value)

                return

            rpc_event = get_function_event(events, rpc_exec_mode.value)
            # verify Node ID for this rpc event.
            self.assertEqual(rpc_event.node_id, self.rank)
            # Ensure recording of remote events.
            remote_events = {event for event in events if event.node_id == dst} - {
                rpc_event
            }
            self.assertGreaterEqual(len(remote_events), 1)
            for remote_event in remote_events:
                self.assertEqual(remote_event.node_id, dst)

            if use_record_function:
                scope_event = get_function_event(events, "foo")
                # Since RPC call is within the scope, its CPU interval should be
                # contained within foo's interval.
                self.assertLessEqual(
                    scope_event.time_range.start, rpc_event.time_range.start
                )
                self.assertGreaterEqual(
                    scope_event.time_range.end, rpc_event.time_range.end
                )
            # the sender, dest worker, function run, and type of RPC should all
            # be recorded.
            self_worker_name = worker_name(self.rank)
            dst_worker_name = worker_name(dst)
            self.check_profiling_info(
                self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode
            )
            if use_record_function:
                # verify order by ensuring that the outer context comes
                # before the rpc event.
                foo_event_ix = next(
                    i for i, event in enumerate(events) if "foo" in event.name
                )
                rpc_event_idx = next(
                    i
                    for i, event in enumerate(events)
                    if rpc_exec_mode.value in event.name
                )
                self.assertLess(foo_event_ix, rpc_event_idx)

    def _run_test_profiler_with_sync_rpc_udf(self):
        # [FUNCTION DOC] HELPER: Profiles sync RPC of a UDF with and without record_function.
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_sleep_func, args=(1,), use_record_function=True
        )

    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        # [FUNCTION DOC] Runs the sync UDF profiler check.
        self._run_test_profiler_with_sync_rpc_udf()

    @dist_init
    def test_profiler_with_sync_rpc_udf_single_threaded(self):
        # [FUNCTION DOC] Runs the sync UDF profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_sync_rpc_udf()

    def _run_test_profiler_with_sync_rpc_builtin(self):
        # [FUNCTION DOC] HELPER: Profiles sync RPC of a builtin with and without record_function.
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        # [FUNCTION DOC] Runs the sync builtin profiler check.
        self._run_test_profiler_with_sync_rpc_builtin()

    @dist_init
    def test_profiler_with_sync_rpc_builtin_single_threaded(self):
        # [FUNCTION DOC] Runs the sync builtin profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_sync_rpc_builtin()

    def _run_test_profiler_with_async_rpc_udf(self):
        # [FUNCTION DOC] HELPER: Profiles async RPC of a UDF with and without record_function, and
        # with the kineto profiler (which must not enable RPC profiling).
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_sleep_func, args=(1,), use_record_function=True
        )
        # Test to ensure that kineto profiler enabled in RPC does not enable
        # RPC profiling (it is unsupported) and does not result in issues.
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_sleep_func, args=(1,), kineto_profile=True
        )

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        # [FUNCTION DOC] Runs the async UDF profiler check.
        self._run_test_profiler_with_async_rpc_udf()

    @dist_init
    def test_profiler_with_async_rpc_udf_single_threaded(self):
        # [FUNCTION DOC] Runs the async UDF profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_async_rpc_udf()

    def _run_test_profiler_with_async_rpc_builtin(self):
        # [FUNCTION DOC] HELPER: Profiles async RPC of a builtin with and without record_function.
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        # [FUNCTION DOC] Runs the async builtin profiler check.
        self._run_test_profiler_with_async_rpc_builtin()

    @dist_init
    def test_profiler_with_async_rpc_builtin_single_threaded(self):
        # [FUNCTION DOC] Runs the async builtin profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_async_rpc_builtin()

    def _run_test_profiler_with_remote_udf(self):
        # [FUNCTION DOC] HELPER: Profiles remote calls of a UDF, with record_function, and to the
        # current worker.
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), use_record_function=True
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), dst=self.rank
        )

    @dist_init
    def test_profiler_with_remote_udf(self):
        # [FUNCTION DOC] Runs the remote UDF profiler check.
        self._run_test_profiler_with_remote_udf()

    @dist_init
    def test_profiler_with_remote_udf_single_threaded(self):
        # [FUNCTION DOC] Runs the remote UDF profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_remote_udf()

    def _run_test_profiler_with_remote_builtin(self):
        # [FUNCTION DOC] HELPER: Profiles remote calls of a builtin, with record_function, and to
        # the current worker.
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            dst=self.rank,
        )

    @dist_init
    def test_profiler_with_remote_builtin(self):
        # [FUNCTION DOC] Runs the remote builtin profiler check.
        self._run_test_profiler_with_remote_builtin()

    @dist_init
    def test_profiler_with_remote_builtin_single_threaded(self):
        # [FUNCTION DOC] Runs the remote builtin profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_remote_builtin()

    def _run_test_profiler_with_script_async_rpc(self):
        # [FUNCTION DOC] HELPER: Profiles async RPC of a scripted function with and
        # without record_function.
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_async_rpc(self):
        # [FUNCTION DOC] Runs the scripted async profiler check.
        self._run_test_profiler_with_script_async_rpc()

    @dist_init
    def test_profiler_with_script_async_rpc_single_threaded(self):
        # [FUNCTION DOC] Runs the scripted async profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_script_async_rpc()

    def _run_test_profiler_with_script_sync_rpc(self):
        # [FUNCTION DOC] HELPER: Profiles sync RPC of a scripted function with and
        # without record_function.
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_sync_rpc(self):
        # [FUNCTION DOC] Runs the scripted sync profiler check.
        self._run_test_profiler_with_script_sync_rpc()

    @dist_init
    def test_profiler_with_script_sync_rpc_single_threaded(self):
        # [FUNCTION DOC] Runs the scripted sync profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_script_sync_rpc()

    def _run_test_profiler_with_script_remote_rpc(self):
        # [FUNCTION DOC] HELPER: Profiles remote calls of a scripted function with and without
        # record_function, and to the current worker.
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), dst=self.rank
        )

    @dist_init
    def test_profiler_with_script_remote_rpc(self):
        # [FUNCTION DOC] Runs the scripted remote profiler check.
        self._run_test_profiler_with_script_remote_rpc()

    @dist_init
    def test_profiler_with_script_remote_rpc_single_threaded(self):
        # [FUNCTION DOC] Runs the scripted remote profiler check (single-threaded variant).
        # KNOWN ISSUE: identical to the non-single-threaded test.
        self._run_test_profiler_with_script_remote_rpc()

    def _assert_top_level_events(
        self, process_global_events, expected_top_level_event_names
    ):
        # [FUNCTION DOC] HELPER: Asserts that the top-level (non-nested) event names across
        # per-thread event lists equal expected_top_level_event_names. An event counts as top level
        # if it starts after the end of the previous top-level event on the same thread; names are
        # compared after sorting.
        top_level_event_names = []
        for thread_local_events in process_global_events:
            # Get top-level events from all events happened on a thread.
            last_end_time = 0
            for event in thread_local_events:
                event_name = event.name
                time_range = event.time_range
                if time_range.start > last_end_time:
                    top_level_event_names.append(event_name)
                    last_end_time = time_range.end
        top_level_event_names = sorted(top_level_event_names)
        expected_top_level_event_names = sorted(expected_top_level_event_names)
        self.assertEqual(
            top_level_event_names,
            expected_top_level_event_names,
            f"Expected events {expected_top_level_event_names}, but got {top_level_event_names}",
        )

    @dist_init
    def test_server_process_global_profiler(self):
        # [FUNCTION DOC] On rank 0, enters nested server-side process-global profilers (outer then
        # inner) on the next worker through RRef proxies, runs add and sub there, exits both,
        # fetches the events from each and checks that the inner profile holds only sub and the
        # outer holds sub and add; also calls key_averages on both.
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)

        x = torch.tensor(1)
        y = torch.tensor(2)

        outer_profile_rref = rpc.remote(
            dst_worker_name, rpc._server_process_global_profile
        )
        outer_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        inner_profile_rref = rpc.remote(
            dst_worker_name, rpc._server_process_global_profile
        )
        inner_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        inner_profile_rref.rpc_sync().__exit__(None, None, None)
        outer_profile_rref.rpc_sync().__exit__(None, None, None)

        inner_events = rpc.rpc_sync(
            dst_worker_name, get_events_from_profile, (inner_profile_rref,)
        )
        expected_inner_events = ["aten::sub"]
        expected_outer_events = expected_inner_events + ["aten::add"]

        self._assert_top_level_events(inner_events, expected_inner_events)
        outer_events = rpc.rpc_sync(
            dst_worker_name, get_events_from_profile, (outer_profile_rref,)
        )
        self._assert_top_level_events(outer_events, expected_outer_events)

        inner_profile_rref.rpc_sync().key_averages()
        outer_profile_rref.rpc_sync().key_averages()

    @dist_init
    def test_async_record_function_double_end_callbacks(self):
        # [FUNCTION DOC] On rank 1, checks that finishing a record_function scope's callbacks on a
        # future twice raises RuntimeError saying it can only be called once.
        num_sleep_seconds = 1
        if self.rank == 1:
            # Validate that calling the function twice results in an error.
            with _profile():
                with torch.autograd.profiler.record_function("foo") as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    rf._call_end_callbacks_on_future(fut)
                    with self.assertRaisesRegex(
                        RuntimeError, "can only be called once."
                    ):
                        rf._call_end_callbacks_on_future(fut)
                fut.wait()

    @dist_init
    def test_async_record_function_legacy(self):
        # [FUNCTION DOC] On rank 1, checks the legacy profiler record-function operators
        # (enter, end-callbacks-on-future, exit) work around an async RPC; kept for TorchScript
        # backward compatibility.
        # Test the legacy _record_function ops work
        # Note: These exist for backward compatibility with TorchScript
        num_sleep_seconds = 1
        if self.rank == 1:
            with _profile():
                try:
                    handle = torch.ops.profiler._record_function_enter("foo", None)
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    torch.ops.profiler._call_end_callbacks_on_jit_fut(handle, fut)
                finally:
                    torch.ops.profiler._record_function_exit(handle)

                fut.wait()

    @dist_init
    def test_async_record_function_cbs_jit_call(self):
        # [FUNCTION DOC] On rank 1, attaches a record_function to an async RPC of a scripted
        # function through the JIT future callback operator, checks the returned future gives the
        # same value as the RPC, and that the profile contains the event named by the function's
        # qualified name.
        if self.rank == 1:
            with _profile() as pf:
                key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(my_script_func),
                    "worker1",
                    "worker0",
                )
                with torch.autograd.profiler.record_function(key) as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_script_func, args=(torch.tensor(1),)
                    )
                    # Intentionally calling record_function internals
                    fut = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                        rf.record, fut
                    )
                result = fut.wait()
                # Validate that the profiling future returns the same value as the RPC
                # future.
                expected = torch.add(torch.tensor(1), torch.tensor(1))
                self.assertEqual(result, expected)
            events = pf.function_events
            rpc_event = get_function_event(
                events, torch._jit_internal._qualified_name(my_script_func)
            )
            self.assertTrue(
                torch._jit_internal._qualified_name(my_script_func) in rpc_event.name
            )

    @dist_init
    def test_py_class_constructor(self):
        # [FUNCTION DOC] Checks that a class constructor can be sent as an RPC target and the
        # returned instance has the expected attribute.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        # [FUNCTION DOC] Checks that a bound instance method can be sent as an RPC target.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass(2).my_instance_method, args=(n,)
        )
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        # [FUNCTION DOC] Checks that a class method can be sent as an RPC target.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_class_method, args=(n, n + 1)
        )
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        # [FUNCTION DOC] Checks that a static method can be sent as an RPC target.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_static_method, args=(n + 10,)
        )
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        # [FUNCTION DOC] Issues two async RPCs (a static method and the builtin min) to the same
        # destination and checks both results.
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_info = rpc.get_worker_info(worker_name(dst_rank))
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        # [FUNCTION DOC] Checks that an RPC to a function returning nothing yields None.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        # [FUNCTION DOC] Checks an RPC of a Python function with tensor arguments.
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_tensor_function,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, my_tensor_function(torch.ones(n, n), torch.ones(n, n)))

    @dist_init
    def test_py_tensors_multi_async_call(self):
        # [FUNCTION DOC] Sends 100 async RPCs with growing tensor sizes and checks each result
        # in order.
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        for i in range(100):
            fut = rpc.rpc_async(
                worker_name(dst_rank),
                my_tensor_function,
                args=(torch.ones(i, i), torch.ones(i, i)),
            )
            futs.append(fut)

        for j, val in enumerate(torch.futures.wait_all(futs)):
            self.assertEqual(
                val, my_tensor_function(torch.ones(j, j), torch.ones(j, j))
            )

    @dist_init
    def test_py_tensors_in_container(self):
        # [FUNCTION DOC] Checks an RPC of a function whose arguments are a list of tensors, a
        # namedtuple of nested tensor containers and a dict of tensors.
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [torch.ones(n, n), torch.ones(n, n)]
        b = TensorClass(build_complex_tensors())
        c = {"foo": torch.ones(n, n), "bar": torch.ones(n, n)}
        ret = rpc.rpc_sync(
            worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c)
        )
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        # [FUNCTION DOC] Checks an RPC whose argument is an object with custom pickling hooks that
        # nest the RPC pickler.
        n = self.rank + 1
        dst_rank = n % self.world_size

        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            run_nested_pickle,
            args=(MyPickleClass(), torch.ones(2, 2)),
        )

        m = MyPickleClass()
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    @dist_init
    def test_py_function_exception(self):
        # [FUNCTION DOC] Checks that calling a function with the wrong number of arguments raises
        # TypeError at the caller.
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaises(TypeError):
            rpc.rpc_sync(worker_name(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        # [FUNCTION DOC] Checks that an exception raised in a user function propagates as
        # ValueError to the caller's wait and that the message is also logged to stderr on the
        # worker, using barriers around the captured-output context. Side effect: captures
        # process stderr.
        with captured_output() as (_, err):
            # This barrier prevents a race condition where the main thread has
            # not entered the context manager when the remote function runs.
            initialize_pg(self.file_init_method, self.rank, self.world_size)
            dist.barrier()
            n = self.rank + 1
            dst_rank = n % self.world_size
            fut = rpc.rpc_async(worker_name(dst_rank), raise_func)
            with self.assertRaisesRegex(ValueError, expected_err):
                fut.wait()
            # This barrier prevents a race condition where the main thread exits
            # context manager before the remote function has ran.
            dist.barrier()

        # Validate that trainers log errors when running functions.
        stderr_lines = err.getvalue()
        self.assertTrue(expected_err in stderr_lines)

    @dist_init
    def test_py_raise_in_user_func_escaped_str(self):
        # [FUNCTION DOC] Checks that a remote error message with newlines is unescaped (the message
        # is unchanged by unicode_escape decoding).
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(worker_name(dst_rank), raise_func_escape)
        try:
            fut.wait()
        except ValueError as e:
            msg = str(e)
            # Ensure newlines are unescaped to provide a better repr of error.
            self.assertEqual(msg, msg.encode("utf-8").decode("unicode_escape"))
        else:
            self.assertTrue(False, "expected raise_func_escape to raise ValueError.")

    @dist_init
    def test_nested_rpc(self):
        # [FUNCTION DOC] Runs the nested RPC helper with nested_rpc.
        self._nested_rpc(nested_rpc, torch.ones(2, 2) + 1)

    @dist_init
    def test_stress_light_rpc(self):
        # [FUNCTION DOC] Runs the stress helper with the light payload.
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        # [FUNCTION DOC] Runs the stress helper with the heavy payload, 20 repeats.
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_stress_heavy_rpc_torchscript(self):
        # [FUNCTION DOC] Runs the stress helper with the TorchScript heavy payload, 20 repeats.
        self._stress_test_rpc(
            heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),)
        )

    @dist_init
    def test_builtin_remote_ret(self):
        # [FUNCTION DOC] Runs the builtin remote return helper with ones tensors.
        self._builtin_remote_ret(
            torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2
        )

    @dist_init
    def test_builtin_remote_self(self):
        # [FUNCTION DOC] Runs the builtin remote-to-self helper with ones tensors.
        self._builtin_remote_self(
            torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2
        )

    @staticmethod
    def _multi_args_fn(n, sparse=False):
        # [FUNCTION DOC] HELPER: Static; returns a pair of tensors for builtin add: two sparse
        # tensors if sparse is true, otherwise two ones tensors of size n by n.
        if sparse:
            return (build_sparse_tensor(), build_sparse_tensor())
        else:
            return (torch.ones(n, n), torch.ones(n, n))

    @dist_init
    def test_multi_builtin_remote_ret(self):
        # [FUNCTION DOC] Runs the multi remote call helper with torch.add and dense tensors.
        self._test_multi_remote_call(torch.add, False, args_fn=RpcTest._multi_args_fn)

    @dist_init
    def test_py_udf_remote(self):
        # [FUNCTION DOC] Checks a remote call of a Python function with keyword arguments.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(rref.to_here(), my_function(n, n + 1, n + 2))

    @staticmethod
    def _multi_kwargs_fn(n, sparse=False):
        # [FUNCTION DOC] HELPER: Static; returns a mapping of keyword arguments a, b and c as
        # sparse tensors if sparse is true, otherwise as n by n ones tensors.
        if sparse:
            return {
                "a": build_sparse_tensor(),
                "b": build_sparse_tensor(),
                "c": build_sparse_tensor(),
            }
        else:
            return {"a": torch.ones(n, n), "b": torch.ones(n, n), "c": torch.ones(n, n)}

    @dist_init
    def test_multi_py_udf_remote(self):
        # [FUNCTION DOC] Runs the multi remote call helper with my_function and keyword arguments.
        self._test_multi_remote_call(
            my_function, False, kwargs_fn=RpcTest._multi_kwargs_fn
        )

    @dist_init
    def test_py_rref_args(self):
        # [FUNCTION DOC] Runs the RRef-arguments helper with an RRef built from tensor adds.
        self._py_rref_args(
            torch.ones(2, 2), 1, torch.ones(2, 2), 2, torch.ones(2, 2) * 2 + 3
        )

    @dist_init
    def test_py_rref_args_user_share(self):
        # [FUNCTION DOC] Runs the RRef-arguments helper in which the RRefs are shared with a third
        # user rank.
        self._py_rref_args_user_share(
            torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_py_rpc_rref_args(self):
        # [FUNCTION DOC] Runs the helper that passes RRefs as arguments of a sync RPC.
        self._py_rpc_rref_args(
            torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_nested_remote(self):
        # [FUNCTION DOC] Runs the nested remote helper.
        self._nested_remote(nested_remote, torch.ones(2, 2) + 3)

    @dist_init
    def test_nested_rref(self):
        # [FUNCTION DOC] Runs the nested RRef helper.
        self._nested_rref(nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)

    @dist_init
    def test_nested_rref_stress(self):
        # [FUNCTION DOC] Runs the nested RRef stress helper.
        self._nested_rref_stress(
            nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2
        )

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        # [FUNCTION DOC] Starts a chain of fire-and-forget nested async RPCs and returns
        # immediately; checks that graceful shutdown handles those in-flight messages.
        # This test will exit right away, but there will be a chain of async
        # RPCs. The termination algorithm should detect those messages properly.
        # Otherwise, some peer could exit early, leaving others to timeout
        # errors or connection closed errors.
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size

        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)

    @dist_init
    def test_remote_with_exception(self):
        # [FUNCTION DOC] Checks that fetching an RRef whose remote function raised ValueError
        # re-raises it, and that a self-owned RRef with wrong arguments raises TypeError.
        n = self.rank + 1
        dst_rank = n % self.world_size
        # check ref to other workers
        rref = rpc.remote(worker_name(dst_rank), raise_func)
        with self.assertRaises(ValueError):
            rref.to_here()
        # check ref to itself
        rref = rpc.remote(worker_name(self.rank), no_result, args=(10,))
        with self.assertRaises(TypeError):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        # [FUNCTION DOC] Checks that an RPC can return an RRef created on a third worker and that
        # it can be fetched.
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.rpc_sync(
            worker_name(dst_rank1),
            rpc_return_rref,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        # [FUNCTION DOC] Builds a chain of RRefs through ranks with the recursive helper and walks
        # the chain, checking the final fetched value.
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1))

        ret_rref = rref_forward_chain(dst_rank, self.world_size, rref, ttl)

        for _ in range(ttl):
            self.assertEqual(len(ret_rref), 1)
            ret_rref = ret_rref[0].to_here()

        ret = ret_rref
        self.assertEqual(ret, torch.add(torch.ones(n, n), 1))

    @dist_init
    def test_local_rref_no_fork(self):
        # [FUNCTION DOC] Checks that a locally created RRef holds its value without any fork.
        local_rref = RRef(35)
        self.assertEqual(local_rref.local_value(), 35)

    @dist_init
    def test_local_value_not_on_owner(self):
        # [FUNCTION DOC] Checks that calling local_value on a user RRef (not the owner) raises
        # RuntimeError with a detailed message naming the RRef ids, the user and the owner.
        # ensure that an error message is thrown if a user tries to call
        # local_value() on a non-owning node.
        next_rank = (self.rank + 1) % self.world_size
        rref = rpc.remote(
            worker_name(next_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        with self.assertRaisesRegex(
            RuntimeError,
            (
                rf"For UserRRef\(rref_id=GloballyUniqueId\(created_on={self.rank}, local_id=0\), "
                rf"fork_id=GloballyUniqueId\(created_on={self.rank}, local_id=1\)\), "
                r"can't call localValue\(\) on user "
                rf"WorkerInfo\(id={self.rank}, name={worker_name(self.rank)}\). "
                rf"Call it on owner WorkerInfo\(id={next_rank}, name={worker_name(next_rank)}\)"
            ),
        ):
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        # [FUNCTION DOC] Gets a list of RRefs owned by another worker, increments each remote
        # object by 10 through calls to the owner, and checks the final values are 11, 12 and 13.
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref_list = rpc.rpc_sync(
            worker_name(dst_rank), get_rref_list, args=([1, 2, 3],)
        )

        for rref in rref_list:
            rpc.rpc_sync(
                rref.owner(),
                _call_method_on_rref,
                args=(MyClass.increment_value, rref, 10),
            )

        rets = [
            rpc.rpc_sync(
                rref.owner(), _call_method_on_rref, args=(MyClass.get_value, rref)
            )
            for rref in rref_list
        ]

        self.assertEqual(rets, [11, 12, 13])

    @dist_init
    def _test_rref_type(self, blocking):
        # [FUNCTION DOC] HELPER: Checks the RRef _get_type API with blocking or non-blocking
        # mode: the first call launches an RPC to the owner, later calls are served from cache
        # without any new RPC (and non-blocking calls return the same cached future), and the
        # type of a user-defined class RRef is also correct. The dist_init decorator is applied
        # to this helper.
        def launched_rpc(events):
            # [FUNCTION DOC] HELPER: Returns whether any profiler event is the async RPC for the
            # owner-side type query.
            expected_name = f"rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner"
            return any(e.name.startswith(expected_name) for e in events)

        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))

        with _profile() as p:
            t = rref._get_type(blocking=blocking)
            if not blocking:
                t = t.wait()

        self.assertTrue(launched_rpc(p.function_events))
        expected_type = type(torch.ones(2))
        self.assertEqual(t, expected_type)

        futs = []

        def verify(fut):
            # [FUNCTION DOC] HELPER: Done callback; asserts the future value equals the
            # expected type.
            self.assertEqual(fut.value(), expected_type)

        with _profile() as p:
            for _ in range(10):
                t = rref._get_type(blocking=blocking)
                if not blocking:
                    futs.append(t)
                    t.add_done_callback(verify)
                    t = t.wait()
                self.assertEqual(t, expected_type)

        if not blocking:
            # Note that cached calls with blocking=False all return the same
            # cached original future.
            first_fut = futs[0]
            for f in futs[1:]:
                self.assertTrue(f is first_fut)
        # Ensure we never launch another RPC, other than for the very
        # first call.
        self.assertFalse(launched_rpc(p.function_events))
        self.assertEqual(t, type(torch.ones(2)))

        rref = rpc.remote(dst, MyClass, args=(0,))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_blocking(self):
        # [FUNCTION DOC] Runs the RRef type check in blocking mode.
        self._test_rref_type(blocking=True)

    def test_rref_type_non_blocking(self):
        # [FUNCTION DOC] Runs the RRef type check in non-blocking mode.
        self._test_rref_type(blocking=False)

    @dist_init
    def _test_rref_type_with_error(self, blocking):
        # [FUNCTION DOC] HELPER: Checks that _get_type on an RRef whose creation failed raises
        # ValueError, inline when blocking or on wait when not. The dist_init decorator is applied
        # to this helper.
        # KNOWN ISSUE: a comment says 10 ms timeout but no timeout is set.
        dst = worker_name((self.rank + 1) % self.world_size)
        # 10 ms timeout
        rref = rpc.remote(dst, raise_func)
        # Blocking: error raised inline
        if blocking:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                rref._get_type(blocking=blocking)
        else:
            # Non-blocking: Immediately return future, block on wait
            fut = rref._get_type(blocking=blocking)
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()

    def test_rref_type_with_error_blocking(self):
        # [FUNCTION DOC] Runs the RRef type error check in blocking mode.
        self._test_rref_type_with_error(blocking=True)

    def test_rref_type_with_error_non_blocking(self):
        # [FUNCTION DOC] Runs the RRef type error check in non-blocking mode.
        self._test_rref_type_with_error(blocking=False)

    @dist_init
    def _test_rref_type_owner(self, blocking):
        # [FUNCTION DOC] HELPER: Checks _get_type on locally owned RRefs (a tensor and an object)
        # in blocking or non-blocking mode. The dist_init decorator is applied to this helper.
        rref = RRef(torch.ones(2) + 1)
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, type(torch.ones(2)))

        rref = RRef(MyClass(0))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_owner_blocking(self):
        # [FUNCTION DOC] Runs the owner RRef type check in blocking mode.
        self._test_rref_type_owner(blocking=True)

    def test_rref_type_owner_non_blocking(self):
        # [FUNCTION DOC] Runs the owner RRef type check in non-blocking mode.
        self._test_rref_type_owner(blocking=False)

    @staticmethod
    def _slow_add(x, y):
        # [FUNCTION DOC] HELPER: Static; sleeps one second and returns x plus y.
        time.sleep(1)
        return x + y

    @dist_init
    def test_rref_type_slow_init(self):
        # [FUNCTION DOC] Checks that the type of an RRef created by a slow function is eventually
        # reported correctly.
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, RpcTest._slow_add, args=(torch.ones(2), 1))
        self.assertEqual(rref._get_type(), type(torch.ones(2)))

    @dist_init
    def test_owner_equality(self):
        # [FUNCTION DOC] Checks equality and hashing of RRef owners (WorkerInfo): same owner
        # compares equal, different owners differ, and owners work as dictionary keys.
        a = RRef(40)
        b = RRef(50)

        other_rank = (self.rank + 1) % self.world_size
        other_a = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        other_b = rpc.remote(
            worker_name(other_rank), torch.add, args=(torch.ones(1), 1)
        )
        other_a.to_here()  # to ensure clean termination
        other_b.to_here()

        self.assertNotEqual(a.owner(), 23)
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertNotEqual(a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertEqual(a.owner(), a.owner())
        self.assertEqual(a.owner(), b.owner())
        self.assertEqual(a.owner(), rpc.get_worker_info())
        x = {}
        x[a.owner()] = a
        x[other_a.owner()] = other_a
        self.assertEqual(x[a.owner()], a)
        self.assertEqual(x[b.owner()], a)
        self.assertEqual(x[other_a.owner()], other_a)
        self.assertEqual(x[other_b.owner()], other_a)
        self.assertEqual(len(x), 2)

    @dist_init
    def test_pass_local_rrefs(self):
        # [FUNCTION DOC] Checks that a locally owned RRef can be passed as an argument to sync,
        # async and remote calls.
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker = worker_name(dst_rank)

        rref = RRef(40)
        self.assertEqual(
            rpc.rpc_sync(dst_worker, add_rref_to_value, args=(rref, 50)), 90
        )
        self.assertEqual(
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 50)).wait(), 90
        )
        self.assertEqual(
            rpc.remote(dst_worker, add_rref_to_value, args=(rref, 50)).to_here(), 90
        )

    @dist_init
    def test_remote_same_worker(self):
        # [FUNCTION DOC] Checks that an RRef created on a worker can be consumed by a remote call
        # running on the same worker.
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 2)
        )
        rref_b = rpc.remote(
            worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1)
        )
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init(setup_rpc=True)
    def test_call_method_on_rref(self):
        # [FUNCTION DOC] Checks that an instance method can be called on a remote object by using
        # the owner of its RRef as the call destination, for sync, async and remote modes, then
        # reads back the combined state.
        """
        Tests that it is possible to call an instance method on a remote object
        by using rref.owner() as destination of the call.
        """
        vals = [10, 2, 5, 7]
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)

        # creates a remote object
        rref = rpc.remote(dst_worker, MyClass, args=(vals[0],))

        # modifies state of the remote object
        rpc.rpc_sync(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[1]),
        )
        rpc.rpc_async(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[2]),
        ).wait()
        rpc.remote(
            rref.owner(),
            _call_method_on_rref,
            args=(MyClass.increment_value, rref, vals[3]),
        ).to_here()

        # queries state of the remote object
        result = rpc.rpc_sync(
            dst_worker, _call_method_on_rref, args=(MyClass.get_value, rref)
        )

        self.assertEqual(result, sum(vals))

    # Notice `rpc.api.shutdown()` accesses
    # `_delete_all_user_and_unforked_owner_rrefs` through
    # `torch.distributed.rpc.api`, so patching
    # `torch.distributed.rpc._delete_all_user_and_unforked_owner_rrefs` will
    # not help.
    @mock.patch.object(
        torch.distributed.rpc.api, "_delete_all_user_and_unforked_owner_rrefs"
    )
    def _test_rref_leak(
        self, _mock_delete_all_user_and_unforked_owner_rrefs, ignore_leak
    ):
        # [FUNCTION DOC] HELPER: Initializes RPC, creates an RRef that is never deleted, and then
        # shuts down gracefully with the RRef deletion function mocked out. When ignore_leak is
        # false the shutdown must raise RuntimeError about a leaked RRef; when true the leak is
        # ignored. Side effect: sets the module flag _ignore_rref_leak in rpc.api.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        rref = rpc.remote(  # noqa: F841
            worker_name((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.ones(2, 2), 1),
        )

        import torch.distributed.rpc.api as api

        if ignore_leak:
            api._ignore_rref_leak = True
            rpc.shutdown(graceful=True)
        else:
            api._ignore_rref_leak = False
            with self.assertRaisesRegex(RuntimeError, "Leaking RRef"):
                rpc.shutdown(graceful=True)

    @dist_init(setup_rpc=False)
    def test_rref_leak(self):
        # [FUNCTION DOC] Runs the RRef leak check expecting an error.
        self._test_rref_leak(ignore_leak=False)

    @dist_init(setup_rpc=False)
    def test_ignore_rref_leak(self):
        # [FUNCTION DOC] Runs the RRef leak check ignoring the leak.
        self._test_rref_leak(ignore_leak=True)

    @dist_init
    def test_rref_str(self):
        # [FUNCTION DOC] Checks the string forms of an owner RRef and a user RRef (ids and
        # fork ids).
        rref1 = RRef(self.rank)
        id_class = "GloballyUniqueId"
        self.assertEqual(
            f"OwnerRRef({id_class}(created_on={self.rank}, local_id=0))",
            rref1.__str__(),
        )

        dst_rank = (self.rank + 1) % self.world_size
        rref2 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(
            rref2.__str__(),
            f"UserRRef(RRefId = {id_class}(created_on={self.rank}, local_id=1), "
            f"ForkId = {id_class}(created_on={self.rank}, local_id=2))",
        )

    @dist_init
    def test_rref_get_future(self):
        # [FUNCTION DOC] On rank 0, checks that the creation future can be obtained from RRefs
        # created for a builtin, a Python function and a scripted function.
        # Tests that we can obtain the future corresponding to the creation of
        # the RRef on remote end
        if self.rank == 0:
            # Builtin
            rref = rpc.remote(worker_name(1), torch.add, args=(1, 1))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

            # UDF
            rref = rpc.remote(worker_name(1), foo_add, args=())
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

            # Script
            rref = rpc.remote(worker_name(1), my_script_func, args=(torch.tensor(1),))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

    @dist_init
    def test_rref_context_debug_info(self):
        # [FUNCTION DOC] Checks the RRef context debug info counters (owner RRef count and pending
        # users) after local RRef creation, after sharing an RRef as an argument, and after remote
        # calls, using barriers around each check.
        # This test checks local states that are modified by remote workers.
        # This means that we would need barrier before and after every check.
        # The barrier before the check makes sure that all previous states are
        # cleared globally, the barrier after ensures that no following states
        # change gets into the current check.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        # Check 1: local RRef does not update owners_ map or add a pending user.
        #################################################

        rref1 = RRef(self.rank)

        # don't need a barrier here as local RRef is handled by this thread
        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertIn("num_pending_users", info)
        # RRef on local value is not added to context until shared across RPC
        self.assertEqual(0, int(info["num_owner_rrefs"]))
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after the check 1
        dist.barrier()

        # Check 2: Sharing RRef as an arg should update owners_ map
        ###########################################################

        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), set_global_rref, args=(rref1,))

        # barrier before check 2
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(1, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))
        # barrier after check 2
        dist.barrier()

        # clear states for check 2
        rpc.rpc_sync(worker_name(dst_rank), clear_global_rref)

        # Wait for owner rref to be cleared.
        while int(info["num_owner_rrefs"]) != 0:
            info = _rref_context_get_debug_info()
            time.sleep(0.1)
        dist.barrier()

        # Check 3: rpc.remote call should update owners_ map
        ####################################################
        rref2 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        rref3 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        rref2.to_here()
        rref3.to_here()

        # barrier before check 3
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()

        info = _rref_context_get_debug_info()
        self.assertIn("num_owner_rrefs", info)
        self.assertEqual(2, int(info["num_owner_rrefs"]))
        # no pending users since the fork is finished
        self.assertEqual(0, int(info["num_pending_users"]))

        # barrier after check 3
        dist.barrier()

    @dist_init
    def test_disable_gil_profiling(self):
        # [FUNCTION DOC] Checks that GIL wait time profiling is off by default and appears in the
        # agent debug info after enabling it. Side effect: enables GIL profiling.
        # test that rpc.enable_gil_profiling(false) will result in
        # GIL wait time not being recorded.

        # GIL profiling should be disabled by default.
        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertRaises(KeyError, lambda: info["agent.gil_average_wait_time_us"])
        rpc.enable_gil_profiling(True)
        rpc.rpc_sync(
            worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn("agent.gil_average_wait_time_us", info)

    @dist_init(setup_rpc=False)
    def test_local_shutdown(self):
        # [FUNCTION DOC] Checks that RPC can be initialized and then locally shut down without
        # sending any message.
        # test that we can start RPC and then immediately locally shutdown
        # without sending any messages.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        # pass in graceful=False to ensure that we don't wait for other workers.
        rpc.shutdown(graceful=False)

    @dist_init
    def test_debug_info(self):
        # [FUNCTION DOC] Checks that the combined debug info has exactly the union of the RRef,
        # agent and autograd debug info keys, with no overlap.
        # only test keys in this test case. Values should be covered by
        # individual module debug info tests
        import torch.distributed.autograd as dist_autograd

        info = _get_debug_info()
        rref_info = _rref_context_get_debug_info()
        agent_info = rpc.api._get_current_rpc_agent().get_debug_info()
        autograd_info = dist_autograd._get_debug_info()
        common_keys = rref_info.keys() & agent_info.keys() & autograd_info.keys()
        self.assertEqual(0, len(common_keys))
        expected = {}
        expected.update(rref_info)
        expected.update(agent_info)
        expected.update(autograd_info)
        self.assertEqual(info.keys(), expected.keys())

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        IS_MACOS,
        "Test is flaky on MacOS since libuv error handling is not as robust as TCP",
    )
    def test_handle_send_exceptions(self):
        # [FUNCTION DOC] Checks that sending to a worker that has failed raises RuntimeError with a
        # shutdown error, instead of crashing. Sets a 10 second RPC timeout and ends with
        # non-graceful shutdown. Skipped (passes) on MacOS under sandcastle.
        # test that if a callee node has gone down, we raise an appropriate
        # exception instead of just crashing.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        rpc._set_rpc_timeout(10)
        # This barrier is needed to ensure that some workers do not exit before
        # others have been brought up.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        if self.rank == 1:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker = worker_name(dst_rank)
            # allow destination worker to exit without joining
            error_str = self.get_shutdown_error_regex()
            wait_until_node_failure(dst_rank, error_str)
            fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.ones(1), 3))
            # Shutdown sequence is not very well defined and as a result
            # we can see any of the error messages defined in get_shutdown_error_regex.
            with self.assertRaisesRegex(RuntimeError, error_str):
                fut.wait()
        # exit all workers non-gracefully.
        rpc.shutdown(graceful=False)

    @dist_init
    def test_deadlock(self):
        # [FUNCTION DOC] Reproduces a known deadlock scenario: rank 1 waits for a slow RPC, then
        # each rank initializes a process group if not already initialized.
        # this test is copied from https://github.com/pytorch/pytorch/issues/45089
        if self.rank == 1:
            dst1 = worker_name((self.rank + 1) % self.world_size)
            x = torch.ones(2)
            y = torch.ones(2)
            rpc.rpc_async(dst1, RpcTest._slow_add, args=(x, y), timeout=15).wait()

        dist_initialized = dist.is_initialized()
        if not dist_initialized:
            dist.init_process_group(
                backend="gloo",
                init_method=self.file_init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

    @dist_init(setup_rpc=False)
    def test_local_shutdown_with_rpc(self):
        # [FUNCTION DOC] Checks that RPC can be initialized, used, and then locally shut down after
        # a barrier.
        # test that we can start RPC, send RPCs, and then run local shutdown.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        n = self.rank + 1
        dst_rank = n % self.world_size
        rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # A barrier is needed to ensure that all RPCs are processed.
        # Otherwise, some RPCs can timeout since the receiving end
        # has terminated.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        # pass in graceful=False to ensure that we don't wait for other workers.
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    def test_set_and_get_default_rpc_timeout(self):
        # [FUNCTION DOC] Checks that a timeout set in the backend options is reported
        # by get_rpc_timeout.
        timeout = 0.5

        # A new `RpcBackendOptions` is constructed
        # when accessing `self.rpc_backend_options`.
        rpc_backend_options = self.rpc_backend_options
        rpc_backend_options.rpc_timeout = timeout

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )
        set_timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, set_timeout)
        rpc.shutdown()

    @dist_init
    def test_default_timeout_used(self):
        # [FUNCTION DOC] Checks that the default RPC timeout applies when none is given, that
        # changing the default affects only later calls, that zero disables the timeout, and
        # restores the default at the end. Side effect: changes the global timeout.
        """
        Tests that if no timeout is passed into rpc_async and rpc_sync, then the
        default timeout is used.
        """
        dst_rank = (self.rank + 1) % self.world_size
        rpc._set_rpc_timeout(0.001)  # 1 ms
        # futures should time out and be marked with an exception indicating it as such.
        futs = [
            rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=())
            for _ in range(10)
        ]
        expected_error = self.get_timeout_error_regex()
        for fut in futs:
            with self.assertRaisesRegex(RuntimeError, expected_error):
                fut.wait()

        # ensure that if a new timeout is set old futures don't time out but new ones do.
        rpc._set_rpc_timeout(200)  # 200 seconds
        # create a longstanding RPC.
        fut1 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        # now, set a short timeout.
        rpc._set_rpc_timeout(0.001)
        # fut2 should time out, fut1 should not.
        fut2 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut2.wait()
        fut1.wait()

        # Zero timeout means infinity, so future should run to completion.
        rpc._set_rpc_timeout(0)
        rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=()).wait()

        # reset to default timeout so shutdown messages can process cleanly.
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init
    def test_rpc_timeouts(self):
        # [FUNCTION DOC] Checks per-call timeouts for async and sync RPC: expiry raises
        # RuntimeError, a per-call timeout overrides the default, and zero means no timeout. Resets
        # the default at the end.
        # TODO: enable timeouts for rpc.remote/RRef (https://github.com/pytorch/pytorch/issues/33803)
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)
        timeout = 0.1  # 100 ms
        expected_error = self.get_timeout_error_regex()
        # Test async UDF
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if there is no timeout and we use the default
        # RPC timeout.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,)).wait()

        # Test sync UDF
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=timeout)

        # Ensure run to completion if there is no timeout and we use the default
        # RPC timeout.
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # If we set a default timeout for RPCs, it should be respected, though
        # still overridden if we pass in a different timeout to the APIs.
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))

        # The RPCs should run to completion since we override the timeout.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=5).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=5)
        # Passing in a zero timeout should ensure that the RPC won't time out.
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=0).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=0)
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    def test_dist_init_decorator(self):
        # [FUNCTION DOC] Checks that the dist_init decorator works both with arguments and without,
        # by wrapping a trivial function and calling it.
        @dist_init(setup_rpc=False)
        def test_func(self):
            # [FUNCTION DOC] HELPER: Trivial wrapped function returning a fixed string.
            return "expected result"

        self.assertEqual(test_func(self), "expected result")

        @dist_init
        def test_func(self):
            # [FUNCTION DOC] HELPER: Trivial wrapped function returning a fixed string.
            return "expected result"

        self.assertEqual(test_func(self), "expected result")

    def test_use_rpc_pickler(self):
        # [FUNCTION DOC] Checks that the RPC pickler context manager swaps the default pickler for
        # the duration and restores the internal pickler afterward.
        class TestPickler:
            # [CLASS DOC] HELPER: Empty marker class used as a fake pickler.
            pass

        test_pickler = TestPickler()
        with _use_rpc_pickler(test_pickler):
            self.assertTrue(torch.distributed.rpc.api._default_pickler is test_pickler)
        self.assertTrue(
            torch.distributed.rpc.api._default_pickler is _internal_rpc_pickler
        )

    @dist_init
    def test_wait_all(self):
        # [FUNCTION DOC] Checks that the wait-all context manager collects futures created by async
        # RPC in a thread-local list, and removes the list on exit.
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            fut = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
            self.assertTrue(len(_thread_local_var.future_list) == 1)
            self.assertTrue(
                isinstance(_thread_local_var.future_list[0], torch._C.Future)
            )
        self.assertTrue(fut.done())
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_multiple_call(self):
        # [FUNCTION DOC] Checks that the wait-all context manager collects 20 futures while sync
        # calls are interleaved, and that the thread-local list is removed afterward.
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            for i in range(20):
                fut = rpc.rpc_async(dst, torch.add, (torch.ones(i, i), 1))
                res = rpc.rpc_sync(dst, torch.add, (torch.ones(i, i), 1))
                self.assertEqual(res, torch.ones(i, i) + 1)
                self.assertEqual(fut.wait(), torch.ones(i, i) + 1)
            self.assertTrue(len(_thread_local_var.future_list) == 20)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_timeout(self):
        # [FUNCTION DOC] Checks that leaving the wait-all context manager raises the timeout
        # RuntimeError for a timed-out pending future.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error), _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            timeout = 0.1  # 100 ms
            rpc.rpc_async(dst, my_sleep_func, args=(1,), timeout=timeout)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_user_func(self):
        # [FUNCTION DOC] Checks that the wait-all context manager raises ValueError from a pending
        # future that failed in the user function.
        with self.assertRaises(ValueError), _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            rpc.rpc_async(dst, raise_func)
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_wait_all_raise_in_body(self):
        # [FUNCTION DOC] Checks that an exception raised in the body of the wait-all context
        # manager propagates and the thread-local list is cleaned up.
        with self.assertRaises(ValueError), _wait_all():
            raise_func()
        self.assertFalse(hasattr(_thread_local_var, "future_list"))

    @dist_init
    def test_custom_exception_throw_during_reconstruction(self):
        # [FUNCTION DOC] On ranks other than 0, checks that an exception that cannot be rebuilt on
        # the caller still yields RuntimeError mentioning the original remote exception and the
        # CustomException type.
        """
        Test that we still throw info about the remote side exception even when
        we cannot recreate it on client side.
        """
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank != 0:
            exc_caught = False
            dst = worker_name(0)
            try:
                rpc.rpc_sync(dst, custom_raise_func, args=())
            except RuntimeError as e:
                exc_caught = True
                msg = str(e)
                print(f"Got msg {msg}")
                self.assertTrue("Original exception on remote side was" in msg)
                self.assertTrue("CustomException" in msg)
            except BaseException as e:
                raise RuntimeError(f"Failure - expected RuntimeError, got {e}") from e
            finally:
                self.assertTrue(exc_caught)

        dist.barrier()

    timed_out_rpc_event = None

    @staticmethod
    def timed_out_rpc():
        # [FUNCTION DOC] HELPER: Static; blocks until the shared test event is set.
        RpcTest.timed_out_rpc_event.wait()

    @dist_init
    def test_wait_all_exit_early_python(self):
        # [FUNCTION DOC] Checks that waiting on several futures raises the first error (ValueError
        # from a UDF) without waiting for a blocked future; releases the event afterward.
        # Initialize the event in the subprocess.
        RpcTest.timed_out_rpc_event = Event()

        # Wait for all processes to initialize event.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, raise_func)
        fut3 = rpc.rpc_async(dst, raise_func)

        # We should receive the error from fut2
        with self.assertRaisesRegex(ValueError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])

        # Unblock RPC thread for fut1
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_wait_all_exit_early_builtin(self):
        # [FUNCTION DOC] Same as test_wait_all_exit_early_python but with failing builtin calls
        # (mismatched tensor sizes) raising RuntimeError.
        # Initialize the event in the subprocess.
        RpcTest.timed_out_rpc_event = Event()

        # Wait for all processes to initialize event.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))
        fut3 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))

        # We should receive the error from fut2
        with self.assertRaisesRegex(RuntimeError, "size of tensor"):
            torch.futures.wait_all([fut1, fut2, fut3])

        # Unblock RPC thread for fut1
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_wait_all_exit_early_script_function(self):
        # [FUNCTION DOC] Same as test_wait_all_exit_early_python but with failing scripted function
        # calls raising RuntimeError.
        # Initialize the event in the subprocess.
        RpcTest.timed_out_rpc_event = Event()

        # Wait for all processes to initialize event.
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()

        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))
        fut3 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))

        # We should receive the error from fut2
        with self.assertRaisesRegex(RuntimeError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])

        # Unblock RPC thread for fut1
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_function_not_on_callee(self):
        # [FUNCTION DOC] Checks that calling a function that was removed from the callee module
        # raises RuntimeError that the RPC pickler does not serialize it. Side effect: removes
        # foo_add from the module on rank 1.
        # test that if a function does not exist on a callee, we don't crash,
        # instead we get an AttributeError indicating that the func does not exist.
        this_module = sys.modules[__name__]
        caller_worker = "worker0"
        callee_worker = "worker1"

        if self.rank == 1:
            # Use delattr to remove the binding of a func on this nodes
            delattr(this_module, "foo_add")
            # notify remote end that we have removed it.
            rpc.rpc_sync(caller_worker, set_value, args=(self.rank,))

        if self.rank == 0:
            # func exists on caller, but not callee.
            # wait for remote end to remove the binding of foo_add func.
            wait_for_value_future()
            # Ensure that we have the attribute on this module. Otherwise, the test could fail due to a caller-side pickling error.
            self.assertTrue(hasattr(this_module, "foo_add"))
            with self.assertRaisesRegex(RuntimeError, "RPC pickler does not serialize"):
                rpc.rpc_sync(callee_worker, foo_add, args=())

    @dist_init
    def test_non_garbage_collected_user_rref_due_to_local_circular_dependency(self):
        # [FUNCTION DOC] Creates a reference cycle between two objects and an RRef held in one of
        # them so that the RRef is not garbage collected before shutdown; checks that termination
        # still works.
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        a = MyClass(1)
        b = MyClass(2)

        # This is to make Python not garbage collect a and b.
        a.other = b
        b.other = a

        n = self.rank
        a.rref = rpc.remote(dst_worker_name, torch.add, args=(torch.ones(n, n), 2))

    @dist_init(setup_rpc=False)
    def test_use_rref_after_shutdown(self):
        # [FUNCTION DOC] Checks that using a user RRef after a graceful shutdown (which deletes
        # local user RRefs) raises RuntimeError.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        # pass in graceful=True to ensure that local UserRRefs are deleted.
        rpc.shutdown(graceful=True)

        with self.assertRaisesRegex(
            RuntimeError, "Cannot call to_here\\(\\) on it after deletion."
        ):
            rref.to_here()

        with self.assertRaisesRegex(
            RuntimeError, "Cannot call fork an UserRRef after deletion."
        ):
            import torch.distributed.rpc.internal as internal

            internal.serialize(rref)

    @staticmethod
    def _return_gpu_tensor():
        # [FUNCTION DOC] HELPER: Static; returns a random 3 by 3 tensor placed on CUDA device 0.
        return torch.rand(3, 3).cuda(0)

    @staticmethod
    def _return_gpu_tensor_list():
        # [FUNCTION DOC] HELPER: Static; returns a list of two random 3 by 3 tensors, one on CUDA
        # device 0 and one on device 1.
        return [torch.rand(3, 3).cuda(0), torch.rand(3, 3).cuda(1)]

    @staticmethod
    def _gpu_tensor_list_arg(tensor_list):
        # [FUNCTION DOC] HELPER: Static; ignores the given tensor list and returns a random CPU 3
        # by 3 tensor.
        return torch.rand(3, 3)

    def _create_rref(self):
        # [FUNCTION DOC] HELPER: Creates an RRef on the worker two ranks ahead that holds a 2 by 2
        # tensor of ones; returns the RRef.
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote(
            worker_name(owner_rank), torch.add, args=(torch.zeros(2, 2), 1)
        )

    @dist_init
    def test_user_rrefs_confirmed(self):
        # [FUNCTION DOC] Checks that a user RRef passed to another worker via sync RPC is already
        # confirmed by its owner.
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync(worker_name(dst_rank), check_rref_confirmed, args=(rref,))
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        # [FUNCTION DOC] Checks that a user RRef passed via a remote call is confirmed by its
        # owner, using the fetched result.
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote(worker_name(dst_rank), check_rref_confirmed, args=(rref,))
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_py_pickle_not_supported(self):
        # [FUNCTION DOC] Checks that saving an RRef with the regular Python pickler (torch.save)
        # raises RuntimeError.
        local_rref = RRef(35)
        with (
            TemporaryFileName() as fname,
            self.assertRaisesRegex(
                RuntimeError, "Can not pickle rref in python pickler"
            ),
        ):
            torch.save(local_rref, fname)

    @dist_init
    def test_remote_throw(self):
        # [FUNCTION DOC] Checks that fetching an RRef whose remote function raised an error raises
        # an exception with the original message.
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            raise_or_inc,
            args=(torch.ones(2),),
        )
        with self.assertRaisesRegex(Exception, ".*Expected error.*"):
            rref.to_here()

    @dist_init
    def test_non_cont_tensors(self):
        # [FUNCTION DOC] On rank 0, sends a non-contiguous tensor view and its contiguous copy over
        # RPC and checks the returned view is equal and still non-contiguous.
        if self.rank == 0:
            # Create a non-contiguous tensor.
            t = torch.rand(5, 5)
            t_view = t.narrow(1, 2, 2)
            self.assertFalse(t_view.is_contiguous())
            t_cont = t_view.contiguous()
            self.assertTrue(t_cont.is_contiguous())
            self.assertEqual(t_view, t_cont)

            # Send non-cont tensor over RPC.
            next_rank = (self.rank + 1) % self.world_size
            t_ret = rpc.rpc_sync(
                worker_name(next_rank), non_cont_test, args=(t_view, t_cont)
            )

            # Verify the returned tensor.
            self.assertEqual(t_view, t_ret)
            self.assertFalse(t_ret.is_contiguous())

    @dist_init
    def test_callback_simple(self):
        # [FUNCTION DOC] Checks that a then-callback runs with the future result and that waiting
        # on the original future still works afterward.
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        def callback(fut):
            # [FUNCTION DOC] HELPER: Callback; asserts the result and resolves the outer future
            # with the result plus one.
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            set_by_cb.set_result(ret.clone() + 1)

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )

        fut.then(callback)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        self.assertEqual(set_by_cb.result(), torch.ones(n, n) * 2 + 1)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_wrong_arg_num(self):
        # [FUNCTION DOC] Checks that a callback taking the wrong number of arguments yields a
        # future that fails with RuntimeError about missing positional arguments.
        n = self.rank + 1

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )

        cb_fut = fut.then(my_function)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        with self.assertRaisesRegex(
            RuntimeError, "my\\_function\\(\\) missing 2 required positional arguments"
        ):
            cb_fut.wait()

    @dist_init
    def test_callback_wrong_arg_type(self):
        # [FUNCTION DOC] Checks that a callback that fails with a type error yields a future that
        # raises RuntimeError about the unsupported operand types.
        dst = worker_name((self.rank + 1) % self.world_size)

        fut0 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1))
        fut1 = fut0.then(lambda x: x + 1)

        with self.assertRaisesRegex(
            RuntimeError, "unsupported operand type\\(s\\) for \\+"
        ):
            fut1.wait()

    @dist_init
    def test_callback_multi(self):
        # [FUNCTION DOC] Attaches 10 callbacks (each adding its index) to one future and checks
        # every callback result and the original result.
        num_cbs = 10
        n = self.rank + 1

        def callback(idx, fut):
            # [FUNCTION DOC] HELPER: Callback; asserts the future value and returns it plus idx.
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            return ret + idx

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )

        cb_futs = [fut.then(partial(callback, idx)) for idx in range(num_cbs)]

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        for idx in range(num_cbs):
            self.assertEqual(cb_futs[idx].wait(), torch.ones(n, n) * 2 + idx)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_chain(self):
        # [FUNCTION DOC] Chains 20 callbacks on one future, each adding one, and checks the
        # final value.
        n = self.rank + 1

        def callback(fut):
            # [FUNCTION DOC] HELPER: Callback; returns the waited value plus one.
            return fut.wait() + 1

        fut = rpc.rpc_async(
            worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), 1)
        )

        num_cbs = 20
        for _ in range(num_cbs):
            fut = fut.then(callback)

        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_callback_in_rpc(self):
        # [FUNCTION DOC] Checks a UDF that uses a future callback inside the
        # callee (add_use_future_cb).
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(dst1, add_use_future_cb, args=(dst2, torch.ones(2, 2), 1, 2))
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_callback_with_ret(self):
        # [FUNCTION DOC] Checks a callback that issues another async RPC and returns its
        # waited result.
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            # [FUNCTION DOC] HELPER: Callback; issues an async add RPC chained with a callback
            # adding one, and returns the waited result.
            fut2 = rpc.rpc_async(dst, torch.add, args=(fut0.wait(), 1)).then(
                lambda fut1: fut1.wait() + 1
            )

            return fut2.wait()

        fut3 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1)).then(callback)

        self.assertEqual(fut3.wait(), torch.ones(2, 2) + 3)

    @dist_init
    def test_callback_with_error(self):
        # [FUNCTION DOC] Checks that a callback that observes a failed future and raises a
        # different error yields a future raising that error.
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            # [FUNCTION DOC] HELPER: Callback; asserts waiting on the failed future raises
            # ValueError, then raises RuntimeError.
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut0.wait()
            raise RuntimeError("Another expected error")

        fut1 = rpc.rpc_async(dst, raise_func).then(callback)
        with self.assertRaisesRegex(RuntimeError, "Another expected error"):
            fut1.wait()

    @dist_init
    def test_callback_none(self):
        # [FUNCTION DOC] Checks that passing None to then raises TypeError about incompatible
        # function arguments.
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments."):
            rpc.rpc_async(dst, raise_func).then(None)

    @dist_init
    def test_add_done_callback(self):
        # [FUNCTION DOC] Checks add_done_callback runs by adding a later then-callback to wait for
        # completion and asserting that a flag set by the done callback is true.
        set_by_cb = False
        n = self.rank + 1

        def callback(fut):
            # [FUNCTION DOC] HELPER: Done callback; waits on the future and sets the flag in the
            # enclosing scope.
            nonlocal set_by_cb
            fut.wait()
            set_by_cb = True

        fut = rpc.rpc_async(
            worker_name(n % self.world_size),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )

        fut.add_done_callback(callback)
        fut_then = fut.then(lambda _: True)

        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

        # We have no guarantee that the add_done_callback fn will execute before the test finishes.
        # Adding a 'then' callback that runs afterwards to guarantee we wait for the first callback
        fut_then.wait()
        self.assertTrue(set_by_cb)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_mark_future_twice(self):
        # [FUNCTION DOC] Checks that setting a result on an already completed RPC future
        # raises RuntimeError.
        fut = rpc.rpc_async(
            worker_name((self.rank + 1) % self.world_size),
            torch.add,
            args=(torch.zeros(2, 2), 1),
        )
        self.assertEqual(fut.wait(), torch.zeros(2, 2) + 1)
        with self.assertRaisesRegex(
            RuntimeError, "Future can only be marked completed once"
        ):
            fut.set_result(1)

    @dist_init
    def test_pickle_future(self):
        # [FUNCTION DOC] Checks that a torch future cannot be pickled when sent as an argument via
        # sync, async or remote calls (RuntimeError).
        fut = torch.futures.Future()
        errMsg = "Can not pickle torch.futures.Future"

        dst = worker_name((self.rank + 1) % self.world_size)
        with TemporaryFileName(), self.assertRaisesRegex(RuntimeError, errMsg):
            rpc.rpc_sync(dst, fail_on_fut, args=(fut,))

        with TemporaryFileName(), self.assertRaisesRegex(RuntimeError, errMsg):
            rpc.rpc_async(dst, fail_on_fut, args=(fut,))

        with TemporaryFileName(), self.assertRaisesRegex(RuntimeError, errMsg):
            rpc.remote(dst, fail_on_fut, args=(fut,))

    @dist_init
    def test_future_done(self):
        # [FUNCTION DOC] Checks that a future reports done after being waited on.
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, torch.add, args=(torch.zeros(2), 1))
        fut.wait()
        self.assertTrue(fut.done())

    @dist_init
    def test_future_done_exception(self):
        # [FUNCTION DOC] Checks that a failed future reports done after its error was raised
        # from wait.
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, raise_func)
        with self.assertRaisesRegex(ValueError, "Expected error"):
            fut.wait()
        self.assertTrue(fut.done())

    def _test_future_cb(self, func):
        # [FUNCTION DOC] HELPER: Calls func on the next worker, which uses another worker (two
        # ranks ahead) and a callback internally; asserts the result is ones plus 3.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(dst1, func, args=(dst2, torch.ones(2, 2), 1, 2))
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_future_in_rpc(self):
        # [FUNCTION DOC] Runs the future callback helper with the set-result variant.
        self._test_future_cb(add_use_future_set_result)

    @dist_init
    def test_future_nested_callback(self):
        # [FUNCTION DOC] Runs the future callback helper with the nested-callback variant.
        self._test_future_cb(add_use_future_nested_cb)

    def _test_async_function_raise(self, mode):
        # [FUNCTION DOC] HELPER: Asserts that calling an async_execution function that raises,
        # using the given mode, raises RuntimeError with the expected message.
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size), async_raise_func, mode
            )

    @dist_init
    def test_async_function_raise(self):
        # [FUNCTION DOC] Runs the async function raise check in sync mode.
        self._test_async_function_raise(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_raise_async(self):
        # [FUNCTION DOC] Runs the async function raise check in async mode.
        self._test_async_function_raise(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_raise_remote(self):
        # [FUNCTION DOC] Runs the async function raise check in remote mode.
        self._test_async_function_raise(RPCExecMode.REMOTE)

    def _test_async_function_wrong_return_type(self, mode):
        # [FUNCTION DOC] HELPER: Asserts that calling an async_execution function that returns a
        # non-future, in the given mode, raises RuntimeError about needing a future.
        errMsg = (
            "Functions decorated with @rpc\\.async_function must return a "
            "torch\\.futures\\.Future object,"
        )
        with self.assertRaisesRegex(RuntimeError, errMsg):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size), async_wrong_type, mode
            )

    @dist_init
    def test_async_function_wrong_return_type(self):
        # [FUNCTION DOC] Runs the wrong return type check in sync mode.
        self._test_async_function_wrong_return_type(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_wrong_return_type_async(self):
        # [FUNCTION DOC] Runs the wrong return type check in async mode.
        self._test_async_function_wrong_return_type(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        # [FUNCTION DOC] Runs the wrong return type check in remote mode.
        self._test_async_function_wrong_return_type(RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_simple(self):
        # [FUNCTION DOC] Checks a simple async_execution add function reached through
        # another worker.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        ret = rpc.rpc_sync(dst1, async_add, args=(dst2, torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _test_async_function(self, fn, mode=RPCExecMode.SYNC):
        # [FUNCTION DOC] HELPER: Runs fn with the next worker as callee, the worker two ranks ahead
        # as target, and arguments of ones, 1 and 2 in the given mode (default sync); asserts the
        # result is ones plus 3.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        args = (dst2, torch.ones(2, 2), 1, 2)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + 3)

    @dist_init
    def test_async_function_with_future_ctor(self):
        # [FUNCTION DOC] Checks an async function that builds its future by constructor.
        self._test_async_function(async_add_with_future_ctor)

    @dist_init
    def test_async_function_with_future_ctor_remote(self):
        # [FUNCTION DOC] Same as test_async_function_with_future_ctor but via remote calls.
        self._test_async_function(async_add_with_future_ctor, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_chained(self):
        # [FUNCTION DOC] Checks an async function that chains with a callback.
        self._test_async_function(async_add_chained)

    @dist_init
    def test_async_function_chained_remote(self):
        # [FUNCTION DOC] Same as test_async_function_chained but via remote calls.
        self._test_async_function(async_add_chained, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_nested(self):
        # [FUNCTION DOC] Checks a nested async function.
        self._test_async_function(async_add_nested)

    @dist_init
    def test_async_function_nested_remote(self):
        # [FUNCTION DOC] Same as test_async_function_nested but via remote calls.
        self._test_async_function(async_add_nested, RPCExecMode.REMOTE)

    @dist_init
    def test_async_static_method(self):
        # [FUNCTION DOC] Checks an async_execution static method.
        self._test_async_function(AsyncExecutionClass.static_async_add)

    @dist_init
    def test_async_static_method_remote(self):
        # [FUNCTION DOC] Same as test_async_static_method but via remote calls.
        self._test_async_function(
            AsyncExecutionClass.static_async_add, RPCExecMode.REMOTE
        )

    @dist_init
    def test_async_class_method(self):
        # [FUNCTION DOC] Checks an async_execution class method.
        self._test_async_function(AsyncExecutionClass.class_async_add)

    @dist_init
    def test_async_class_method_remote(self):
        # [FUNCTION DOC] Same as test_async_class_method but via remote calls.
        self._test_async_function(
            AsyncExecutionClass.class_async_add, RPCExecMode.REMOTE
        )

    def _test_test_async_class_rref_proxy(self, mode=RPCExecMode.SYNC):
        # [FUNCTION DOC] HELPER: Creates an RRef of AsyncExecutionClass and calls its static, class
        # and bound async_execution methods through the sync, async or remote proxy per mode;
        # asserts the combined result equals 3 times 4 times x.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        rref = rpc.remote(dst1, AsyncExecutionClass)

        x = torch.ones(2, 2)
        y = torch.ones(2, 2) + 1
        if mode == RPCExecMode.SYNC:
            ret = rref.rpc_sync().static_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().class_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().bound_async_add(dst2, x, x, y)
        elif mode == RPCExecMode.ASYNC:
            ret = rref.rpc_async().static_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().class_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().bound_async_add(dst2, x, x, y).wait()
        elif mode == RPCExecMode.REMOTE:
            ret = rref.remote().static_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().class_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().bound_async_add(dst2, x, x, y).to_here()

        self.assertEqual(ret, 3 * 4 * x)

    @dist_init
    def test_async_class_rref_proxy(self):
        # [FUNCTION DOC] Runs the RRef proxy async-class check in sync mode.
        self._test_test_async_class_rref_proxy()

    @dist_init
    def test_async_class_rref_proxy_async(self):
        # [FUNCTION DOC] Runs the RRef proxy async-class check in async mode.
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.ASYNC)

    @dist_init
    def test_async_class_rref_proxy_remote(self):
        # [FUNCTION DOC] Runs the RRef proxy async-class check in remote mode.
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.REMOTE)

    def _test_async_function_multi(self, fn, mode=RPCExecMode.SYNC):
        # [FUNCTION DOC] HELPER: Runs fn on the next worker with 20 repetitions of step 3 and a
        # target two ranks ahead in the given mode; asserts the result equals ones plus 60.
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)

        num = 20
        step = 3
        args = (dst2, torch.ones(2, 2), num, step)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + num * step)

    @dist_init
    def test_async_function_multi_chained(self):
        # [FUNCTION DOC] Runs the multi-step helper with the chained function in sync mode.
        self._test_async_function_multi(async_add_chained_multi)

    @dist_init
    def test_async_function_multi_chained_async(self):
        # [FUNCTION DOC] Runs the multi-step helper with the chained function in async mode.
        self._test_async_function_multi(async_add_chained_multi, RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_multi_chained_remote(self):
        # [FUNCTION DOC] Runs the multi-step helper with the chained function in remote mode.
        self._test_async_function_multi(async_add_chained_multi, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_multi_fanout(self):
        # [FUNCTION DOC] Runs the multi-step helper with the fan-out function in sync mode.
        self._test_async_function_multi(async_add_multi_fanout)

    @dist_init
    def test_async_function_multi_fanout_async(self):
        # [FUNCTION DOC] Runs the multi-step helper with the fan-out function in async mode.
        self._test_async_function_multi(async_add_multi_fanout, RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_multi_fanout_remote(self):
        # [FUNCTION DOC] Runs the multi-step helper with the fan-out function in remote mode.
        self._test_async_function_multi(async_add_multi_fanout, RPCExecMode.REMOTE)

    def _test_return_future(self, mode):
        # [FUNCTION DOC] HELPER: Asserts that returning a torch future from a normal remote
        # function raises RuntimeError about pickling futures, for the given mode.
        with self.assertRaisesRegex(
            RuntimeError, "Can not pickle torch.futures.Future"
        ):
            self._run_func_in_mode(
                worker_name((self.rank + 1) % self.world_size), return_future, mode
            )

    @dist_init
    def test_return_future(self):
        # [FUNCTION DOC] Runs the return-future check in sync mode.
        self._test_return_future(RPCExecMode.SYNC)

    @dist_init
    def test_return_future_async(self):
        # [FUNCTION DOC] Runs the return-future check in async mode.
        self._test_return_future(RPCExecMode.ASYNC)

    @dist_init
    def test_return_future_remote(self):
        # [FUNCTION DOC] Runs the return-future check in remote mode.
        self._test_return_future(RPCExecMode.REMOTE)

    @dist_init
    def test_rref_timeout(self):
        # [FUNCTION DOC] On rank 0, checks that an RRef creation with a 10 ms timeout fails: the
        # creation future times out, a later fetch raises RuntimeError about RRef creation, and the
        # owner and fork counts settle.
        # This test is similar to ones in FaultyProcessGroupTest, but is meant to be
        # run with other backends besides ProcessGroup.
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 10 ms timeout
        rref = rpc.remote(dst_worker, my_sleep_func, args=(2,), timeout=0.01)
        # Future corresponding to the remote creation should time out.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        wait_until_owners_and_forks_on_rank(1, 1, rank=1)

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "init_pg_then_rpc does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614.",
    )
    def test_init_pg_then_rpc(self):
        # [FUNCTION DOC] Initializes a gloo process group before RPC, then checks RPC and the
        # process group both work; skipped (passes) under TCP init.
        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        # Test RPC.
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(
            worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test PG
        dist.barrier()

        rpc.shutdown()

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "init_rpc_then_pg does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614.",
    )
    def test_init_rpc_then_pg(self):
        # [FUNCTION DOC] Initializes RPC before a gloo process group, then checks RPC and the
        # process group both work; skipped (passes) under TCP init.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        dist.init_process_group(
            backend="gloo",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Test RPC.
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(
            worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1)
        )
        self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test PG
        dist.barrier()

        rpc.shutdown()

    @dist_init
    def test_wait_all_with_exception(self):
        # [FUNCTION DOC] Checks that wait_all over futures that all fail raises ValueError.
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = [rpc.rpc_async(dst, raise_func) for _ in range(10)]

        with self.assertRaisesRegex(ValueError, "Expected error"):
            torch.futures.wait_all(futs)

    @dist_init
    def test_wait_all_with_partial_exception(self):
        # [FUNCTION DOC] Checks that wait_all raises ValueError when only the last of many
        # futures fails.
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = [
            rpc.rpc_async(dst, torch.add, args=(torch.ones(2), 1)) for _ in range(10)
        ]

        futs.append(rpc.rpc_async(dst, raise_func))

        with self.assertRaisesRegex(ValueError, "Expected error"):
            torch.futures.wait_all(futs)

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("RPC_INIT_WITH_TCP", None) == "1",
        "Test does not work with TCP init, see https://github.com/pytorch/pytorch/issues/46491",
    )
    def test_init_rpc_twice(self):
        # [FUNCTION DOC] Checks that RPC can be initialized, shut down, and initialized again with
        # a different init method, and still works. Skipped (passes) under TCP init.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )
        rpc.shutdown()

        # Wait for all init to complete.
        dist.barrier()

        # Use a different file name for the next initialization
        new_backend_options = self.rpc_backend_options
        new_backend_options.init_method += "init_2"

        # Ensure rpc initialization works again.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=new_backend_options,
        )

        # Verify RPCs work after re-init.
        dst = worker_name((self.rank + 1) % self.world_size)
        rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))
        rpc.rpc_sync(dst, foo_add, args=())

        rpc.shutdown()

    def test_wrong_types(self):
        # [FUNCTION DOC] Checks that init_rpc raises TypeError for a string backend and for options
        # that are not an RpcBackendOptions instance.
        with self.assertRaisesRegex(
            TypeError,
            "Argument backend must be a member of BackendType",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend="TENSORPIPE",
            )

        with self.assertRaisesRegex(
            TypeError,
            "Argument rpc_backend_options must be an instance of RpcBackendOptions",
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=self.rpc_backend,
                rpc_backend_options={"init_method": self.init_method},
            )

    def test_cannot_infer_backend_from_options(self):
        # [FUNCTION DOC] Checks that init_rpc raises TypeError when the backend is omitted and the
        # options come from an unknown class.
        # An exception should be raised if the backend isn't specified but
        # options are given which are not an instance of any of the known
        # agents' option classes.
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(TypeError, "Could not infer backend for options"):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                # Do _not_ pass backend.
                rpc_backend_options=rpc_backend_options,
            )

    @dist_init
    def test_owner_rref_backward(self):
        # [FUNCTION DOC] Checks RRef backward on an owner RRef: without a context, within a
        # distributed autograd context, with retained graph (double backward), and the error cases
        # for non-differentiable tensors, non-scalar outputs, an unknown context id and a
        # non-tensor value.
        dst = worker_name((self.rank + 1) % self.world_size)
        t1 = torch.rand(10, 10, requires_grad=True)
        rref = rpc.RRef(t1.sum() + t1.sum())
        rref.backward()
        expected_grad = torch.ones_like(t1) * 2
        self.assertEqual(expected_grad, t1.grad)

        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id)
            self.assertEqual(expected_grad, dist_autograd.get_gradients(context_id)[t1])

        # Double backward.
        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(
                expected_grad * 2, dist_autograd.get_gradients(context_id)[t1]
            )

        # Test errors.
        with self.assertRaisesRegex(
            RuntimeError, "tensors does not require grad and does not have a grad_fn"
        ):
            rpc.RRef(torch.rand(10)).backward()

        with self.assertRaisesRegex(
            RuntimeError, "grad can be implicitly created only for scalar outputs"
        ):
            rpc.RRef(torch.rand(10, requires_grad=True)).backward()

        with self.assertRaisesRegex(
            RuntimeError, "Could not find autograd context with id: 100"
        ):
            rpc.RRef(torch.rand(10, requires_grad=True).sum()).backward(100)

        with self.assertRaisesRegex(
            RuntimeError, "RRef should contain a tensor for .backward()"
        ):
            rpc.RRef("foo").backward()

    @staticmethod
    def _sum(x):
        # [FUNCTION DOC] HELPER: Static; returns the sum of x.
        return x.sum()

    @staticmethod
    def _identity(x):
        # [FUNCTION DOC] HELPER: Static; returns x unchanged.
        return x

    @dist_init
    def test_user_rref_backward(self):
        # [FUNCTION DOC] Checks RRef backward on a user RRef with a distributed autograd context
        # (including retain graph and gradient accumulation), and the errors for a non-tensor value
        # and for a missing context id.
        dst = worker_name((self.rank + 1) % self.world_size)
        t = torch.rand(10, requires_grad=True)
        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._sum, args=(t,))
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(
                torch.ones_like(t) * 2, dist_autograd.get_gradients(context_id)[t]
            )

        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._identity, args=("foo",))
            with self.assertRaisesRegex(
                RuntimeError, "RRef should contain a tensor for .backward()"
            ):
                rref.backward(context_id)

            with self.assertRaisesRegex(
                RuntimeError,
                "User RRefs require 'dist_autograd_ctx_id' to be specified",
            ):
                rref.backward()

    @dist_init(setup_rpc=False)
    def test_shutdown_errors(self):
        # [FUNCTION DOC] Checks that errors during shutdown surface: on non-leader ranks, with the
        # broadcast and the RRef deletion functions monkey-patched to fail, shutdown raises the
        # RRef error; on rank 0 the all-gather times out. Restores the patched functions. Side
        # effect: temporarily patches rpc.api.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        if self.rank != 0:
            og_func = rpc.api._broadcast_to_followers
            og_rref_func = rpc.api._delete_all_user_and_unforked_owner_rrefs

            # Monkey-patch _broadcast_to_followers to fail, which would ensure
            # _all_gather on leader raises an exception.
            def raise_error(sequence_id, objects_map):
                # [FUNCTION DOC] HELPER: Replacement for the broadcast to followers; calls the
                # original and then raises RuntimeError.
                og_func(sequence_id, objects_map)
                raise RuntimeError("simulation")

            # Monkey-patch _delete_all_user_and_unforked_owner_rrefs to fail,
            # which would ensure barrier is not called on followers.
            def rref_error():
                # [FUNCTION DOC] HELPER: Replacement for the RRef deletion function;
                # raises RuntimeError.
                raise RuntimeError("simulation rref")

            try:
                rpc.api._broadcast_to_followers = raise_error
                rpc.api._delete_all_user_and_unforked_owner_rrefs = rref_error
                with self.assertRaisesRegex(RuntimeError, "simulation rref"):
                    rpc.shutdown()
            finally:
                rpc.api._broadcast_to_followers = og_func
                rpc.api._delete_all_user_and_unforked_owner_rrefs = og_rref_func
        else:
            with self.assertRaisesRegex(RuntimeError, "timed out in _all_gather"):
                rpc.shutdown()

        dist.barrier()

    @dist_init
    def test_my_parameter_server(self):
        # [FUNCTION DOC] Runs the parameter server helper with dense gradients.
        self._my_parameter_server(False)


class CudaRpcTest(RpcAgentTestFixture):
    # [CLASS DOC] CUDA-specific RPC tests, mixed into the RPC agent fixture.
    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/154587")
    @skip_if_lt_x_gpu(2)
    @dist_init
    def test_profiler_remote_cuda(self):
        # [FUNCTION DOC] On rank 1, profiles remote calls to two ranks that use CUDA devices 0 and
        # 1; checks that async events have no device time and that remote events for the expected
        # ops have kernels on the right device and positive device time, and that all expected
        # remote events appear. Needs at least two GPUs; disabled on Linux by skipIf.
        if self.rank != 1:
            return

        dst_cuda_0 = (self.rank + 1) % self.world_size
        dst_cuda_1 = (self.rank + 2) % self.world_size
        dst_worker_cuda_0 = worker_name(dst_cuda_0)
        dst_worker_cuda_1 = worker_name(dst_cuda_1)

        with _profile(use_cuda=True) as p:
            fut1 = rpc.rpc_async(dst_worker_cuda_0, udf_with_torch_ops, args=(0,))
            fut2 = rpc.rpc_async(dst_worker_cuda_1, udf_with_torch_ops, args=(1,))
            fut1.wait()
            fut2.wait()

        def get_name(event):
            # [FUNCTION DOC] HELPER: Strips everything up to and including the remote-op marker
            # from an event name.
            return event.name[event.name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR) :]

        function_events = p.function_events
        for event in function_events:
            if event.is_async:
                self.assertEqual(0, event.device_time_total)
                self.assertEqual([], event.kernels)
                self.assertEqual(0, event.device_time)
            else:
                if event.node_id == 1:
                    continue
                self.assertTrue(event.node_id in [dst_cuda_0, dst_cuda_1])
                if get_name(event) in EXPECTED_REMOTE_EVENTS:
                    self.assertGreater(event.device_time_total, 0)
                    self.assertEqual(1, len(event.kernels))
                    kernel = event.kernels[0]
                    if event.node_id == dst_cuda_0:
                        self.assertEqual(kernel.device, 0)
                    if event.node_id == dst_cuda_1:
                        self.assertEqual(kernel.device, 1)
                    self.assertGreater(event.device_time, 0)

        # Validate that EXPECTED_REMOTE_EVENTS is a subset of remotely profiled
        # events.
        remote_events = [event for event in function_events if event.is_remote]
        remote_event_names = [
            get_name(event)
            for event in remote_events
            if get_name(event) in EXPECTED_REMOTE_EVENTS
        ]
        self.assertEqual(set(remote_event_names), set(EXPECTED_REMOTE_EVENTS))


class TensorPipeAgentRpcTest(RpcAgentTestFixture, RpcTestCommon):
    # [CLASS DOC] Tests specific to the TensorPipe agent (option validation, timeouts, thread
    # counts, sparse tensor variants of the common tests and dynamic RPC initialization).
    def test_mismatched_type_for_options(self):
        # [FUNCTION DOC] Checks that init_rpc raises TypeError when the TensorPipe backend is
        # requested with options that are not TensorPipe options.
        # An exception should be raised if the options are not an instance of
        # TensorPipeRpcBackendOptions.
        rpc_backend_options = FooBackendOptions(self.init_method)

        with self.assertRaisesRegex(
            TypeError, "`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`"
        ):
            rpc.init_rpc(
                name=worker_name(self.rank),
                rank=self.rank,
                world_size=self.world_size,
                backend=rpc.BackendType.TENSORPIPE,
                rpc_backend_options=rpc_backend_options,
            )

    def test_infer_backend_from_options(self):
        # [FUNCTION DOC] Checks that TensorPipe options alone are enough for init_rpc to select the
        # TensorPipe agent.
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.init_method, _transports=tp_transports()
        )

        rpc.init_rpc(
            name=worker_name(self.rank),
            rank=self.rank,
            world_size=self.world_size,
            # Do _not_ pass backend.
            rpc_backend_options=rpc_backend_options,
        )

        self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)

    # FIXME Merge this test with the corresponding one in RpcTest.
    @dist_init(setup_rpc=False)
    def test_set_and_get_num_worker_threads(self):
        # [FUNCTION DOC] Checks that the thread count in the options (27) is reported by the agent
        # debug info. Needs dist_init without automatic setup. FIXME in code: merge with the
        # equivalent test in RpcTest.
        NUM_THREADS = 27
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_worker_threads=NUM_THREADS,
            _transports=tp_transports(),
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertEqual(int(info["agent.thread_pool_size"]), NUM_THREADS)
        rpc.shutdown()

    # FIXME Merge this test with the corresponding one in RpcTest.
    @unittest.skipIf(IS_MACOS, "https://github.com/pytorch/pytorch/issues/70546")
    @dist_init(setup_rpc=False)
    def test_tensorpipe_set_default_timeout(self):
        # [FUNCTION DOC] Checks that a default timeout given in the TensorPipe options is
        # returned by get_rpc_timeout. Skipped on MacOS. FIXME in code: merge with the equivalent
        # test in RpcTest.
        # Set a high timeout since it doesn't affect test runtime and ensures
        # the test doesn't erroneously timeout due to slow machines.
        timeout = 100
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=self.rpc_backend_options.init_method,
            num_worker_threads=self.rpc_backend_options.num_worker_threads,
            rpc_timeout=timeout,
            _transports=tp_transports(),
        )
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        default_timeout = rpc.get_rpc_timeout()
        self.assertEqual(default_timeout, timeout)
        rpc.shutdown()

    # FIXME Merge this test with the corresponding one in RpcTest.
    @dist_init(setup_rpc=False)
    def test_tensorpipe_options_throw_on_timedelta_timeout(self):
        # [FUNCTION DOC] Checks that constructing TensorPipe options with a timedelta timeout
        # raises TypeError. FIXME in code: merge with the equivalent test in RpcTest.
        from datetime import timedelta

        timeout = timedelta()
        # Ensure that constructing TensorPipeRpcBackendOptions with timedelta fails
        with self.assertRaisesRegex(TypeError, "incompatible constructor arguments"):
            rpc.TensorPipeRpcBackendOptions(
                init_method=self.rpc_backend_options.init_method,
                num_worker_threads=self.rpc_backend_options.num_worker_threads,
                rpc_timeout=timeout,
            )

    @dist_init
    def _test_rref_get_type_timeout(self, blocking):
        # [FUNCTION DOC] HELPER: Checks that _get_type with a 0.5 second timeout on an RRef whose
        # creation takes longer raises the timeout RuntimeError, inline if blocking or on wait
        # otherwise; then waits for the creation to finish to avoid a shutdown race. The dist_init
        # decorator is applied to this helper.
        # Test where we try to get the type of a RRef from an owner, but RRef
        # creation is slower than timeout passed into _get_type.
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.5
        expected_err = self.get_timeout_error_regex()
        # Blocking: blocks on inline call
        if blocking:
            with self.assertRaisesRegex(RuntimeError, expected_err):
                slow_rref._get_type(timeout=timeout, blocking=blocking)
        # Non-blocking: blocks on wait
        else:
            fut = slow_rref._get_type(timeout=timeout, blocking=blocking)
            with self.assertRaisesRegex(RuntimeError, expected_err):
                fut.wait()

        # FIXME We wait until the remote completed creating the OwnerRRef
        # because there's currently a race if we shut down RPC before that.
        slow_rref.to_here()

    def test_rref_get_type_timeout_blocking(self):
        # [FUNCTION DOC] Runs the get-type timeout check in blocking mode.
        self._test_rref_get_type_timeout(blocking=True)

    def test_rref_get_type_timeout_non_blocking(self):
        # [FUNCTION DOC] Runs the get-type timeout check in non-blocking mode.
        self._test_rref_get_type_timeout(blocking=False)

    @dist_init
    def test_op_with_invalid_args(self):
        # [FUNCTION DOC] Checks that calling torch.add remotely with no arguments raises
        # RuntimeError about no matching schema.
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(
            RuntimeError,
            "Overloaded torch operator invoked from Python failed to match any schema",
        ):
            rpc.rpc_sync(dst, torch.add, args=())

    def _test_rref_proxy_timeout(self, rref_proxy_api):
        # [FUNCTION DOC] HELPER: For the named RRef proxy API, checks that a per-call timeout on a
        # slow method raises the timeout RuntimeError, and that a short timeout on a slow-to-create
        # RRef also times out (at proxy creation, since the type query blocks on creation). Waits
        # for creation to finish at the end.
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2),))
        # Ensure RRef is created on remote node.
        rref.to_here()
        rref_api = getattr(rref, rref_proxy_api)
        self.assertTrue(
            rref_api is not None, f"Failed to get RRef proxy api: {rref_proxy_api}"
        )
        expected_error = self.get_timeout_error_regex()
        timeout = 2
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_slow_method(torch.ones(2, 2))
            if rref_api == rref.rpc_async:
                result.wait()
            elif rref_api == rref.remote:
                result._get_future().wait()

        # Case where rpc.remote() is stuck and exceeds timeout
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.01
        rref_api = getattr(slow_rref, rref_proxy_api)
        # Note that even when we call rref.rpc_async() in this case, we
        # time out in future creation, not waiting for future. This is because
        # rref proxy function calls rref._get_type before returning future,
        # which blocks on the RRef being created on owner node, until the
        # specified timeout.
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_instance_method(torch.ones(2, 2))
            # rpc_async returns immediately and surfaces a timeout through wait()
            if rref_api == slow_rref.rpc_async:
                result.wait()

        # FIXME We wait until the remote completed creating the OwnerRRef
        # because there's currently a race if we shut down RPC before that.
        slow_rref.to_here()

    @dist_init
    def test_rref_proxy_timeout(self):
        # [FUNCTION DOC] Runs the RRef proxy timeout check for the sync, async and remote proxies.
        for rpc_api in ["rpc_sync", "rpc_async", "remote"]:
            self._test_rref_proxy_timeout(rpc_api)

    @dist_init
    def test_send_to_rank_sparse(self):
        # [FUNCTION DOC] Checks sending coalesced and uncoalesced sparse tensors by rank in sync,
        # async and remote modes.
        dst_rank = (self.rank + 1) % self.world_size

        # Test sparse tensor
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor()
            y = build_sparse_tensor()
            expected_tensor = x + y
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor(coalesce=True)
            y = build_sparse_tensor(coalesce=True)
            expected_tensor = x + y
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)

    @dist_init
    def test_self_py_udf_remote_sparse(self):
        # [FUNCTION DOC] Sparse tensor version of the self remote UDF helper.
        self._self_py_udf_remote(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
        )

    @dist_init
    def test_self_remote_rref_as_rpc_arg_sparse(self):
        # [FUNCTION DOC] Sparse tensor version of the RRef-as-RPC-argument helper.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(
            dst, build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg_sparse(self):
        # [FUNCTION DOC] Sparse tensor version of the RRef-as-RPC-argument helper with self
        # as destination.
        self._self_remote_rref_as_rpc_arg(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
        )

    @dist_init
    def test_self_remote_rref_as_remote_arg_sparse(self):
        # [FUNCTION DOC] Sparse tensor version of the RRef-as-remote-argument helper.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(
            dst, build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor()
        )

    @dist_init
    def test_self_remote_rref_as_self_remote_arg_sparse(self):
        # [FUNCTION DOC] Sparse tensor version of the RRef-as-remote-argument helper with self
        # as destination.
        self._self_remote_rref_as_remote_arg(
            rpc.get_worker_info(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
        )

    def test_world_size_one_sparse(self):
        # [FUNCTION DOC] Runs the world-size-one helper with sparse tensors.
        self._world_size_one(build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_multi_rpc_sparse(self):
        # [FUNCTION DOC] Runs the multi-RPC helper with sparse tensors.
        self._multi_rpc(True)

    def test_wait_all_workers_sparse(self):
        # [FUNCTION DOC] Runs the wait-all-workers helper with the sparse heavy payload.
        self._wait_all_workers(heavy_rpc_sparse, build_sparse_tensor())

    def test_wait_all_workers_twice_sparse(self):
        # [FUNCTION DOC] Runs the wait-all-workers-twice helper with the sparse heavy payload.
        self._wait_all_workers_twice(heavy_rpc_sparse, build_sparse_tensor())

    @dist_init
    def test_py_sparse_tensors_in_container(self):
        # [FUNCTION DOC] Checks an RPC that sums a list of sparse tensors.
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [build_sparse_tensor(), build_sparse_tensor()]
        ret = rpc.rpc_sync(worker_name(dst_rank), my_container_sum, args=(a,))
        self.assertEqual(ret, my_container_sum(a))

    @dist_init
    def test_nested_rpc_sparse(self):
        # [FUNCTION DOC] Runs the nested RPC helper with sparse tensors.
        self._nested_rpc(nested_rpc_sparse, build_sparse_tensor() * 2)

    @dist_init
    def test_stress_heavy_rpc_sparse(self):
        # [FUNCTION DOC] Runs the stress helper with the sparse heavy payload.
        self._stress_test_rpc(
            heavy_rpc_sparse, repeat=20, args=(build_sparse_tensor(),)
        )

    @dist_init
    def test_builtin_remote_ret_sparse(self):
        # [FUNCTION DOC] Runs the builtin remote return helper with sparse tensors.
        self._builtin_remote_ret(
            build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 2
        )

    @dist_init
    def test_builtin_remote_self_sparse(self):
        # [FUNCTION DOC] Runs the builtin remote-to-self helper with sparse tensors.
        self._builtin_remote_self(
            build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 2
        )

    @dist_init
    def test_multi_builtin_remote_ret_sparse(self):
        # [FUNCTION DOC] Runs the multi remote call helper with sparse tensors and builtin add.
        self._test_multi_remote_call(torch.add, True, args_fn=RpcTest._multi_args_fn)

    @dist_init
    def test_multi_py_udf_remote_sparse(self):
        # [FUNCTION DOC] Runs the multi remote call helper with sparse tensors and
        # keyword arguments.
        self._test_multi_remote_call(
            my_function, True, kwargs_fn=RpcTest._multi_kwargs_fn
        )

    @dist_init
    def test_py_rref_args_sparse(self):
        # [FUNCTION DOC] Runs the RRef arguments helper with sparse tensors.
        self._py_rref_args(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 4,
        )

    @dist_init
    def test_py_rref_args_user_share_sparse(self):
        # [FUNCTION DOC] Runs the RRef user-share helper with sparse tensors.
        self._py_rref_args_user_share(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 6,
        )

    @dist_init
    def test_py_rpc_rref_args_sparse(self):
        # [FUNCTION DOC] Runs the RPC RRef arguments helper with sparse tensors.
        self._py_rpc_rref_args(
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor(),
            build_sparse_tensor() * 6,
        )

    @dist_init
    def test_nested_remote_sparse(self):
        # [FUNCTION DOC] Runs the nested remote helper with sparse tensors.
        self._nested_remote(
            nested_remote_sparse, build_sparse_tensor() + build_sparse_tensor()
        )

    @dist_init
    def test_nested_rref_sparse(self):
        # [FUNCTION DOC] Runs the nested RRef helper with sparse tensors.
        self._nested_rref(
            nested_rref_sparse, build_sparse_tensor() * 2, build_sparse_tensor() * 2
        )

    @dist_init
    def test_nested_rref_stress_sparse(self):
        # [FUNCTION DOC] Runs the nested RRef stress helper with sparse tensors.
        self._nested_rref_stress(
            nested_rref_sparse, build_sparse_tensor() * 2, build_sparse_tensor() * 2
        )

    @dist_init
    def test_my_parameter_server_sparse(self):
        # [FUNCTION DOC] Runs the parameter server helper with sparse gradients.
        self._my_parameter_server(True)

    # Test init_rpc without world_size argument
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc(self):
        # [FUNCTION DOC] Checks that init_rpc without a world size (dynamic RPC) works and
        # shuts down.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            rpc_backend_options=self.rpc_backend_options,
        )
        rpc.shutdown()

    # Dynamic RPC new ranks communicate with existing ranks
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_new_rank_can_communicated_with_existing_rank(self):
        # [FUNCTION DOC] Checks that, with dynamic RPC, ranks that join after rank 0 can call rank
        # 0, using barriers to order the joins and the calls.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        if self.rank == 0:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # Rank 0 will be initialized with RPC after this barrier
        dist.barrier()

        if self.rank != 0:
            # Newly joined ranks will be able to communicate with rank 0, since that was created first
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )
            result = rpc.rpc_sync(
                worker_name(0), torch.add, args=(torch.tensor(1), torch.tensor(1))
            )
            self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)

        # Barrier to ensure that all rpc_sync calls are finished
        dist.barrier()
        rpc.shutdown()

    # Dynamic RPC existing ranks can communicate with new ranks
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank(self):
        # [FUNCTION DOC] Checks that, with dynamic RPC, rank 0 can call the ranks that joined
        # after it.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        if self.rank == 0:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # Rank 0 will be initialized with RPC after this barrier
        dist.barrier()

        # Rest of ranks join after barrier
        if self.rank != 0:
            # Newly joined ranks will be able to communicate with rank 0, since that was created first
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        dist.barrier()
        if self.rank == 0:
            for i in range(1, self.world_size):
                result = rpc.rpc_sync(
                    worker_name(i), torch.add, args=(torch.tensor(1), torch.tensor(1))
                )
                self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)

        # Barrier to ensure that all rpc_sync calls are finished
        dist.barrier()
        rpc.shutdown()

    # Dynamic RPC existing ranks can communicate with new ranks using CUDA rpc
    @skip_if_lt_x_gpu(2)
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank_cuda(self):
        # [FUNCTION DOC] Checks dynamic RPC with CUDA device maps (rank 0 maps devices 1 to 0 and 0
        # to 1 for every other rank before init); requires at least two GPUs.
        # KNOWN ISSUE: the actual CUDA RPC calls and assertions are commented out under a TODO
        # (allocator not initialized for device 1), so the test only exercises init, barriers
        # and shutdown.
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        if self.rank == 0:
            options = self.rpc_backend_options
            for i in range(1, self.world_size):
                dst = worker_name(i)
                options.set_device_map(dst, {1: 0})
                options.set_device_map(dst, {0: 1})
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=options,
            )

        # Rank 0 will be initialized with RPC after this barrier
        dist.barrier()

        # Rest of ranks join after barrier
        if self.rank != 0:
            # Newly joined ranks will be able to communicate with rank 0, since that was created first
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                rpc_backend_options=self.rpc_backend_options,
            )

        # TODO: Cuda RPC is failing due to:
        # terminate called after throwing an instance of 'c10::Error'
        # what():  0 <= device && static_cast<size_t>(device) < device_allocator.size()
        # INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1937,
        # please report a bug to PyTorch. Allocator not initialized for device 1: did you call init?
        # dist.barrier()
        # if self.rank == 0:
        #     for i in range(1, self.world_size):
        #         x = torch.ones(2)
        #         result_on_device_0 = rpc.rpc_sync(worker_name(i), torch.add, args=(x.to(0), 1))
        #         result_on_device_1 = rpc.rpc_sync(worker_name(i), torch.add, args=(x.to(1), 1))
        #         self.assertEqual(torch.add(torch.ones(2), 1), result_on_device_0)
        #         self.assertEqual(torch.device('cuda:0'), result_on_device_0.device)
        #         self.assertEqual(torch.add(torch.ones(2), 1), result_on_device_1)
        #         self.assertEqual(torch.device('cuda:1'), result_on_device_1.device)

        # Barrier to ensure that all rpc_sync calls are finished
        dist.barrier()
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc_without_rank(self):
        # [FUNCTION DOC] Checks that dynamic init_rpc without a rank raises ValueError for file
        # init and TCP init (rank parameter missing) and for env init (RANK environment
        # variable expected).
        # default initialization uses file init
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=self.rpc_backend_options,
            )

        # env init
        with self.assertRaisesRegex(ValueError, "environment variable RANK expected"):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="env://")
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=rpc_backend_options,
            )

        # tcp init
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method="tcp://127.0.0.1:23456"
            )
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rpc_backend_options=rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_dynamic_and_static_init_rpc_together(self):
        # [FUNCTION DOC] Checks that adding a dynamically initialized member to a statically
        # initialized group (world size minus one) raises RuntimeError about mixing static and
        # dynamic members.
        # Initialize a static rpc group with size = self.world_size - 1
        dist.init_process_group(
            backend="gloo",
            init_method=self.file_init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        world_size_minus_one = self.world_size - 1
        if self.rank < world_size_minus_one:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=world_size_minus_one,
                rpc_backend_options=self.rpc_backend_options,
            )

        dist.barrier()

        # Attempt to add an additional dynamic group member
        if self.rank == world_size_minus_one:
            # Expect error message to be thrown
            with self.assertRaisesRegex(
                RuntimeError,
                "RPC group mixes statically and dynamically\
 initialized members which is not supported.",
            ):
                rpc.init_rpc(
                    name=worker_name(self.rank),
                    backend=self.rpc_backend,
                    rank=self.rank,
                    rpc_backend_options=self.rpc_backend_options,
                )


class TensorPipeAgentCudaRpcTest(RpcAgentTestFixture, RpcTestCommon):
    # [CLASS DOC] CUDA tests for the TensorPipe agent: device map validation and use, custom
    # streams, RRef synchronization, CUDA futures and async_execution with CUDA futures. Mixed with
    # RpcTestCommon for the shared future-extraction helper.
    def _test_device_maps(self, options, errMsg):
        # [FUNCTION DOC] HELPER: Asserts that init_rpc with the given options raises ValueError
        # matching errMsg, and that no RPC agent remains set afterward.
        with self.assertRaisesRegex(ValueError, errMsg):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

        self.assertFalse(rpc.api._is_current_rpc_agent_set())

    @skip_if_lt_x_gpu(2)
    def test_device_maps_wrong_worker_name(self):
        # [FUNCTION DOC] Checks that a device map naming a non-existent worker is rejected at init.
        options = self.rpc_backend_options
        options.set_device_map("none_exist", {0: 1})

        self._test_device_maps(
            options,
            errMsg="Node worker0 has invalid target node names in its device maps",
        )

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_local_device(self):
        # [FUNCTION DOC] Checks that a device map whose source device index is out of range is
        # rejected at init.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {torch.cuda.device_count(): 0})

        self._test_device_maps(
            options,
            errMsg="Node worker0 has source devices with invalid indices in its device map for worker1",
        )

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_remote_device(self):
        # [FUNCTION DOC] Checks that a device map whose target device index is out of range is
        # rejected at init.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: torch.cuda.device_count()})

        self._test_device_maps(
            options,
            errMsg="Node worker0 has target devices with invalid indices in its device map for worker1",
        )

    @skip_if_lt_x_gpu(2)
    def test_device_maps_many_to_one(self):
        # [FUNCTION DOC] Checks that a many-to-one device map (duplicated target) is rejected
        # at init.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})
        options.set_device_map(dst, {0: 0})

        self._test_device_maps(
            options,
            errMsg="Node worker0 has duplicated target devices in its device map for worker1",
        )

    @skip_if_lt_x_gpu(2)
    def test_device_maps_one_to_many(self):
        # [FUNCTION DOC] On rank 0, checks that set_device_map raises ValueError when a source
        # device is mapped to a second target (only one-to-one maps are allowed).
        if self.rank == 0:
            options = self.rpc_backend_options
            dst = worker_name((self.rank + 1) % self.world_size)
            options.set_device_map(dst, {0: 1})
            with self.assertRaisesRegex(
                ValueError, "`set_device_map` only supports 1-to-1 mapping"
            ):
                options.set_device_map(dst, {0: 0})

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_min_device(self):
        # [FUNCTION DOC] Checks that set_device_map raises RuntimeError for negative device indices
        # on either side.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, "Device index must not be negative"):
            options.set_device_map(dst, {-1: 0})

        with self.assertRaisesRegex(RuntimeError, "Device index must not be negative"):
            options.set_device_map(dst, {0: -1})

    @staticmethod
    def _gpu_add(x, y):
        # [FUNCTION DOC] HELPER: Static; requires both tensors on CUDA device 1 and returns their
        # sum moved to device 0; otherwise raises ValueError.
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 1]):
            return (x + y).to(0)
        else:
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(2)
    def test_device_maps_gpu(self):
        # [FUNCTION DOC] Checks a device map {0: 1, 1: 0}: tensors sent from device 0 arrive on
        # device 1 and the result comes back on device 1 with the right value.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: 1, 1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add,
            args=(torch.zeros(2).to(0), torch.ones(2).to(0)),
        )
        self.assertEqual(ret.device, torch.device(1))
        self.assertEqual(ret, (torch.zeros(2) + torch.ones(2)).to(1))
        rpc.shutdown()

    @staticmethod
    def _gpu_add_given_devices(x, y, x_to, y_to, z_to):
        # [FUNCTION DOC] HELPER: Static; checks the devices (or cpu) of x and y against x_to and
        # y_to, raising ValueError on mismatch, and returns x and y moved to z_to and added.
        x_device = "cpu" if x.device.type == "cpu" else x.device.index
        y_device = "cpu" if y.device.type == "cpu" else y.device.index
        if x_device == x_to and y_device == y_to:
            return x.to(z_to) + y.to(z_to)
        else:
            raise ValueError("Wrong device affinity")

    def _test_device_maps_gpu(
        self, x_from, y_from, z_to, device_map, dst=None, fn=None
    ):
        # [FUNCTION DOC] HELPER: Sets the device map for the destination (default next worker),
        # initializes RPC, sends x and y created on x_from and y_from, calls fn on the destination,
        # and checks the result lives on the inverse-mapped device z_from and has the expected
        # value; then shuts down.
        fn = TensorPipeAgentCudaRpcTest._gpu_add_given_devices if fn is None else fn
        x_to = device_map[x_from]
        y_to = device_map[y_from]

        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size) if dst is None else dst
        options.set_device_map(dst, device_map)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        x = torch.zeros(2).to(x_from)
        y = torch.ones(2).to(y_from)

        ret = rpc.rpc_sync(dst, fn, args=(x, y, x_to, y_to, z_to))

        reverse_device_map = {device_map[k]: k for k in device_map}
        z_from = reverse_device_map[z_to]

        ret_device = "cpu" if ret.device.type == "cpu" else ret.device.index
        self.assertEqual(ret_device, z_from)
        self.assertEqual(ret, torch.ones(2).to(z_from))

        rpc.shutdown()

    def test_device_map_cpu(self):
        # [FUNCTION DOC] Device map check from cpu to cpu.
        self._test_device_maps_gpu(
            x_from="cpu",
            y_from="cpu",
            z_to="cpu",
            device_map={"cpu": "cpu"},
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,
        )

    @skip_if_lt_x_gpu(1)
    def test_device_map_cpu_to_gpu_default(self):
        # [FUNCTION DOC] Device map check from cpu to GPU 0.
        self._test_device_maps_gpu(
            x_from="cpu",
            y_from="cpu",
            z_to=0,
            device_map={"cpu": 0},
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_cpu_to_gpu_non_default(self):
        # [FUNCTION DOC] Device map check from cpu to GPU 1.
        self._test_device_maps_gpu(
            x_from="cpu",
            y_from="cpu",
            z_to=1,
            device_map={"cpu": 1},
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,
        )

    @skip_if_lt_x_gpu(1)
    def test_device_map_gpu_to_cpu_default(self):
        # [FUNCTION DOC] Device map check from GPU 0 to cpu.
        self._test_device_maps_gpu(
            x_from=0,
            y_from=0,
            z_to="cpu",
            device_map={0: "cpu"},
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_to_cpu_non_default(self):
        # [FUNCTION DOC] Device map check from GPU 1 to cpu.
        self._test_device_maps_gpu(
            x_from=1,
            y_from=1,
            z_to="cpu",
            device_map={1: "cpu"},
            fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices,
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default(self):
        # [FUNCTION DOC] Device map check, GPU 0 to GPU 0.
        self._test_device_maps_gpu(x_from=0, y_from=0, z_to=0, device_map={0: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default(self):
        # [FUNCTION DOC] Device map check, GPU 1 to GPU 1.
        self._test_device_maps_gpu(x_from=1, y_from=1, z_to=1, device_map={1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default_to_non_default(self):
        # [FUNCTION DOC] Device map check, GPU 0 to GPU 1.
        self._test_device_maps_gpu(x_from=0, y_from=0, z_to=1, device_map={0: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default_to_default(self):
        # [FUNCTION DOC] Device map check, GPU 1 to GPU 0.
        self._test_device_maps_gpu(x_from=1, y_from=1, z_to=0, device_map={1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_1(self):
        # [FUNCTION DOC] Mixed inputs from GPU 0 and 1, identity device map, result on GPU 0.
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_2(self):
        # [FUNCTION DOC] Mixed inputs from GPU 0 and 1, identity device map, result on GPU 1.
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_3(self):
        # [FUNCTION DOC] Mixed inputs from GPU 1 and 0, identity device map, result on GPU 0.
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_4(self):
        # [FUNCTION DOC] Mixed inputs from GPU 1 and 0, identity device map, result on GPU 1.
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_5(self):
        # [FUNCTION DOC] Mixed inputs from GPU 0 and 1, swapped device map, result on GPU 0.
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_6(self):
        # [FUNCTION DOC] Mixed inputs from GPU 0 and 1, swapped device map, result on GPU 1.
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_7(self):
        # [FUNCTION DOC] Mixed inputs from GPU 1 and 0, swapped device map, result on GPU 0.
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_8(self):
        # [FUNCTION DOC] Mixed inputs from GPU 1 and 0, swapped device map, result on GPU 1.
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_1(self):
        # [FUNCTION DOC] Same as the mixed 1 check (identity map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0: 0, 1: 1},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_2(self):
        # [FUNCTION DOC] Same as the mixed 2 check (identity map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0: 0, 1: 1},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_3(self):
        # [FUNCTION DOC] Same as the mixed 3 check (identity map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0: 0, 1: 1},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_4(self):
        # [FUNCTION DOC] Same as the mixed 4 check (identity map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0: 0, 1: 1},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_5(self):
        # [FUNCTION DOC] Same as the mixed 5 check (swapped map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=0,
            device_map={0: 1, 1: 0},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_6(self):
        # [FUNCTION DOC] Same as the mixed 6 check (swapped map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=0,
            y_from=1,
            z_to=1,
            device_map={0: 1, 1: 0},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_7(self):
        # [FUNCTION DOC] Same as the mixed 7 check (swapped map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=0,
            device_map={0: 1, 1: 0},
            dst=worker_name(self.rank),
        )

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_8(self):
        # [FUNCTION DOC] Same as the mixed 8 check (swapped map) but the destination is the
        # current worker.
        self._test_device_maps_gpu(
            x_from=1,
            y_from=0,
            z_to=1,
            device_map={0: 1, 1: 0},
            dst=worker_name(self.rank),
        )

    @staticmethod
    def _gpu_add_multi_gpu(x, y):
        # [FUNCTION DOC] HELPER: Static; requires x on CUDA device 1 and y on device 0 and returns
        # the pair (x moved to 0 plus y, x minus y moved to 1); otherwise raises ValueError.
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 0]):
            return x.to(0) + y, x - y.to(1)
        else:
            raise ValueError("Wrong device affinity")

    def _test_device_maps_multi_gpu(self, dst):
        # [FUNCTION DOC] HELPER: Sets device maps {0: 1} and {1: 0} for dst, initializes RPC, calls
        # the two-GPU add function with x on GPU 0 and y on GPU 1 and checks the devices (1 and 0)
        # and values of both results; then shuts down.
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        x = torch.zeros(2).to(0)
        y = torch.ones(2).to(1)
        rets = rpc.rpc_sync(
            dst, TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu, args=(x, y)
        )

        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu(self):
        # [FUNCTION DOC] Runs the multi-GPU device map check against the next worker.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_multi_gpu(dst)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu_self(self):
        # [FUNCTION DOC] Runs the multi-GPU device map check against the current worker.
        dst = worker_name(self.rank)
        self._test_device_maps_multi_gpu(dst)

    @staticmethod
    def _gpu_add_return_to_gpu(x, y):
        # [FUNCTION DOC] HELPER: Static; requires CPU inputs and returns four results placed on
        # GPUs 0, 1, 2 and 3 (sum, difference, product, quotient); otherwise raises ValueError.
        if x.device.type == "cpu" and y.device.type == "cpu":
            return (x + y).to(0), (x - y).to(1), (x * y).to(2), (x / y).to(3)
        else:
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(2)
    def test_device_maps_in_options(self):
        # [FUNCTION DOC] Checks that device maps passed through the TensorPipe options constructor
        # (instead of set_device_map) work, using the two-GPU add function.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=options.init_method,
                num_worker_threads=options.num_worker_threads,
                device_maps={dst: {0: 1, 1: 0}},
                _transports=tp_transports(),
            ),
        )

        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu,
            args=(torch.zeros(2).to(0), torch.ones(2).to(1)),
        )
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    def _test_device_maps_return_to_gpu(self, dst):
        # [FUNCTION DOC] HELPER: Sets a cyclic device map across four GPUs for dst, calls the
        # function returning tensors on four GPUs, and checks the result devices and values; then
        # shuts down.
        options = self.rpc_backend_options

        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 2})
        options.set_device_map(dst, {2: 3})
        options.set_device_map(dst, {3: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        rets = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._gpu_add_return_to_gpu,
            args=(torch.zeros(2), torch.ones(2)),
        )
        for i in range(len(rets)):
            self.assertEqual(rets[i].device, torch.device((3 + i) % 4))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(3))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        self.assertEqual(rets[2], (torch.zeros(2) * torch.ones(2)).to(1))
        self.assertEqual(rets[3], (torch.zeros(2) / torch.ones(2)).to(2))
        rpc.shutdown()

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu(self):
        # [FUNCTION DOC] Runs the four-GPU return check against the next worker.
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_return_to_gpu(dst)

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu_self(self):
        # [FUNCTION DOC] Runs the four-GPU return check against the current worker.
        dst = worker_name(self.rank)
        self._test_device_maps_return_to_gpu(dst)

    @staticmethod
    def _add_to_gpu(x, y):
        # [FUNCTION DOC] HELPER: Static; returns x plus y moved to GPU 0.
        return (x + y).to(0)

    def _test_device_maps_missing_config(self, mode):
        # [FUNCTION DOC] HELPER: Without any device map, sends a CUDA tensor (sync or remote mode)
        # and asserts RuntimeError that the backend only supports CPU tensors by default; then
        # checks RPC still works. Raises ValueError for other modes.
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = (
            "TensorPipe RPC backend only supports CPU tensors by default.*"
            "`set_device_map` on `TensorPipeRpcBackendOptions`"
        )

        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(dst, torch.add, args=(torch.zeros(2).to(0), 1))
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(dst, torch.add, args=(torch.zeros(2).to(0), 1)).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # make sure RPC is still functioning
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    def _test_device_maps_missing_config_response(self, mode):
        # [FUNCTION DOC] HELPER: Without any device map, calls a function that returns a CUDA
        # tensor (sync or remote mode) and asserts RuntimeError that the response device mapping is
        # unavailable; then checks RPC still works. Raises ValueError for other modes.
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = "Response device mapping is not available"

        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(
                    dst,
                    TensorPipeAgentCudaRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1),
                )
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(
                    dst,
                    TensorPipeAgentCudaRpcTest._add_to_gpu,
                    args=(torch.zeros(2), 1),
                ).to_here()
            else:
                raise ValueError(f"unexpected mode {mode}")

        # make sure RPC is still functioning
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config(self):
        # [FUNCTION DOC] Runs the missing device map request check in sync mode.
        self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    def test_device_maps_missing_config_not_timeout(self):
        # [FUNCTION DOC] Checks that the missing device map error is raised faster than the RPC
        # timeout (no hang); manages init and shutdown manually.
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        timeout = rpc.get_rpc_timeout()

        tik = time.time()
        self._test_device_maps_missing_config(RPCExecMode.SYNC)
        rpc.shutdown()
        tok = time.time()

        self.assertTrue(tok - tik < timeout)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_loop(self):
        # [FUNCTION DOC] Repeats the missing device map request check more times than the number of
        # worker threads to check no threads leak.
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response(self):
        # [FUNCTION DOC] Runs the missing device map response check in sync mode.
        self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response_loop(self):
        # [FUNCTION DOC] Repeats the missing device map response check more times than the number
        # of worker threads.
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote(self):
        # [FUNCTION DOC] Runs the missing device map request check in remote mode.
        self._test_device_maps_missing_config(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote_response(self):
        # [FUNCTION DOC] Runs the missing device map response check in remote mode.
        self._test_device_maps_missing_config_response(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_remote(self):
        # [FUNCTION DOC] Checks a remote call with device map {1: 0} that returns a tensor on GPU 0
        # which is mapped back to device 1 on fetch.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        rref = rpc.remote(
            dst, TensorPipeAgentCudaRpcTest._add_to_gpu, args=(torch.zeros(2), 1)
        )

        self.assertEqual(rref.to_here().device.index, 1)
        self.assertEqual(rref.to_here(), torch.ones(2).to(1))

        rpc.shutdown()

    @staticmethod
    def _slow_add_on_user_stream(x, y):
        # [FUNCTION DOC] HELPER: Static; adds x and y on a user-created CUDA stream after a long
        # GPU sleep, with stream synchronization and record_stream calls, and returns the sum.
        s0 = torch.cuda.current_stream(x.device)
        s1 = torch.cuda.Stream(device=x.device)
        s1.wait_stream(s0)
        x.record_stream(s1)
        y.record_stream(s1)
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            z = x + y
        s0.wait_stream(s1)
        z.record_stream(s0)
        return z

    def _test_custom_stream(self, fn, device_map):
        # [FUNCTION DOC] HELPER: Sets the given device map for the next worker, initializes RPC,
        # runs fn on that destination, and shuts down.
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, device_map)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        fn(dst)

        rpc.shutdown()

    def _test_stream_sync(self, dst):
        # [FUNCTION DOC] HELPER: Sends a GPU tensor to the destination, which adds it on a custom
        # stream, and checks the result equals twice x.
        x = torch.ones(2, 2).to(0)
        ret = rpc.rpc_sync(
            dst, TensorPipeAgentCudaRpcTest._slow_add_on_user_stream, args=(x, x)
        )
        self.assertEqual(ret, 2 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream(self):
        # [FUNCTION DOC] Runs the custom stream sync check with map cuda:0 to cuda:1.
        self._test_custom_stream(self._test_stream_sync, {"cuda:0": "cuda:1"})

    def _test_stream_multi_async(self, dst):
        # [FUNCTION DOC] HELPER: Sends 20 async calls of the slow user-stream add and checks
        # each result.
        futs = []
        for i in range(20):
            x = torch.ones(2, 2).to(0) * i
            futs.append(
                rpc.rpc_async(
                    dst,
                    TensorPipeAgentCudaRpcTest._slow_add_on_user_stream,
                    args=(x, x),
                )
            )

        for i in range(20):
            self.assertEqual(futs[i].wait(), 2 * torch.ones(2, 2).to(0) * i)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_multi(self):
        # [FUNCTION DOC] Runs the multiple async custom stream check.
        self._test_custom_stream(self._test_stream_multi_async, {"cuda:0": "cuda:1"})

    @staticmethod
    def _nested_slow_add_on_user_stream(dst, x, y, z):
        # [FUNCTION DOC] HELPER: Static; makes a nested sync RPC to dst for the slow user-stream
        # add of x and y and then adds z with the same slow function locally.
        ret = rpc.rpc_sync(
            dst, TensorPipeAgentCudaRpcTest._slow_add_on_user_stream, args=(x, y)
        )

        return TensorPipeAgentCudaRpcTest._slow_add_on_user_stream(ret, z)

    def _test_stream_nested_sync(self, dst):
        # [FUNCTION DOC] HELPER: Calls the nested slow add through dst with a second worker as
        # nested target, and checks the result equals six times x.
        x = torch.ones(2, 2).to(0)
        y = torch.ones(2, 2).to(0) * 2
        z = torch.ones(2, 2).to(0) * 3
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(
            dst,
            TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream,
            args=(nested_dst, x, y, z),
        )
        self.assertEqual(ret, 6 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested(self):
        # [FUNCTION DOC] Runs the nested custom stream check with a two-way device map.
        self._test_custom_stream(
            self._test_stream_nested_sync, {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}
        )

    def _test_stream_nested_multi_async(self, dst):
        # [FUNCTION DOC] HELPER: On rank 0 only, sends five async nested slow-add calls with
        # distinct inputs and checks each sum.
        if self.rank == 0:
            futs = []
            n = 5
            xs, ys, zs = [], [], []
            for i in range(n):
                x = torch.ones(2, 2).to(0) * (i - 1)
                y = torch.ones(2, 2).to(0) * i
                z = torch.ones(2, 2).to(0) * (i + 1)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                nested_dst = worker_name((self.rank + 2) % self.world_size)
                futs.append(
                    rpc.rpc_async(
                        dst,
                        TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream,
                        args=(nested_dst, x, y, z),
                    )
                )

            for i in range(n):
                self.assertEqual(futs[i].wait(), xs[i] + ys[i] + zs[i])

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested_multi(self):
        # [FUNCTION DOC] Runs the nested multiple async custom stream check with a two-way
        # device map.
        self._test_custom_stream(
            self._test_stream_nested_multi_async,
            {"cuda:0": "cuda:1", "cuda:1": "cuda:0"},
        )

    @staticmethod
    def _gpu_add_wrong_gpus(x, y):
        # [FUNCTION DOC] HELPER: Static; if both tensors are CUDA, returns the CPU copy of x plus
        # the CUDA copy of y (which mixes devices); otherwise raises ValueError.
        if x.is_cuda and y.is_cuda:
            return x.cpu() + y.cuda()
        else:
            raise ValueError("Wrong device affinity")

    @skip_if_lt_x_gpu(1)
    def test_device_mismatch(self):
        # [FUNCTION DOC] Checks that a function mixing devices raises RuntimeError that tensors
        # must be on the same device.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        x = torch.zeros(2).to(0)
        y = torch.ones(2).to(0)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected all tensors to be on the same device, but found at least two devices",
        ):
            rpc.rpc_sync(
                dst, TensorPipeAgentCudaRpcTest._gpu_add_wrong_gpus, args=(x, y)
            )

        rpc.shutdown()

    def _test_rref_synchronization(self, local_device, remote_device):
        # [FUNCTION DOC] HELPER: Initializes RPC with a device map from local_device to
        # remote_device; on rank 1 repeatedly (10 times) builds a CNN remotely and compares the
        # output of the remote call fetched with to_here with the sync call result, checking proper
        # stream synchronization of to_here.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {local_device: remote_device})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 1:
            # This test compares rref.rpc_sync().forward(x) vs rref.remote().forward(x).to_here()
            # If to_here() is properly synchronized with forward(x) the results must be identical
            # This test needs multiple iterations and significant batch size to simulate real
            # training of a CNN of MNIST-like data.
            # see https://github.com/pytorch/pytorch/issues/54771
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                x = torch.randn(200, 1, 28, 28).to(local_device)
                actual = rref.remote().forward(x).to_here()
                expected = rref.rpc_sync().forward(x)
                self.assertEqual(actual, expected)

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_to_here_synchronization1(self):
        # [FUNCTION DOC] RRef to_here synchronization check, cuda:0 to cuda:0.
        self._test_rref_synchronization("cuda:0", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization2(self):
        # [FUNCTION DOC] RRef to_here synchronization check, cuda:1 to cuda:0.
        self._test_rref_synchronization("cuda:1", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization3(self):
        # [FUNCTION DOC] RRef to_here synchronization check, cuda:1 to cuda:1.
        self._test_rref_synchronization("cuda:1", "cuda:1")

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization4(self):
        # [FUNCTION DOC] RRef to_here synchronization check, cuda:0 to cuda:1.
        self._test_rref_synchronization("cuda:0", "cuda:1")

    def _test_rref_as_arg_synchronization(
        self, local_device, remote_device, devicesOptions=None
    ):
        # [FUNCTION DOC] HELPER: Like _test_rref_synchronization but the model input is an RRef
        # passed as an argument; also sets a reverse device map for the input source and optionally
        # the allowed devices per rank (devicesOptions).
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {local_device: remote_device})

        input_src = worker_name((self.rank - 1 + self.world_size) % self.world_size)
        options.set_device_map(input_src, {remote_device: local_device})

        if devicesOptions is not None:
            options.set_devices(devicesOptions[self.rank])

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 1:
            # This test compares rref.rpc_sync().forward(x) vs rref.remote().forward(x).to_here()
            # If to_here() is properly synchronized with forward(x) the results must be identical
            # This test needs multiple iterations and significant batch size to simulate real
            # training of a CNN of MNIST-like data.
            # see https://github.com/pytorch/pytorch/issues/54771
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                rref_x = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                actual = rref.remote().forward(rref_x, True).to_here()
                expected = rref.rpc_sync().forward(rref_x, True)
                self.assertEqual(actual, expected)

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization1(self):
        # [FUNCTION DOC] RRef as argument synchronization check, cuda:0 to cuda:0.
        self._test_rref_as_arg_synchronization("cuda:0", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization2(self):
        # [FUNCTION DOC] RRef as argument synchronization check, cuda:1 to cuda:0.
        self._test_rref_as_arg_synchronization("cuda:1", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization3(self):
        # [FUNCTION DOC] RRef as argument synchronization check, cuda:1 to cuda:1.
        self._test_rref_as_arg_synchronization("cuda:1", "cuda:1")

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization4(self):
        # [FUNCTION DOC] RRef as argument synchronization check, cuda:0 to cuda:1.
        self._test_rref_as_arg_synchronization("cuda:0", "cuda:1")

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization5(self):
        # [FUNCTION DOC] RRef as argument synchronization check, cuda:0 to cuda:0 with explicit
        # device options per rank.
        self._test_rref_as_arg_synchronization(
            "cuda:0",
            "cuda:0",
            [["cuda:0"] for _ in range(4)],  # devicesOptions
        )

    @staticmethod
    def _rref_relay(rref):
        # [FUNCTION DOC] HELPER: Static; returns the fetched value of the given RRef (relays an
        # RRef result).
        return rref.to_here()

    def _test_rref_forward_synchronization(self, local_device, remote_device):
        # [FUNCTION DOC] HELPER: Rank 0 builds a model on rank 1, sends an RRef input, and fetches
        # the forward output RRef through a relay on rank 2, then compares with the sync call
        # result; device maps are configured per rank (0, 1 and 2) so RRef forward CUDA stream
        # synchronization is exercised.
        options = self.rpc_backend_options

        input_src = worker_name(0)
        model_dst = worker_name(1)
        out_relay = worker_name(2)

        if self.rank == 0:
            # for 1) model construction 2) forward execution
            options.set_device_map(model_dst, {local_device: remote_device})

            # Forward output will be first copied to the relay node before
            # returning to the worker. This is intentional, to test RRef
            # forward CUDA stream synchronizations.
            options.set_device_map(out_relay, {local_device: local_device})
        elif self.rank == 1:
            # worker1 hosts the model and runs forward. The forward functions
            # calls RRef.to_here(), hence needs to configure the device map
            options.set_device_map(input_src, {remote_device: local_device})
        elif self.rank == 2:
            # worker2 will get the out RRef and call to_here() and hence, needs
            # to configure device map.
            options.set_device_map(model_dst, {local_device: remote_device})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        if self.rank == 0:
            # This test compares rref.rpc_sync().forward(x) vs rref.remote().forward(x).to_here()
            # If to_here() is properly synchronized with forward(x) the results must be identical
            # This test needs multiple iterations and significant batch size to simulate real
            # training of a CNN of MNIST-like data.
            # see https://github.com/pytorch/pytorch/issues/54771
            rref = rpc.remote(model_dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                rref_input = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                rref_out = rref.remote().forward(rref_input, True)
                out = rpc.remote(
                    out_relay, TensorPipeAgentCudaRpcTest._rref_relay, args=(rref_out,)
                ).to_here()
                expected = rref.rpc_sync().forward(rref_input, True)
                self.assertEqual(out, expected)

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_forward_synchronization1(self):
        # [FUNCTION DOC] RRef forward synchronization check, cuda:0 to cuda:0.
        self._test_rref_forward_synchronization("cuda:0", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization2(self):
        # [FUNCTION DOC] RRef forward synchronization check, cuda:0 to cuda:1.
        self._test_rref_forward_synchronization("cuda:0", "cuda:1")

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization3(self):
        # [FUNCTION DOC] RRef forward synchronization check, cuda:1 to cuda:0.
        self._test_rref_forward_synchronization("cuda:1", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization4(self):
        # [FUNCTION DOC] RRef forward synchronization check, cuda:1 to cuda:1.
        self._test_rref_forward_synchronization("cuda:1", "cuda:1")

    def _test_owner_rref_forward_synchronization(self, local_device, remote_device):
        # [FUNCTION DOC] HELPER: On rank 0 only, initializes RPC with a single worker named w0,
        # creates a large Linear model through a remote call on itself and moves it to
        # remote_device, then 30 times runs forward on random data and checks that the sum computed
        # through an RRef of the output equals the local sum (owner RRef stream synchronization);
        # then shuts down.
        if self.rank == 0:
            options = self.rpc_backend_options
            options.set_device_map("w0", {local_device: remote_device})
            rpc.init_rpc("w0", rank=0, world_size=1, rpc_backend_options=options)

            model = (
                rpc.remote("w0", torch.nn.Linear, (2048, 20000))
                .remote()
                .to(remote_device)
            )
            for _ in range(30):
                data = torch.rand(2048, 2048).to(local_device)
                output = model.rpc_sync().forward(data)
                # to_here() internally calls localValue as the caller is
                # the owner of the RRef.
                v0 = rpc.RRef(output).remote().sum().to_here().item()
                v1 = output.sum().item()
                self.assertEqual(v0, v1)

            rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_owner_rref_forward_synchronization1(self):
        # [FUNCTION DOC] Owner RRef forward synchronization check, cuda:0 to cuda:0.
        self._test_owner_rref_forward_synchronization("cuda:0", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization2(self):
        # [FUNCTION DOC] Owner RRef forward synchronization check, cuda:0 to cuda:1.
        self._test_owner_rref_forward_synchronization("cuda:0", "cuda:1")

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization3(self):
        # [FUNCTION DOC] Owner RRef forward synchronization check, cuda:1 to cuda:0.
        self._test_owner_rref_forward_synchronization("cuda:1", "cuda:0")

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization4(self):
        # [FUNCTION DOC] Owner RRef forward synchronization check, cuda:1 to cuda:1.
        self._test_owner_rref_forward_synchronization("cuda:1", "cuda:1")

    @staticmethod
    def _return_tensor_view(i):
        # [FUNCTION DOC] HELPER: Static; on GPU 0's current stream builds a tensor scaled by i,
        # sleeps on the GPU, and returns a view (first split) so serialization creates a new tensor
        # outside the function.
        with torch.cuda.stream(torch.cuda.current_stream(0)):
            x = torch.ones(1000, 200).cuda(0) * i
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            # serialization of the return value will create a new tensor from the
            # view, which is done outside of the user function.
            return x.split(100)[0]

    @skip_if_lt_x_gpu(1)
    def test_tensor_view_as_return_value(self):
        # [FUNCTION DOC] Checks that returning a tensor view from a CUDA function over RPC gives
        # correct values for five concurrent async calls.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 0})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        futs = [
            rpc.rpc_async(
                dst, TensorPipeAgentCudaRpcTest._return_tensor_view, args=(i,)
            )
            for i in range(5)
        ]

        for i in range(5):
            self.assertEqual(torch.ones(100, 200) * i, futs[i].wait())

        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_devices_option_mismatch(self):
        # [FUNCTION DOC] Checks that init raises ValueError about unexpected source devices when
        # the explicit devices list does not include the source device used in the device map.
        with self.assertRaisesRegex(
            ValueError,
            "Node worker0 has unexpected source devices in its device map for worker1",
        ):
            dst = worker_name((self.rank + 1) % self.world_size)
            options = self.rpc_backend_options
            options.set_device_map(dst, {0: 0})
            options.set_devices([1])

            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

            rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_devices_option_mismatch_reverse(self):
        # [FUNCTION DOC] Checks that init raises ValueError about unexpected target devices when
        # the explicit devices list is inconsistent with the target device used in the device map.
        with self.assertRaisesRegex(
            ValueError,
            "Node worker0 has unexpected target devices in its device map for worker1",
        ):
            dst = worker_name((self.rank + 1) % self.world_size)

            options = rpc.TensorPipeRpcBackendOptions(
                init_method=self.rpc_backend_options.init_method,
                num_worker_threads=self.rpc_backend_options.num_worker_threads,
                device_maps={dst: {0: 1}},
                devices=[0],
            )

            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )

            rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_int(self):
        # [FUNCTION DOC] Checks that a future can be created with a device given as an integer.
        Future(devices=[0])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_str(self):
        # [FUNCTION DOC] Checks that a future can be created with a device given as a string.
        Future(devices=["cuda:0"])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_device(self):
        # [FUNCTION DOC] Checks that a future can be created with a torch device object.
        Future(devices=[torch.device("cuda", 0)])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_not_cuda(self):
        # [FUNCTION DOC] Checks that creating a future with a cpu device raises ValueError that
        # devices must have indices.
        with self.assertRaisesRegex(
            ValueError, "Expected devices to have indices, got cpu"
        ):
            Future(devices=["cpu"])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a plain tensor.
        self._test_cuda_future_extraction(
            wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=False
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a tensor inside a list.
        self._test_cuda_future_extraction(
            wrapper=lambda t: [t], unwrapper=operator.itemgetter(0), sparse_tensor=False
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a tensor inside a custom class.
        self._test_cuda_future_extraction(
            wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=False
        )

    @skip_if_lt_x_gpu(2)
    def test_cuda_future_callback_changes_devices(self):
        # [FUNCTION DOC] Checks that a callback on a future with two devices can return a tensor on
        # another device and that stream synchronization is correct for the child future.
        # We check proper CUDA stream synchronization by filling the tensor with
        # the expected value in one stream, and reading it from another stream.
        tensor0 = torch.zeros((100,), device="cuda:0")
        tensor1 = torch.zeros((100,), device="cuda:1")
        parent_future = Future(devices=["cuda:0", "cuda:1"])

        def cb(fut):
            # [FUNCTION DOC] HELPER: Callback; copies the parent's tensor into the device 1 tensor
            # without blocking and returns it.
            t0 = fut.value()
            tensor1.copy_(t0, non_blocking=True)
            return tensor1

        child_future = parent_future.then(cb)
        with torch.cuda.device("cuda:0"):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor0.fill_(1)
                parent_future.set_result(tensor0)
        with torch.cuda.device("cuda:1"):
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())

    @skip_if_lt_x_gpu(2)
    def test_cuda_future_value_on_bad_device(self):
        # [FUNCTION DOC] Checks that a future whose result is on a device not in its device list
        # raises ValueError on wait, while a child future whose callback ignores the parent
        # still succeeds.
        tensor0 = torch.zeros((100,), device="cuda:0")
        tensor1 = torch.zeros((100,), device="cuda:1")
        parent_future = Future(devices=["cuda:1"])

        # As a plus, we test that futures still invoke callbacks even in case of
        # error, and that the child futures are successful if those callbacks
        # don't access the parent future.
        def cb(fut):
            # [FUNCTION DOC] HELPER: Callback; on device 1 sleeps on the GPU, fills tensor1 with
            # ones and returns it.
            with torch.cuda.device("cuda:1"):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor1.fill_(1)
                return tensor1

        child_future = parent_future.then(cb)
        with torch.cuda.device("cuda:0"):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor0.fill_(1)
                parent_future.set_result(tensor0)
        with self.assertRaisesRegex(
            ValueError,
            r"The result contained tensors residing on device\(s\) cuda:0 "
            r"which are not among the expected device\(s\) cuda:1",
        ):
            parent_future.wait()
        with torch.cuda.device("cuda:1"):
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())

    @skip_if_lt_x_gpu(1)
    def test_async_execution_with_cuda_future(self):
        # [FUNCTION DOC] Checks that an async_execution function returning a CUDA future produces a
        # correctly synchronized result with a device map.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        t = torch.zeros((100,), device="cuda:0")
        fut = rpc.rpc_async(dst, async_cuda_sleep_and_set_to_one, args=(t,))
        another_stream = torch.cuda.Stream("cuda:0")
        with torch.cuda.stream(another_stream):
            self.assertTrue(torch.eq(fut.wait(), 1).all().item())

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_async_execution_nested_with_cuda_future(self):
        # [FUNCTION DOC] Checks the nested async_execution CUDA function (sum of three ones tensors
        # equals 3) with a device map.
        dst = worker_name((self.rank + 1) % self.world_size)
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        a = torch.ones((100,), device="cuda:0")
        b = torch.ones((100,), device="cuda:0")
        c = torch.ones((100,), device="cuda:0")
        fut = rpc.rpc_async(dst, async_cuda_nested_add, args=(nested_dst, a, b, c))
        another_stream = torch.cuda.Stream("cuda:0")
        with torch.cuda.stream(another_stream):
            self.assertTrue(torch.eq(fut.wait(), 3).all().item())

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_modify_tensor_inplace(self):
        # [FUNCTION DOC] Checks that modifying the tensor of a completed CUDA future in place does
        # not crash (documented as undefined behavior).
        tensor = torch.zeros((100,), device="cuda:0")
        future = Future(devices=["cuda:0"])
        future.set_result(tensor)
        # It's weird to modify the value of a future once it's complete, but
        # technically possible. Currently this is considered undefined behavior
        # (in practice the future will ignore the modification and still
        # synchronize with the original value). We could one day add logic to
        # detect and warn or throw in such cases, but for now we just check that
        # this doesn't crash.
        tensor.fill_(1)
        future.wait()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_replace_tensor(self):
        # [FUNCTION DOC] Checks that replacing a tensor in a list held by a completed CUDA future
        # (deleting the original) does not crash.
        tensor_list = [torch.zeros((100,), device="cuda:0")]
        future = Future(devices=["cuda:0"])
        future.set_result(tensor_list)
        # It's weird to modify the value of a future once it's complete, but
        # technically possible. Currently this is considered undefined behavior
        # (in practice the future will ignore the modification and still
        # synchronize with the original value). We could one day add logic to
        # detect and warn or throw in such cases, but for now we just check that
        # this doesn't crash.
        # We set things up so that the original tensor contained in the list
        # gets deleted once we replace it with the other one. This will
        # invalidate any cached information held by the future.
        tensor_list[0] = torch.ones((100,), device="cuda:0")
        future.wait()

    @skip_if_lt_x_gpu(1)
    def test_rref_with_unpickleable_attributes(self):
        # [FUNCTION DOC] Checks that an RRef of an object with unpicklable attributes (lock, event,
        # thread) holding a CUDA tensor works with remote method calls.
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {"cuda:0": "cuda:0"})

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )

        rref = rpc.remote(dst, TensorWrapper, args=(torch.zeros(42, device="cuda:0"),))
        rref.rpc_sync().increase(1)
        ret = rref.rpc_sync().sum()
        self.assertEqual(ret, 42)

        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_sparse_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a sparse CUDA tensor.
        self._test_cuda_future_extraction(
            wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=True
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_sparse_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a sparse CUDA tensor inside a list.
        self._test_cuda_future_extraction(
            wrapper=lambda t: [t], unwrapper=operator.itemgetter(0), sparse_tensor=True
        )

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_sparse_tensor(self):
        # [FUNCTION DOC] Runs the future extraction helper with a sparse CUDA tensor inside a
        # custom class.
        self._test_cuda_future_extraction(
            wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=True
        )
