# mypy: allow-untyped-defs
# [MODULE DOC] Support for RRef helper proxies (rpc_sync / rpc_async / remote on an RRef):
# the RRefProxy class plus helpers that invoke a method on the RRef's owner via RPC.
# No import-time side effects.
from functools import partial

import torch
from torch.futures import Future

from . import functions, rpc_async
from .constants import UNSET_RPC_TIMEOUT


def _local_invoke(rref, func_name, args, kwargs):
    # [FUNCTION DOC] HELPER: runs on the RRef owner; looks up the named method on the
    # RRef's local value and calls it with args and kwargs, returning the result.
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    # [FUNCTION DOC] HELPER: same as _local_invoke but decorated as async-execution so the
    # RPC callee treats the returned Future as the eventual response.
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    # [FUNCTION DOC] HELPER: invokes method func_name of the RRef's value on its owner using
    # rpc_api (sync, async or remote). First fetches the RRef's value type (non-blocking),
    # then picks the async-execution invoker if the method carries the async marker
    # (skipped for script modules). For non-async APIs it waits for the type and returns
    # the RPC result; for rpc_async it returns a Future completed with the call's result,
    # or with the exception raised at any stage. Errors from type fetch surface on wait.
    def _rref_type_cont(rref_fut):
        # [FUNCTION DOC] HELPER: continuation once the RRef type is known; chooses the invoker
        # and issues the RPC to the owner with the saved args/kwargs/timeout, returning
        # whatever rpc_api returns.
        rref_type = rref_fut.value()

        _invoke_func = _local_invoke
        # Bypass ScriptModules when checking for async function attribute.
        bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(
            rref_type, torch._C.ScriptModule
        )
        if not bypass_type:
            func = getattr(rref_type, func_name)
            if hasattr(func, "_wrapped_async_rpc_function"):
                _invoke_func = _local_invoke_async_execution

        return rpc_api(
            rref.owner(),
            _invoke_func,
            args=(rref, func_name, args, kwargs),
            timeout=timeout,
        )

    rref_fut = rref._get_type(timeout=timeout, blocking=False)

    if rpc_api is not rpc_async:
        rref_fut.wait()
        return _rref_type_cont(rref_fut)
    else:
        # A little explanation on this.
        # rpc_async returns a Future pointing to the return value of `func_name`, it returns a `Future[T]`
        # Calling _rref_type_cont from the `then` lambda causes Future wrapping. IOW, `then` returns a `Future[Future[T]]`
        # To address that, we return a Future that is completed with the result of the async call.
        result: Future = Future()

        def _wrap_rref_type_cont(fut):
            # [FUNCTION DOC] HELPER: runs the type continuation and chains its Future to
            # complete the outer result; any exception is set on the outer Future.
            try:
                _rref_type_cont(fut).then(_complete_op)
            except BaseException as ex:
                result.set_exception(ex)

        def _complete_op(fut):
            # [FUNCTION DOC] HELPER: copies the value of fut into the outer result Future,
            # or sets the raised exception on it.
            try:
                result.set_result(fut.value())
            except BaseException as ex:
                result.set_exception(ex)

        rref_fut.then(_wrap_rref_type_cont)
        return result


# This class manages proxied RPC API calls for RRefs. It is entirely used from
# C++ (see python_rpc_handler.cpp).
class RRefProxy:
    # [CLASS DOC] Proxy returned by RRef helper accessors. Holds the RRef, the RPC API
    # function to use and the timeout; any attribute access yields a callable that performs
    # the corresponding method call on the RRef owner through that API.
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        # [FUNCTION DOC] Stores the RRef, RPC API function and timeout (default: unset
        # sentinel meaning use the agent default) on the instance.
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        # [FUNCTION DOC] Returns a partially applied invoker for the method func_name; the
        # call is made on the owner when the result is called with arguments.
        return partial(
            _invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout
        )
