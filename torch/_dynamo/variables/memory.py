# [MODULE DOC] Dynamo support for torch.cuda.MemPool objects and the torch.cuda.use_mem_pool
# context manager. Defines two custom ops (mempool::begin and mempool::end) marked as FX
# side-effect ops, plus CUDAMemPoolVariable (tracks a MemPool) and CUDAMemPoolContextVariable
# (emits the begin/end ops and annotates enclosed FX nodes with the pool index and device).
# Import-time side effects: registers the two custom ops, their fake implementations, and
# marks both op overloads as having side effects so they are not dead-code eliminated.
from contextlib import ExitStack
from typing import Any, TYPE_CHECKING

import torch
from torch.fx import has_side_effect, Proxy

from ..bytecode_transformation import create_call_function
from ..exc import raise_type_error, unimplemented
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, CallFunctionNoArgsSource, ImportSource
from .base import Member, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


def _get_mempool_by_index(index: int) -> torch.cuda.MemPool:
    # [FUNCTION DOC] HELPER: Looks up the user object registered at the given external-object
    # index and returns it as a MemPool. Raises RuntimeError if the object is not a MemPool.
    mempool = get_external_object_by_index(index)
    if not isinstance(mempool, torch.cuda.MemPool):
        raise RuntimeError(
            f"use_mem_pool expected a torch.cuda.MemPool object at index {index}"
        )
    return mempool


def _current_cuda_device_source() -> CallFunctionNoArgsSource:
    # [FUNCTION DOC] HELPER: Builds a Source that, at runtime, calls torch.cuda.current_device
    # with no arguments, so a guard can be installed on the current CUDA device index.
    return CallFunctionNoArgsSource(
        AttrSource(AttrSource(ImportSource("torch"), "cuda"), "current_device")
    )


# These marker ops are intentional even though Inductor also annotates enclosed
# FX nodes. The annotations tell Inductor which scheduler nodes allocate in a
# user pool, while begin/end keep the pool lifetime visible as ordered FX
# side-effects for non-Inductor backends.
@custom_op("mempool::begin", mutates_args=())
def begin_mempool(device_index: int, mempool_index: int) -> None:
    # [FUNCTION DOC] Custom op mempool::begin. Resolves the MemPool from its external-object
    # index and routes the current thread's CUDA allocations on the given device to that pool.
    # Raises RuntimeError (via the lookup) if the index does not hold a MemPool. Returns nothing.
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool(device_index, mempool.id)


@begin_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of mempool::begin; does nothing.
    pass


has_side_effect(torch.ops.mempool.begin.default)


@custom_op("mempool::end", mutates_args=())
def end_mempool(device_index: int, mempool_index: int) -> None:
    # [FUNCTION DOC] Custom op mempool::end. Resolves the MemPool from its external-object
    # index, stops routing the current thread's allocations on the device to the pool, then
    # releases the pool. Raises RuntimeError if the index does not hold a MemPool.
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_endAllocateToPool(device_index, mempool.id)
    torch.cuda.memory._cuda_releasePool(device_index, mempool.id)


@end_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of mempool::end; does nothing.
    pass


has_side_effect(torch.ops.mempool.end.default)


class CUDAMemPoolVariable(VariableTracker):
    # [CLASS DOC] Tracks a real torch.cuda.MemPool during tracing. State: proxy (FX proxy for the
    # pool), value (the real MemPool), user_object_index (index in the external-object table
    # used to fetch the pool at runtime). Exposes the "id" member; reconstructs either from its
    # source or by loading the external object by index.
    """Represents a torch.cuda.MemPool object."""

    def __init__(
        self,
        proxy: Proxy,
        value: torch.cuda.MemPool,
        user_object_index: int,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the proxy, the real MemPool and its external-object index;
        # remaining keyword arguments go to the base VariableTracker constructor.
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the Python type represented, torch.cuda.MemPool.
        return torch.cuda.MemPool

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Returns the underlying real MemPool object.
        return self.value

    def _id(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Implements the "id" member. If the variable has a source,
        # installs an identity-match guard on it. Returns a constant variable holding the
        # pool's real id.
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
        return ConstantVariable.create(self.value.id)

    tp_members = {
        "id": Member(_id),
    }

    def as_proxy(self) -> Proxy:
        # [FUNCTION DOC] Returns the FX proxy associated with this pool.
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that recreates the pool value. With a source, defers to
        # the base class reconstruction; otherwise emits a call to the external-object getter
        # with the stored index as the single argument. Writes to the codegen output only.
        if self.source:
            return super().reconstruct(codegen)

        codegen.add_push_null(
            lambda: codegen.load_import_from(
                torch._dynamo.graph_bytecode_inputs.__name__,
                "get_external_object_by_index",
            )
        )
        codegen.append_output(codegen.create_load_const(self.user_object_index))
        codegen.extend_output(create_call_function(1, False))


class CUDAMemPoolContextVariable(ContextWrappingVariable):
    # [CLASS DOC] Models the torch.cuda.use_mem_pool context manager. State: mempool (a
    # CUDAMemPoolVariable) and device_index (int CUDA device). Entering emits the mempool::begin
    # op and annotates enclosed FX nodes with pool index/device; exiting emits mempool::end.
    # Graph breaks inside the context are unsupported.
    """Represents the torch.cuda.use_mem_pool context manager."""

    def __init__(
        self,
        mempool: CUDAMemPoolVariable,
        device_index: int,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the pool variable and device index, then initializes the base
        # context wrapper with no target values and no initial values.
        self.mempool = mempool
        self.device_index = device_index
        super().__init__(target_values=(), initial_values=None, **kwargs)

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        mempool: VariableTracker,
        device: VariableTracker | None = None,
        **kwargs: Any,
    ) -> "CUDAMemPoolContextVariable":
        # [FUNCTION DOC] Factory for the context variable. Raises a Python TypeError (via
        # raise_type_error) if mempool is not a CUDAMemPoolVariable. If device is missing or
        # constant None, installs an equality guard on the current CUDA device and uses the
        # current device index; if device is a Python constant, converts it to a device index;
        # otherwise triggers a graph break (unimplemented). Returns the new variable.
        if not isinstance(mempool, CUDAMemPoolVariable):
            raise_type_error(
                tx,
                "torch.cuda.use_mem_pool() expected a torch.cuda.MemPool argument",
            )

        if device is None or (
            isinstance(device, ConstantVariable) and device.value is None
        ):
            install_guard(
                _current_cuda_device_source().make_guard(GuardBuilder.EQUALS_MATCH)
            )
            device_index = torch.cuda.current_device()
        elif device.is_python_constant():
            device_index = torch.cuda._utils._get_device_index(
                device.as_python_constant()
            )
        else:
            unimplemented(
                gb_type="torch.cuda.use_mem_pool with non-constant device",
                context=f"device={device}",
                explanation="Dynamo requires the device argument to torch.cuda.use_mem_pool to be a Python constant.",
                hints=[],
            )

        return CUDAMemPoolContextVariable(mempool, device_index, **kwargs)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Enters the context during tracing. Opens FX annotation (pool index and
        # device) and node-metadata preservation scopes on an exit stack, registers a cleanup
        # hook that closes the stack, and emits a call_function proxy for the mempool::begin op.
        # Returns a constant None variable.
        stack = ExitStack()
        stack.enter_context(
            torch.fx.traceback.annotate(
                {
                    "mempool": self.mempool.user_object_index,
                    "mempool_device": self.device_index,
                }
            )
        )
        stack.enter_context(torch.fx.traceback.preserve_node_meta())
        self.set_cleanup_hook(tx, lambda: stack.close())
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.begin,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        return ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Exits the context during tracing: emits a call_function proxy for the
        # mempool::end op, then asserts the cleanup hook state is consistent (cleanup_assert).
        # Exception arguments are ignored. Returns a constant None variable.
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.end,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        self.cleanup_assert()
        return ConstantVariable.create(None)

    def module_name(self) -> str:
        # [FUNCTION DOC] Returns the module name of the wrapped context manager, torch.cuda.
        return "torch.cuda"

    def fn_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the wrapped context manager function, use_mem_pool.
        return "use_mem_pool"

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Always triggers a graph break (unimplemented) because reconstructing
        # the eager context on a resume path would unbalance the begin/end marker ops.
        # Supporting graph breaks safely requires reconstructing the eager
        # torch.cuda.use_mem_pool context on the resume path while keeping the
        # compiled begin/end marker ops balanced around each compiled segment.
        unimplemented(
            gb_type="torch.cuda.use_mem_pool graph break",
            context=str(self),
            explanation="Dynamo doesn't support graph breaks inside torch.cuda.use_mem_pool.",
            hints=[],
        )
