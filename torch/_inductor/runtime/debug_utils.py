# [MODULE DOC] Debug helpers that track inductor runtime tensor allocations and frees per step (via weak references to storage)
# and compare them to expected lists, plus registration of a custom ordered effectful op that performs the check inside compiled
# graphs. Module state: a thread-local holding the current memory tracker. Import-time side effect: sets the importing thread's tracker to None.
import functools
import logging
import threading
import weakref

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

local = threading.local()
local.memory_tracker = None


class BufferMemoryTracker:
    # [CLASS DOC] Tracks which named tensors were allocated and which died since the last step. State: a weak mapping name -> storage, sets of
    # names that died or were added since the last step, and a flag to raise errors (config mode assert) instead of logging.
    """
    Tracks inductor runtime allocations and deallocations to compare against
    expected behavior.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the weak tensor mapping and empty added/died sets; reads the memory lifecycle tracking config to decide
        # whether violations raise.
        self.tensor_tracker: dict[str, torch.storage.UntypedStorage] = (
            weakref.WeakValueDictionary()  # type: ignore[assignment]
        )
        self.died_since_last_step: OrderedSet[str] = OrderedSet()
        self.added_since_last_step: OrderedSet[str] = OrderedSet()
        self.error = (
            torch._inductor.config.test_configs.track_memory_lifecycle == "assert"
        )

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] Records the name as added in this step, stores a weak reference to the tensor storage under the name and arranges for
        # the name to be added to the died set when the storage is garbage collected.
        storage = tensor.untyped_storage()

        self.added_since_last_step.add(name)
        self.tensor_tracker[name] = storage

        def on_tensor_death() -> None:
            # [FUNCTION DOC] HELPER: Finalizer callback that marks the tensor name as dead in the current step.
            self.died_since_last_step.add(name)

        weakref.finalize(storage, on_tensor_death)

    def advance_step(self) -> None:
        # [FUNCTION DOC] Clears the added and died sets, starting a new step.
        self.died_since_last_step.clear()
        self.added_since_last_step.clear()

    def log_or_raise(self, msg: str) -> None:
        # [FUNCTION DOC] HELPER: Raises RuntimeError with the message in assert mode, otherwise logs it at info level.
        if self.error:
            raise RuntimeError(msg)
        else:
            log.info(msg)

    def check_step_delta(
        self,
        expected_allocated: list[str],
        expected_freed: list[str],
        is_final_step: bool,
    ) -> None:
        # [FUNCTION DOC] Compares this step's deaths and allocations with the expected lists and reports (log or raise) missing deaths (skipped on
        # the final step), unexpected deaths, unexpected allocations and missing allocations. Then starts a new step; on the final step also
        # resets the thread's global tracker to None. In assert mode the first violation raises RuntimeError, so later checks and the reset are skipped.
        """Check only the delta changes since last step"""

        # Check expected deaths - we don't currently distinguish between nodes which die in last step
        # and are returned as outputs, so skip if final_step.
        if not is_final_step:
            missing_deaths = OrderedSet(expected_freed) - self.died_since_last_step
            if missing_deaths:
                self.log_or_raise(
                    f"Expected tensors to die but still alive: {missing_deaths}"
                )

        # Check for unexpected deaths
        unexpected_deaths = self.died_since_last_step - OrderedSet(expected_freed)
        if unexpected_deaths:
            self.log_or_raise(f"Unexpected tensor deaths: {unexpected_deaths}")

        # Check newly alive tensors - separate messages like deaths
        actual_allocated = self.added_since_last_step
        expected_allocated_set = OrderedSet(expected_allocated)

        extra_alive = actual_allocated - expected_allocated_set
        if extra_alive:
            self.log_or_raise(f"Unexpected allocated tensors: {extra_alive}")

        missing_alive = expected_allocated_set - actual_allocated
        if missing_alive:
            self.log_or_raise(
                f"Expected allocated tensors but missing: {missing_alive}"
            )

        # Reset for next step
        self.advance_step()

        if is_final_step:
            local.memory_tracker = None


def get_mem_tracker() -> BufferMemoryTracker:
    # [FUNCTION DOC] HELPER: Returns the thread-local memory tracker, creating it if needed.
    # KNOWN ISSUE: the thread-local attribute is only initialized for the importing thread, so first use on another thread raises AttributeError.
    if local.memory_tracker is None:
        local.memory_tracker = BufferMemoryTracker()
    return local.memory_tracker


def track_tensor(tensor: torch.Tensor, name: str) -> None:
    # [FUNCTION DOC] HELPER: Registers the tensor's storage under the name with the current tracker.
    get_mem_tracker().set_tensor(name, tensor)


def tracked_empty_strided(
    size: list[int],
    stride: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    # [FUNCTION DOC] Allocates an uninitialized tensor with the given size, stride, dtype and device, registers it for tracking under the name
    # and returns it.
    o = torch.empty_strided(size, stride, dtype=dtype, device=device)
    track_tensor(o, name)
    return o


def check_memory_step(
    allocated: list[str], freed: list[str], is_final_step: bool = False
) -> None:
    # [FUNCTION DOC] Runs the step delta check on the current thread's tracker with the expected allocated and freed names. Returns nothing;
    # may raise RuntimeError in assert mode.
    tracker = get_mem_tracker()
    tracker.check_step_delta(allocated, freed, is_final_step)


@functools.lru_cache(None)
def register_check_mem_op() -> None:
    # [FUNCTION DOC] Cached (runs once). Defines a custom library fragment with a check_memory_step op taking allocated names, freed names and a
    # final-step flag, implements it with the Python check function for the backend-select key, keeps the library alive for the process, and
    # registers the op as an ordered effectful op so it is not reordered or removed. Side effects: global operator registration.
    lib = torch.library.Library("_inductor_debug", "FRAGMENT")
    # Keep `lib` alive: this function is lru_cached, so it runs once, but the
    # registered op must remain accessible via torch.ops for the rest of the
    # process's lifetime. Letting `lib` die triggers the finalizer, which
    # clears torch.ops._inductor_debug.check_memory_step.
    torch.library._keep_alive.append(lib)
    lib.define(
        "check_memory_step(str[] allocated, str[] freed, bool is_final_step) -> ()"
    )
    lib.impl("check_memory_step", check_memory_step, "BackendSelect")
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    _register_effectful_op(
        torch.ops._inductor_debug.check_memory_step.default,
        _EffectType.ORDERED,
    )
