# [MODULE DOC] Defines the Stateful runtime-checkable protocol used by distributed checkpointing
# for objects that expose state_dict and load_state_dict, plus the StatefulT bound type variable.
# No import-time side effects beyond defining these names.
from typing import Any, TypeVar
from typing_extensions import Protocol, runtime_checkable


__all__ = ["Stateful", "StatefulT"]


@runtime_checkable
class Stateful(Protocol):
    # [CLASS DOC] Structural protocol (runtime checkable, so isinstance works) for objects that can be
    # checkpointed and restored. Implementers provide state_dict and load_state_dict.
    """
    Stateful protocol for objects that can be checkpointed and restored.
    """

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Protocol stub: return the object's state as a dict of string keys to values.
        # The result is what gets saved, and it is also called during load to obtain in-place targets.
        # Body is empty (protocol method).
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        .. warning::
            Because of the inplace nature of restoring a checkpoint, this function
            is also called during `torch.distributed.checkpoint.load`.


        Returns:
            Dict: The objects state dict
        """

        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Protocol stub: restore the object's state from the given state dict; returns nothing.
        # Body is empty (protocol method).
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """

        ...


StatefulT = TypeVar("StatefulT", bound=Stateful)
