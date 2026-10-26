# [MODULE DOC] Distributed autotuning of GEMM kernels for Inductor. When enabled via
# config.distributed_max_autotune_gemm, each rank autotunes only a share of the mm ops
# and the other ranks receive placeholder buffers; after scheduling, results are exchanged
# with an all-gather and the placeholders are replaced by real autotuned buffers.
# Holds module-level state: a lazily created autotune process group. No import-time side effects.
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import sympy

import torch._logging
import torch.distributed as dist
import torch.fx
from torch.utils._ordered_set import OrderedSet

from . import config, select_algorithm
from .ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    MultiTemplateBuffer,
    OperationBuffer,
    StorageBox,
    TensorBox,
)
from .kernel_inputs import KernelInputs, MMKernelInputs
from .scheduler import SchedulerNode
from .virtualized import NullHandler, V


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


_DISTRIBUTED_AUTOTUNE_KEY = "distributed_autotune"

_AUTOTUNE_PG: dist.ProcessGroup | None = None


@dataclasses.dataclass
class _DistributedAutotuneState:
    # [CLASS DOC] HELPER: Per-graph counters stored in the virtualized distributed-autotune state.
    # autotuned_index is the running operator index used to pick the owning rank (round robin);
    # autotuned_local_count counts ops this rank tuned, used as a sanity check later.
    """
    State used to track autotuning during a graph_context()
    """

    # This is the next operator index. Used to figure out which rank should do
    # the autotuning.
    autotuned_index: int = 0

    # For debugging - used to make sure that we autotune the same number of
    # local operators that we expected to.
    autotuned_local_count: int = 0


@dataclasses.dataclass
class _DistributedAutotuneInfo:
    # [CLASS DOC] HELPER: Metadata attached to an FX node: the global op index and whether this rank tunes it.
    index: int
    local: bool


def get_autotune_pg() -> dist.ProcessGroup | None:
    # [FUNCTION DOC] Return the dedicated autotune process group, creating it on first use under a fixed tag.
    # Returns None if torch.distributed is unavailable or not initialized. Raises an error if the
    # created group does not include this rank. Caches the group in a module global.
    if dist.is_available() and dist.is_initialized():
        global _AUTOTUNE_PG
        if _AUTOTUNE_PG is None:
            autotune_pg = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_distributed_autotune_pg"
            )
            if autotune_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Autotune process group must include all ranks")
            _AUTOTUNE_PG = autotune_pg
        return _AUTOTUNE_PG

    return None


def schedule(scheduler: torch._inductor.scheduler.Scheduler) -> None:
    # [FUNCTION DOC] Finish distributed autotuning for a scheduler: autotune this rank's nodes, all-gather the
    # results from all ranks, then replace remote placeholder nodes with real buffers.
    # Raises an error if the distributed_max_autotune_gemm config is off. Mutates scheduler nodes.
    """
    Finish the distributed autotuning by propagating the autotuning results
    between the ranks and then replacing the placeholder with the real Buffer.
    """
    if not config.distributed_max_autotune_gemm:
        raise AssertionError("distributed_max_autotune_gemm config must be enabled")
    autotune_results = _autotune_local_nodes(scheduler)
    choices_by_index = _sync(autotune_results)
    _autotune_remote_nodes(scheduler, choices_by_index)


@contextlib.contextmanager
def graph_context() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager wrapped around graph processing. Installs a fresh distributed autotune
    # state in the virtualized state and resets it to a null handler on exit (even on error).
    # Raises an error if a state is already active (nesting is not allowed).
    """
    Wrapped around processing a graph, sets up figuring out which ranks tune
    which shapes.
    """
    if isinstance(
        V.get_distributed_autotune_state(check_poisoned=False),  # type: ignore[call-arg]
        _DistributedAutotuneState,
    ):
        raise AssertionError(
            "distributed autotune state is already active, nested graph_context is not allowed"
        )
    V.set_distributed_autotune_state(_DistributedAutotuneState())
    try:
        yield
    finally:
        V.set_distributed_autotune_state(NullHandler())


def maybe_autotune_remote(
    name: str, choices: list[ChoiceCaller], inputs: list[Buffer], layout: Layout
) -> TensorBox | None:
    # [FUNCTION DOC] Called by an op such as mm to decide where to autotune. Returns None when the feature is
    # off, no autotune group exists, there is at most one choice, or this rank owns the op
    # (round robin on the running index versus group rank). Otherwise returns a TensorBox wrapping
    # a placeholder buffer for a remote rank. Side effects: bumps the state's index (and local
    # count when local) and records index/local info in the current FX node's metadata.
    """
    Used by an op (like `mm`) to determine if the op should be autotuned
    locally (returns None) or remotely (returns a placeholder Buffer).
    """
    if not config.distributed_max_autotune_gemm:
        return None

    if not (autotune_pg := get_autotune_pg()):
        return None

    if len(choices) <= 1:
        return None

    state = V.distributed_autotune_state
    index = state.autotuned_index
    state.autotuned_index += 1
    local = index % autotune_pg.size() == autotune_pg.rank()

    V.current_node.meta[_DISTRIBUTED_AUTOTUNE_KEY] = _DistributedAutotuneInfo(
        index, local
    )
    if local:
        state.autotuned_local_count += 1
        return None

    return torch._inductor.ir.TensorBox.create(
        _DistributedAutotuneBuffer(name, inputs, layout)
    )


class _DistributedAutotuneBuffer(MultiTemplateBuffer):
    # [CLASS DOC] A MultiTemplateBuffer placeholder for a kernel being autotuned on another rank.
    # Stores the kernel name; has no real choices. After scheduling, autotune() builds the real
    # buffer from serialized results and the scheduler swaps it in.
    """
    A MultiTemplateBuffer which represents a kernel being autotuned on a
    different rank. When `schedule` is called this will be replaced by the
    "real" buffer.
    """

    # Name of the kernel being autotuned.
    _kernel_name: str

    def __init__(
        self,
        kernel_name: str,
        inputs: list[Buffer],
        layout: Layout,
    ) -> None:
        # [FUNCTION DOC] Create the placeholder buffer with empty choices and a timing function that always fails;
        # records the kernel name.
        super().__init__(
            layout,
            inputs,
            choice_timings_fn=self._dummy_choice_timings,
            unfiltered_choices=[],
            allowed_prologue_inps=OrderedSet({}),
        )

        self._kernel_name = kernel_name

    def _dummy_choice_timings(
        self, _hint_override: int | None
    ) -> dict[ChoiceCaller, float]:
        # [FUNCTION DOC] HELPER: Timing callback that must never run; raises NotImplementedError because a remote
        # autotune placeholder was never filled in.
        # This should never get called. It means that a remote autotune was
        # scheduled but never filled in.
        raise NotImplementedError

    def autotune(self, ser_choice: _SerializedChoice) -> TensorBox:
        # [FUNCTION DOC] Build the final TensorBox from a serialized choice received from another rank.
        # Temporarily clears the graph scheduler, rebuilds the choice for this buffer's layout and
        # original inputs, runs the algorithm selection with that single choice, and returns the
        # result. Raises an error if the layout is not a Layout or the result is not a TensorBox.
        """
        Given a _SerializedChoice (autotune results from another rank)
        compute the final TensorBox.
        """

        from .select_algorithm import autotune_select_algorithm

        with patch.object(V.graph, "scheduler", None):
            kernel_inputs = MMKernelInputs([*self.original_inputs])
            if not isinstance(self.layout, Layout):
                raise AssertionError(f"Expected Layout, got {type(self.layout)}")
            choice = ser_choice.get_choice(self.layout, kernel_inputs)
            buffer, _ = autotune_select_algorithm(
                self._kernel_name,
                [choice],
                kernel_inputs.nodes(),
                self.layout,
            )
            if not isinstance(buffer, TensorBox):
                raise AssertionError(f"Expected TensorBox, got {type(buffer)}")
            return buffer


# Can we make this async?
def _sync(autotune_results: list[_SerializedChoice]) -> Sequence[_SerializedChoice]:
    # [FUNCTION DOC] HELPER: All-gather every rank's serialized autotune results over the autotune group and
    # return them as one list indexed by op index. Raises an error if the group is missing,
    # an entry has the wrong type, an index is duplicated, or the counts disagree.
    """
    Perform the all_gather to collect the autotune results from all the ranks.
    """

    autotune_pg = get_autotune_pg()
    if not autotune_pg:
        raise AssertionError("autotune process group is not initialized")

    # Perform allgather
    all_states: list[list[_SerializedChoice]] = [None] * autotune_pg.size()  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_states, autotune_results, group=autotune_pg)

    node_count = sum(len(x) for x in all_states)
    # It's faster to briefly lie about the type than to unzip the results and append.
    choices_by_index: list[_SerializedChoice] = [None] * node_count  # type: ignore[list-item]

    check_count = 0
    for other_results in all_states:
        for choice in other_results:
            if not isinstance(choice, _SerializedChoice):
                raise AssertionError(f"Expected _SerializedChoice, got {type(choice)}")
            if choices_by_index[choice.index] is not None:
                raise AssertionError(f"duplicate choice for index {choice.index}")
            choices_by_index[choice.index] = choice
            check_count += 1

    if node_count != check_count:
        raise AssertionError(f"count mismatch: {node_count} != {check_count}")
    return choices_by_index


class _SerializedChoice:
    # [CLASS DOC] HELPER: Picklable form of an autotune choice (op index, template uid, parsed kwargs),
    # since the real choice object cannot be serialized. get_choice() reconstructs it later.
    # Only the mm extern kernel and the mm Triton template are supported.
    """
    This is a serializer for the autotune choice. KernelTemplateChoice can't
    be serialized directly (the template and inputs prevent this) so we need to
    serialize it by parts and reconstruct later on.
    """

    def __init__(self, index: int, choice: ChoiceCaller) -> None:
        # [FUNCTION DOC] HELPER: Record the op index, derive the template uid from the choice, and parse the
        # choice description into keyword arguments.
        self.index = index
        self.template_uid = _SerializedChoice._template_uid_from_choice(choice)
        self.kwargs = self._compute_kwargs(choice.description)

    def get_choice(self, layout: Layout, inputs: KernelInputs) -> ChoiceCaller | None:
        # [FUNCTION DOC] Rebuild the ChoiceCaller for the given layout and kernel inputs from stored template uid
        # and kwargs. If BLOCK_K is present, computes the EVEN_K flag from the K dimension of the first
        # input. Returns the choice of the reconstructed template choice.
        """
        Deserialize the ChoiceCaller and return it.
        """

        template = self._template_from_uid()

        kwargs = {**self.kwargs}
        if "BLOCK_K" in kwargs:
            # TODO: Do we really need to externally compute this value? If it's
            # needed I'm surprised it's not just part of the original template
            # description.
            # This needs the actual 'k' to figure out the value.
            k = inputs.nodes()[0].get_size()[1]
            kwargs["EVEN_K"] = sympy.gcd(k, kwargs["BLOCK_K"]) == kwargs["BLOCK_K"]

        extra_kwargs: dict[str, Any] = {}
        from .kernel_template_choice import (
            DictKernelTemplateParams,
            KernelTemplateChoice,
        )

        params = DictKernelTemplateParams(kwargs)
        ktc = KernelTemplateChoice(template, params, extra_kwargs, layout, inputs)
        return ktc.choice

    @staticmethod
    def _compute_kwargs(description: str) -> dict[str, int | str | bool]:
        # [FUNCTION DOC] HELPER: Parse a comma separated key=value description into a mapping. Values are
        # booleans, digit-only integers, or single-quoted strings (quotes stripped). Empty
        # description gives an empty mapping. Raises an error for an unquoted non-numeric value.
        # KNOWN ISSUE: negative or non-integer numeric values are not digits and would hit the quote check.
        """
        Given a template description turn it into input kwargs.
        """
        if not description:
            return {}

        # TODO: It seems like it would be better if the template could provide
        # this directly instead of having to parse a string.
        kwargs: dict[str, int | str | bool] = {}
        for cfg in description.split(","):
            key, val = cfg.split("=", 1)
            key, val = key.strip(), val.strip()
            if val == "True":
                kwargs[key] = True
            elif val == "False":
                kwargs[key] = False
            elif val.isdigit():
                kwargs[key] = int(val)
            else:
                if not (val.startswith("'") and val.endswith("'")):
                    raise AssertionError(
                        f"expected string value to be single-quoted, got {val!r}"
                    )
                kwargs[key] = val[1:-1]
        return kwargs

    @staticmethod
    def _template_uid_from_choice(choice: ChoiceCaller) -> str:
        # [FUNCTION DOC] HELPER: Map a choice to a dotted template path: the extern mm kernel or the mm Triton
        # template. Raises RuntimeError for any other kernel or choice type.
        """
        Given a ChoiceCaller figure out which template represents it. This
        is reversed by _template_from_uid().
        """

        # We need a better way to do this - right now we need to add each
        # supported template directly.
        if isinstance(choice, select_algorithm.ExternKernelCaller):
            if choice.choice.name == "mm":
                return "torch._inductor.kernel.mm.aten_mm"
            else:
                raise RuntimeError(f"TODO: kernel {choice.choice.name!r}")
        elif isinstance(choice, select_algorithm.TritonTemplateCaller):
            return "torch._inductor.kernel.mm.mm_template"
        else:
            raise RuntimeError(f"TODO: {type(choice)}")

    def _template_from_uid(self) -> Any:
    # [FUNCTION DOC] HELPER: Resolve the dotted template uid by looking up the first part in module globals
    # and following the rest as attributes. Returns the template object.
        """
        See _template_uid_from_choice().
        """
        parts = self.template_uid.split(".")
        obj = globals()[parts[0]]
        for k in parts[1:]:
            obj = getattr(obj, k)
        return obj


def _autotune_local_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
) -> list[_SerializedChoice]:
    # [FUNCTION DOC] HELPER: Autotune the multi-template buffers this rank owns (marked local in node metadata),
    # skipping remote placeholders and unmarked nodes, forcing selection of the minimum choice.
    # Returns serialized choices. Raises an error if the number found differs from the count
    # recorded during lowering.
    """
    Go through the nodes in the scheduler and autotune the kernels which
    should be autotuned by this rank.
    """

    autotune_results: list[_SerializedChoice] = []

    for node in scheduler.nodes:
        if not isinstance(node, SchedulerNode):
            continue

        if (inner_node := node.node) is None:
            continue

        if isinstance(inner_node, _DistributedAutotuneBuffer):
            # This is marked for remote autotuning.
            continue

        if not isinstance(inner_node, MultiTemplateBuffer):
            continue

        if (origin_node := inner_node.origin_node) is None:
            continue

        if (meta := origin_node.meta) is None:
            continue

        info = meta.get(_DISTRIBUTED_AUTOTUNE_KEY)
        if info is None:
            continue

        if not info.local:
            raise AssertionError(
                "expected info.local to be True for locally autotuned node"
            )

        # We force autotuning here
        # Still takes advantage of async precompile
        # We need all the configs before fusion
        min_choice, _ = inner_node.get_min_choice()

        choice = _SerializedChoice(info.index, min_choice)
        autotune_results.append(choice)

    state = V.distributed_autotune_state
    if len(autotune_results) != state.autotuned_local_count:
        raise AssertionError(
            f"incorrect local autotuned nodes found ({len(autotune_results)} != {state.autotuned_local_count})"
        )
    return autotune_results


def _autotune_remote_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
    choices_by_index: Sequence[_SerializedChoice],
) -> None:
    # [FUNCTION DOC] HELPER: For each scheduler node holding a remote placeholder buffer, autotune it using the
    # gathered choice for its index and replace the node in the scheduler. Raises an error on
    # missing origin node, wrong buffer types, or a layout mismatch.
    """
    Go through the nodes in the scheduler and autotune the nodes that were
    autotuned on remote ranks.
    """

    for i, node in enumerate(scheduler.nodes):
        if isinstance(node, SchedulerNode) and isinstance(
            (dist_node := node.node), _DistributedAutotuneBuffer
        ):
            if dist_node.origin_node is None:
                raise AssertionError("dist_node.origin_node must not be None")
            info = dist_node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_KEY]
            out_tensorbox = dist_node.autotune(choices_by_index[info.index])

            out_storage = out_tensorbox.data
            if not isinstance(out_storage, StorageBox):
                raise AssertionError(f"Expected StorageBox, got {type(out_storage)}")
            out_buffer = out_storage.data
            if not isinstance(out_buffer, OperationBuffer):
                raise AssertionError(
                    f"Expected OperationBuffer, got {type(out_buffer)}"
                )

            if out_buffer.layout != dist_node.layout:
                raise AssertionError(
                    f"layout mismatch: {out_buffer.layout} != {dist_node.layout}"
                )

            scheduler._replace_node(out_buffer, dist_node, i, node)
