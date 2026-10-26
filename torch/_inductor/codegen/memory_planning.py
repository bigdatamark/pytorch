# mypy: allow-untyped-defs
# [MODULE DOC] Static memory planning for Inductor wrapper codegen. Takes the
# list of allocate/free/reuse lines emitted by the wrapper, groups buffers that
# share storage, computes live ranges, and packs buffers into pools (a tree of
# temporal and spatial splits) so one backing allocation serves many buffers.
# Main contents: LiveRange(s), the allocation tree nodes (Allocation, Empty,
# TemporalSplit, SpatialSplit), AllocationPool(s), BufferGroup, the pool-based
# wrapper lines, and MemoryPlanner. No import-time side effects.
from __future__ import annotations

import collections
import dataclasses
import itertools
import pprint
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Max

from .. import config
from ..utils import _align, align, cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
    AllocateLine,
    BufferLike,
    FreeIfNotReusedLine,
    MemoryPlanningLine,
    NullLine,
    ReuseLine,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy


@dataclasses.dataclass
class LiveRange:
    # [CLASS DOC] Dataclass holding a half-open interval (begin inclusive, end
    # exclusive) of program timesteps (numbers, possibly infinite) during which
    # a tensor is live. Treated as immutable; supports containment test, union
    # hull, and length.
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """

    begin: float  # int | +/-inf
    end: float  # int | +/-inf

    def contains(self, other: LiveRange):
        # [FUNCTION DOC] Returns True if other lies entirely inside this range
        # (this begin <= other begin and other end <= this end). No side effects.
        """Is other entirely within self"""
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        # [FUNCTION DOC] Returns a new LiveRange spanning both ranges (min of
        # begins, max of ends), so any gap between them is included.
        """Combine two ranges using a union operation"""
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        # [FUNCTION DOC] Returns end minus begin. Note len() requires an int
        # result, so infinite or float endpoints would fail when called via len.
        return self.end - self.begin


class LiveRanges:
    # [CLASS DOC] A possibly non-contiguous set of live regions stored as a
    # sorted list of disjoint LiveRange objects in the attribute ranges.
    # Touching or overlapping inputs are merged at construction.
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        # [FUNCTION DOC] Sorts the input ranges by begin, then merges any that
        # overlap or touch (end >= next begin) into a single range, storing the
        # result in self.ranges. Empty input gives an empty list.
        # Raises AssertionError if the sorted order invariant is violated.
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            if self.ranges[-1].begin > r.begin:
                raise AssertionError("ranges must be sorted by begin")
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        # [FUNCTION DOC] Returns True if any range in self overlaps any range in
        # other, using a merge-style sweep over both sorted lists (the one
        # starting earlier is checked against the other's start, else popped).
        # Does not mutate either object (works on copies).
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            if left[0].begin > right[0].begin:
                raise AssertionError("left should begin no later than right")
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        # [FUNCTION DOC] Property: begin of the first range. Raises IndexError if empty.
        return self.ranges[0].begin

    @property
    def end(self):
        # [FUNCTION DOC] Property: end of the last range. Raises IndexError if empty.
        return self.ranges[-1].end

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Debug string listing the class name and each range.
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"


class AllocationTreeNode:
    # [CLASS DOC] Abstract interface for nodes of the allocation tree inside a
    # pool. Subclasses report live ranges and byte sizes (a concrete hint and a
    # symbolic runtime expression), accept new blocks via allocate, and are
    # resolved to concrete offsets via finalize.
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        # [FUNCTION DOC] Default: tries to place block in this node; the base
        # version never places anything and returns False. Subclasses return
        # True after placing (and marking) the block. is_last says whether
        # this node may grow to fit.
        """
        Try to assign block to a memory location in this pool.  Return True if
        an assignment was made.
        """
        return False

    def get_live_ranges(self) -> LiveRanges:
        # [FUNCTION DOC] Abstract: union of live ranges of all allocations below
        # this node. Raises NotImplementedError in the base class.
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    def get_size_hint(self) -> int:
        # [FUNCTION DOC] Abstract: byte size computed from example-input
        # (hint) sizes. Raises NotImplementedError in the base class.
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    def get_symbolic_size(self) -> sympy.Expr:
        # [FUNCTION DOC] Abstract: symbolic byte size needed at runtime.
        # Raises NotImplementedError in the base class.
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    def finalize(self, pool, offset) -> AllocationTreeNode:
        # [FUNCTION DOC] Hook run once all allocation is done, given the owning
        # pool and this node's byte offset. Base version returns self unchanged;
        # subclasses may return a simplified replacement node.
        """Called after all allocations have been made"""
        return self

    def is_empty(self):
        # [FUNCTION DOC] Returns True if the node is only padding. Base: False.
        return False


@dataclasses.dataclass
class Allocation(AllocationTreeNode):
    # [CLASS DOC] Leaf of the allocation tree: the memory block for one buffer
    # (node) with its live range, byte size hint and symbolic size. After
    # finalize it records its pool and byte offset; allocated flags that it was
    # placed. earliest_available is set only when sizes contain unbacked
    # symbols, restricting the block to pools that start no earlier than it.
    """
    Represents memory allocated to a given node in the allocation pool.
    """

    node: BufferLike
    live_range: LiveRange
    size_hint: int
    symbolic_size: sympy.Expr
    allocated: bool = False
    pool: AllocationPool | None = None
    offset: sympy.Expr | None = None
    earliest_available: float | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] If any dimension of the buffer layout size contains
        # unbacked symbols, sets earliest_available to the start of this
        # block's live range; otherwise leaves it unset.
        has_unbacked_sym = False
        for s in self.node.get_layout().size:
            if free_unbacked_symbols(s):
                has_unbacked_sym = True
                break

        if has_unbacked_sym:
            self.earliest_available = self.get_live_ranges().begin

    @property
    def device(self):
        # [FUNCTION DOC] Property: device of the underlying buffer node.
        return self.node.get_device()

    def get_live_ranges(self):
        # [FUNCTION DOC] Returns a LiveRanges containing just this block's live range.
        return LiveRanges([self.live_range])

    def get_size_hint(self):
        # [FUNCTION DOC] Returns the stored byte size hint.
        return self.size_hint

    def get_symbolic_size(self):
        # [FUNCTION DOC] Returns the stored symbolic byte size.
        return self.symbolic_size

    def mark_allocated(self):
        # [FUNCTION DOC] Sets the allocated flag. Raises AssertionError if it was already set.
        if self.allocated:
            raise AssertionError("block already allocated")
        self.allocated = True

    def finalize(self, pool, offset):
        # [FUNCTION DOC] Records the owning pool and byte offset on this block
        # and returns self. Raises AssertionError if already finalized.
        if not (self.pool is None and self.offset is None):
            raise AssertionError("block already finalized")
        self.pool = pool
        self.offset = offset
        return self

    def codegen_alloc_from_pool(self, wrapper):
        # [FUNCTION DOC] Asks the wrapper for code that views this block's
        # slice of the pool (pool name, offset, dtype, shape, stride) and
        # returns the wrapper's result (the expression plus any lines to emit).
        # Raises AssertionError if no pool is assigned yet.
        if not self.pool:
            raise AssertionError("block has no pool assigned")
        node = self.node
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        return wrapper.codegen_alloc_from_pool(
            self.pool.name, self.offset, node.get_dtype(), shape, stride
        )

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Debug string with node name, range, sizes, pool name and offset.
        return (
            f"{self.__class__.__name__}("
            f"node={self.node.get_name()}, "
            f"live_range={self.live_range}, "
            f"size_hint={self.size_hint}, "
            f"symbolic_size={self.symbolic_size}, "
            f"pool={self.pool.name if self.pool else None}, "
            f"offset={self.offset})"
        )

    def get_earliest_available(self):
        # [FUNCTION DOC] Returns earliest_available (None unless unbacked symbols are involved).
        return self.earliest_available


@dataclasses.dataclass
class Empty(AllocationTreeNode):
    # [CLASS DOC] Padding node representing unused space of size_hint bytes in
    # a pool. Has no live ranges and symbolic size zero; exists only so parent
    # nodes compute correct size hints.
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """

    size_hint: int

    def get_live_ranges(self):
        # [FUNCTION DOC] Returns an empty LiveRanges.
        return LiveRanges([])

    def get_size_hint(self):
        # [FUNCTION DOC] Returns the padding size in bytes.
        return self.size_hint

    def get_symbolic_size(self):
        # [FUNCTION DOC] Returns zero (padding does not count toward the runtime size).
        return 0

    def is_empty(self):
        # [FUNCTION DOC] Always True.
        return True


class MemorySplitProtocol(Protocol):
    # [CLASS DOC] HELPER: Typing protocol describing what ClearCacheOnAllocateMixin
    # needs from its host: cached get_live_ranges/get_size_hint/get_symbolic_size
    # methods and an _allocate method.
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]

    def _allocate(self, block: Allocation, is_last: bool) -> bool: ...
    # [FUNCTION DOC] HELPER: Protocol stub; implementers try to place block and return success.


class ClearCacheOnAllocateMixin(MemorySplitProtocol):
    # [CLASS DOC] HELPER: Mixin that wraps a subclass's _allocate so that the
    # cached live-range and size results are invalidated whenever a block is
    # successfully placed.
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """

    def allocate(self, block: Allocation, is_last: bool):
        # [FUNCTION DOC] Calls _allocate; if it placed the block, clears the
        # caches. Returns whether the block was placed.
        is_allocated = self._allocate(block, is_last)
        if is_allocated:
            self.clear_cache()
        return is_allocated

    def clear_cache(self):
        # [FUNCTION DOC] HELPER: Drops the memoized results of the three cached getters on this object.
        self.get_live_ranges.clear_cache(self)
        self.get_size_hint.clear_cache(self)
        self.get_symbolic_size.clear_cache(self)


@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    # [CLASS DOC] Tree node holding a list of child nodes that occupy the same
    # memory slot at different times (their live ranges never overlap). Its size
    # is the largest child; its live ranges are the union of the children's.
    # Cached getters are invalidated on allocation. finalize may collapse to the
    # single child.
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """

    allocations: list[AllocationTreeNode]

    def _allocate(self, block: Allocation, is_last: bool):
        # [FUNCTION DOC] Tries to place block in this slot. Fails (False) if the
        # block is larger than the slot and this is not the last node, or if it
        # overlaps in time with more than one child. If exactly one child
        # overlaps, delegates to that child. Otherwise marks the block
        # allocated, drops a lone Empty placeholder, and adds the block: as-is
        # on exact size fit, wrapped in a SpatialSplit with padding if smaller,
        # or (only when is_last) grows the slot by padding every existing child.
        # Mutates self.allocations; returns True on placement.
        # Raises AssertionError if growth is attempted when not is_last.
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        block_live = block.get_live_ranges()
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        if len(overlapping) > 1:
            # TODO(jansel): we could try harder here by merging overlapping in space
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()

            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            if slot_size == block_size:
                # perfect fit
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(
                    SpatialSplit.create(block, slot_size - block_size)
                )
            else:  # grow this allocation
                if not is_last:
                    raise AssertionError("can only grow allocation when is_last")
                self.allocations = [
                    *(
                        SpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        # [FUNCTION DOC] Cached. Returns the merged LiveRanges of all children.
        return LiveRanges(
            itertools.chain.from_iterable(
                x.get_live_ranges().ranges for x in self.allocations
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        # [FUNCTION DOC] Cached. Returns the maximum child size hint, or 0 if there are no children.
        if not self.allocations:
            return 0
        return max(x.get_size_hint() for x in self.allocations)

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        # [FUNCTION DOC] Cached. Returns the symbolic maximum of child symbolic sizes, or 0 if empty.
        if not self.allocations:
            return 0  # type: ignore[return-value]
        return Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        # [FUNCTION DOC] True when the only child is itself empty padding.
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        # [FUNCTION DOC] Finalizes each child at the same pool and offset
        # (children share the slot), clears caches, and returns the single
        # child if only one remains, else self.
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self


@dataclasses.dataclass
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    # [CLASS DOC] Tree node with two TemporalSplit children, left and right,
    # laid out one after the other in memory (right starts at the aligned end of
    # left). Size is aligned left size plus right size. Allocation tries left
    # first (never growing it), then right.
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """

    left: TemporalSplit
    right: TemporalSplit

    @staticmethod
    def create(left, extra_space):
        # [FUNCTION DOC] Builds a SpatialSplit whose left holds the given node
        # and whose right holds an Empty padding of extra_space bytes.
        # Raises AssertionError if left is not an AllocationTreeNode or
        # extra_space is not a positive int.
        if not isinstance(left, AllocationTreeNode):
            raise AssertionError(f"expected AllocationTreeNode, got {type(left)}")
        if not (isinstance(extra_space, int) and extra_space >= 1):
            raise AssertionError(
                f"expected positive int extra_space, got {extra_space}"
            )
        return SpatialSplit(TemporalSplit([left]), TemporalSplit([Empty(extra_space)]))

    def _allocate(self, block: Allocation, is_last: bool):
        # [FUNCTION DOC] Tries to place block in left (never allowed to grow),
        # else in right (growth allowed per is_last). Returns True on success.
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    @cache_on_self
    def get_live_ranges(self):
        # [FUNCTION DOC] Cached. Returns merged live ranges of left and right.
        return LiveRanges(
            itertools.chain(
                self.left.get_live_ranges().ranges, self.right.get_live_ranges().ranges
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        # [FUNCTION DOC] Cached. Left size hint rounded up to the alignment plus right size hint.
        return _align(self.left.get_size_hint()) + self.right.get_size_hint()

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        # [FUNCTION DOC] Cached. Symbolic left size aligned up plus symbolic right size.
        return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()

    def finalize(self, pool, offset):
        # [FUNCTION DOC] Finalizes left at offset and right at offset plus the
        # aligned symbolic size of left, clears caches, and returns left alone
        # if right is just padding, else self.
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )
        self.clear_cache()
        if self.right.is_empty():
            return self.left
        return self


@dataclasses.dataclass
class AllocationPool:
    # [CLASS DOC] One backing allocation for a device, described by a tree
    # (root TemporalSplit) of buffer blocks. can_expand controls whether new
    # blocks may extend it; restrict_live_range limits which blocks may join
    # (set when unbacked-size blocks are present); name is set at finalize;
    # names_to_del lists tensor names to free; creation_cache maps pool view
    # expressions to the first tensor name made from them. Identity-based
    # equality and hashing.
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """

    device: torch.device
    root: TemporalSplit
    can_expand: bool = True
    restrict_live_range: LiveRange | None = None
    name: str | None = None
    names_to_del: list[str] = dataclasses.field(default_factory=list)
    creation_cache: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Updates restrict_live_range from every Allocation already in the root.
        for block in self.root.allocations:
            if isinstance(block, Allocation):
                self.update_restrict_live_range(block)

    def allocate(self, block: Allocation, is_last: bool):
        # [FUNCTION DOC] Tries to place block in this pool. Rejects it if it
        # falls outside restrict_live_range or if its earliest_available is
        # later than the pool's start. Otherwise tries the tree (growth only if
        # is_last and can_expand), and if that fails and growth is allowed,
        # appends the block at the end. Returns True when placed and updates
        # restrict_live_range.
        if (
            self.restrict_live_range is not None
            and not self.restrict_live_range.contains(block.live_range)
        ):
            return False

        block_earliest_available = block.get_earliest_available()
        pool_begin = self.root.get_live_ranges().begin
        if block_earliest_available and block_earliest_available > pool_begin:
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            self.update_restrict_live_range(block)
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False

    def update_restrict_live_range(self, block: Allocation):
        # [FUNCTION DOC] If block has an earliest_available time, sets
        # restrict_live_range to start there (or the smaller of the old begin and
        # it) and extend to infinity. No effect for blocks without one.
        if block_earliest_available := block.get_earliest_available():
            if self.restrict_live_range is None:
                self.restrict_live_range = LiveRange(
                    block_earliest_available, float("inf")
                )
            else:
                self.restrict_live_range = LiveRange(
                    min(self.restrict_live_range.begin, block_earliest_available),
                    self.restrict_live_range.end,
                )

    def allocate_at_end(self, block):
        # [FUNCTION DOC] Marks block allocated and makes it a new region placed
        # after the whole current tree, growing the pool. Always returns True.
        block.mark_allocated()
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
        self.update_restrict_live_range(block)
        return True

    def finalize(self, name):
        # [FUNCTION DOC] Names the pool, records the name in names_to_del, and
        # finalizes the tree with offset zero (note: the returned root from the
        # tree finalize is discarded). Raises AssertionError if already named.
        if self.name:
            raise AssertionError("pool already finalized")
        self.name = name
        self.names_to_del.append(name)
        self.root.finalize(self, 0)

    def codegen_create(self, wrapper, code: IndentedBuffer):
        # [FUNCTION DOC] Writes the line creating the pool tensor into code.
        # If a top-level block is as large as the whole pool, the pool is
        # created with that block's dtype/shape/stride; otherwise a flat byte
        # (uint8) tensor of the total symbolic size is created.
        # Raises AssertionError if the pool has no name yet.
        if not self.name:
            raise AssertionError("pool must be finalized before codegen_create")
        nbytes = self.root.get_symbolic_size()
        for block in self.root.allocations:
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                node = block.node
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                return
        else:
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )

    def codegen_destroy(self, wrapper, code: IndentedBuffer):
        # [FUNCTION DOC] Writes the line freeing all names in names_to_del into code.
        code.writeline(wrapper.make_free_by_names(self.names_to_del))

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Identity equality (dataclass field equality is disabled).
        return self is other

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Identity-based hash, consistent with equality.
        return id(self)


@dataclasses.dataclass
class AllocationPools:
    # [CLASS DOC] Registry mapping each device to its ordered list of
    # AllocationPool objects. Routes blocks to pools, creating new pools when
    # needed, depending on config.memory_pool; finalize names all pools.
    """
    Collection of many AllocationPool objects grouped by device.
    """

    device_to_pools: dict[torch.device, list[AllocationPool]] = dataclasses.field(
        default_factory=dict
    )

    def get_pools(self, block):
        # [FUNCTION DOC] Returns the pool list for the block's device, creating an empty list if new.
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        # [FUNCTION DOC] Places an intermediate block into the first pool of
        # its device that accepts it (only the last pool may grow). If none
        # accepts, creates a new pool holding the block (expandable unless
        # config.memory_pool is "none") and marks the block allocated.
        pools = self.get_pools(block)

        for pool in pools:
            if pool.allocate(block, is_last=pool is pools[-1]):
                return

        # everything is full, make a new pool
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=config.memory_pool != "none",
            )
        )
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        # [FUNCTION DOC] Places an output block. With memory_pool "outputs" or
        # "combined" and an existing pool, appends it at the end of the last
        # pool; otherwise creates a new pool for it (expandable only for
        # "combined") and marks it allocated.
        """Outputs get different pools so memory gets freed properly"""
        pools = self.get_pools(block)
        if pools and config.memory_pool in ("outputs", "combined"):
            pools[-1].allocate_at_end(block)
        else:
            # create a new pool
            block.mark_allocated()
            pools.append(
                AllocationPool(
                    block.device,
                    TemporalSplit([block]),
                    can_expand=config.memory_pool == "combined",
                )
            )

    def finalize(self):
        # [FUNCTION DOC] Names every pool across all devices with consecutive
        # names (pool0, pool1, ...) and finalizes their trees.
        """Called at the end of allocation process"""
        for i, pool in enumerate(
            itertools.chain.from_iterable(self.device_to_pools.values())
        ):
            pool.finalize(f"pool{i}")

    def pprint(self):
        # [FUNCTION DOC] HELPER: Debug aid printing each pool's name, live ranges and tree to stdout.
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)


class BufferGroup:
    # [CLASS DOC] A set of buffer names that share one storage because of
    # in-place reuse. State: node (the originally allocated buffer), names
    # (all aliases), is_output (any alias is a graph output), allocation (set by
    # make_allocation), live_range (hull of all timesteps of use, initially
    # empty/inverted).
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: BufferLike):
        # [FUNCTION DOC] Starts a group for node with its name as the only alias,
        # not an output, no allocation, and an empty live range.
        self.node = node
        self.names = [node.get_name()]
        self.is_output = False
        self.allocation: Allocation | None = None
        self.live_range = LiveRange(float("inf"), -float("inf"))

    def update_usage(self, timestep: int):
        # [FUNCTION DOC] Widens live_range so it covers the given timestep
        # (begin becomes the minimum, end the maximum seen).
        """Expand self.live_range to include timestep"""
        self.live_range = LiveRange(
            min(timestep, self.live_range.begin),
            max(timestep, self.live_range.end),
        )

    def sym_nbytes(self):
        # [FUNCTION DOC] Returns the symbolic byte size: layout storage size times dtype item size.
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize

    def make_allocation(self):
        # [FUNCTION DOC] Creates the Allocation for this group from its node,
        # live range and byte size (hint obtained from the graph size variables
        # with a fallback of 64 for unbacked symbols). Stores it in allocation.
        # Raises AssertionError if already allocated or if live ranges were
        # never computed (begin not an int).
        if self.allocation:
            raise AssertionError("multiple allocations")
        if not isinstance(self.live_range.begin, int):
            raise AssertionError("live ranges not computed")
        nbytes = self.sym_nbytes()
        # For now, fallback value will be used if we encounter an unbacked SymInt. The longer-term plan is to have
        # size_hint() use better heuristics for unbackeds, at which point the fallback value will be ignored.
        size_hint = V.graph.sizevars.optimization_hint(nbytes, fallback=64)
        self.allocation = Allocation(
            self.node,
            self.live_range,
            size_hint=size_hint,
            symbolic_size=nbytes,
        )

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Debug string with names, output flag and live range.
        # KNOWN ISSUE: the produced string is missing its closing parenthesis.
        return (
            f"{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, "
            f"live_range={self.live_range}"
        )


@dataclasses.dataclass
class PoolMemoryPlanningLine(MemoryPlanningLine):
    # [CLASS DOC] Base for wrapper lines tied to a BufferGroup: stores the
    # group and the timestep (set during live-range computation) at which the
    # line executes.
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""

    group: BufferGroup
    timestep: int | None = None

    @property
    def node(self):
        # [FUNCTION DOC] Property: the buffer node of the group.
        return self.group.node


@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    # [CLASS DOC] Wrapper line that emits code defining a buffer as a view into
    # its pool. is_first_pool_usage is set by the planner so the first such line
    # for a pool also emits pool creation.
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        # [FUNCTION DOC] Emits code for this allocation into code: creates the
        # pool if this is its first use, registers the group's names for later
        # freeing, writes the pool view lines, then either aliases an earlier
        # tensor with identical pool view or declares the new tensor and records
        # it in the pool's creation cache.
        # Raises AssertionError if the group has no allocation with a pool.
        allocation = self.group.allocation
        if not (allocation and allocation.pool):
            raise AssertionError("group must have an allocation with a pool")
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.extend(self.group.names)
        alloc_from_pool, allocation_lines_to_write = allocation.codegen_alloc_from_pool(
            self.wrapper
        )
        code.writelines(allocation_lines_to_write)
        if alloc_from_pool in pool.creation_cache:
            code.writeline(
                self.wrapper.make_tensor_alias(
                    name, pool.creation_cache[alloc_from_pool], "alloc"
                )
            )
        else:
            pool.creation_cache[alloc_from_pool] = name
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )


@dataclasses.dataclass
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    # [CLASS DOC] Wrapper line for freeing a pooled buffer. Emits code only
    # when is_last_pool_usage is set (the final use of its pool), in which case
    # the whole pool's names are freed.
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""

    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        # [FUNCTION DOC] If this is the pool's last usage, writes the pool
        # destroy code into code; otherwise emits nothing.
        # Raises AssertionError if the group has no allocation with a pool.
        if self.is_last_pool_usage:
            if not (self.group.allocation and self.group.allocation.pool):
                raise AssertionError("group must have an allocation with a pool")
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)


@dataclasses.dataclass
class MemoryPlanner:
    # [CLASS DOC] Runs the memory planning passes over wrapper lines. State:
    # wrapper (codegen object), pools (AllocationPools filled by planning),
    # buffer_groups (None until compute_buffer_groups runs, which may happen
    # only once per planner). Entry point is plan.
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: list[BufferGroup] | None = None

    def plan(self, lines: list[Any]) -> list[Any]:
        # [FUNCTION DOC] Copies the line list, then runs in order: drop removed
        # buffers, convert to pool lines, compute live ranges, allocate groups
        # into pools, mark first/last pool usage. Returns the new list (the
        # input list is not mutated, but line objects may be).
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        # [FUNCTION DOC] In place: replaces allocate, free-if-not-reused and
        # reuse lines whose buffer is in the graph's removed-buffer set with a
        # no-op line.
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        # drop any removed buffers
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def compute_buffer_groups(self, lines):
        # [FUNCTION DOC] Builds a group per allocate line; a reuse line adds the
        # new name to the old buffer's group when the old one has a group
        # (buffers from extern allocations are not grouped). Marks a group as
        # output if any alias is a graph output or is consumed by the OUTPUT
        # node. Stores the unique groups in self.buffer_groups and returns the
        # mapping from every name to its group.
        # Raises AssertionError on duplicate allocations/groups or if already computed.
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                if name in name_to_group:
                    raise AssertionError(f"duplicate allocation for {name}")
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                if new_name in name_to_group:
                    raise AssertionError(f"duplicate group for reused {new_name}")
                # TODO(jansel): we should support reusing buffers created via ExternKernelAlloc
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = OrderedSet(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names) or any(
                any(user.node.get_name() == "OUTPUT" for user in buf.users)
                for name in group.names
                if (buf := V.graph.scheduler.name_to_buf.get(name)) is not None
            )

        if self.buffer_groups is not None:
            raise AssertionError("buffer_groups already computed")
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        # [FUNCTION DOC] Computes buffer groups, then in place swaps allocate
        # and free lines of grouped buffers for pool-based lines, and tells
        # reuse lines of grouped buffers not to delete the old buffer.
        # Raises AssertionError if a free line is already marked as reused.
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    lines[i] = AllocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, FreeIfNotReusedLine):
                if line.is_reused:
                    raise AssertionError("expected line not to be reused")
                if line.node.get_name() in name_to_group:
                    lines[i] = DeallocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, ReuseLine):
                if line.node.get_name() in name_to_group:
                    line.delete_old = False

    def compute_live_ranges(self, lines):
        # [FUNCTION DOC] Walks the lines assigning a timestep to each run of
        # consecutive memory planning lines (non-planning lines separate runs
        # and add no step). Pool lines record their timestep and extend their
        # group's live range. Output groups are extended to one step past the
        # end. Raises AssertionError if buffer groups were not computed.
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)
                        line.timestep = timestep
            else:
                worklist.popleft()

        timestep += 1
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

    def allocate_groups(self):
        # [FUNCTION DOC] Creates an Allocation per group, splits them into
        # outputs (unless the pool mode is "combined") and intermediates, places
        # outputs sorted by increasing size then longer life first, then
        # intermediates by decreasing size then longer life first, and
        # finalizes the pools. Raises AssertionError on an invalid
        # config.memory_pool value or missing groups.
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        if config.memory_pool not in ("none", "intermediates", "outputs", "combined"):
            raise AssertionError(f"invalid memory_pool {config.memory_pool}")
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")

        for group in self.buffer_groups:
            group.make_allocation()

        outputs: list[Allocation] = []
        intermediates: list[Allocation] = []
        for group in self.buffer_groups:
            if not group.allocation:
                raise AssertionError("group has no allocation")
            if group.is_output and config.memory_pool != "combined":
                outputs.append(group.allocation)
            else:
                intermediates.append(group.allocation)

        for block in sorted(
            outputs,
            key=lambda x: (
                x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate_output(block)

        for block in sorted(
            intermediates,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate(block)

        self.pools.finalize()

    def mark_first_last_usage(self, lines):
        # [FUNCTION DOC] Flags the first allocation line of each pool (forward
        # scan) so the pool gets created, and the last free line seen for each
        # pool (backward scan) as last usage only if the pool's live ranges end
        # at or before that line's timestep. Mutates the lines. Raises
        # AssertionError if a group lacks an allocation or pool.
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
        seen = OrderedSet[AllocationPool]()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)

        seen = OrderedSet[AllocationPool]()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_last_pool_usage = (
                        pool.root.get_live_ranges().end <= line.timestep
                    )
                    seen.add(pool)
