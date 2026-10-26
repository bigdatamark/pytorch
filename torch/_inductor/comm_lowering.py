# [MODULE DOC] Inductor lowerings for distributed collectives (c10d functional ops and symmetric-memory ops).
# Defines helpers to realize tensors as communication buffers (CommBufferLayout), a per-graph skip-wait registry,
# and two registration functions that install lowerings into the Inductor lowering table.
# Import-time side effects: creates a module-level logger and an ordered set of buffers whose wait can be skipped.
# mypy: allow-untyped-defs
import logging

import torch
import torch.utils._pytree as pytree
from torch._inductor.utils import is_symbolic
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .virtualized import V


log = logging.getLogger(__name__)


# NOTE [lowering-time collective optimization]
#
# In collective communication libraries such as NCCL, every rank maintains
# communication buffers that are remotely accessible by some peers. Depending
# on the underlying transport, remote accessibility may be established via
# mechanisms such as ib_reg_mr, CUDA P2P, or CUDA multicast. Typically, these
# buffers are private to the communication library by default, and
# communication ops copy user data in and out of these buffers.
#
# To prevent these copies, an optimization commonly known as "user buffer
# registration" can be employed. This allows direct establishment of remote
# accessibility on user buffers, eliminating the need for copying. However,
# this optimization introduces stringent usage requirements, which are
# typically hard to satisfy without being intrusive to the user code:
#
# - Establishing remote accessibility is expensive and often done ahead of
# time. In such implementations, all ranks must agree on the set of allocations
# used for every collective op. Failing to meet this requirement can
# lead to runtime errors or even silent correctness issues.
# - Even if the collective communication library supports gracefully falling
# back to "unregistered" implementations, the fallback mechanism would nullify
# the optimization.
# - Some communication mechanisms impose stricter requirements than others. For
# example, CUDA's multicast + multi-mem instructions require all ranks to agree
# not only on the allocations used for every collective but also on the offsets
# within these allocations.
#
# To support all different mechanisms with optimal results, we aim to satisfy
# the strictest requirement for this family of optimizations - we ensure that
# every collective op invocation is guaranteed to operate on the same
# allocation, at the same offset, in every iteration.
#
# For eligible collective ops, we identify communication buffers at lowering
# time and optionally choose to lower the op to a different kernel
# (communication libraries like NCCL handle both registered and non-registered
# buffers transparently within the same op, though some may require different
# ops for different cases). Later, the codegen will perform "persistent
# allocation" to satisfy the aforementioned constraints, and optionally,
# perform buffer planning to optimize overall memory usage.
def can_realize_as_comm_buffer(
    x: ir.TensorBox, comm_buffer_type: ir.CommBufferType
) -> bool:
    # [FUNCTION DOC] Decide whether an input tensor box can be turned into a communication buffer of the given type.
    # Returns True for pointwise/reduction loops, for buffers already using a comm layout, fixed layouts, and flexible
    # layouts with a non-symbolic element count; returns False for buffers whose allocation Inductor does not control
    # (should_allocate false) or any other layout. No mutation.
    """
    Check if an input can be realized as a comm buffer of the specified
    `comm_buffer_type`.
    """
    data = _get_data(x)

    if isinstance(data, ir.Loops):
        return True

    # We cannot realize buffers as comm buffers if we don't control their
    # allocation.
    if isinstance(data, ir.Buffer) and not data.should_allocate():
        return False

    layout = data.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return True

    if isinstance(layout, ir.FixedLayout):
        return True

    if isinstance(layout, ir.FlexibleLayout) and not is_symbolic(data.get_numel()):
        return True

    return False


def realize_as_comm_buffer(
    x: ir.TensorBox,
    comm_buffer_type: ir.CommBufferType,
    group_name: "torch.distributed.distributed_c10d.GroupName",
) -> None:
    # [FUNCTION DOC] Realize the tensor box and switch its underlying buffer layout to a CommBufferLayout wrapping the old layout.
    # No-op if already a comm buffer. Mutates the buffer's layout attribute. Raises an assertion error if the data is
    # not a buffer, the layout is neither flexible nor fixed, or the buffer has a symbolic element count.
    """
    Realize an input as a comm buffer of the specified `comm_buffer_type`.

    Specifically, this realizes the underlying buffer if it's still unrealized
    and changes the layout of the buffer to `ir.CommBufferLayout`.
    """
    x.realize()
    buffer = _get_data(x)
    if not isinstance(buffer, ir.Buffer):
        raise AssertionError(f"expected an `ir.Buffer`, got {type(buffer)}")

    layout = buffer.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return

    # The buffer may have already been frozen to FixedLayout if it was used
    # by another operation before the comm operation.
    if not isinstance(layout, (ir.FlexibleLayout, ir.FixedLayout)):
        raise AssertionError(
            "A buffer can only be realized as a comm buffer if it "
            f"has `FlexibleLayout` or `FixedLayout` (got {layout})."
        )

    if is_symbolic(buffer.get_numel()):
        raise AssertionError(
            "A buffer with symbolic shape cannot be converted to "
            f"a comm buffer (got {layout})."
        )

    buffer.layout = ir.CommBufferLayout(
        layout=layout,
        comm_buffer_type=comm_buffer_type,
        group_name=group_name,
    )


def _get_data(x: ir.TensorBox) -> ir.IRNode:
    # [FUNCTION DOC] HELPER: Unwrap a tensor box (through view and storage box layers) to the underlying IR node.
    # Raises an assertion error if the box holds neither a view nor a storage box, or if the unwrapped view is not a view or mutable box.
    if isinstance(x.data, ir.BaseView):
        # TensorBox -> *View -> StorageBox -> IRNode
        node = x.data.unwrap_view()
        if not isinstance(node, (ir.BaseView, ir.MutableBox)):
            raise AssertionError(
                f"expected an `ir.BaseView` or `ir.MutableBox`, got {type(node)}"
            )
        return node.data
    elif isinstance(x.data, ir.StorageBox):
        # TensorBox -> StorageBox -> IRNode
        return x.data.data
    else:
        raise AssertionError(
            "Expect the data attr of a `TensorBox` to be either "
            f"an `ir.BaseView` or `ir.StorageBox` (got {x.data})."
        )


_bufs_to_skip_wait = OrderedSet[tuple[int, str]]()


def mark_as_skip_wait(x: ir.IRNode) -> None:
    # [FUNCTION DOC] Record the node (keyed by current graph identity and buffer name) so its later wait lowering is skipped.
    # Used when a non-blocking collective was lowered as a blocking one. Mutates the module-level skip set.
    """
    If a non-blocking collective is lowered as a blocking collective, the wait
    node in the original graph becomes useless and we can skip the lowering it.
    """
    _bufs_to_skip_wait.add((id(V.graph), x.get_name()))


def should_skip_wait(x: ir.IRNode) -> bool:
    # [FUNCTION DOC] Return True if the node was previously marked via mark_as_skip_wait for the current graph.
    return (id(V.graph), x.get_name()) in _bufs_to_skip_wait


def _should_lower_as_one_shot_all_reduce(
    inp: ir.TensorBox,
    reduce_op: str,
    group_name: "torch.distributed.distributed_c10d.GroupName",
):
    # [FUNCTION DOC] HELPER: Decide whether an all-reduce should use the symmetric-memory one-shot kernel.
    # True only if auto-selection is enabled in config, symmetric memory is enabled for the group, the input can be
    # realized as a symm-mem comm buffer, the reduce op is sum, and the input byte size is within the configured threshold.
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)
        and reduce_op == "sum"
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes
    )


def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    # [FUNCTION DOC] HELPER: Realize the input as a symm-mem comm buffer and create a fallback-kernel call to the one-shot all-reduce op.
    # Returns the result wrapped as tensor box(es). Side effect: changes the input buffer layout.
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)
    return pytree.tree_map(
        ir.TensorBox.create,
        ir.FallbackKernel.create(
            torch.ops.symm_mem.one_shot_all_reduce.default,
            inp,
            reduce_op,
            group_name,
        ),
    )


def _create_out_of_place(kernel, inputs, *args) -> ir.IRNode:
    # [FUNCTION DOC] HELPER: Create an out-of-place collective kernel node for the given op and inputs, returning it as a tensor box.
    # Raises an assertion error if the created node is not an IR node.
    node = ir._CollectiveKernel.create_out_of_place(kernel, inputs, *args)
    if not isinstance(node, ir.IRNode):
        raise AssertionError(f"expected an `ir.IRNode`, got {type(node)}")
    return ir.TensorBox.create(node)


def register_comm_lowerings():
    # [FUNCTION DOC] Register lowerings for the c10d functional collectives (all_reduce, all_gather, reduce_scatter, all_to_all,
    # broadcast, wait_tensor, isend/irecv, batched p2p, shard_dim_alltoall).
    # Returns silently (after an info log) if the distributed functional ops are not built. Side effect: populates the
    # global lowering table and layout constraints (inputs constrained to FX strides).
    """
    Register lowerings for the comm subsystem.
    """
    try:
        _ = torch.ops._c10d_functional.all_reduce
    except AttributeError:
        log.info(
            "Inductor support for distributed collectives depends on building "
            "torch.distributed"
        )
        return

    from .lowering import (
        add_layout_constraint,
        clone,
        constrain_to_fx_strides,
        copy_,
        register_lowering,
    )

    def register_comm_lowering(fn):
        # [FUNCTION DOC] HELPER: Register fn as a lowering and constrain its inputs to the strides seen in the FX graph. Returns the registered lowering decorator result.
        add_layout_constraint(fn, constrain_to_fx_strides)
        return register_lowering(fn)

    c10d = torch.ops._c10d_functional

    @register_comm_lowering(c10d.all_reduce)  # type: ignore[misc]
    def _all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        # [FUNCTION DOC] Lower the out-of-place all_reduce. Uses the symm-mem one-shot kernel when eligible; otherwise clones the input,
        # optionally marks the clone as not fusable (when compute/comm overlap reordering is on), makes it contiguous and
        # emits an in-place all_reduce kernel on the clone. Returns the reduced tensor box.
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            return _one_shot_all_reduce(inp, reduce_op, group_name)

        # Lower as c10d.all_reduce_
        inp = clone(inp)
        if config.reorder_for_compute_comm_overlap:
            # The horizontal fusion of this clone often severely delays the
            # scheduling of the all_reduce_ node. Horizontally fusing this
            # clone can almost never out-perform scheduling the all_reduce_
            # earlier. Also in most cases, this clone is eliminated via
            # in-place reuse. Therefore, we tell the scheduler to not fuse it.
            inp.realize()
            V.graph.no_fuse_buffer_names.add(inp.get_name())
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        # Because we are lowering as inplace c10d.all_reduce_, we should generate
        # _AllReduce_Kernel instead of _AllReduceKernel.
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_)  # type: ignore[misc]
    def _all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        # [FUNCTION DOC] Lower the in-place all_reduce. If eligible for the one-shot symm-mem kernel, copies the result into the input and
        # marks the copy so its wait is skipped; otherwise makes the input contiguous and emits an in-place all_reduce kernel.
        # Returns the input box.
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            ret = copy_(
                inp,
                _one_shot_all_reduce(inp, reduce_op, group_name),
            )
            mark_as_skip_wait(ret)
            return inp

        # Lower as c10d.all_reduce_
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        # [FUNCTION DOC] Lower coalesced all_reduce: clone every input, emit an in-place coalesced all_reduce kernel, return the clones.
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_comm_lowering(c10d.all_reduce_coalesced_)
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        # [FUNCTION DOC] Lower in-place coalesced all_reduce: emit the in-place collective kernel over the inputs and return them.
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_comm_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        # [FUNCTION DOC] Lower all_gather_into_tensor as an out-of-place collective kernel; returns the output tensor box.
        return _create_out_of_place(
            c10d.all_gather_into_tensor.default,
            inp,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        # [FUNCTION DOC] Lower coalesced all_gather_into_tensor as an out-of-place collective; returns a structure of tensor boxes.
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_gather_into_tensor_coalesced.default,
                inputs,
                group_size,
                group_name,
            ),
        )

    @register_comm_lowering(c10d.all_gather_into_tensor_out)
    def _all_gather_into_tensor_out(inp, group_size, group_name, *, out):
        # [FUNCTION DOC] Lower all_gather_into_tensor with a caller-provided output: emit an in-place collective writing to out and return out.
        ir._CollectiveKernel.create_inplace(
            c10d.all_gather_into_tensor_out.default,
            inp,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        # [FUNCTION DOC] Lower reduce_scatter_tensor as an out-of-place collective kernel; returns the output tensor box.
        return _create_out_of_place(
            c10d.reduce_scatter_tensor.default,
            inp,
            reduce_op,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.reduce_scatter_tensor_out)
    def _reduce_scatter_tensor_out(inp, reduce_op, group_size, group_name, *, out):
        # [FUNCTION DOC] Lower reduce_scatter_tensor with a caller-provided output: emit an in-place collective writing to out and return out.
        ir._CollectiveKernel.create_inplace(
            c10d.reduce_scatter_tensor_out.default,
            inp,
            reduce_op,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor_coalesced)
    def _reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
        # [FUNCTION DOC] Lower coalesced reduce_scatter_tensor as an out-of-place collective; returns a structure of tensor boxes.
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.reduce_scatter_tensor_coalesced.default,
                inputs,
                reduce_op,
                group_size,
                group_name,
            ),
        )

    @register_comm_lowering(c10d.all_to_all_single)
    def _all_to_all_single(inp, output_split_sizes, input_split_sizes, group_name):
        # [FUNCTION DOC] Lower all_to_all_single as an out-of-place collective with the given split sizes; returns the output tensor box.
        return _create_out_of_place(
            c10d.all_to_all_single.default,
            inp,
            output_split_sizes,
            input_split_sizes,
            group_name,
        )

    @register_comm_lowering(c10d.broadcast)
    def _broadcast(inp, src, group_name):
        # [FUNCTION DOC] Lower out-of-place broadcast: clone the input, broadcast in place from src, return the clone.
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(c10d.broadcast_)
    def _broadcast_(inp, src, group_name):
        # [FUNCTION DOC] Lower in-place broadcast: emit the in-place broadcast kernel on the input and return it.
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(torch.ops._dtensor.shard_dim_alltoall)
    def _shard_dim_alltoall(inp, gather_dim, shard_dim, group_name):
        # [FUNCTION DOC] Lower the DTensor shard-dim all-to-all (gather_dim, shard_dim) as an out-of-place collective; returns the output box.
        return _create_out_of_place(
            torch.ops._dtensor.shard_dim_alltoall.default,
            inp,
            gather_dim,
            shard_dim,
            group_name,
        )

    @register_comm_lowering(c10d.wait_tensor)
    def _wait_tensor(inp):
        # [FUNCTION DOC] Lower wait_tensor: returns the input unchanged if its wait was marked skippable, else emits a wait kernel then returns the input.
        if should_skip_wait(inp):
            return inp

        ir._WaitKernel.create_wait(c10d.wait_tensor.default, inp)
        return inp

    @register_comm_lowering(c10d.isend)  # type: ignore[misc]
    def _isend(inp, dst, tag, group_name):
        # [FUNCTION DOC] Lower isend: make the input contiguous and emit an out-of-place collective send; returns the placeholder output box.
        inp = ir.ExternKernel.require_contiguous(inp)
        return _create_out_of_place(c10d.isend.default, inp, dst, tag, group_name)

    @register_comm_lowering(c10d.irecv)  # type: ignore[misc]
    def _irecv(inp, src, tag, group_name):
        # [FUNCTION DOC] Lower irecv: make the input contiguous, emit an in-place receive collective into it, and return the input.
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.irecv.default, inp, src, tag, group_name
        )
        return inp

    @register_comm_lowering(c10d.batch_p2p_ops)  # type: ignore[misc]
    def _batch_p2p_ops(op_list, peer_list, tag_list, tensors, group_name):
        # [FUNCTION DOC] Lower batched point-to-point ops. Makes tensors contiguous, runs the kernel under the fake mode to get example outputs
        # (raises an assertion error if unbacked symbols appear), realizes all tensor args and marks receive buffers as mutated,
        # then builds one packed collective node. For receives it records mutation outputs and aliases and returns the tensor itself;
        # for sends it returns a zero-element placeholder output. Returns a list of results aligned with op_list.
        tensors = [ir.ExternKernel.require_contiguous(t) for t in tensors]
        kernel = c10d.batch_p2p_ops.default
        with V.graph.fake_mode:
            result = ir._CollectiveKernel.process_kernel(
                kernel,
                op_list,
                peer_list,
                tag_list,
                tensors,
                group_name,
            )
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        for op, tensor_arg in zip(op_list, tensor_args):
            tensor_arg.realize()
            if op == "irecv":
                V.graph.mark_buffer_mutated(tensor_arg.get_name())

        device = tensor_args[0].get_device_or_error()
        packed = ir._CollectiveKernel(
            ir.MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        results = []
        for i, (op, t, ex_out) in enumerate(zip(op_list, tensors, example_output)):
            if op == "irecv":
                packed.mutation_outputs.append(
                    ir.MutationOutput(ir.NoneLayout(device=device), t, packed)
                )
                packed.alias_names.append(t.get_name())
                results.append(t)
            else:
                # isend: 0-element placeholder output connected to the collective
                placeholder = ir.MultiOutput(
                    ir._CollectiveKernel.tensor_to_layout(ex_out),
                    packed,
                    [(list, i)],
                )
                results.append(ir.TensorBox.create(placeholder))
        return results


def register_symm_mem_lowerings():
    # [FUNCTION DOC] Register lowerings for symmetric-memory ops (one-shot/two-shot/multimem all-reduce, all-gather, reduce-scatter,
    # all-to-all-vdev, tile reduce, low-contention collectives). Returns silently (info log) if the symm_mem ops are unavailable.
    # Side effects: registers out-variant mappings for the one-shot all-reduce ops and populates the lowering table.
    """
    Register lowerings for symmetric memory (symm_mem) operations.
    """
    try:
        symm_mem = torch.ops.symm_mem
        # Check for an actual operation, not just the namespace.
        # torch.ops.symm_mem is a lazy namespace that always exists,
        # but the operations may not exist on non-CUDA platforms or
        # when USE_DISTRIBUTED is disabled.
        _ = symm_mem.one_shot_all_reduce
    except AttributeError:
        log.info("symm_mem ops not available, skipping symm_mem lowerings")
        return

    from torch._library._out_variant import register_out_variant

    # Register manual out variant mappings for symm_mem ops.
    register_out_variant(
        symm_mem.one_shot_all_reduce.default,
        symm_mem.one_shot_all_reduce_out.default,
    )
    register_out_variant(
        symm_mem.one_shot_all_reduce_copy.default,
        symm_mem.one_shot_all_reduce_copy_out.default,
    )

    from .lowering import register_lowering

    def _copy_input_to_comm_buffer(
        inp: ir.TensorBox,
        comm_buffer_type: ir.CommBufferType,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        # [FUNCTION DOC] HELPER: Fallback when Inductor does not own the input allocation: realize it and copy it via an identity pointwise
        # node into a new buffer, which is then realized as a comm buffer. Returns the copy.
        """
        Fallback: insert a Pointwise identity copy allocated in P2P via
        CommBufferLayout.  Used when we don't control the input's allocation.
        """
        inp.realize()
        copy = ir.Pointwise.create(
            device=inp.get_device(),
            dtype=inp.get_dtype(),
            inner_fn=inp.make_loader(),
            ranges=inp.get_size(),
        )
        realize_as_comm_buffer(copy, comm_buffer_type, group_name)
        return copy

    def _maybe_realize_symm_mem(
        inp: ir.TensorBox,
        group_name: str,  # type: ignore[arg-type]
    ) -> ir.TensorBox:
        # [FUNCTION DOC] HELPER: Ensure the input lives in symmetric memory. If it can be realized as a comm buffer, switch its layout in place
        # and return it; otherwise return a comm-buffer copy (costing an extra kernel).
        """
        Ensure inp is in P2P memory for a symm_mem collective.

        If inductor controls the buffer's allocation (ComputedBuffer,
        or any buffer with FlexibleLayout/FixedLayout), switch its
        layout to CommBufferLayout in-place, zero-copy.

        If inductor does not control allocation (e.g. InputBuffer),
        insert a Pointwise identity copy into a new CommBufferLayout buffer.
        This adds an extra Triton kernel. Returns the possibly new TensorBox.

        TODO(tianrengao): eliminate the extra kernel for static-shape
        InputBuffers by pre-allocating P2P memory in the wrapper and DMA .copy_()
        """
        if can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM):
            realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)  # type: ignore[arg-type]
            return inp
        else:
            return _copy_input_to_comm_buffer(
                inp,
                ir.CommBufferType.SYMM_MEM,
                group_name,  # type: ignore[arg-type]
            )

    def _create_low_contention_ag_out(
        kernel: torch._ops.OpOverload,
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        # [FUNCTION DOC] HELPER: Allocate the output of a low-contention all-gather as a fixed-layout symm-mem comm buffer whose first dimension is the
        # input's first dimension times the group size, and create an extern kernel node writing into it. Realizes the input. Returns the output box.
        from torch.distributed import distributed_c10d as c10d

        inp.realize()
        group_size = c10d._get_group_size_by_name(group_name)
        out_size = [inp.get_size()[0] * group_size, *inp.get_size()[1:]]
        layout = ir.CommBufferLayout(
            ir.FixedLayout(
                device=inp.get_device_or_error(),
                dtype=inp.get_dtype(),
                size=out_size,
            ),
            ir.CommBufferType.SYMM_MEM,
            group_name,
        )
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=layout,
                inputs=[inp],
                constant_args=(group_name,),
                op_overload=kernel,
            )
        )

    @register_lowering(symm_mem.one_shot_all_reduce)
    def _symm_mem_one_shot_all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower one_shot_all_reduce: move the input into symm-mem, call the fallback kernel, return the result boxes.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce.default,
                inp,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_out)
    def _symm_mem_one_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower one_shot_all_reduce_out: move the input into symm-mem and call the fallback kernel writing into out.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_copy)
    def _symm_mem_one_shot_all_reduce_copy(
        symm_buffer: ir.TensorBox,
        local_input: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower one_shot_all_reduce_copy: move the symm buffer into symm-mem and call the fallback kernel with the local input.
        symm_buffer = _maybe_realize_symm_mem(symm_buffer, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_copy.default,
                symm_buffer,
                local_input,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_copy_out)
    def _symm_mem_one_shot_all_reduce_copy_out(
        symm_buffer: ir.TensorBox,
        local_input: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower one_shot_all_reduce_copy_out: like the copy variant but with a caller-provided output.
        symm_buffer = _maybe_realize_symm_mem(symm_buffer, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_copy_out.default,
                symm_buffer,
                local_input,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.two_shot_all_reduce_)
    def _symm_mem_two_shot_all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower in-place two_shot_all_reduce_: move input into symm-mem, call the fallback kernel, return the input.
        inp = _maybe_realize_symm_mem(inp, group_name)
        ir.FallbackKernel.create(
            symm_mem.two_shot_all_reduce_.default,
            inp,
            reduce_op,
            group_name,
        )
        return inp

    @register_lowering(symm_mem.two_shot_all_reduce_out)
    def _symm_mem_two_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        output: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower two_shot_all_reduce_out: move input into symm-mem and call the fallback kernel writing into output.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.two_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                output,
            ),
        )

    @register_lowering(symm_mem.multimem_all_reduce_)
    def _symm_mem_multimem_all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower in-place multimem_all_reduce_: move input into symm-mem, call the fallback kernel, return the input.
        inp = _maybe_realize_symm_mem(inp, group_name)
        ir.FallbackKernel.create(
            symm_mem.multimem_all_reduce_.default,
            inp,
            reduce_op,
            group_name,
        )
        return inp

    @register_lowering(symm_mem.multimem_one_shot_all_reduce)
    def _symm_mem_multimem_one_shot_all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower multimem_one_shot_all_reduce: move input into symm-mem and call the fallback kernel; returns result boxes.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_all_reduce.default,
                inp,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.multimem_one_shot_all_reduce_out)
    def _symm_mem_multimem_one_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower multimem_one_shot_all_reduce_out: move input into symm-mem, call the fallback kernel writing into out.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.multimem_one_shot_reduce_out)
    def _symm_mem_multimem_one_shot_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        root: int,
        group_name: str,
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower multimem_one_shot_reduce_out: move input into symm-mem, call the fallback kernel with the root rank and out.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_reduce_out.default,
                inp,
                reduce_op,
                root,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.multimem_all_gather_out)
    def _symm_mem_multimem_all_gather_out(
        inp: ir.TensorBox,
        group_name: str,
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower multimem_all_gather_out: move input into symm-mem, call the fallback kernel writing into out.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_all_gather_out.default,
                inp,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.reduce_scatter_out)
    def _symm_mem_reduce_scatter_out(
        inp: ir.TensorBox,
        group_name: str,
        split_last_dim: bool,
        output: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower symm_mem reduce_scatter_out: move input into symm-mem, call the fallback kernel (with split_last_dim) writing into output.
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.reduce_scatter_out.default,
                inp,
                group_name,
                split_last_dim,
                output,
            ),
        )

    @register_lowering(symm_mem.all_to_all_vdev)
    def _symm_mem_all_to_all_vdev(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower all_to_all_vdev: move input and output into symm-mem and call the fallback kernel; returns None (outputs are in-place).
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev.default,
            inp,
            out,
            in_splits,
            out_splits_offsets,
            group_name,
        )
        return None

    @register_lowering(symm_mem.all_to_all_vdev_2d)
    def _symm_mem_all_to_all_vdev_2d(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
        major_align=None,
    ):
        # [FUNCTION DOC] Lower all_to_all_vdev_2d: move input and output into symm-mem and call the fallback kernel with major_align; returns None.
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev_2d.default,
            inp,
            out,
            in_splits,
            out_splits_offsets,
            group_name,
            major_align,
        )
        return None

    @register_lowering(symm_mem.all_to_all_vdev_2d_offset)
    def _symm_mem_all_to_all_vdev_2d_offset(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits_offsets: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower all_to_all_vdev_2d_offset: move input and output into symm-mem and call the fallback kernel; returns None.
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev_2d_offset.default,
            inp,
            out,
            in_splits_offsets,
            out_splits_offsets,
            group_name,
        )
        return None

    @register_lowering(symm_mem.tile_reduce)
    def _symm_mem_tile_reduce(
        in_tile: ir.TensorBox,
        out_tile: ir.TensorBox,
        root: int,
        group_name: str,
        reduce_op: str = "sum",
    ):
        # [FUNCTION DOC] Lower tile_reduce: move both tiles into symm-mem and call the fallback kernel with root and reduce op; returns None.
        in_tile = _maybe_realize_symm_mem(in_tile, group_name)
        out_tile = _maybe_realize_symm_mem(out_tile, group_name)
        ir.FallbackKernel.create(
            symm_mem.tile_reduce.default,
            in_tile,
            out_tile,
            root,
            group_name,
            reduce_op,
        )
        return None

    @register_lowering(symm_mem.multi_root_tile_reduce)
    def _symm_mem_multi_root_tile_reduce(
        in_tiles,  # list of TensorBox
        out_tile: ir.TensorBox,
        roots,  # list of int
        group_name: str,
        reduce_op: str = "sum",
    ):
        # [FUNCTION DOC] Lower multi_root_tile_reduce: move every input tile (replacing entries in the passed list in place) and the output tile into symm-mem,
        # then call the fallback kernel; returns None.
        for i, in_tile in enumerate(in_tiles):
            in_tiles[i] = _maybe_realize_symm_mem(in_tile, group_name)
        out_tile = _maybe_realize_symm_mem(out_tile, group_name)
        ir.FallbackKernel.create(
            symm_mem.multi_root_tile_reduce.default,
            in_tiles,
            out_tile,
            roots,
            group_name,
            reduce_op,
        )
        return None

    @register_lowering(symm_mem._low_contention_all_gather)
    def _symm_mem_low_contention_all_gather(
        inp: ir.TensorBox,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower low-contention all-gather as an out-of-place collective kernel (so wait lowering can track input lifetime); returns the output box.
        # Use _CollectiveKernel so that _WaitKernel.get_volatile_reads()
        # can track the input's lifetime through wait_tensor, preventing
        # the memory planner from reusing the input buffer while the
        # backend stream is still reading it.
        return _create_out_of_place(
            symm_mem._low_contention_all_gather.default,
            inp,
            group_name,
        )

    @register_lowering(symm_mem._low_contention_all_gather_ce_multicast)
    def _symm_mem_low_contention_all_gather_ce_multicast(
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ):
        # [FUNCTION DOC] Lower the copy-engine multicast low-contention all-gather by allocating a symm-mem output and delegating to the out variant.
        return _create_low_contention_ag_out(
            symm_mem._low_contention_all_gather_ce_multicast_out.default,
            inp,
            group_name,
        )

    @register_lowering(symm_mem._low_contention_all_gather_ce_multicast_out)
    def _symm_mem_low_contention_all_gather_ce_multicast_out(
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
        out: ir.TensorBox,
    ):
        # [FUNCTION DOC] Lower the out variant of the copy-engine multicast all-gather: realizes the input, requires out to be convertible to a symm-mem comm buffer
        # (raises a runtime error otherwise), converts it, and calls the fallback kernel.
        inp.realize()
        if can_realize_as_comm_buffer(out, ir.CommBufferType.SYMM_MEM):
            realize_as_comm_buffer(out, ir.CommBufferType.SYMM_MEM, group_name)
        else:
            raise RuntimeError(
                "_low_contention_all_gather_ce_multicast_out requires an "
                "Inductor-owned output buffer that can be realized as symmetric "
                "memory; this indicates an invalid lowering."
            )
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem._low_contention_all_gather_ce_multicast_out.default,
                inp,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem._low_contention_reduce_scatter)
    def _symm_mem_low_contention_reduce_scatter(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # [FUNCTION DOC] Lower low-contention reduce_scatter as an out-of-place collective kernel; returns the output box.
        # Use _CollectiveKernel so that _WaitKernel.get_volatile_reads()
        # can track the input's lifetime through wait_tensor, preventing
        # the memory planner from reusing the input buffer while the
        # backend stream is still reading it.
        return _create_out_of_place(
            symm_mem._low_contention_reduce_scatter.default,
            inp,
            reduce_op,
            group_name,
        )
