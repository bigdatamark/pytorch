# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for the FSDP2 (fully_shard) all-gather extension protocol, in which a
# tensor subclass (or a plain tensor with monkey-patched methods) supplies fsdp_pre_all_gather
# / fsdp_post_all_gather to control what is all-gathered and how the gathered result is turned
# back into the unsharded parameter. Contents: pre/post hooks for TwoTensor (v1 and v2 pre
# signatures), a wrapper subclass BFloat16AllGatherTensor that gathers in bfloat16, a shared
# mixin, and a multi-process train-parity test plus multi-thread end-to-end, monkey-patch,
# outer size/stride and HSDP mesh tests. Import-time side effect: defines module-level
# device_type from get_devtype().

import contextlib
import copy
import functools
import math
import threading
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.two_tensor import TwoTensor


device_type = torch.device(get_devtype())


def two_tensor_fsdp_pre_all_gather_v1(
    self, mesh: DeviceMesh
) -> tuple[tuple[torch.Tensor, ...], Any]:
    # [FUNCTION DOC] HELPER: Version 1 pre-all-gather hook for TwoTensor, installed as a method.
    # Ignores the mesh and returns the two inner tensors a and b as the all-gather inputs with
    # no metadata (None).
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_pre_all_gather_v2(
    self,
    mesh: DeviceMesh,
    outer_size: torch.Size,
    outer_stride: tuple[int, ...],
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy,
) -> tuple[tuple[torch.Tensor, ...], Any]:
    # [FUNCTION DOC] HELPER: Version 2 pre-all-gather hook for TwoTensor; accepts the extra outer
    # size, outer stride, module and mixed precision policy arguments but ignores them. Returns
    # the inner tensors a and b as all-gather inputs and None as metadata.
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_post_all_gather(
    self,
    all_gather_outputs: tuple[torch.Tensor, ...],
    metadata: Any,
    param_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    # [FUNCTION DOC] HELPER: Post-all-gather hook for TwoTensor. Expects the two gathered tensors
    # and None metadata (raises AssertionError otherwise). If out is given, it must be a
    # TwoTensor: when the gathered dtype already equals param_dtype it checks that the gathered
    # storages are the ones backing out.a and out.b, else it checks out's dtypes and copies the
    # gathered data into them; returns None. Without out, builds a TwoTensor cast to param_dtype
    # and returns it together with the gathered tensors as the ones to free.
    if metadata is not None:
        raise AssertionError(f"Expected metadata to be None, got {metadata}")
    a, b = all_gather_outputs
    if out is not None:
        if not isinstance(out, TwoTensor):
            raise AssertionError(f"Expected TwoTensor, got {type(out)}")
        if a.dtype == param_dtype:
            if a.untyped_storage().data_ptr() != out.a.untyped_storage().data_ptr():
                raise AssertionError("a storage data_ptr mismatch with out.a")
            if b.untyped_storage().data_ptr() != out.b.untyped_storage().data_ptr():
                raise AssertionError("b storage data_ptr mismatch with out.b")
        else:
            if out.a.dtype != param_dtype:
                raise AssertionError(f"out.a dtype {out.a.dtype} != {param_dtype}")
            if out.b.dtype != param_dtype:
                raise AssertionError(f"out.b dtype {out.b.dtype} != {param_dtype}")
            out.a.copy_(a)
            out.b.copy_(b)
        return
    tensors_to_free = (a, b)
    # If the cast is real, then the all-gather outputs will not alias the
    # returned `TwoTensor`'s `a` and `b`
    two_tensor = TwoTensor(a, b).to(param_dtype)
    return two_tensor, tensors_to_free


class BFloat16AllGatherTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass around a single data tensor used to test the
    # all-gather extension with a custom bfloat16 all-gather. Holds the wrapped tensor in _data
    # and a flag _pad_in_pre_all_gather controlling whether uneven shards are padded before the
    # gather. Every dispatched op unwraps to _data and rewraps tensor results with the same flag
    # (mixed flags raise AssertionError).
    @staticmethod
    def __new__(cls, data: torch.Tensor, pad_in_pre_all_gather: bool = True):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass with the same shape, stride,
        # storage offset, dtype and device as data; the extra flag is only consumed in __init__.
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            data.stride(),
            data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, pad_in_pre_all_gather: bool = True):
        # [FUNCTION DOC] HELPER: Stores the wrapped data tensor in _data and the padding
        # flag in _pad_in_pre_all_gather.
        self._data = data
        self._pad_in_pre_all_gather = pad_in_pre_all_gather

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ) -> tuple[tuple[torch.Tensor, ...], Any]:
        # [FUNCTION DOC] Pre-all-gather extension hook. Requires a 1D mesh (AssertionError
        # otherwise). If the outer size of dim 0 is not divisible by the mesh size and padding is
        # enabled, returns a one-element tuple with a bfloat16 buffer of per-rank padded dim-0
        # size holding the local data in its leading rows (rest uninitialized), else returns the
        # local data cast to bfloat16 (not wrapped in a tuple). Metadata is always None.
        if mesh.ndim != 1:
            raise AssertionError(f"Expected mesh.ndim == 1, got {mesh.ndim}")
        mesh_size = mesh.size()
        requires_padding = outer_size[0] % mesh_size != 0
        if requires_padding and self._pad_in_pre_all_gather:
            sharded_padded_size = list(outer_size)
            sharded_padded_size[0] = math.ceil(outer_size[0] / mesh_size)
            padded_out = torch.empty(
                sharded_padded_size, dtype=torch.bfloat16, device=self.device
            )
            padded_out[: self._data.size(0)].copy_(self._data)
            return (padded_out,), None
        else:
            return self._data.to(torch.bfloat16), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        # [FUNCTION DOC] Post-all-gather extension hook. Requires None metadata and a single
        # bfloat16 gathered tensor (AssertionError otherwise). If out is given, copies the
        # gathered tensor into it while preserving its version counter and returns None.
        # Otherwise upcasts to param_dtype and returns that tensor along with (gathered, upcast)
        # as tensors to free.
        if metadata is not None:
            raise AssertionError(f"Expected metadata to be None, got {metadata}")
        (tensor,) = all_gather_outputs
        if tensor.dtype != torch.bfloat16:
            raise AssertionError(
                f"Expected tensor.dtype == torch.bfloat16, got {tensor.dtype}"
            )
        if out is not None:
            with _unsafe_preserve_version_counter(out):
                out.copy_(tensor)
            return
        upcast_tensor = tensor.to(param_dtype)
        return upcast_tensor, (tensor, upcast_tensor)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] Dispatch handler: unwraps every instance of this class in args and
        # kwargs to its _data, runs the op, and wraps every tensor in the result back into this
        # class using the padding flag seen on the inputs. Raises AssertionError if inputs
        # disagree on the flag.
        pad_in_pre_all_gather = None

        def unwrap(x: cls):
            # [FUNCTION DOC] HELPER: Returns the wrapped _data of x and records its padding flag
            # in the enclosing scope; raises AssertionError if a previously seen flag differs.
            nonlocal pad_in_pre_all_gather
            if pad_in_pre_all_gather is None:
                pad_in_pre_all_gather = x._pad_in_pre_all_gather
            else:
                if pad_in_pre_all_gather != x._pad_in_pre_all_gather:
                    raise AssertionError(
                        f"pad_in_pre_all_gather mismatch: {pad_in_pre_all_gather} vs {x._pad_in_pre_all_gather}"
                    )
            return x._data

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )
        return pytree.tree_map_only(
            torch.Tensor, lambda x: cls(x, pad_in_pre_all_gather), out
        )

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Traceable-subclass flatten: reports the inner tensor attribute
        # names (only _data) and no extra context.
        return ["_data"], None

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors, outer_size: torch.Size, outer_stride: tuple[int, ...]
    ):
        # [FUNCTION DOC] HELPER: Traceable-subclass unflatten: returns the inner _data tensor
        # unwrapped (not a new subclass instance); the size and stride are ignored.
        return inner_tensors["_data"]

    def __repr__(self):
        # [FUNCTION DOC] HELPER: String form: class name wrapping the repr of the data tensor.
        return f"{self.__class__.__name__}({self._data})"


class TestFullyShardAllGatherExtensionsCommon:
    # [CLASS DOC] HELPER: Mixin of shared test utilities for the all-gather extension tests;
    # combined with a multi-process or multi-thread FSDP test base. Provides a world size of 2, a
    # context manager that installs TwoTensor hooks, and a model builder.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Default world size of 2 (overridden by the multi-thread class).
        return 2

    @contextlib.contextmanager
    def _patch_two_tensor_fsdp_all_gather(self, pre_all_gather_version: int):
        # [FUNCTION DOC] HELPER: Context manager that monkey-patches TwoTensor class-level
        # fsdp_pre_all_gather (version 1 or 2 signature, per the argument) and
        # fsdp_post_all_gather, then synchronizes ranks with a barrier. On exit it barriers again
        # and, under a lock so only one thread does it, deletes the patched attributes. Side
        # effect: mutates the global TwoTensor class.
        lock = threading.Lock()
        if pre_all_gather_version == 1:
            TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather_v1
        elif pre_all_gather_version == 2:
            TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather_v2
        TwoTensor.fsdp_post_all_gather = two_tensor_fsdp_post_all_gather
        dist.barrier()
        try:
            yield
        finally:
            dist.barrier()
            with lock:  # only one thread needs to delete
                if hasattr(TwoTensor, "fsdp_pre_all_gather"):
                    delattr(TwoTensor, "fsdp_pre_all_gather")
                if hasattr(TwoTensor, "fsdp_post_all_gather"):
                    delattr(TwoTensor, "fsdp_post_all_gather")

    def _init_two_tensor_mlp(self) -> nn.Module:
        # [FUNCTION DOC] HELPER: Builds a sequential of three bias-free MLPs of width 8
        # whose two linear weights are replaced by parameters wrapping a TwoTensor of the
        # weight and a clone. Bias is disabled because the reference model's bias grad would
        # become a TwoTensor.
        # Disable bias because the reference model will end up with a bias
        # gradient that is a `TwoTensor`, whereas the FSDP model does not
        model = nn.Sequential(*[MLP(8, bias=False) for _ in range(3)])
        for mlp in model:
            mlp.in_proj.weight = nn.Parameter(
                TwoTensor(mlp.in_proj.weight, mlp.in_proj.weight.clone())
            )
            mlp.out_proj.weight = nn.Parameter(
                TwoTensor(mlp.out_proj.weight, mlp.out_proj.weight.clone())
            )
        return model


class TestFullyShardAllGatherExtensionsMultiProcess(
    TestFullyShardAllGatherExtensionsCommon, FSDPTest
):
    # [CLASS DOC] Multi-process (needs 2 GPUs) train-parity test of the all-gather extension
    # against an unsharded reference model.
    @skip_if_lt_x_gpu(2)
    def test_all_gather_extensions_train_parity(self):
        # [FUNCTION DOC] Runs the train parity subtest with both reshard_after_forward values
        # under the version 1 and then the version 2 TwoTensor pre-all-gather patch. Skipped with
        # fewer than 2 GPUs.
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=1):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=2):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )

    def _test_all_gather_extensions_train_parity(self, reshard_after_forward: bool):
        # [FUNCTION DOC] HELPER: Shards a reference copy of the TwoTensor MLP per layer and for
        # the root with fully_shard, then runs 10 iterations of forward/backward on both. The
        # reference gradients are manually all-reduced and averaged. Asserts equal losses and
        # sharded parameter parity before and after each optimizer step (Adam), alternating the
        # zero_grad set_to_none mode.
        torch.manual_seed(42)
        model = self._init_two_tensor_mlp()
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=True)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                if _model is ref_model:
                    for _, param in _model.named_parameters():
                        dist.all_reduce(param.grad)
                        param.grad.detach().div_(self.world_size)
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(self, ref_model, model)


class TestFullyShardAllGatherExtensionsMultiThread(
    TestFullyShardAllGatherExtensionsCommon, FSDPTestMultiThread
):
    # [CLASS DOC] Multi-thread (8 simulated ranks) tests of the all-gather extension: end to end
    # with meta init, monkey-patched hooks, outer size/stride handling and HSDP meshes.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: World size of 8 for the multi-thread tests.
        return 8

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Device used for materializing the model, from
        # the module-level device_type.
        return torch.device(device_type)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extensions_end_to_end(self):
        # [FUNCTION DOC] Runs the end-to-end subtest for both reshard_after_forward values under
        # the version 1 and version 2 TwoTensor pre-all-gather patches. Needs at least one GPU.
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=1):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=2):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )

    def _test_all_gather_extensions_end_to_end(self, reshard_after_forward: bool):
        # [FUNCTION DOC] HELPER: Builds the TwoTensor MLP on the meta device, shards it with a
        # bfloat16 param dtype, materializes it with to_empty, initializes with truncated normal,
        # and runs 3 forward/backward/Adam steps just to check that no error is raised.
        # Check that we can run the meta-device initialization flow
        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extensions_monkey_patch(self):
        # [FUNCTION DOC] Checks the extension works with plain tensors whose pre/post hooks are
        # patched on as instance attributes after materialization (no subclass needed). The hooks
        # gather in bfloat16 and upcast, and the test asserts the pre hook ran in this thread
        # (thread-local flag). Runs 3 training steps after patching each weight's local tensor.
        tls = threading.local()
        tls.ran_pre_all_gather = False

        # Define a pre/post-all-gather pair that quantizes to bf16 for the
        # all-gather and de-quantizes back to the parameter dtype
        def fsdp_pre_all_gather(
            self,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            # [FUNCTION DOC] HELPER: Pre-all-gather hook that records in thread-local state
            # that it ran and returns the tensor cast to bfloat16 as the sole all-gather input,
            # with no metadata.
            nonlocal tls
            tls.ran_pre_all_gather = True
            return (self.to(torch.bfloat16),), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            # [FUNCTION DOC] HELPER: Post-all-gather hook (no grad). Requires None metadata and a
            # bfloat16 gathered tensor (AssertionError otherwise). With out, copies into it
            # preserving the version counter and returns None; otherwise returns the tensor
            # upcast to param_dtype plus (gathered, upcast) as tensors to free.
            (tensor,) = all_gather_outputs
            if metadata is not None:
                raise AssertionError(f"Expected metadata to be None, got {metadata}")
            if tensor.dtype != torch.bfloat16:
                raise AssertionError(
                    f"Expected tensor.dtype == torch.bfloat16, got {tensor.dtype}"
                )
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(tensor)
                return
            upcast_tensor = tensor.to(param_dtype)
            return upcast_tensor, (tensor, upcast_tensor)

        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        # Monkey patch the pre/post-all-gather functions *after* `to_empty()`
        # since the local tensor objects change from materialization
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                # Need to use `_local_tensor` to patch the tensor object
                local_param = param._local_tensor
                # Monkey patch on the `torch.Tensor` as instance methods to
                # show that the extension can work even without a subclass
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(
                    local_param
                )
                local_param.fsdp_post_all_gather = fsdp_post_all_gather.__get__(
                    local_param
                )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()
        if not tls.ran_pre_all_gather:
            raise AssertionError("Expected tls.ran_pre_all_gather to be True")

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_outer_size_stride(self):
        # [FUNCTION DOC] Verifies the outer size and stride passed to the pre-all-gather hook
        # are enough to pad uneven shards: wraps all weights of a dim-3 MLP in
        # BFloat16AllGatherTensor, shards with reshard_after_forward False (workaround for
        # issue 154836) and runs one forward/backward/fused AdamW step. Requires world size of
        # at least 2.
        """
        NOTE: We cannot easily test the incorrect case where the user-defined
        ``fsdp_pre_all_gather`` does not correctly pad the local tensor because
        only some ranks may require padding, in which case only those ranks
        will error out and the all-gather will timeout.
        """
        if self.world_size < 2:
            raise AssertionError(
                f"Assumes world size of at least 2 but got {self.world_size=}"
            )
        model = MLP(dim=3, dim_multiplier=3)
        for module in model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if "weight" in param_name:
                    param = nn.Parameter(BFloat16AllGatherTensor(param))
                    setattr(module, param_name, param)
        # need to fix reshard_after_forward=True
        # https://github.com/pytorch/pytorch/issues/154836
        fully_shard(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 3), device=device_type)
        loss = model(inp).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_hsdp_mesh(self):
        # [FUNCTION DOC] Checks that with an HSDP 2D mesh (replicate size 2) the pre-all-gather
        # hook receives only the 1D shard mesh. Patches identity hooks on every weight's local
        # tensor, runs one forward and asserts the recorded mesh has ndim 1 and size equal to
        # the shard size.
        tls = threading.local()
        replicate_size = 2
        shard_size = self.world_size // replicate_size
        mesh = init_device_mesh(
            device_type.type,
            (replicate_size, shard_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )

        def fsdp_pre_all_gather(
            self,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            # [FUNCTION DOC] HELPER: Pre-all-gather hook that records the received mesh in
            # thread-local state and returns the tensor itself as the sole all-gather input,
            # with no metadata.
            nonlocal tls
            tls.mesh = mesh
            return (self,), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            # [FUNCTION DOC] HELPER: Post-all-gather hook (no grad): returns None when out is
            # given, else returns the single gathered tensor and a tuple containing it as
            # tensors to free.
            (tensor,) = all_gather_outputs
            if out is not None:
                return
            return tensor, (tensor,)

        model = self._init_two_tensor_mlp()
        for mlp in model:
            fully_shard(mlp, mesh=mesh)
        fully_shard(model, mesh=mesh)
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                # Need to use `_local_tensor` to patch the tensor object
                local_param = param._local_tensor
                # Monkey patch on the `torch.Tensor` as instance methods to
                # show that the extension can work even without a subclass
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(
                    local_param
                )
                local_param.fsdp_post_all_gather = fsdp_post_all_gather.__get__(
                    local_param
                )

        inp = torch.randn((2, 8), device=device_type)
        model(inp)
        # Check that FSDP passes only the shard mesh to the pre-all-gather
        self.assertEqual(tls.mesh.ndim, 1)
        self.assertEqual(tls.mesh.size(), shard_size)


if __name__ == "__main__":
    run_tests()
