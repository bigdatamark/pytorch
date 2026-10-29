# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for FSDP state dict handling: full, local (flattened) and sharded state dict
# types. Covers save/load round trips with activation checkpoint wrappers, shared parameters and
# modules, rank0-only plus CPU offload, fp16 and mixed precision, buffers, ignored modules,
# skipped modules, loading into plain local modules, key names, torch.save/load, multi-backend
# process groups, world size one, empty ranks, config type errors, and 4-GPU local state dict
# resharding. Import-time side effects: exits the process if distributed is unavailable or dev-
# asan is on; defines module constants (shapes, state dict type name mapping) and device_type from
# the current accelerator (or cpu); registers parametrized tests for TestFSDPStateDict.

import io
import itertools
import sys
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed._shard.sharded_tensor.metadata import (
    MEM_FORMAT_ENCODING,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec, ShardMetadata
from torch.distributed._state_dict_utils import (
    _all_gather_sharded_tensor,
    _gather_state_dict,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import FSDP_PREFIX
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM
from torch.distributed.fsdp.wrap import enable_wrap, ModuleWrapPolicy, wrap
from torch.distributed.remote_device import _remote_device
from torch.nn import Linear, Module, TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _assert_module_states,
    _broadcast_state_dict,
    _get_state_dict,
    _zero_model,
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    get_full_params,
    SkipModel,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

INNER_SHAPE = [4, 4]
OUTER_SHAPE = [4, 5]
BUFFER_SHAPE = [5, 5]

NON_ROOT_FSDP_PREFIX = "non_fsdp_lin"

_UNFLATTENED_STATE_DICT_IMPLS = ["state_dict", "sharded_state_dict"]
_FLATTENED_STATE_DICT_IMPLS = ["local_state_dict"]
_SUPPORTED_STATE_DICT_IMPLS = (
    _UNFLATTENED_STATE_DICT_IMPLS + _FLATTENED_STATE_DICT_IMPLS
)

STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class Model(Module):
    # [CLASS DOC] HELPER: Small model with an inner linear (4x4) and outer linear (4x5),
    # optionally with FSDP wrapping the inner one, with persistent and non-persistent buffers
    # on both, ignored-module and fp16 mixed precision options. The forward applies the inner
    # layer twice on the same input.
    def __init__(
        self,
        wrap_fsdp,
        register_buffers=False,
        ignore_inner=False,
        mixed_precision=False,
        process_group=None,
    ):
        # [FUNCTION DOC] HELPER: Builds the layers. Inner is wrapped in FSDP when
        # requested (optionally ignoring itself and using fp16 mixed precision for params,
        # reduce and buffers, on the given process group). When register_buffers is set,
        # adds a persistent buffer and a non-persistent buffer to both layers.
        super().__init__()
        self.inner = Linear(*INNER_SHAPE)
        if register_buffers:
            self.inner.buffer = nn.Buffer(torch.randn(BUFFER_SHAPE))
            self.inner.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )
        if wrap_fsdp:
            self.inner = FSDP(
                self.inner,
                ignored_modules=([self.inner] if ignore_inner else []),
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
                if mixed_precision
                else None,
                process_group=process_group,
            )
        self.outer = Linear(*OUTER_SHAPE)
        if register_buffers:
            self.outer.buffer = nn.Buffer(torch.randn(BUFFER_SHAPE))
            self.outer.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the inner layer twice on the input, sums the two
        # results and applies the outer layer.
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestDummyModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Seeded small model with shared pieces: net3 is the same module as
    # net2, and shared_parameter is the same parameter as random_parameter. Used to check
    # shared state dict entries.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Seeds the RNG with 0, then builds two linear+relu blocks,
        # the aliasing net3 and shared_parameter, and an uninitialized parameter of 10
        # elements.
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net3 = self.net2
        self.random_parameter = nn.Parameter(torch.Tensor(10))
        self.shared_parameter = self.random_parameter

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1, then net2, then net3 (the same module as
        # net2) to the input.
        return self.net3(self.net2(self.net1(x)))

    def get_input(self):
        # [FUNCTION DOC] HELPER: Returns a random 8x8 input tensor on the test device.
        return torch.rand(8, 8, device=device_type)


class TestFSDPStateDict(FSDPTest):
    # [CLASS DOC] Two-GPU test case for FSDP state_dict and load_state_dict across the full,
    # local and sharded state dict types. Holds model factory helpers (simple, nested, multi-
    # buffer, non-FSDP root), a state dict context helper, comparison helpers and the tests
    # that use them.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: number of ranks, the smaller of the accelerator count
        # and 2.
        return min(torch.accelerator.device_count(), 2)

    def _broadcast_state_dict(self, state_dict):
        # [FUNCTION DOC] HELPER: Broadcasts the given state dict from rank 0 to all ranks
        # via the shared test utility and returns the result.
        return _broadcast_state_dict(self.rank, state_dict)

    def _state_compare(self, model, model_new, assert_fn, state_generator="parameters"):
        # [FUNCTION DOC] HELPER: Gets the lists of parameters (or buffers) of both models,
        # asserts their lengths match, then applies the given assertion function to the
        # two lists.
        state_base = list(getattr(model, state_generator)())
        state_new = list(getattr(model_new, state_generator)())
        # Regardless of `assert_fn`, the number of parameters should be the same
        self.assertEqual(len(state_base), len(state_new))
        assert_fn(state_base, state_new)

    def _compare_models(
        self, model, model_new, assert_fn, check_fp16=False, check_buffers=True
    ):
        # [FUNCTION DOC] HELPER: Inside summon_full_params of both models, compares
        # parameters (and buffers when either model has buffers and check_buffers is set)
        # with the given assertion function, which must be assertEqual or assertNotEqual
        # (raises AssertionError otherwise). With check_fp16, also asserts all parameters
        # of the new model are float16.
        if assert_fn not in (self.assertEqual, self.assertNotEqual):
            raise AssertionError(
                f"Expected assert_fn in (self.assertEqual, self.assertNotEqual), got {assert_fn}"
            )
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_new):
                self._state_compare(model, model_new, assert_fn)
                if check_buffers:
                    has_buffers = any(
                        len(list(m.buffers())) for m in (model, model_new)
                    )
                    if has_buffers:
                        self._state_compare(
                            model, model_new, assert_fn, state_generator="buffers"
                        )
                if check_fp16:
                    for tensor in model_new.parameters():
                        self.assertEqual(tensor.dtype, torch.float16)

    def _get_simple_nested_model(
        self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs
    ):
        # [FUNCTION DOC] HELPER: Returns a model of two bias-free 10x10 linears. With wrap
        # set, the first linear is wrapped in FSDP and the pair is wrapped in FSDP again,
        # passing along the given FSDP arguments; with checkpoint_wrap, the linears and
        # the sequence get checkpoint wrappers before FSDP. Without wrap, returns the
        # plain sequence.
        if wrap:
            lin1 = nn.Linear(10, 10, bias=False).to(device_type)
            lin2 = nn.Linear(10, 10, bias=False).to(device_type)
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                lin2 = checkpoint_wrapper(lin2)
            seq = nn.Sequential(FSDP(lin1, *fsdp_args, **fsdp_kwargs), lin2)
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            model = nn.Sequential(
                nn.Linear(10, 10, bias=False).to(device_type),
                nn.Linear(10, 10, bias=False).to(device_type),
            )
        return model

    def _get_simple_model(self, *fsdp_args, checkpoint_wrap=False, **fsdp_kwargs):
        # [FUNCTION DOC] HELPER: Returns a single bias-free 10x10 linear on the device
        # wrapped in FSDP with the given arguments, with a checkpoint wrapper applied
        # underneath when requested.
        lin = nn.Linear(10, 10, bias=False).to(device_type)
        if checkpoint_wrap:
            lin = checkpoint_wrapper(lin)
        model = FSDP(lin, *fsdp_args, **fsdp_kwargs)
        return model

    def _get_multibuffer_nested_model(
        self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs
    ):
        # [FUNCTION DOC] HELPER: Like the simple nested model but with a batch norm layer
        # between two linears. The linear FSDP uses the given mixed precision (popped from
        # kwargs); the batch norm FSDP uses float32 for everything when mixed precision
        # was given, otherwise none. Optional checkpoint wrapping and wrap switch behave
        # as in the simple nested model.
        full_p = torch.float32
        lin_mp = fsdp_kwargs.pop("mixed_precision", None)
        bn_mp = (
            MixedPrecision(param_dtype=full_p, reduce_dtype=full_p, buffer_dtype=full_p)
            if lin_mp
            else None
        )
        if wrap:
            lin1 = nn.Linear(10, 10, bias=False).to(device_type)
            bn1 = nn.BatchNorm1d(10).to(device_type)
            lin2 = nn.Linear(10, 10, bias=False).to(device_type)
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                bn1 = checkpoint_wrapper(bn1)
                lin2 = checkpoint_wrapper(lin2)
            seq = nn.Sequential(
                FSDP(lin1, *fsdp_args, mixed_precision=lin_mp, **fsdp_kwargs),
                FSDP(bn1, *fsdp_args, mixed_precision=bn_mp, **fsdp_kwargs),
                lin2,
            )
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            model = nn.Sequential(
                nn.Linear(10, 10, bias=False).to(device_type),
                nn.BatchNorm1d(10).to(device_type),
                nn.Linear(10, 10, bias=False).to(device_type),
            )
        return model

    def _get_non_fsdp_root_module(self, *fsdp_args, wrap=True, **fsdp_kwargs):
        # [FUNCTION DOC] HELPER: Builds a container module whose root is not FSDP: a plain
        # linear followed by two independently built nested models (FSDP wrapped unless
        # wrap is false). Returns it.
        class FSDPContainer(nn.Module):
            # [CLASS DOC] HELPER: Module chaining a plain linear and two sub-models in
            # order.
            def __init__(self, fsdp_1, fsdp_2):
                # [FUNCTION DOC] HELPER: Creates the plain bias-free 10x10 linear
                # and stores the two sub-models.
                super().__init__()
                self.non_fsdp_lin = nn.Linear(10, 10, bias=False).to(device_type)
                self.fsdp_1 = fsdp_1
                self.fsdp_2 = fsdp_2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the plain linear, then the first
                # and second sub-models.
                x = self.non_fsdp_lin(x)
                x = self.fsdp_1(x)
                x = self.fsdp_2(x)
                return x

        return FSDPContainer(
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
        )

    def _get_state_dict_mgr(
        self,
        model: nn.Module,
        state_dict_type: str,
        state_dict_rank0_and_offload: bool,
    ):
        # [FUNCTION DOC] HELPER: Returns the context manager from FSDP.state_dict_type for
        # the model with the state dict type named by the string (full, local or sharded)
        # and a matching config. The flag sets rank0_only and offload_to_cpu for the full
        # config, and offload_to_cpu for the others. Raises ValueError for an unsupported
        # type name.
        _state_dict_type = STATE_DICT_MAPPING[state_dict_type]
        if state_dict_type == "state_dict":
            config = FullStateDictConfig(
                rank0_only=state_dict_rank0_and_offload,
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "local_state_dict":
            config = LocalStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "sharded_state_dict":
            config = ShardedStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        else:
            raise ValueError("Unsupported state_dict_type")
        return FSDP.state_dict_type(model, _state_dict_type, config)

    def _validate_state_dict_contents(
        self, model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=None
    ):
        # [FUNCTION DOC] HELPER: When the rank0-and-offload flag is set: on rank 0 asserts
        # the state dict is non-empty and every entry (except ignored keys) is on CPU; on
        # other ranks asserts the state dict is empty if the model is an FSDP instance.
        # Does nothing otherwise.
        if state_dict_rank0_and_offload:
            if self.rank == 0:
                self.assertNotEqual(fsdp_state_dict, {})
                for key, tensor in fsdp_state_dict.items():
                    if ignore_keys and key in ignore_keys:
                        continue
                    self.assertEqual(
                        tensor.device,
                        torch.device("cpu"),
                        lambda msg: f"{msg}\n{key} is unexpectedly on device {tensor.device}",
                    )
            else:
                # For non-FSDP roots, the non FSDP portion can still have parameters on rank 0,
                # so bypass the check for now.
                if isinstance(model, FSDP):
                    self.assertEqual(
                        fsdp_state_dict,
                        {},
                        lambda msg: f"{msg}\nExpected empty state_dict but got {fsdp_state_dict} on rank {dist.get_rank()}",
                    )

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize(
        "checkpoint_wrap",
        ["source", "dest", "both", "source_after_wrap", "both_after_wrap"],
    )
    @parametrize("rank0_only_and_offload", [False, True])
    def test_fsdp_state_dict_with_activation_checkpoint(
        self, state_dict_type, checkpoint_wrap, rank0_only_and_offload
    ):
        # [FUNCTION DOC] Saves a state dict of a simple or nested FSDP model, then loads
        # it into a freshly built and zeroed model, with checkpoint wrappers applied to
        # the source, destination or both (before or after FSDP wrapping, via activation
        # checkpointing). Checks models differ before loading and match after a strict
        # load, broadcasting from rank 0 when rank0 only is used. Needs 2 GPUs.
        """Tests saving the state dict, zeroing a target model's parameters, and
        loading the state dict, where the source and target models may have a
        checkpoint wrapper."""

        def apply_ac_to_linears(model) -> None:
            # [FUNCTION DOC] HELPER: Applies non-reentrant activation checkpointing to
            # every linear submodule of the model.
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: isinstance(submodule, nn.Linear),
            )

        for model_call in [
            partial(self._get_simple_model),
            partial(self._get_simple_nested_model),
        ]:
            model = model_call(checkpoint_wrap=(checkpoint_wrap in ("source", "both")))
            if checkpoint_wrap in ("source_after_wrap", "both_after_wrap"):
                apply_ac_to_linears(model)
            with self._get_state_dict_mgr(
                model, state_dict_type, rank0_only_and_offload
            ):
                state_dict = _gather_state_dict(_get_state_dict(model, False, False))
                # Possibly wrap new model in activation checkpoint wrapper to test save/
                # load with this wrapper
                model_new = model_call(
                    checkpoint_wrap=(checkpoint_wrap in ("dest", "both"))
                )
                if checkpoint_wrap == "both_after_wrap":
                    apply_ac_to_linears(model_new)
                _zero_model(model_new)
                self._compare_models(model, model_new, self.assertNotEqual)
                if rank0_only_and_offload:
                    state_dict = self._broadcast_state_dict(state_dict)
                # Would fail if checkpoint_wrapper did not correctly implement state_dict pre/post hooks
                model_new.load_state_dict(state_dict, strict=True)
                self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("rank0_only_and_offload", [False, True])
    def test_state_dict_with_manual_ac_wrapper(
        self,
        state_dict_type: str,
        rank0_only_and_offload: bool,
    ):
        # [FUNCTION DOC] Builds a transformer with shared params twice, wrapping layers
        # with FSDP manually, once with a checkpoint wrapper inside each FSDP. Saves both
        # state dicts, asserts the keys are identical, loads each back, and compares the
        # models. Returns early for sharded state dict with rank0 and offload
        # (unsupported). Needs 2 GPUs.
        """
        Tests saving and loading a state dict for a model manually wrapped with
        ``FSDP(CheckpointWrapper(module))``, where the ``CheckpointWrapper`` is
        wrapped before FSDP.

        TODO: Investigate why the test above does not cover everything in this
        test and de-duplicate afterwards.
        """
        if state_dict_type == "sharded_state_dict" and rank0_only_and_offload:
            return  # not supported
        model_ac = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
        )
        # Manually wrap FSDP without AC
        model_no_ac = deepcopy(model_ac)
        for i, layer in enumerate(model_no_ac.transformer.encoder.layers):
            model_no_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_no_ac.transformer.decoder.layers):
            model_no_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_no_ac.transformer = FSDP(model_no_ac.transformer)

        # Manually wrap FSDP with AC as `FSDP(CheckpointWrapper(module))`
        for i, layer in enumerate(model_ac.transformer.encoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_ac.transformer.decoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_ac.transformer = FSDP(model_ac.transformer)

        # Save, load, and compare the two models
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_no_ac = model_no_ac.state_dict()
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_ac = model_ac.state_dict()
        self.assertEqual(state_dict_ac.keys(), state_dict_no_ac.keys())
        if rank0_only_and_offload:
            state_dict_no_ac = self._broadcast_state_dict(state_dict_no_ac)
            state_dict_ac = self._broadcast_state_dict(state_dict_ac)
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            model_no_ac.load_state_dict(state_dict_no_ac)
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            model_ac.load_state_dict(state_dict_ac)
        self._compare_models(model_ac, model_no_ac, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_with_shared_parameters(self, state_dict_type):
        # [FUNCTION DOC] Wraps a shared-parameter transformer with a policy on encoder and
        # decoder layers, saves its state dict and loads it into a second zeroed model
        # (including buffers); only checks that saving and loading raise no error. Needs 2
        # GPUs.
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        model_creator = partial(
            TransformerWithSharedParams.init,
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            {"auto_wrap_policy": auto_wrap_policy},
        )

        fsdp_model = model_creator()
        with self._get_state_dict_mgr(fsdp_model, state_dict_type, False):
            state_dict = fsdp_model.state_dict()

        new_model = model_creator()
        _zero_model(new_model, zero_buffers=True)
        with self._get_state_dict_mgr(new_model, state_dict_type, False):
            new_model.load_state_dict(state_dict)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    def test_state_dict_rank0_offload_save_load_flow(self, use_orig_params: bool):
        # [FUNCTION DOC] Emulates a rank0-only checkpoint flow: saves a full state dict to
        # CPU on rank 0 only from an FSDP model with nonzero tensors, loads it into a
        # plain model on rank 0 only (other ranks differ), wraps it with FSDP with
        # sync_module_states, and checks that all ranks agree and match the original
        # model. Parametrized on use_orig_params. Needs 2 GPUs.
        """Tests saving a model checkpoint only on rank 0 and loading it only
        on rank 0 with ``sync_module_states=True`` to emulate the workflow to
        avoid redundant CPU memory usage."""
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": use_orig_params,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs,
        )
        # Force model parameters and buffers to be nonzero
        with FSDP.summon_full_params(fsdp_model):
            for tensor in itertools.chain(
                fsdp_model.parameters(), fsdp_model.buffers()
            ):
                if torch.count_nonzero(tensor) == 0:
                    with torch.no_grad():
                        tensor.add_(torch.ones_like(tensor))
        with self._get_state_dict_mgr(fsdp_model, "state_dict", True):
            state_dict = deepcopy(_get_state_dict(fsdp_model))
        # Initialize a non-wrapped model on all ranks
        new_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
        )
        _zero_model(new_model, zero_buffers=True)
        # Only load the checkpoint on rank 0
        if self.rank == 0:
            new_model.load_state_dict(state_dict, strict=True)
        _assert_module_states(
            new_model,
            process_group=self.process_group,
            assert_fn=self.assertNotEqual,
        )
        # Broadcast the module states from rank 0 with `sync_module_states=True`
        new_fsdp_model = FSDP(
            new_model,
            device_id=torch.accelerator.current_device_index(),
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
        )
        # Check FSDP models are equal across ranks
        with FSDP.summon_full_params(new_fsdp_model):
            _assert_module_states(
                new_fsdp_model,
                process_group=self.process_group,
                assert_fn=self.assertEqual,
            )
        # Check FSDP models correctly loaded the checkpoint
        with FSDP.summon_full_params(fsdp_model):
            with FSDP.summon_full_params(new_fsdp_model):
                params = list(fsdp_model.parameters())
                params_new = list(new_fsdp_model.parameters())
                self.assertEqual(params, params_new)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("fp16", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("use_orig_params", [True, False])
    def test_basic_save_and_load_state_dict(
        self,
        state_dict_type: str,
        cpu_offload: bool,
        fp16: bool,
        state_dict_rank0_and_offload: bool,
        use_orig_params: bool,
    ):
        # [FUNCTION DOC] Saves a state dict from a root-less, nested or simple FSDP model
        # (with optional fp16, CPU offload and rank0 offload), validates contents and
        # dtype, loads it into a zeroed copy and checks models match. Returns early for
        # combinations that are not supported (rank0 offload with non-full types, or
        # use_orig_params with local state dict). Needs 2 GPUs.
        """
        Tests that we can save a state_dict and load it into a blank model
        with various configs such as fp16 and cpu offload and parameters
        match as expected.
        """
        if (state_dict_rank0_and_offload and state_dict_type != "state_dict") or (
            use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS
        ):
            return  # not supported
        device = torch.device(self.rank)
        for model_call in [
            partial(
                self._get_non_fsdp_root_module,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
            partial(
                self._get_simple_nested_model,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
            partial(
                self._get_simple_model,
                cpu_offload=cpu_offload,
                use_orig_params=use_orig_params,
            ),
        ]:
            model = model_call()
            if fp16:
                model.half()
            # Run a forward/backward to compute gradients to test the case
            # where there are gradients populated
            inp = torch.randn((3, 10), device=device)
            if fp16:
                inp = inp.half()
            model(inp).sum().backward()

            ctx = self._get_state_dict_mgr(
                model, state_dict_type, state_dict_rank0_and_offload
            )
            with ctx:
                fsdp_state_dict = _get_state_dict(
                    model, cpu_offload.offload_params, fp16
                )

            ignore_keys = [k for k in fsdp_state_dict if NON_ROOT_FSDP_PREFIX in k]

            self._validate_state_dict_contents(
                model,
                fsdp_state_dict,
                state_dict_rank0_and_offload,
                ignore_keys=ignore_keys,
            )
            if fp16:
                # Verify fp16 is the type
                for tensor in fsdp_state_dict.values():
                    self.assertEqual(tensor.dtype, torch.float16)

            model_new = model_call()
            if not cpu_offload.offload_params:
                model_new = model_new.to(device_type)
            if fp16:
                model_new.half()
            # Run a forward/backward to compute gradients to test the case
            # where there are gradients populated
            inp = torch.randn((3, 10), device=device)
            if fp16:
                inp = inp.half()
            model_new(inp).sum().backward()

            # zero the model to ensure parameters are different.
            _zero_model(model_new, zero_buffers=True)
            self._compare_models(model, model_new, self.assertNotEqual)

            # Verify parameters are the same in the new model.
            if state_dict_rank0_and_offload:
                fsdp_state_dict = self._broadcast_state_dict(fsdp_state_dict)
            with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
                model_new.load_state_dict(fsdp_state_dict, strict=True)

            self._compare_models(model, model_new, self.assertEqual, check_fp16=fp16)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("mixed_precision", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("use_orig_params", [True, False])
    def test_buffers_save_and_load_state_dict(
        self,
        state_dict_type: str,
        cpu_offload: bool,
        mixed_precision: bool,
        state_dict_rank0_and_offload: bool,
        use_orig_params: bool,
    ):
        # [FUNCTION DOC] Like the basic save/load test but for a model with linear and
        # batch norm buffers, optionally with fp16 mixed precision: saves the state dict,
        # validates it, loads it into a zeroed copy and checks that parameters and buffers
        # match. Unsupported combinations return early. Needs 2 GPUs.
        """
        Tests that we can save a state_dict and load it for modules with persistent buffers, including
        in the context of non-default mixed precision, different ``state_dict_type`` s and CPU offloading.
        """
        if (state_dict_rank0_and_offload and state_dict_type != "state_dict") or (
            use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS
        ):
            return  # not supported
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None
        )
        model_call = partial(
            self._get_multibuffer_nested_model,
            cpu_offload=cpu_offload,
            use_orig_params=use_orig_params,
            mixed_precision=mixed_precision,
        )
        model = model_call()
        ctx = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with ctx:
            fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, False)

        self._validate_state_dict_contents(
            model, fsdp_state_dict, state_dict_rank0_and_offload
        )

        model_new = model_call()
        if not cpu_offload.offload_params:
            model_new = model_new.to(device_type)

        # zero the model to ensure parameters are different.
        _zero_model(model_new, zero_buffers=True)
        self._compare_models(model, model_new, self.assertNotEqual)

        # Verify parameters are the same in the new model.
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(fsdp_state_dict)
        with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
            model_new.load_state_dict(fsdp_state_dict, strict=True)

        self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("mixed_precision", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    def test_save_and_load_after_forward_state_dict(
        self, state_dict_type, mixed_precision, state_dict_rank0_and_offload
    ):
        # [FUNCTION DOC] Trains a nested model for six steps (optionally with fp16 mixed
        # precision), checks the loss dtype and that parameters changed, saves a copy of
        # the state dict (cloning the tensors), checks the saved dtype is float32, zeroes
        # the model, loads the state dict and checks the full parameters equal the trained
        # ones. Returns early for rank0 offload with non-full types. Needs 2 GPUs.
        """
        Test that saving after some training results in params being updated as
        expected.
        """
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        torch.accelerator.set_device_index(self.rank)
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None
        )
        model = self._get_simple_nested_model(mixed_precision=mixed_precision)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_params = get_full_params(model)
        for _ in range(6):
            inp = torch.randn(1, 10, device=torch.accelerator.current_device_index())
            output = model(*inp)
            loss = output.sum()
            expected_dtype = torch.float32 if mixed_precision is None else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)
            loss.backward()
            optim.step()

        trained_params = get_full_params(model)
        # Ensure some training occurred
        self.assertNotEqual(initial_params, trained_params)
        # Save a copy of the state_dict
        fsd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with fsd_mgr:
            state_dict = model.state_dict()
            if state_dict_type == "state_dict":
                state_dict = {k: v.clone() for k, v in state_dict.items()}
            else:
                for sharded_tensor in state_dict.values():
                    shard = sharded_tensor._local_shards[0]
                    shard.tensor = shard.tensor.clone().detach_()
        self._validate_state_dict_contents(
            model, state_dict, state_dict_rank0_and_offload
        )
        _zero_model(model)

        # Ensure checkpointed params have the full param dtype
        for tensor in state_dict.values():
            self.assertEqual(tensor.dtype, torch.float32)

        # Load state_dict into zeroed model
        if state_dict_rank0_and_offload:
            state_dict = self._broadcast_state_dict(state_dict)

        with FSDP.state_dict_type(model, STATE_DICT_MAPPING[state_dict_type]):
            model.load_state_dict(state_dict, strict=True)
        loaded_params = get_full_params(model)
        self.assertEqual(loaded_params, trained_params)

    def _initialize_model(
        self,
        wrap_fsdp: bool,
        wrap_ddp: bool = True,
        register_buffers: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Seeds the RNG and builds the test Model on the device.
        # Wraps in FSDP if wrap_fsdp is set, else in DistributedDataParallel if wrap_ddp
        # is set, else returns the plain model.
        # keep everything deterministic for input data
        torch.manual_seed(0)

        model = Model(wrap_fsdp, register_buffers=register_buffers).to(device_type)
        if wrap_fsdp:
            model = FSDP(model)
        elif wrap_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_type: str):
        # [FUNCTION DOC] HELPER: Static. Returns the model state dict under the given type
        # name; raises ValueError for an unknown name.
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            raise ValueError(f"No state_dict type for {state_dict_type}") from e

        with FSDP.state_dict_type(model, enum_val):
            return model.state_dict()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_type: str, state_dict: dict[str, Any]
    ):
        # [FUNCTION DOC] HELPER: Static. Strict-loads the state dict into the model under
        # the given type name and returns the load result; raises ValueError for an
        # unknown name.
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            raise ValueError(f"No state_dict for {state_dict_type}") from e

        with FSDP.state_dict_type(model, enum_val):
            return model.load_state_dict(state_dict, strict=True)

    def _dist_train(
        self, wrap_fsdp: bool, state_dict_type: str = "", move_to_cpu: bool = False
    ):
        # [FUNCTION DOC] HELPER: Trains the test model with SGD for three steps on random
        # data (FSDP wrapped or DDP wrapped). For FSDP, saves the state dict (optionally
        # moving tensors to CPU), loads it into a zeroed blank FSDP model and returns its
        # full parameters; otherwise returns the parameter list of the DDP model.
        # TODO: Move this test to common_fsdp.
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(
            64, 4, requires_grad=True, device=torch.device(device_type)
        )
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        if wrap_fsdp:
            blank_model = FSDP(Model(True).to(device_type))
            _zero_model(blank_model)
            state_dict = self._state_dict(model, state_dict_type)
            if move_to_cpu:
                for key in list(state_dict.keys()):
                    tensor = state_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        state_dict[key] = tensor.cpu()
                    else:
                        shards = tensor.local_shards()
                        if shards:
                            shards[0].tensor = shards[0].tensor.cpu()

            self._load_state_dict(blank_model, state_dict_type, state_dict)
            return get_full_params(blank_model)
        else:
            return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_save_load_flow(self, state_dict_type):
        # [FUNCTION DOC] Runs the save/load flow subtest for the given state dict type
        # with and without moving the saved tensors to CPU. Needs 2 GPUs.
        self.run_subtests(
            {"move_to_cpu": [True, False]},
            self._test_state_dict_save_load_flow,
            state_dict_type=state_dict_type,
        )

    def _test_state_dict_save_load_flow(self, state_dict_type, move_to_cpu):
        # [FUNCTION DOC] HELPER: Asserts that parameters from FSDP training followed by a
        # state dict save and load equal parameters from the DDP training.
        fsdp_params = self._dist_train(
            wrap_fsdp=True,
            state_dict_type=state_dict_type,
            move_to_cpu=move_to_cpu,
        )
        ddp_params = self._dist_train(wrap_fsdp=False)
        self.assertEqual(ddp_params, fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    def test_fsdp_state_dict_keys(self, state_dict_type):
        # [FUNCTION DOC] Checks key names of the state dict: for local state dict only the
        # flat parameter names (root and inner), for full and sharded the same keys as an
        # unwrapped local model. Needs 2 GPUs.
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == "local_state_dict":
            self.assertEqual({FLAT_PARAM, f"inner.{FLAT_PARAM}"}, state_dict.keys())
        elif state_dict_type in ("state_dict", "sharded_state_dict"):
            # Keys should match local model.
            local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
            local_keys = local_model.state_dict().keys()
            self.assertEqual(state_dict.keys(), local_keys)
        else:
            raise NotImplementedError(f"No test for {state_dict_type}!")

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("fsdp_root", [True, False])
    def test_state_dict_load_into_local_module(
        self,
        state_dict_type,
        state_dict_rank0_and_offload,
        fsdp_root,
    ):
        # [FUNCTION DOC] Trains an FSDP (or non-FSDP-root) model, takes a full or sharded
        # state dict (optionally rank0 and offload), validates it, loads it strictly into
        # a zeroed non-FSDP model and checks the parameters equal the trained FSDP ones.
        # Returns early for rank0 offload with sharded type. Needs 2 GPUs.
        """
        Tests that FSDP's state_dict can be loaded into a local model.
        """
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        if not fsdp_root:
            model = self._get_non_fsdp_root_module()
        else:
            model = self._initialize_model(wrap_fsdp=True, register_buffers=True)
        optim = SGD(model.parameters(), lr=0.1)
        if not fsdp_root:
            in_data = torch.randn(
                1, 10, requires_grad=True, device=torch.device(device_type)
            )
        else:
            in_data = torch.rand(
                64, 4, requires_grad=True, device=torch.device(device_type)
            )
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        with FSDP.summon_full_params(model):
            fsdp_params = deepcopy(list(model.parameters()))

        # get FSDP state_dict. Note that by default we return full_state_dict.
        sd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with sd_mgr:
            fsdp_state_dict = model.state_dict()

        ignore_keys = [k for k in fsdp_state_dict if NON_ROOT_FSDP_PREFIX in k]
        self._validate_state_dict_contents(
            model,
            fsdp_state_dict,
            state_dict_rank0_and_offload,
            ignore_keys=ignore_keys,
        )
        # Create zeroed local model
        if not fsdp_root:
            blank_local_model = self._get_non_fsdp_root_module(wrap=False)
        else:
            blank_local_model = self._initialize_model(
                wrap_fsdp=False, wrap_ddp=False, register_buffers=True
            )

        # Nothing should be FSDP
        for mod in blank_local_model.modules():
            self.assertFalse(isinstance(mod, FSDP))

        for param in blank_local_model.parameters():
            with torch.no_grad():
                param.zero_()

        fsdp_state_dict = _gather_state_dict(fsdp_state_dict)

        # Load fsdp's full state dict into the local and verify params are as
        # expected.
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(fsdp_state_dict)

        blank_local_model.load_state_dict(fsdp_state_dict, strict=True)
        local_params = list(blank_local_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("double_nest", [True])
    def test_state_dict_skip_module(self, state_dict_type, double_nest):
        # [FUNCTION DOC] Wraps a SkipModel while temporarily removing one inner linear so
        # FSDP skips it, then checks the state dict keys (full and sharded, rank 0) equal
        # those of the unwrapped model, loads it into a second FSDP and into a local model
        # and compares parameters. Needs 2 GPUs.
        # KNOWN ISSUE: a TODO notes skipped parameters should be excluded from
        # FSDP.state_dict but this is not asserted.
        torch.accelerator.set_device_index(self.rank)

        def _create_module(wrap_fsdp=True):
            # [FUNCTION DOC] HELPER: Builds a SkipModel (optionally under an FSDP wrap
            # context), removes the skipped linear attribute, wraps the rest,
            # reattaches the linear, and returns the wrapped module along with the
            # names of parameters that belong to the skipped linear.
            LINEAR_SKIP = "linear_skip"
            ctx = enable_wrap(wrapper_cls=FSDP) if wrap_fsdp else nullcontext()
            with ctx:
                module = SkipModel(double_nest=double_nest)
                # Full name of linear_skip param tensors in SkipModel, as would be
                # stored in checkpoint.
                linear_skip_tensor_names = [
                    k for k in dict(module.named_parameters()) if LINEAR_SKIP in k
                ]
                # skip SkipModule
                linear_skip = getattr(module, LINEAR_SKIP)
                delattr(module, LINEAR_SKIP)
                # Wrap FSDP
                fsdp = wrap(module)
                # reattach
                setattr(module, LINEAR_SKIP, linear_skip)
                return fsdp, linear_skip_tensor_names

        fsdp, _ = _create_module()
        # Run a forward pass
        inp = torch.randn((1, 10), device=torch.accelerator.current_device_index())
        loss = fsdp(inp)
        loss.sum().backward()

        with FSDP.state_dict_type(fsdp, STATE_DICT_MAPPING[state_dict_type]):
            state_dict = fsdp.state_dict()
        if self.rank == 0 and state_dict_type != "local_state_dict":
            sd_keys = list(state_dict.keys())
            expected = list(SkipModel(double_nest=False).state_dict().keys())
            self.assertEqual(sorted(sd_keys), sorted(expected))
            # TODO: parameters in linear_skip_tensor_names should not be handled
            # by FSDP.state_dict(). Have a check once this is implemented in
            # FSDP.state_dict().

        # Check that it can be loaded into FSDP.
        new_fsdp, _ = _create_module()
        _zero_model(new_fsdp)
        for p1, p2 in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertNotEqual(p1, p2)
        with FSDP.state_dict_type(new_fsdp, STATE_DICT_MAPPING[state_dict_type]):
            if state_dict_type != "local_state_dict":
                # FlatParameter has not supported deepcopy yet.
                state_dict = deepcopy(state_dict)
            new_fsdp.load_state_dict(state_dict, strict=True)
        for p1, p2 in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertEqual(p1, p2)

        # Test that the checkpoint can be loaded into a local model.
        local, _ = _create_module(wrap_fsdp=False)
        for param in local.parameters():
            with torch.no_grad():
                param.zero_()

        with fsdp.summon_full_params(fsdp):
            for p1, p2 in zip(fsdp.parameters(), local.parameters()):
                self.assertNotEqual(p1, p2)

        if state_dict_type == "local_state_dict":
            return
        state_dict = _gather_state_dict(state_dict)
        with fsdp.summon_full_params(fsdp):
            if self.rank == 0:
                local.load_state_dict(state_dict, strict=True)
                for p1, p2 in zip(fsdp.parameters(), local.parameters()):
                    self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_wrong_state_dict_config(self):
        # [FUNCTION DOC] Checks that setting a full state dict type with a local state
        # dict config raises a RuntimeError mentioning the expected config type. Needs 2
        # GPUs.
        model = FSDP(Model(wrap_fsdp=True).to(device_type))
        with self.assertRaisesRegex(RuntimeError, "Expected state_dict_config of type"):
            with model.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, LocalStateDictConfig()
            ):
                pass

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("prefix", [True, False])
    @parametrize("ignore_inner", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_state_dict_with_ignored_modules(
        self, state_dict_type, prefix, ignore_inner, mixed_precision
    ):
        # [FUNCTION DOC] Wraps a model with ignored modules (outer, optionally inner) and
        # optional fp16 mixed precision, saves the state dict with an optional key prefix,
        # and checks that ignored parameters and all buffers are not cloned (same data
        # pointer), buffers stay float32, the state dict loads into an unwrapped model,
        # and a second save keeps the same data pointers. Needs 2 GPUs.
        # Initialize an FSDP-wrapped model with an ignored module that includes
        # both parameters and a buffer
        model = Model(
            wrap_fsdp=True,
            register_buffers=True,
            ignore_inner=ignore_inner,
            mixed_precision=mixed_precision,
        ).to(device_type)
        ignored_modules = [model.outer]
        ignored_tensor_to_tensor_name = {
            model.outer.bias: "outer.bias",
            model.outer.weight: "outer.weight",
        }
        if ignore_inner:
            ignored_tensor_to_tensor_name = {
                **ignored_tensor_to_tensor_name,
                model.inner.bias: "inner.bias",
                model.inner.weight: "inner.weight",
            }
        # Note that when model.inner is not ignored this test also ensures
        # non-ignored buffers are not cloned.
        buffer_to_buffer_name = {
            model.inner.buffer: "inner.buffer",
            model.outer.buffer: "outer.buffer",
        }
        # expect fp16 model.inner.buffer with mixed_precisions
        # expect fp32 sd.inner.buffer after restoring to original precision
        # so skip AssertEqual
        if mixed_precision and not ignore_inner:
            buffer_to_buffer_name.pop(model.inner.buffer)

        fsdp_model = FSDP(
            model,
            ignored_modules=ignored_modules,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None,
        )
        prefix_str = "foo." if prefix else ""
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd1 = _gather_state_dict(fsdp_model.state_dict(prefix=prefix_str))
        with FSDP.summon_full_params(fsdp_model):
            fsdp_params = deepcopy(list(fsdp_model.parameters()))
        # Check that the ignored parameters and all buffers are not cloned
        for tensor, tensor_name in {
            **ignored_tensor_to_tensor_name,
            **buffer_to_buffer_name,
        }.items():
            prefixed_tensor_name = f"{prefix_str}{tensor_name}"
            self.assertTrue(prefixed_tensor_name in sd1)
            self.assertEqual(
                tensor.data_ptr(),
                sd1[prefixed_tensor_name].data_ptr(),
                lambda msg: f"{msg}\n{prefixed_tensor_name}",
            )
        # should not apply mixed_precision to ignored buffers
        for buffer_name in buffer_to_buffer_name.values():
            prefixed_buffer_name = f"{prefix_str}{buffer_name}"
            self.assertTrue(prefixed_buffer_name in sd1)
            self.assertEqual(sd1[prefixed_buffer_name].dtype, torch.float32)
        # Check that the state dict can be loaded into a non-wrapped version of
        # the model
        nonwrapped_model = Model(wrap_fsdp=False, register_buffers=True).to(device_type)
        for param in nonwrapped_model.parameters():
            with torch.no_grad():
                param.zero_()

        to_load = {k[len(prefix_str) :]: v for k, v in sd1.items()}
        nonwrapped_model.load_state_dict(to_load, strict=True)
        local_params = list(nonwrapped_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)
        # Check that if we save a state dict again, the ignored parameters and
        # buffer still have the same data pointer
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd2 = fsdp_model.state_dict(prefix=prefix_str)
        for tensor, tensor_name in {
            **ignored_tensor_to_tensor_name,
            **buffer_to_buffer_name,
        }.items():
            prefixed_tensor_name = f"{prefix_str}{tensor_name}"
            self.assertTrue(prefixed_tensor_name in sd2)
            self.assertEqual(tensor.data_ptr(), sd2[prefixed_tensor_name].data_ptr())
            self.assertEqual(
                sd1[prefixed_tensor_name].data_ptr(),
                sd2[prefixed_tensor_name].data_ptr(),
            )

    @skip_if_lt_x_gpu(2)
    def test_state_dict_type(self):
        # [FUNCTION DOC] Checks that entering and leaving the state_dict_type context
        # restores the default (full) state dict type on every FSDP submodule. Needs 2
        # GPUs.
        module = SkipModel(double_nest=True)
        with enable_wrap(wrapper_cls=FSDP):
            fsdp = wrap(module)
        with FSDP.state_dict_type(fsdp, StateDictType.LOCAL_STATE_DICT):
            pass
        for module in FSDP.fsdp_modules(fsdp):
            self.assertEqual(module._state_dict_type, StateDictType.FULL_STATE_DICT)

    @skip_if_lt_x_gpu(2)
    def test_local_state_dict_with_empty_ranks(self):
        # [FUNCTION DOC] Uses a one-element parameter so that on world size 2 one rank has
        # an empty shard. With local state dict, saves a state dict, changes the
        # parameter, loads it back and checks the original value returns. Needs 2 GPUs.
        class Model(Module):
            # [CLASS DOC] HELPER: Module with a one-element scalar parameter whose
            # forward returns the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a one-element tensor and a
                # parameter holding it.
                super().__init__()
                self.my_tensor = torch.full((1,), 3.1415926)
                self.my_parameter = nn.Parameter(self.my_tensor)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Ignores the input and returns the
                # parameter.
                return self.my_parameter

        model = FSDP(Model().to(device_type))
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            out = model(None)
            out.backward()

            state_dict = deepcopy(model.state_dict())
            with torch.no_grad():
                with FSDP.summon_full_params(model):
                    self.assertEqual(model.my_parameter.item(), 3.1415926)
                    model.my_parameter.copy_(torch.full((1,), 1.75).to(device_type))
                    self.assertEqual(model.my_parameter.item(), 1.75)
            model.load_state_dict(state_dict)
            with FSDP.summon_full_params(model):
                self.assertEqual(model.my_parameter.item(), 3.1415926)

    @skip_if_lt_x_gpu(2)
    def test_torch_save_load(self):
        # [FUNCTION DOC] Saves a local state dict to an in-memory buffer with torch.save
        # and reloads it with safe globals for the sharded tensor classes, then checks
        # every entry equals the original (comparing local shards for sharded tensors).
        # Needs 2 GPUs.
        model = Model(wrap_fsdp=True).to(device_type)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()
            checkpoint = io.BytesIO()
            torch.save(state_dict, checkpoint)
            checkpoint.seek(0)
            with torch.serialization.safe_globals(
                [
                    Shard,
                    ShardMetadata,
                    ShardedTensor,
                    ShardedTensorMetadata,
                    TensorProperties,
                    MEM_FORMAT_ENCODING,
                    _remote_device,
                    getattr,
                    ShardedTensor.ProcessGroupState,
                    ChunkShardingSpec,
                ]
            ):
                state_dict_saved = torch.load(checkpoint)
            for k, v in state_dict_saved.items():
                if isinstance(v, ShardedTensor):
                    self.assertEqual(
                        v._local_shards[0].tensor, state_dict[k]._local_shards[0].tensor
                    )
                else:
                    self.assertEqual(v, state_dict[k])

    @skip_if_lt_x_gpu(2)
    def test_shared_module_and_shared_parameter(self):
        # [FUNCTION DOC] Checks that a full state dict of a model with a shared parameter
        # and a shared module holds equal values for the aliased entries. Needs 2 GPUs.
        model = FSDP(TestDummyModel().to(device_type))
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
            self.assertEqual(
                state_dict["random_parameter"], state_dict["shared_parameter"]
            )
            self.assertEqual(state_dict["net2.0.bias"], state_dict["net3.0.bias"])
            self.assertEqual(state_dict["net2.0.weight"], state_dict["net3.0.weight"])

    @skip_if_lt_x_gpu(2)
    def test_full_state_dict_missing_unexpected_keys_cleaned(self):
        # [FUNCTION DOC] Removes one key and adds an unexpected key to a state dict and
        # loads it non-strictly, checking exactly one missing and one unexpected key are
        # reported and neither contains the FSDP internal prefix. Needs 2 GPUs.
        model = self._get_simple_nested_model()
        sd = model.state_dict()
        # Create a missing key
        sd.pop(next(iter(sd.keys())))
        # Create an unexpected key
        sd["unexpected"] = torch.ones(1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) != 1:
            raise AssertionError(f"Expected len(missing) == 1, got {len(missing)}")
        if len(unexpected) != 1:
            raise AssertionError(
                f"Expected len(unexpected) == 1, got {len(unexpected)}"
            )
        self.assertTrue(FSDP_PREFIX not in missing[0])
        self.assertTrue(FSDP_PREFIX not in unexpected[0])

    @skip_if_lt_x_gpu(2)
    def test_sharded_load_multi_backend_pg(self):
        # [FUNCTION DOC] Creates a process group that spans cpu via gloo and the device
        # backend, builds a transformer in it, saves a sharded state dict, zeroes
        # parameters, optionally moves the state dict to CPU, loads it and checks
        # parameters are restored. Needs 2 GPUs.
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": True,
        }
        for load_cpu in [True, False]:
            with self.subTest(load_cpu=load_cpu):
                backend = torch.distributed.get_default_backend_for_device(device_type)
                pg = dist.new_group(backend=f"cpu:gloo,{device_type}:{backend}")
                fsdp_model = TransformerWithSharedParams.init(
                    pg,
                    FSDPInitMode.RECURSIVE,
                    DEVICEInitMode.DEVICE_BEFORE,
                    fsdp_kwargs,
                )
                FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
                sharded = fsdp_model.state_dict()
                param_copy = [t.clone().detach_() for t in fsdp_model.parameters()]
                with torch.no_grad():
                    for p in fsdp_model.parameters():
                        p.zero_()

                if load_cpu:
                    # Offload to CPU to simulate CPU state_dict load
                    for k, v in sharded.items():
                        sharded[k] = v.cpu()

                fsdp_model.load_state_dict(sharded)
                for p1, p2 in zip(param_copy, fsdp_model.parameters()):
                    self.assertEqual(
                        p1,
                        p2,
                        lambda msg: f"{msg}\nnot equal: {p1.sum()} vs {p2.sum()}",
                    )

    @skip_if_lt_x_gpu(2)
    def test_world_size_one(self):
        # [FUNCTION DOC] Creates a single-rank process group per rank, builds a model on
        # it, and saves and loads a sharded state dict without error, then barriers on the
        # default group. Needs 2 GPUs.
        my_pg = None
        for i in range(self.world_size):
            pg = dist.new_group(ranks=[i])
            if i == self.rank:
                my_pg = pg

        model = TransformerWithSharedParams.init(
            my_pg,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
        )
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            model.load_state_dict(state_dict)

        dist.barrier()


class TestFSDPStateDict4GPUs(FSDPTest):
    # [CLASS DOC] Four-GPU test case for state dict behavior that needs more ranks than the
    # main test case (local state dict resharding).
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: number of ranks, all accelerator devices.
        return torch.accelerator.device_count()

    @skip_if_lt_x_gpu(4)
    def test_local_state_dict_reshard(self):
        # [FUNCTION DOC] Trains one step on 4 ranks, takes a local (sharded flat
        # parameter) state dict, and mimics resharding to 2 ranks by all-gathering each
        # sharded tensor and re-chunking it over a new 2-rank group. Loads this into a
        # model on the new group and checks that the full state dicts of the original and
        # resharded models are equal on ranks 0 and 1. Needs 4 GPUs.
        """
        This test demonstrates the ability to do resharding when using
        local_state_dict. Although we do not recommend users to use
        local_state_dict, there are still some corner cases that
        using local_state_dict is a better solution.
        """
        model = FSDP(Model(wrap_fsdp=True)).to(device_type)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        batch = torch.randn(4, 4, device=torch.accelerator.current_device_index())
        output = model(batch)
        loss = output.sum()
        loss.backward()
        optim.step()
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

        rank = dist.get_rank()
        new_pg = dist.new_group(ranks=[0, 1])
        resharded_state_dict = {}
        # Mimic resharding from 4 GPUs to 2 GPUs
        for key, value in state_dict.items():
            if isinstance(value, ShardedTensor):
                full_flat_param = _all_gather_sharded_tensor(value)
                if rank < 2:
                    full_numel = full_flat_param.size()
                    chunks = full_flat_param.chunk(2)
                    flat_param = chunks[rank]
                    shard_offset = 0 if rank == 0 else chunks[0].numel()
                    local_shards = [
                        Shard.from_tensor_and_offsets(flat_param, [shard_offset], rank)
                    ]
                    sharded_tensor = init_from_local_shards(
                        local_shards, full_numel, process_group=new_pg
                    )
                    resharded_state_dict[key] = sharded_tensor
            else:
                if rank < 2:
                    resharded_state_dict[key] = value

        if rank < 2:
            model2 = FSDP(
                Model(wrap_fsdp=True, process_group=new_pg), process_group=new_pg
            ).to(device_type)
            with FSDP.state_dict_type(model2, StateDictType.LOCAL_STATE_DICT):
                model2.load_state_dict(resharded_state_dict)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict1 = model.state_dict()

        if rank < 2:
            with FSDP.state_dict_type(model2, StateDictType.FULL_STATE_DICT):
                full_state_dict2 = model2.state_dict()
            self.assertEqual(full_state_dict1, full_state_dict2)


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
