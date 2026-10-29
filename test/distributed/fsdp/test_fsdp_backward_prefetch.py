# [MODULE DOC] Test of FSDP backward prefetch ordering on a 12 layer transformer: by wrapping the
# runtime function that selects the next handle to prefetch, it records the parameter names of
# each prefetched handle and checks the order and count for BACKWARD_PRE, BACKWARD_POST and no
# prefetch. Defines parameter-name templates and expected-count constants. Import-time side
# effects: exits the interpreter with status 0 when distributed is unavailable or under dev-debug
# ASAN; computes device_type.
# Owner(s): ["oncall: distributed"]

import sys
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_handle_fqns_from_root
from torch.distributed.fsdp._flat_param import HandleTrainingState
from torch.distributed.fsdp._runtime_utils import (
    _get_handle_to_prefetch,
    _get_training_state,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous, get_devtype
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

NUM_ITERS = 2
DECODER_PARAM_FQNS = [
    "decoder.layers.{index}.self_attn.in_proj_weight",
    "decoder.layers.{index}.self_attn.in_proj_bias",
    "decoder.layers.{index}.self_attn.out_proj.weight",
    "decoder.layers.{index}.self_attn.out_proj.bias",
    "decoder.layers.{index}.multihead_attn.in_proj_weight",
    "decoder.layers.{index}.multihead_attn.in_proj_bias",
    "decoder.layers.{index}.multihead_attn.out_proj.weight",
    "decoder.layers.{index}.multihead_attn.out_proj.bias",
    "decoder.layers.{index}.linear1.weight",
    "decoder.layers.{index}.linear1.bias",
    "decoder.layers.{index}.linear2.weight",
    "decoder.layers.{index}.linear2.bias",
    "decoder.layers.{index}.norm1.weight",
    "decoder.layers.{index}.norm1.bias",
    "decoder.layers.{index}.norm2.weight",
    "decoder.layers.{index}.norm2.bias",
    "decoder.layers.{index}.norm3.weight",
    "decoder.layers.{index}.norm3.bias",
]
ENCODER_PARAM_FQNS = [
    "encoder.layers.{index}.self_attn.in_proj_weight",
    "encoder.layers.{index}.self_attn.in_proj_bias",
    "encoder.layers.{index}.self_attn.out_proj.weight",
    "encoder.layers.{index}.self_attn.out_proj.bias",
    "encoder.layers.{index}.linear1.weight",
    "encoder.layers.{index}.linear1.bias",
    "encoder.layers.{index}.linear2.weight",
    "encoder.layers.{index}.linear2.bias",
    "encoder.layers.{index}.norm1.weight",
    "encoder.layers.{index}.norm1.bias",
    "encoder.layers.{index}.norm2.weight",
    "encoder.layers.{index}.norm2.bias",
]
TOTAL_NUM_PREFETCH_FOR_PRE = 12
TOTAL_NUM_PREFETCH_FOR_POST = 11
ENCODER_BEGIN_INDEX_FOR_PRE = 6
ENCODER_BEGIN_INDEX_FOR_POST = 5
ENCODER_PREFETCH_NUM = 5

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestBackwardPrefetch(FSDPTestContinuous):
    # [CLASS DOC] Two-rank FSDP test (FSDPTestContinuous) checking which handles are prefetched
    # during backward for each backward prefetch setting.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning the fixed rank count of 2.
        return 2

    def _dist_train(self, backward_prefetch=BackwardPrefetch.BACKWARD_PRE):
        # [FUNCTION DOC] HELPER: Builds a transformer (d_model 1024, 8 heads) wrapped by FSDP per
        # encoder or decoder layer with the given backward prefetch mode, and an SGD optimizer.
        # Patches the prefetch selection function in the runtime module to record, in the matching
        # training state, the parameter names of each returned handle. For two iterations it trains
        # and then checks: no records for no prefetch; otherwise the expected total (counts plus
        # trailing none results), decoder layers in descending index, then encoder layers in
        # descending index, then none values. Resets the record list each iteration. Asserts the
        # patched function is called with exactly two positional arguments.
        rank = self.rank
        orig_get_handle_to_prefetch = _get_handle_to_prefetch
        torch.manual_seed(0)
        policy = ModuleWrapPolicy(
            {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        )
        model = FSDP(
            nn.Transformer(d_model=1024, nhead=8, device=device_type),
            device_id=device_type.type,
            auto_wrap_policy=policy,
            use_orig_params=True,
            backward_prefetch=backward_prefetch,
        )
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)

        # prepare input
        torch.manual_seed(rank + 1)
        src = torch.randn((10, 1, 1024), device=device_type)
        tgt = torch.randn((20, 1, 1024), device=device_type)

        # monkey patch
        all_handle_fqns: list[list[str]] = []

        def patched_get_handle_to_prefetch(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper around the original prefetch selection function: calls
            # it, then when the current handle's training state matches the configured backward
            # prefetch mode, appends the handle's names (relative to the root) to the enclosing
            # record list. Returns the original result.
            handle = orig_get_handle_to_prefetch(*args, **kwargs)

            self.assertEqual(
                len(args), 2, "expect _get_handle_to_prefetch(state, current_handle)"
            )
            state = args[0]
            current_handle = args[1]
            training_state = _get_training_state(current_handle)
            if (
                training_state == HandleTrainingState.BACKWARD_PRE
                and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
            ) or (
                training_state == HandleTrainingState.BACKWARD_POST
                and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
            ):
                nonlocal all_handle_fqns
                # FQNs prefixed from the root module
                # state._exec_order_data.param_to_fqn
                fqns = _get_handle_fqns_from_root(state, handle)
                all_handle_fqns.append(fqns)
            return handle

        # flat params from prefetch handle should match
        # DECODER_PARAM_FQNS and ENCODER_PARAM_FQNS
        with patch(
            "torch.distributed.fsdp._runtime_utils._get_handle_to_prefetch",
            patched_get_handle_to_prefetch,
        ):
            for _ in range(NUM_ITERS):
                optim.zero_grad()
                loss = model(src, tgt).sum()
                loss.backward()
                optim.step()
                if backward_prefetch is None:
                    self.assertEqual(len(all_handle_fqns), 0)
                    continue
                elif backward_prefetch == BackwardPrefetch.BACKWARD_PRE:
                    # state._exec_order_data.handles_post_forward_order
                    # equals forward order
                    #     encoder 0...5 -> decoder 0...5 -> root
                    # pre-backward hook order
                    #     root -> decoder 5...0 -> encoder 5...0
                    # prefetch order
                    #     decoder 5...0 -> encoder 5...0 -> None
                    # None: when current_handle=encoder 0,
                    #       _get_handle_to_prefetch returns None
                    # +1 is for the above None
                    encoder_begin_index = ENCODER_BEGIN_INDEX_FOR_PRE
                    self.assertEqual(
                        len(all_handle_fqns), TOTAL_NUM_PREFETCH_FOR_PRE + 1
                    )
                elif backward_prefetch == BackwardPrefetch.BACKWARD_POST:
                    # state._exec_order_data.handles_post_forward_order
                    # equals forward order (same as BACKWARD_PRE)
                    #     encoder 0...5 -> decoder 0...5 -> root
                    # post-backward hook (AccumulateGrad) order
                    #     decoder 5, 4...0 -> encoder 5...0 -> root
                    # prefetch order
                    #     decoder 4...0 -> encoder 5...0 -> None -> None
                    # 1st None: when current_handle=encoder 0,
                    #           _get_handle_to_prefetch returns None
                    # 2nd None: when current_handle=root,
                    #           get decoder 5 inside _get_handle_to_prefetch
                    #           but not needed since decoder 5 is computed already
                    # +2 is for the above Nones
                    encoder_begin_index = ENCODER_BEGIN_INDEX_FOR_POST
                    self.assertEqual(
                        len(all_handle_fqns), TOTAL_NUM_PREFETCH_FOR_POST + 2
                    )

                # ith_prefetch: 0, 1st, 2nd, 3rd, 4th ... ith prefetch
                for ith_prefetch, fqns in enumerate(all_handle_fqns):
                    if ith_prefetch >= 0 and ith_prefetch < encoder_begin_index:
                        layer_index = encoder_begin_index - 1 - ith_prefetch
                        self.assertEqual(
                            fqns,
                            [x.format(index=layer_index) for x in DECODER_PARAM_FQNS],
                        )
                    elif (
                        ith_prefetch >= encoder_begin_index
                        and ith_prefetch <= encoder_begin_index + ENCODER_PREFETCH_NUM
                    ):
                        layer_index = (
                            encoder_begin_index + ENCODER_PREFETCH_NUM - ith_prefetch
                        )
                        self.assertEqual(
                            fqns,
                            [x.format(index=layer_index) for x in ENCODER_PARAM_FQNS],
                        )
                    else:
                        self.assertTrue(fqns is None)

                all_handle_fqns = []

    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch(self):
        # [FUNCTION DOC] Runs the training check as subtests for backward prefetch values none,
        # BACKWARD_PRE and BACKWARD_POST, reusing the process group. Requires at least 2 GPUs.
        # subtest reuse process group to shorten test time
        self.run_subtests(
            {
                "backward_prefetch": [
                    None,
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
            },
            self._test_backward_prefetch,
        )

    def _test_backward_prefetch(self, backward_prefetch: BackwardPrefetch):
        # [FUNCTION DOC] HELPER: Subtest body that calls the training check with the given backward
        # prefetch value.
        self._dist_train(backward_prefetch)


if __name__ == "__main__":
    run_tests()
