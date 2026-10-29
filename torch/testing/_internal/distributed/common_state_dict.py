# mypy: allow-untyped-defs

# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Shared test utilities for distributed state dict tests: a mixin that verifies model
# and optimizer state dicts produced by distributed checkpoint APIs against reference (non-
# distributed) ones, plus small embedding modules that exercise state dict hooks and fqn
# modifiers. No import-time side effects beyond imports.

import copy
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
    _PG,
    _STATE,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor


class VerifyStateDictMixin:
    # [CLASS DOC] Mixin for unittest-style test cases (uses assertEqual, assertTrue and similar
    # from the host class). Provides comparison helpers for model state dicts and optimizer state
    # dicts, gathering sharded or DTensor values to full tensors before comparing. Stateless.
    def _compare_tensor(self, orig_tensor, dist_tensor, offload_to_cpu=False):
        # [FUNCTION DOC] HELPER: Asserts that a reference tensor and a possibly distributed tensor
        # are equal. A DTensor or ShardedTensor is first gathered to a full tensor; optionally
        # both sides are moved to CPU. Fails if the result is not a plain tensor or values differ
        # (allclose).
        if isinstance(dist_tensor, (DTensor, ShardedTensor)):
            dist_tensor = _gather_state_dict({"mykey": dist_tensor}).pop("mykey")

        if offload_to_cpu:
            orig_tensor = orig_tensor.cpu()
            dist_tensor = dist_tensor.cpu()
        self.assertTrue(isinstance(dist_tensor, torch.Tensor))
        self.assertTrue(torch.allclose(orig_tensor, dist_tensor))

    def _verify_msd(
        self,
        msd: dict[str, Any],
        dist_msd: dict[str, Any],
        options: StateDictOptions = StateDictOptions(),
        offload_to_cpu=False,
    ) -> None:
        # [FUNCTION DOC] HELPER: Verifies a distributed model state dict against a reference one.
        # Unless frozen parameters are ignored, requires equal lengths and every key present with
        # allclose-equal values; a mismatch is re-raised as AssertionError naming the key. When
        # frozen parameters are ignored, a key missing from the distributed dict must belong to a
        # parameter that does not require grad. Returns None.
        if not options.ignore_frozen_params:
            self.assertEqual(len(msd), len(dist_msd))
        for fqn, param in msd.items():
            dist_param = dist_msd.get(fqn)
            if not options.ignore_frozen_params:
                self.assertIsNotNone(dist_param, f"{fqn=}")
                try:
                    self._compare_tensor(param, dist_param, offload_to_cpu)
                except AssertionError as e:
                    raise AssertionError(
                        f"{fqn} has mismatched value {param} {dist_param}"
                    ) from e
            elif dist_param is None:
                self.assertFalse(param.requires_grad, f"{fqn=}")

    def _verify_osd(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        osd: dict[str, Any],
        dist_osd: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Verifies a distributed optimizer state dict against a reference
        # one. Builds mappings between parameter ids and fully qualified names from the model and
        # optimizer, then checks per-parameter state (counts and tensor values) and param groups.
        # If the distributed dict has more param groups, they are temporarily merged into one for
        # comparison, comparing params by sorted names and other keys by equality; the original
        # groups are restored at the end. Mutates dist_osd temporarily. Returns None.
        # KNOWN ISSUE: in the state loop the distributed state is looked up in the reference
        # states mapping rather than in the distributed states mapping, so the reference state is
        # compared with itself and values are never really checked.
        params = list(chain.from_iterable(g["params"] for g in optim.param_groups))
        param_pid_mapping = dict(zip(params, range(len(params)), strict=True))
        fqn_pid_mapping = {}
        for fqn, param in model.named_parameters():
            pid = param_pid_mapping[param]
            fqn_pid_mapping[fqn] = pid
            fqn_pid_mapping[pid] = fqn
        # Check optimizer_state_dict state

        self.assertEqual(len(osd[_STATE]), len(dist_osd[_STATE]))
        for pid, states in osd[_STATE].items():
            fqn = fqn_pid_mapping[pid]
            dist_states = dist_osd[_STATE].get(fqn, None)
            self.assertIsNotNone(dist_states, fqn)
            self.assertEqual(len(states), len(dist_states))
            for key, state in states.items():
                dist_state = states.get(key, None)
                self.assertIsNotNone(dist_state)
                self._compare_tensor(state, dist_state)

        # Check optimizer_state_dict param_group
        old_dist_osd_pg = dist_osd[_PG]
        if len(osd[_PG]) != len(dist_osd[_PG]):
            self.assertTrue(len(dist_osd[_PG]) > len(osd[_PG]))
            new_pg = copy.deepcopy(dist_osd[_PG][0])
            new_pg["params"] = []
            for dist_group in dist_osd[_PG]:
                new_pg["params"].extend(dist_group["params"])
            dist_osd[_PG] = [new_pg]

        self.assertEqual(len(osd[_PG]), len(dist_osd[_PG]))
        for group, dist_group in zip(osd[_PG], dist_osd[_PG], strict=True):
            self.assertEqual(len(group), len(dist_group))
            for key, value in group.items():
                # Below doesn't work because param_groups can have None
                # values.
                # dist_value = dist_group.get(key, None)
                # self.assertIsNotNone(dist_value, (dist_group, group))
                dist_value = dist_group[key]
                if key == "params":
                    fqns = [fqn_pid_mapping[pid] for pid in value]
                    self.assertEqual(sorted(fqns), sorted(dist_value))
                else:
                    self.assertEqual(value, dist_value)
        dist_osd[_PG] = old_dist_osd_pg

    def _verify_osd_by_load(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        new_optim: torch.optim.Optimizer,
        dist_osd: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Verifies an optimizer state dict by round trip: gathers the
        # distributed optimizer state dict to full values, loads it into new_optim through
        # set_state_dict (with an empty model state dict), then asserts the reference optimizer
        # state dict equals the new one. Uses looser tolerances on XPU devices. Mutates new_optim
        # and the model's optimizer state. Returns None.
        new_dist_osd = _gather_state_dict(dist_osd)
        set_state_dict(
            model,
            optimizers=new_optim,
            model_state_dict={},
            optim_state_dict=new_dist_osd,
        )
        is_xpu = next(model.parameters()).device.type == "xpu"
        tol = {"rtol": 1e-5, "atol": 1e-5} if is_xpu else {}
        self.assertEqual(optim.state_dict(), new_optim.state_dict(), **tol)


class FusionEmbedding(nn.Module):
    # [CLASS DOC] Test module that holds two embedding tables (base and fusion) as submodules,
    # used as a model for state dict tests. No custom behavior.
    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        # [FUNCTION DOC] Creates the base embedding (vocab_size by embed_dim) and the fusion
        # embedding (fusion_vocab_size by embed_dim).
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)


class FusionEmbeddingWithHook(nn.Module):
    # [CLASS DOC] Same layout as FusionEmbedding, but registers a state dict hook and a load-
    # state-dict pre hook that hide the extra embedding level so saved keys look like a single
    # embedding with a weight entry (the pre-fusion key names).
    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        # [FUNCTION DOC] Creates the two embeddings and registers the state dict hook and the load
        # pre hook (with module access).
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self._register_state_dict_hook(FusionEmbeddingWithHook._state_dict_hook)
        self._register_load_state_dict_pre_hook(
            FusionEmbeddingWithHook._load_state_dict_hook, with_module=True
        )

    def _state_dict_hook(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: State dict hook. Renames the prefixed key for the embedding
        # weight to the prefixed plain weight key in the destination mapping, in place. Raises
        # KeyError if the original key is absent. Returns None.
        """Remove "embedding" from the original embedding in the state_dict
        name. This keeps the original state dict name for the embedding
        from before fusing with the FusionEmbedding.
        """
        key = prefix + "embedding.weight"
        new_key = prefix + "weight"
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Load pre hook. If the state dict is non-empty, renames the
        # prefixed plain weight key to the prefixed embedding weight key, in place (the inverse of
        # the state dict hook). Raises KeyError if the key is missing. Returns None.
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        if state_dict:
            key = prefix + "weight"
            new_key = prefix + "embedding.weight"
            state_dict[new_key] = state_dict[key]
            del state_dict[key]


class FusionEmbeddingWithModifier(FusionEmbeddingWithHook):
    # [CLASS DOC] FusionEmbeddingWithHook that also declares the key-to-module mapping used by
    # distributed state dict code, so the renamed weight key can be mapped back to its true
    # module.
    # _fqn_modifiers is a private function as a contract between DSD. When users change the state_dict
    # keys, they need to provide a mapping from the new key to the original key. This is used to ensure
    # consistency between the state_dict keys and fqn.
    def _fqn_modifiers(self) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Returns a mapping from the modified state dict key (weight) to
        # the original submodule name (embedding), as required by the distributed state dict
        # contract.
        return {
            "weight": "embedding",
        }
