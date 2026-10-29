# [PACKAGE DOC]
# Implementation package of MaskedTensor (a tensor paired with a boolean mask
# marking valid elements). Submodules: core (MaskedTensor class and
# is_masked_tensor), binary, unary, reductions and passthrough (operator
# tables and dispatch helpers for each op category). This init re-exports
# is_masked_tensor and MaskedTensor plus the private predicate/apply helpers
# (_is_native_binary, _apply_native_binary, _is_pass_through_fn,
# _apply_pass_through_fn, _is_reduction, _apply_reduction, _is_native_unary,
# _apply_native_unary) used by the core dispatch. Importing it imports all
# those submodules.
# Copyright (c) Meta Platforms, Inc. and affiliates

from .binary import _apply_native_binary, _is_native_binary
from .core import is_masked_tensor, MaskedTensor
from .passthrough import _apply_pass_through_fn, _is_pass_through_fn
from .reductions import _apply_reduction, _is_reduction
from .unary import _apply_native_unary, _is_native_unary
