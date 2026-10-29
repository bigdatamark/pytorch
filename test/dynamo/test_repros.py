# [MODULE DOC] Regression tests for Dynamo (torch.compile) built from reproductions of
# user-reported and model-zoo bugs. Defines small stand-in models and helpers (detectron2,
# HuggingFace Reformer/T5/BigBird/DeBERTa, timm, maml snippets), the large ReproTests case class
# (parametrized tests expanded at import), the device-generic ReproTestsDevice class (instantiated
# for cuda, hpu and xpu only) and CUDAReproTests (CUDA-only). Import-time side effects: registers
# a CPU-only custom op test_sample::foo on a library named test_sample, creates a global CPU
# tensor, optionally imports msgspec and omegaconf when installed, and runs the test instantiation
# helpers at the end. Running the file as a script invokes the Dynamo test runner.
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]
import collections
import contextlib
import copy
import dataclasses
import functools
import gc
import importlib
import inspect
import itertools
import logging
import os
import random
import sys
import types
import typing
import unittest
import warnings
import weakref
from abc import ABC
from collections import defaultdict, namedtuple
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Literal, TypedDict
from unittest import mock

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
import torch._functorch.config
import torch.distributed as dist
import torch.library
import torch.utils._pytree as pytree
from torch import nn
from torch._dynamo.backends.debugging import ExplainWithBackend
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    expectedFailureDynamic,
    rand_strided,
    same,
    skipIfNotPy312,
    skipIfPy312,
)
from torch._inductor.utils import fresh_cache
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    SM70OrLater,
)
from torch.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    serialTest,
    skipIfHpu,
    skipIfRocm,
    skipIfWindows,
    skipIfXpu,
    TEST_WITH_ROCM,
    xfailIfS390X,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import TorchDispatchMode


_orig_module_call = torch.nn.Module.__call__

# Custom operator that only supports CPU and Meta
lib = torch.library.Library("test_sample", "DEF")  # noqa: SCOPED_LIBRARY
lib.define("foo(Tensor self) -> Tensor")
lib.impl("foo", torch.sin, "CPU")


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


_GLOBAL_CPU_TENSOR = torch.randn(3)

HAS_MSGSPEC = importlib.util.find_spec("msgspec")
if HAS_MSGSPEC:
    import msgspec


HAS_OMEGACONF = importlib.util.find_spec("omegaconf")
if HAS_OMEGACONF:
    from omegaconf import OmegaConf


def exists(val):
    # [FUNCTION DOC] HELPER: Returns True when the argument is not None; used by the
    # maybe wrapper.
    return val is not None


def maybe(fn):
    # [FUNCTION DOC] HELPER: Decorator factory that wraps a function so it passes None through
    # unchanged and otherwise forwards all arguments to the wrapped function. Preserves metadata
    # via functools wraps.
    @wraps(fn)
    def inner(x, *args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper produced by maybe; returns the first argument as-is
        # when it is None, otherwise calls the wrapped function with the first argument and any
        # extra args and kwargs.
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def is_fx_tracing_test() -> bool:
    # [FUNCTION DOC] HELPER: Returns True when torch.nn.Module.__call__ has been replaced by
    # something other than the original saved at import time, which is how FX tracing is detected.
    # Copied from an hpc trainer codebase.
    """
    Copied from the hpc trainer codebase
    """
    return torch.nn.Module.__call__ is not _orig_module_call


def has_detectron2():
    # [FUNCTION DOC] HELPER: Reports whether detectron2 is importable by trying to import a
    # private mask_ops symbol; returns True on success and False on ImportError. Used to skip
    # detectron2-dependent tests.
    try:
        from detectron2.layers.mask_ops import _paste_masks_tensor_shape

        return _paste_masks_tensor_shape is not None
    except ImportError:
        return False


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    # [FUNCTION DOC] HELPER: Copy of detectron2 mask pasting. Takes per-instance masks,
    # per-instance boxes, target image height and width and a skip_empty flag. Samples the masks
    # onto an image-space grid with grid_sample (casting integer masks to float first) and returns
    # the sampled masks plus either a pair of slices for the cropped region (when skip_empty and
    # not scripting) or an empty tuple.
    # from detectron2 mask_ops.py

    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=torch.int32
        )
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=torch.int32
        )
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def global_fn(x):
    # [FUNCTION DOC] HELPER: Applies sine elementwise to the input tensor; a module-level function
    # used to test calls to global functions.
    return torch.sin(x)


def cat(tensors, dim=0):
    # [FUNCTION DOC] HELPER: Copy of detectron2 concatenation wrapper. Raises AssertionError if
    # the input is not a list or tuple; returns the single element unchanged when the sequence has
    # length one, otherwise concatenates along the given dim.
    # from detectron2 wrappers.py
    if not isinstance(tensors, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(tensors)}")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x, device=None):
    # [FUNCTION DOC] HELPER: Copy of detectron2 shape-to-tensor wrapper. When scripting, converts
    # the input with as_tensor; when tracing, requires every element to be a tensor
    # (AssertionError otherwise) and stacks them, moving to the device if it differs; in eager
    # mode converts with as_tensor on the given device.
    # from detectron2 wrappers.py
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        if not all(isinstance(t, torch.Tensor) for t in x):
            raise AssertionError("Shape should be tensor during tracing!")
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


fw_graph = [None]
bw_graph = [None]


def aot_graph_capture_backend(gm, args):
    # [FUNCTION DOC] HELPER: Custom Dynamo backend that runs aot_module_simplified using the
    # min-cut rematerialization partitioner with inference input mutations kept. Its forward and
    # backward compilers store the captured graph modules into the module-level lists fw_graph and
    # bw_graph (side effect) and return them unchanged.
    from functorch.compile import min_cut_rematerialization_partition
    from torch._functorch.aot_autograd import aot_module_simplified

    def fw_compiler(gm, _):
        # [FUNCTION DOC] HELPER: Forward compiler for aot_graph_capture_backend; stores the graph
        # module in the global fw_graph slot and returns it.
        fw_graph[0] = gm
        return gm

    def bw_compiler(gm, _):
        # [FUNCTION DOC] HELPER: Backward compiler for aot_graph_capture_backend; stores the graph
        # module in the global bw_graph slot and returns it.
        bw_graph[0] = gm
        return gm

    return aot_module_simplified(
        gm,
        args,
        fw_compiler,
        bw_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )


class Boxes:
    # [CLASS DOC] HELPER: Minimal copy of detectron2 Boxes: holds an N by 4 float32 tensor of box
    # coordinates in the attribute tensor. Supports length and a device property.
    # from detectron2 poolers.py
    def __init__(self, tensor: torch.Tensor):
        # [FUNCTION DOC] HELPER: Converts the input to a float32 tensor on its own device (or
        # CPU), reshapes an empty input to shape N by 4, raises AssertionError unless the result
        # is 2D with last dim 4, and stores it as the tensor attribute.
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        if not (tensor.dim() == 2 and tensor.size(-1) == 4):
            raise AssertionError(
                f"Expected 2D tensor with last dim 4, got {tensor.size()}"
            )
        self.tensor = tensor

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the number of boxes, i.e. the size of the
        # first dimension.
        return self.tensor.shape[0]

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property returning the device of the underlying box tensor.
        return self.tensor.device


def convert_boxes_to_pooler_format(box_lists):
    # [FUNCTION DOC] HELPER: Copy of the detectron2 pooler-format conversion. Concatenates the box
    # tensors of all lists, builds a batch index per box by repeating each list index by that
    # list's length, and returns the indices as a column concatenated in front of the boxes.
    # from detectron2 structures.py
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists], device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(box_lists), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


class _ReversibleFunction(torch.autograd.Function):
    # [CLASS DOC] HELPER: Simplified reversible-residual autograd Function from HuggingFace
    # modeling_reformer. Forward splits the hidden states in two along the last dim, runs the
    # attention half through all layers, saves tensors and metadata on ctx, and returns the two
    # halves concatenated. Backward returns gradients only for the first input.
    # taken from modeling_reformer.py in huggingface
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):
        # [FUNCTION DOC] HELPER: Splits hidden states into two halves, applies each layer to the
        # attention half while collecting outputs (and appending hidden states to the caller's
        # list when output_hidden_states is True), saves detached tensors and layer metadata on
        # ctx, and returns the concatenation of both halves. Mutates the all_hidden_states list
        # passed in.
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            attn_output = layer(attn_output)
            all_buckets = all_buckets + (attn_output,)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        # [FUNCTION DOC] HELPER: Splits the incoming gradient into two halves, discards the first,
        # and returns the second as the gradient of the hidden states followed by eleven None
        # values to match the forward argument count.
        grad_attn_output, grad_hidden_states = torch.chunk(
            grad_hidden_states, 2, dim=-1
        )

        # free memory
        del grad_attn_output

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return (
            grad_hidden_states,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ReformerEncoder(torch.nn.Module):
    # [CLASS DOC] HELPER: Simplified HuggingFace Reformer encoder module: one linear layer kept in
    # a plain list (not a registered submodule), a layer norm and dropout probability 0.5.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the dropout probability, a LayerNorm over 512 features,
        # and a plain list holding one Linear of 256 by 256.
        super().__init__()
        self.dropout = 0.5
        self.layer_norm = torch.nn.LayerNorm(512, eps=1.0e-12)
        self.layers = [torch.nn.Linear(256, 256)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=[None] * 6,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=64,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # [FUNCTION DOC] HELPER: Duplicates the hidden states along the last dim, runs the
        # reversible autograd function over the layers, applies layer norm and then functional
        # dropout using the module training flag, and returns a ReformerEncoderOutput namedtuple
        # holding hidden states and the collected lists. Note the default for head_mask is a
        # shared mutable list.
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []
        past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


class ListConfig:
    # [CLASS DOC] HELPER: Stand-in for omegaconf ListConfig; stores a content list of ValueNode
    # items and iterates over their values through a custom iterator.
    class ValueNode:
        # [CLASS DOC] HELPER: Wraps a single value; provides the node protocol (dereference,
        # missing check, value) expected by ListIterator.
        def __init__(self, value):
            # [FUNCTION DOC] HELPER: Stores the wrapped value.
            self.value = value

        def _dereference_node(self):
            # [FUNCTION DOC] HELPER: Returns the node itself.
            return self

        def _is_missing(self):
            # [FUNCTION DOC] HELPER: Always reports that the node is not missing.
            return False

        def _value(self):
            # [FUNCTION DOC] HELPER: Returns the wrapped value.
            return self.value

    # Based on an example from omegaconfig.listconfig
    class ListIterator(Iterator[Any]):
        # [CLASS DOC] HELPER: Iterator over the content list of a ListConfig, based on the
        # omegaconf implementation. Yields the raw value of each ValueNode and tracks a
        # running index.
        def __init__(self, lst: Any, resolve: bool) -> None:
            # [FUNCTION DOC] HELPER: Stores the resolve flag and an iterator over the content
            # entry in the instance dictionary of the given list config; starts the index at zero.
            self.resolve = resolve
            self.iterator = iter(lst.__dict__["_content"])
            self.index = 0

        def __next__(self) -> Any:
            # [FUNCTION DOC] HELPER: Advances the underlying iterator (propagating StopIteration),
            # dereferences the node when resolve is set and raises AssertionError if it is
            # missing, increments the index and returns the node value; raises AssertionError if
            # the item is not a ValueNode.
            x = next(self.iterator)
            if self.resolve:
                x = x._dereference_node()
                if x._is_missing():
                    raise AssertionError

            self.index = self.index + 1
            if isinstance(x, ListConfig.ValueNode):
                return x._value()
            raise AssertionError

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Iterates with resolution enabled by delegating to the extended
        # iterator method.
        return self._iter_ex(True)

    def _iter_ex(self, resolve: bool) -> Iterator[Any]:
        # [FUNCTION DOC] HELPER: Builds a ListIterator for this config; converts any exception
        # raised during construction into an AssertionError with no chained cause.
        try:
            return ListConfig.ListIterator(self, resolve)
        except Exception:
            raise AssertionError from None

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Fills the content list with value nodes holding 1, 3 and a
        # one-element tensor 7.0.
        self._content = [
            ListConfig.ValueNode(1),
            ListConfig.ValueNode(3),
            ListConfig.ValueNode(torch.tensor([7.0])),
        ]


def longformer_chunk(hidden_states, window_overlap=256):
    # [FUNCTION DOC] HELPER: Turns hidden states into overlapping chunks, adapted from Longformer.
    # Views the input as non-overlapping chunks of twice the window overlap, then returns an
    # as_strided view whose chunk count is doubled minus one by halving the chunk stride. Returns
    # a view; no copy.
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class PartialT5(torch.nn.Module):
    # [CLASS DOC] HELPER: Highly simplified prefix of the T5Attention module with three 512 by 512
    # linear projections q, k and v.
    # Highly simplified T5Attention prefix
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the q, k and v linear layers.
        super().__init__()
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        query_length=None,
    ):
        # [FUNCTION DOC] HELPER: Projects hidden states into per-head query, key and value states
        # (8 heads of dim 64), optionally concatenating or reusing past key/value states, and
        # returns the attention scores (query times transposed key) and value states. Raises
        # AssertionError if past_key_value does not hold exactly two entries. Truncated version of
        # the real model.
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise AssertionError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        def shape(states):
            # [FUNCTION DOC] HELPER: Nested helper that reshapes a projected tensor to batch,
            # heads, length, head dim form by view and transpose.
            """projection"""
            return states.view(batch_size, -1, 8, 64).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            # [FUNCTION DOC] HELPER: Nested helper that projects hidden states (self-attention) or
            # key/value states (cross-attention) through a layer and reshapes them, then
            # concatenates with a past state along the length dimension for self-attention or
            # replaces them with the past state for cross-attention.
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # (truncated here )
        return scores, value_states


class ChunkReformerFeedForward(torch.nn.Module):
    # [CLASS DOC] HELPER: Simplified HuggingFace Reformer feed-forward block using a layer norm
    # and two linear layers, applied through a chunking helper.
    # simplified from HF modeling_reformer.py
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a LayerNorm and two 256 by 256 linear layers named dense
        # and output.
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(256, eps=1e-12)
        self.dense = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 256)

    def forward(self, attention_output):
        # [FUNCTION DOC] HELPER: Adds one to the attention output and passes it through
        # apply_chunking_to_forward with the chunk function.
        return apply_chunking_to_forward(
            self.forward_chunk,
            attention_output + 1,
        )

    def forward_chunk(self, hidden_states):
        # [FUNCTION DOC] HELPER: Applies layer norm, dense layer and output layer in sequence.
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


def apply_chunking_to_forward(forward_fn, *input_tensors):
    # [FUNCTION DOC] HELPER: Simplified HuggingFace chunking helper. Raises AssertionError when
    # there are no input tensors or when their second dimensions differ, raises ValueError when
    # the number of tensors differs from the number of parameters of the forward function
    # (found through signature inspection), otherwise calls the function on all tensors and
    # returns the result.
    # simplified from HF model_utils.py
    if len(input_tensors) == 0:
        raise AssertionError("Expected at least one input tensor")
    tensor_shape = input_tensors[0].shape[1]
    if not all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors):
        raise AssertionError("All input tensors must have the same shape[1]")
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError

    return forward_fn(*input_tensors)


def _validate_model_kwargs(fn, model_kwargs):
    # [FUNCTION DOC] HELPER: Simplified transformers argument validation. Collects keyword names
    # whose value is not None and which are not parameters of the given function; raises
    # ValueError listing them if any exist.
    # simplified from transformers.generation.utils._validate_model_kwargs
    unused_model_args = []
    model_args = set(inspect.signature(fn).parameters)
    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)
    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )


class FakeMamlInner(torch.nn.Module):
    # [CLASS DOC] HELPER: Fake inner network for the MAML repro; one linear layer from 784 to
    # 5 features.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the linear layer.
        super().__init__()
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        # [FUNCTION DOC] HELPER: Flattens the input to two dimensions keeping the batch size and
        # applies the linear layer; the ignored and bn_training arguments are unused.
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):
    # [CLASS DOC] HELPER: Highly simplified version of the finetuning step of maml.meta.Meta.
    # Holds a FakeMamlInner network, an update step count of 10 and a learning rate of 0.4.
    # Highly simplified version of maml.meta.Meta.finetuning
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the inner network and sets the update step count and
        # learning rate.
        super().__init__()
        self.net = FakeMamlInner()
        self.update_step_test = 10
        self.update_lr = 0.4

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        # [FUNCTION DOC] HELPER: Deep-copies the inner network, computes the support loss and
        # gradients, builds fast weights by a gradient step, then under no_grad measures the
        # number of correct query predictions before and after the update. Returns the accuracies
        # as a tensor divided by the query set size. Note only the first two entries of the
        # corrects list are updated.
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetuning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = [
            p[1] - self.update_lr * p[0] for p in zip(grad, net.parameters())
        ]

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        del net

        accs = torch.tensor(corrects) / querysz

        return accs


def softmax_backward_data(parent, grad_output, output, dim, self):
    # [FUNCTION DOC] HELPER: Calls the private softmax backward data op using grad_output, output,
    # the parent dim and the dtype of the last argument; copy of the DeBERTa helper.
    from torch import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


class XSoftmax(torch.autograd.Function):
    # [CLASS DOC] HELPER: Copy of the DeBERTa XSoftmax autograd Function: masked softmax whose
    # masked positions are zeroed in the output.
    # transformers.models.deberta.modeling_deberta.XSoftmax
    @staticmethod
    def forward(self, input, mask, dim):
        # [FUNCTION DOC] HELPER: Records dim on the context, masks out positions where the mask
        # is false by filling with the dtype minimum, applies softmax along dim, zeroes the
        # masked positions in place, saves the output and inverted mask for backward and returns
        # the output.
        self.dim = dim
        rmask = ~(mask.to(torch.bool))
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output, rmask)
        return output

    @staticmethod
    def backward(self, grad_output):
        # [FUNCTION DOC] HELPER: Retrieves the saved output, computes the input gradient with
        # softmax_backward_data and returns it with None for the mask and dim arguments.
        output, _ = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None


class ModelOutput(collections.OrderedDict):
    # [CLASS DOC] HELPER: Ordered dictionary based on HuggingFace ModelOutput; integer keys index
    # into a tuple of values, and attribute assignment of an existing key also updates the item.
    """based on file_utils.py in HuggingFace"""

    def __getitem__(self, k):
        # [FUNCTION DOC] HELPER: String keys are looked up in a copied plain dictionary; any other
        # key indexes into the tuple of values.
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        # [FUNCTION DOC] HELPER: Sets an attribute; when the name is an existing key and the
        # value is not None, also updates the dictionary item through the base class setitem to
        # avoid recursion.
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Sets the dictionary item and also sets the same-named attribute
        # through the base class to avoid recursion.
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        # [FUNCTION DOC] HELPER: Returns the dictionary values as a tuple in key order.
        return tuple(self[k] for k in self.keys())


def create_rand_mask_from_inputs(
    from_blocked_mask,
    to_blocked_mask,
    rand_attn,
    num_attention_heads,
    num_rand_blocks,
    batch_size,
    from_seq_length,
    from_block_size,
):
    # [FUNCTION DOC] HELPER: Copy from HuggingFace BigBird. Gathers rows of the blocked to-mask by
    # flattened random attention indices per head, reshapes to batch, heads, windows and random
    # block elements, and combines with the inner part of the from-mask via an einsum. Returns the
    # random mask tensor.
    """taken from HF modeling_big_bird.py"""
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = torch.stack(
        [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)]
    )
    rand_mask = rand_mask.view(
        batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size
    )
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
    return rand_mask


class SequentialAppendList(torch.nn.Sequential):
    # [CLASS DOC] HELPER: Sequential container from timm vovnet; its forward takes an input and a
    # list, appends each module output (the first module consumes the input, later ones the
    # previous output) to the list.
    """from timm/models/vovnet.py"""

    def forward(self, x: torch.Tensor, concat_list: list[torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs every module in order, appending outputs to concat_list
        # (mutating it), concatenates the list along dim 1 and returns the result together with
        # the list. The annotated return type is tensor only but a tuple is returned.
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x, concat_list


class BatchNormAct2d(torch.nn.BatchNorm2d):
    # [CLASS DOC] HELPER: BatchNorm2d with a fused activation layer, taken from timm.
    """Taken from timm"""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        act_layer=torch.nn.ReLU,
        inplace=True,
    ):
        # [FUNCTION DOC] HELPER: Initializes the BatchNorm2d base with the given options and
        # builds the activation layer from act_layer with the inplace flag.
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.act = act_layer(inplace=inplace)

    @torch.jit.ignore
    def _forward_python(self, x):
        # [FUNCTION DOC] HELPER: Runs the base BatchNorm2d forward; marked to be ignored
        # by TorchScript.
        return super().forward(x)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies batch norm (the jit path calls a method that this class
        # does not define, so it only works in eager use) followed by the activation and returns
        # the result.
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


def get_parameter_dtype(parameter):
    # [FUNCTION DOC] HELPER: From HuggingFace model_utils. Returns the dtype of the first
    # parameter of the module; if there are no parameters (StopIteration), falls back to the first
    # tensor attribute found through the private named members helper.
    """from huggingface model_utils.py"""
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module):
            # [FUNCTION DOC] HELPER: Nested helper returning a list of name and tensor pairs for
            # module attributes that are tensors.
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class DummyConfig:
    # [CLASS DOC] HELPER: Plain config object with Reformer attention layer types and chunk
    # lengths used by the min chunk length helper.
    attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh"]
    lsh_attn_chunk_length = 64
    local_attn_chunk_length = 64


def _get_min_chunk_len(config):
    # [FUNCTION DOC] HELPER: From hf_Reformer. Returns the lsh or local chunk length when only one
    # attention type is used, the smaller of the two when both are used, and raises
    # NotImplementedError for any other layer types.
    """from hf_Reformer"""
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _stable_argsort(vector, dim):
    # [FUNCTION DOC] HELPER: From hf_Reformer. Makes argsort stable by scaling the vector by the
    # dimension size and adding the position offset modulo that size, then returns the argsort
    # along the dim.
    """from hf_Reformer"""
    # this function scales the vector so that torch.argsort is stable.
    # torch.argsort is not stable on its own
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets):
    # [FUNCTION DOC] HELPER: From hf_Reformer. Under no_grad computes a stable argsort of the
    # buckets along the last dim and its inverse permutation (built by scattering arange indices);
    # returns both index tensors.
    """from hf_Reformer"""
    # no gradients are needed
    with torch.no_grad():
        # hash-based sort
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # create simple indices to scatter to, to have undo sort
        indices = (
            torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # get undo sort
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

    return sorted_bucket_idx, undo_sorted_bucket_idx


class CustomList1(list):
    # [CLASS DOC] HELPER: List subclass that is callable (pipes a value through each element in
    # turn) and overrides clear with a no-op so that Dynamo's restricted list subclass handling
    # does not apply.
    def __call__(self, x):
        # [FUNCTION DOC] HELPER: Applies each processor in the list in order to the value and
        # returns the final value.
        for processor in self:
            x = processor(x)
        return x

    def clear(self):
        # [FUNCTION DOC] HELPER: Intentionally does nothing; overriding clear keeps the restricted
        # list subclass variable path from being used.
        pass  # this prevents RestrictedListSubclassVariable from kicking in


class CustomList2(list):
    # [CLASS DOC] HELPER: Callable list subclass with extra helper methods for length and
    # appending twice.
    def __call__(self, x):
        # [FUNCTION DOC] HELPER: Applies each processor in the list in order to the value and
        # returns the result.
        for processor in self:
            x = processor(x)
        return x

    def length_times_10(self):
        # [FUNCTION DOC] HELPER: Returns ten times the number of elements.
        return len(self) * 10

    def append_twice(self, x):
        # [FUNCTION DOC] HELPER: Appends the given item to the list twice by extending with a
        # two-element list.
        self.extend([x, x])


def _merge_criteria_processor_list(default_list, custom_list):
    # [FUNCTION DOC] HELPER: Simplified transformers generation helper. Returns the default list
    # if the custom list is empty; raises ValueError if any custom entry has the exact same type
    # as a default entry; otherwise extends the default list in place with the custom entries and
    # returns it.
    # simplified transformers/generation/utils.py
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                raise ValueError
    default_list.extend(custom_list)
    return default_list


class FeedForwardLayer(nn.Module):
    # [CLASS DOC] HELPER: Feed-forward block of two linear layers with an activation in between
    # and dropout after each layer.
    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1, the stored activation, dropout1, linear2 and
        # dropout2 from the given sizes.
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies linear1, activation, dropout1, linear2 and dropout2 in
        # order and returns the result.
        return self.dropout2(
            self.linear2(self.dropout1(self.activation(self.linear1(x))))
        )


class TransformerEncoderLayer(nn.Module):
    # [CLASS DOC] HELPER: Post-norm transformer encoder layer with self attention and a
    # FeedForwardLayer block, each followed by residual add and layer norm.
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
    ):
        # [FUNCTION DOC] HELPER: Creates multi-head self attention, two layer norms, a dropout and
        # the feed-forward block from the given sizes and options.
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # [FUNCTION DOC] HELPER: Applies self attention then feed-forward, each added to its input
        # and layer normalized; returns the result.
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        # [FUNCTION DOC] HELPER: Self-attention sub-block; runs multi-head attention with the
        # given masks and without weights, takes the output and applies dropout.
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x):
        # [FUNCTION DOC] HELPER: Feed-forward sub-block; delegates to the feed-forward layer.
        return self.ff_block(x)


class MockModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with helper methods for testing method calls on a module; returns
    # comparisons of shapes.
    def inner_fn(self, left, right):
        # [FUNCTION DOC] HELPER: Returns whether the two sequences are equal after conversion
        # to tuples.
        return tuple(left) == tuple(right)

    def fn(self, tensor):
        # [FUNCTION DOC] HELPER: Returns False for an int input; otherwise adds the tensor to
        # itself (result discarded) and returns whether the tensor shape equals the tuple 1, 2, 3.
        if type(tensor) is int:
            return False

        torch.add(tensor, tensor)
        return self.inner_fn(tensor.shape, (1, 2, 3))


class IncByOne:
    # [CLASS DOC] HELPER: Simple class whose instance attribute x is the constructor argument
    # plus one.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the argument plus one as x.
        self.x = x + 1


class IncByTwo:
    # [CLASS DOC] HELPER: Simple class whose instance attribute x is the constructor argument
    # plus two.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the argument plus two as x.
        self.x = x + 2


class LRUCacheWarningTests(LoggingTestCase):
    # [CLASS DOC] Logging test case checking that Dynamo tracing does not emit an lru_cache
    # wrapped function warning.
    @unittest.skipUnless(torch.accelerator.is_available(), "requires accelerator")
    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_issued_during_tracing(self, records):
        # [FUNCTION DOC] Skipped unless an accelerator is available. Temporarily sets the default
        # device to the current accelerator, compiles a small function with the eager backend that
        # calls get_device_module and some pointwise ops, runs it and checks a tensor is returned,
        # restores the previous default device (None when it was cpu) in a finally block, then
        # fails if any captured debug log record contains the lru_cache warning text.
        prev_default = torch._C._get_default_device()
        try:
            torch.set_default_device(torch.accelerator.current_accelerator())

            @torch.compile(backend="eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function calling the device module getter and
                # returning cos then sin of the input.
                torch.get_device_module()
                x = x.cos().sin()
                return x

            result = f(torch.randn(1024))
            self.assertIsInstance(result, torch.Tensor)
        finally:
            if prev_default == "cpu":
                torch.set_default_device(None)
            else:
                torch.set_default_device(prev_default)

        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                self.fail("lru_cache warning was incorrectly logged")


class ReproTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Large collection of Dynamo regression tests, each reproducing a reported bug or
    # a model pattern. Installs a guard manager testing hook in setUp that clones guard managers
    # and checks the clone behaves identically.
    def setUp(self) -> None:
        # [FUNCTION DOC] Runs base setUp, then enters a context manager (stored in an ExitStack on
        # the instance) that installs the guard manager clone testing hook; the utils import falls
        # back to a non-relative import.
        super().setUp()
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            install_guard_manager_testing_hook(self.guard_manager_clone_hook_fn)
        )

    def tearDown(self) -> None:
        # [FUNCTION DOC] Closes the ExitStack, which removes the testing hook, then runs the
        # base tearDown.
        self.exit_stack.close()
        super().tearDown()

    def test_compiled_module_truthiness(self):
        # [FUNCTION DOC] Checks that a compiled ModuleList (eager backend) has the same truthiness
        # as the original, both for an empty list (false) and a non-empty one (true).
        # Test with empty ModuleList
        original_empty = nn.ModuleList()
        compiled_empty = torch.compile(original_empty, backend="eager")
        self.assertEqual(bool(original_empty), bool(compiled_empty))
        self.assertFalse(bool(compiled_empty))
        # Test with non-empty ModuleList
        original_filled = nn.ModuleList([nn.Linear(10, 5)])
        compiled_filled = torch.compile(original_filled, backend="eager")
        self.assertEqual(bool(original_filled), bool(compiled_filled))
        self.assertTrue(bool(compiled_filled))

    def guard_manager_clone_hook_fn(self, guard_manager_wrapper, f_locals, builder):
        # [FUNCTION DOC] HELPER: Guard manager hook. Clones the root guard manager with a
        # predicate accepting everything, wraps it, asserts the string forms of the original and
        # clone are equal and that the clone check passes on the given locals, and also checks the
        # diff guard root when present.
        root = guard_manager_wrapper.root
        cloned_root = root.clone_manager(lambda x: True)
        cloned_wrapper = torch._dynamo.guards.GuardManagerWrapper(cloned_root)
        self.assertEqual(str(guard_manager_wrapper), str(cloned_wrapper))
        self.assertTrue(cloned_root.check(f_locals))
        if guard_manager_wrapper.diff_guard_root:
            self.assertTrue(guard_manager_wrapper.diff_guard_root.check(f_locals))

    def test_linalg_inv_singular_aot_eager_raises(self):
        # [FUNCTION DOC] Checks that linalg.inv on a singular zero matrix raises the linalg error
        # with the diagonal-is-zero message both in eager and when compiled with the aot_eager
        # backend in fullgraph mode.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the matrix inverse of the input.
            return torch.linalg.inv(x)

        x = torch.zeros(2, 2)
        msg = r"linalg\.inv: The diagonal element 1 is zero"

        with self.assertRaisesRegex(torch._C._LinAlgError, msg):
            fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(torch._C._LinAlgError, msg):
            opt_fn(x)

    def test_linalg_inv_check_errors_preserved_in_aot_graph(self):
        # [FUNCTION DOC] Compiles the matrix inverse with the AotEagerAndRecordGraphs backend,
        # checks the result matches eager and that exactly one forward graph was recorded, then
        # asserts the forward graph contains a with_effects node wrapping the linalg check errors
        # op so error checking survives AOT tracing.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the matrix inverse of the input.
            return torch.linalg.inv(x)

        backend = AotEagerAndRecordGraphs()
        x = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)

        fw_graph = backend.fw_graphs[0]
        self.assertTrue(
            any(
                node.target is torch.ops.higher_order.with_effects
                and node.args[1] is torch.ops.aten._linalg_check_errors.default
                for node in fw_graph.graph.nodes
            ),
            fw_graph.code,
        )

    def test_swap_tensors_subclass_not_blocked_by_metaconverter_refcycle(self):
        # [FUNCTION DOC] Disables gc, then checks that converting a TwoTensor-backed nn.Parameter
        # with a MetaConverter adds weak id references to it and that, after the converter and
        # result are deleted, the count returns to baseline immediately by refcount alone (no
        # reference cycle). Re-enables gc if it was enabled and collects in a finally block.
        """Checks that MetaConverter doesn't create refcycles of itself that blocks swap_tensor on subclass tensors"""
        was_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            from torch._subclasses.meta_utils import MetaConverter
            from torch.utils.weak import WeakIdRef

            w_a = torch.randn(4, 4)
            w_b = w_a.clone()
            # For custom tensor parameters, `nn.Parameter` returns the same custom tensor type
            # and sets `_is_param`, which is what Module._apply uses to decide swap_tensors.
            w = nn.Parameter(TwoTensor(w_a, w_b).requires_grad_())

            def weak_id_ref_count(t: torch.Tensor) -> int:
                # [FUNCTION DOC] HELPER: Returns how many of the weak references to the tensor are
                # WeakIdRef instances.
                return sum(isinstance(r, WeakIdRef) for r in weakref.getweakrefs(t))

            baseline_weak_id_refs = weak_id_ref_count(w)
            converter = MetaConverter()
            _ = converter(w)
            after_convert_weak_id_refs = weak_id_ref_count(w)
            self.assertGreater(after_convert_weak_id_refs, baseline_weak_id_refs)

            # If MetaConverter is collectable by refcount, the weakrefs go away immediately.
            # If it's kept alive only by a refcycle, this weakref will persist while gc is
            # disabled.
            del _
            del converter
            self.assertEqual(weak_id_ref_count(w), baseline_weak_id_refs)
        finally:
            if was_gc_enabled:
                gc.enable()
            gc.collect()

    def test_do_paste_mask(self):
        # [FUNCTION DOC] Compiles the detectron2 mask pasting helper with a frame counter backend
        # and runs it five times with different box scales, image sizes and skip_empty values;
        # asserts the frame count is one of two allowed values (dynamic or static shapes) and the
        # op count is one of three allowed values. Clears Dynamo counters first.
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        opt__do_paste_mask = torch.compile(_do_paste_mask, backend=cnt)
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 1,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 3,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 4,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            False,
        )
        # (dynamic shapes, static shapes)
        self.assertIn(cnt.frame_count, (5, 7))
        self.assertIn(cnt.op_count, (92, 106, 119))

    def test_convert_boxes_to_pooler_format(self):
        # [FUNCTION DOC] Compares the compiled pooler-format conversion against eager on two
        # lists of Boxes, then checks the expected frame count and op count, which differ
        # depending on assume_static_by_default (repeat_interleave is data dependent so extra
        # graph breaks occur).
        boxes1 = [
            Boxes(torch.arange(0, 8).reshape((2, 4))),
            Boxes(torch.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(torch.arange(16, 20).reshape((1, 4))),
            Boxes(torch.arange(20, 24).reshape((1, 4))),
        ]
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        fn = convert_boxes_to_pooler_format
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertTrue(same(opt_fn(boxes1), correct1))
        self.assertTrue(same(opt_fn(boxes2), correct2))

        # repeat_interleave is a dynamic shape operator we do not execute/
        # In the future, we could reduce the frame_count down to 1
        # by guarding on the exact values of `Tensor repeats` arg
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """14""")

    def test_boxes_len(self):
        # [FUNCTION DOC] Compiles a function that uses both the builtin length and the dunder
        # length of a Boxes object and adds the tensor; with optimize_assert checks the
        # result equals the tensor plus 4 and expects a single frame with one or two ops
        # depending on static-by-default.
        def fn(boxes):
            # [FUNCTION DOC] HELPER: Returns the builtin length of the boxes plus the dunder
            # length plus the box tensor.
            return len(boxes) + boxes.__len__() + boxes.tensor

        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """1""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """2""")

    def test_metaclass_descriptor(self):
        # [FUNCTION DOC] Checks that a unary-negation slot wrapper of int stored as a class
        # attribute on a class with a custom metaclass (whose own negation returns 999) is
        # resolved as the int wrapper, not the metaclass method. Compiles with the eager backend,
        # fullgraph and dynamic shapes, and expects the result of adding the negated size, i.e.
        # minus 7 for a length-7 input.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that defines a unary negation on classes returning the
            # constant 999.
            def __neg__(cls):
                # [FUNCTION DOC] HELPER: Returns the constant 999 when a class using this
                # metaclass is negated.
                return 999

        class Base(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the Meta metaclass that stores int's negation slot
            # wrapper under a different attribute name.
            # Alias int's unary __neg__ slot wrapper into this class's dict under a
            # different name. __objclass__ == int, __name__ == '__neg__', but looked
            # up on Base, whose metaclass separately defines __neg__.
            sneaky = int.__neg__

        def fn(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the wrapper descriptor attribute
            # belongs to int, applies it to the size of the first dimension (a symbolic int under
            # dynamic shapes) and adds the negated size to the input.
            if Base.sneaky.__objclass__ is not int:
                raise AssertionError("Base.sneaky.__objclass__ is not int")

            n = x.size(0)  # symint under dynamic shapes -> NOT a compile-time constant
            # CPython: Base.sneaky is the unbound int.__neg__ wrapper_descriptor,
            # so this is int.__neg__(n) == -n.
            r = Base.sneaky(n)
            return x + r

        x = torch.zeros(7)
        opt = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
        self.assertEqual(opt(x)[0].item(), -7)

    def _reformer(self, nopython):
        # [FUNCTION DOC] HELPER: Builds a random input, seeds the RNG, runs a deep copy of a
        # ReformerEncoder eagerly, then reseeds and runs a compiled model with a CompileCounter
        # backend (fullgraph set by the nopython argument), asserts the outputs match and returns
        # the counter.
        input = torch.randn([1, 64, 256])
        model = ReformerEncoder()
        torch.manual_seed(1337)
        correct = copy.deepcopy(model)(input)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.manual_seed(1337)
        opt_model = torch.compile(model, backend=cnt, fullgraph=nopython)
        self.assertTrue(same(opt_model(input), correct))
        return cnt

    # https://github.com/pytorch/pytorch/issues/113010
    def test_out_overload_non_contiguous(self):
        # [FUNCTION DOC] Regression test for issue 113010: compares eager and compiled (aot_eager)
        # results of abs with out set to a transposed view, checking both the return value and
        # that the out tensor is updated identically.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Writes the absolute value of x into the transposed view of y
            # and returns that view.
            return torch.abs(x, out=y.T)

        f_compiled = torch.compile(f, backend="aot_eager")

        x_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        x_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        out_ref = f(x_ref, y_ref)
        out_test = f_compiled(x_test, y_test)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(y_ref, y_test)

    # https://github.com/pytorch/pytorch/issues/164555
    def test_out_overload_channels_last(self):
        # [FUNCTION DOC] Regression test for issue 164555: an add with out into an empty_like of a
        # channels_last bfloat16 tensor must give the same values, strides and channels_last
        # contiguity under the eager and aot_eager backends with fullgraph.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a channels_last bfloat16 tensor of ones, adds it to
            # itself into a preallocated empty_like output and returns that output.
            x = torch.ones(2, 2, 2, 3, dtype=torch.bfloat16).to(
                memory_format=torch.channels_last
            )
            out = torch.empty_like(x)

            torch.add(x, x, out=out)
            return out

        eager_out = fn()
        for backend in ("eager", "aot_eager"):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            compiled_out = compiled_fn()

            self.assertEqual(eager_out, compiled_out)
            self.assertEqual(eager_out.stride(), compiled_out.stride())
            self.assertTrue(
                compiled_out.is_contiguous(memory_format=torch.channels_last)
            )

    # https://github.com/pytorch/pytorch/issues/168381
    def test_index_select_contiguous_with_compile(self):
        # [FUNCTION DOC] Regression test for issue 168381: index_select on a permuted and
        # unsqueezed tensor must stay contiguous and match eager values and strides when compiled
        # with the aot_eager_decomp_partition backend.
        def fn(x):
            # [FUNCTION DOC] HELPER: Permutes and unsqueezes the input, then selects index zero
            # along dim 0 with index_select.
            x = x.permute(1, 2, 0)
            x = x.unsqueeze(2)
            idx = torch.tensor([0], device=x.device)
            x = torch.index_select(x, 0, idx)
            return x

        x = torch.randn(6, 3, 6)
        eager_out = fn(x)
        self.assertTrue(eager_out.is_contiguous())
        compiled_fn = torch.compile(fn, backend="aot_eager_decomp_partition")
        compiled_out = compiled_fn(x)
        self.assertTrue(compiled_out.is_contiguous())
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out.stride(), compiled_out.stride())

    # https://github.com/pytorch/pytorch/issues/183771
    @parametrize("backend", ["aot_eager", "inductor"])
    def test_interpolate_nearest_5d_channels_last_stride(self, backend):
        # [FUNCTION DOC] Regression test for issue 183771, parametrized over backend (aot_eager
        # and inductor): nearest interpolation of a permuted 5D input must produce the same values
        # and strides as eager when compiled with fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nearest-neighbor interpolation of the input to spatial size
            # 5, 6, 7.
            return F.interpolate(x, size=(5, 6, 7), mode="nearest")

        x = torch.arange(2 * 5 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 5, 2, 3, 2)
        x = x.permute(0, 4, 1, 2, 3)
        eager_out = fn(x)
        compiled_out = torch.compile(fn, backend=backend, fullgraph=True)(x)

        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out.stride(), compiled_out.stride())

    # https://github.com/pytorch/pytorch/issues/174963
    def test_roll_as_strided_channels_last(self):
        # [FUNCTION DOC] Regression test for issue 174963: a roll followed by as_strided on a
        # channels_last tensor must match eager under the aot_eager_decomp_partition backend.
        def fn(x):
            # [FUNCTION DOC] HELPER: Rolls the input by minus one along dim 0 then returns an
            # as_strided view of shape 5 by 10.
            x = torch.roll(x, shifts=-1, dims=0)
            return torch.as_strided(x, (5, 10), (4, 1), 0)

        x = torch.arange(100).reshape(2, 5, 2, 5).to(memory_format=torch.channels_last)
        eager_out = fn(x)
        compiled_fn = torch.compile(fn, backend="aot_eager_decomp_partition")
        compiled_out = compiled_fn(x)
        self.assertEqual(eager_out, compiled_out)

    # https://github.com/pytorch/pytorch/issues/166626
    def test_inplace_add_from_meta_tensor_factory(self):
        # [FUNCTION DOC] Regression test for issue 166626: an in-place add of a meta device tensor
        # into a tensor created on the input's device must compile with fullgraph on the eager
        # backend and give the same device and values as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates zeros on the input device, adds in place zeros
            # created on the meta device, and returns the first tensor.
            log_det = torch.zeros(x.size(0), device=x.device)
            log_det += torch.zeros(x.size(0), device="meta")
            return log_det

        x = torch.randn(2, 4)
        eager_out = fn(x)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_out = compiled_fn(x)

        self.assertEqual(eager_out.device, compiled_out.device)
        self.assertEqual(eager_out, compiled_out)

    # https://github.com/pytorch/pytorch/issues/109053
    def test_view_dtype_overload(self):
        # [FUNCTION DOC] Regression test for issue 109053: the view-as-dtype overload (to int32)
        # must match eager under aot_eager for inputs that do and do not require grad.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input reinterpreted as int32 via view.
            return x.view(torch.int32)

        f_compiled = torch.compile(f, backend="aot_eager")

        x1 = torch.ones(4, requires_grad=True)
        out_ref = f(x1)
        out_test = f_compiled(x1)
        self.assertEqual(out_ref, out_test)

        x2 = torch.ones(4, requires_grad=False)
        out_ref = f(x2)
        out_test = f_compiled(x2)
        self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/90552
    def test_intermediate_leaf_requires_grad(self):
        # [FUNCTION DOC] Regression test for issue 90552: a compiled function creating an
        # intermediate leaf requiring grad and returning it with a product must give the same leaf
        # gradient after backward as eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates a leaf tensor of ones requiring grad and returns it
            # with its double.
            leaf = torch.ones(2, requires_grad=True)
            return leaf, leaf * 2

        f_compiled = torch.compile(f, backend="aot_eager")
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        leaf, out = f(x)
        leaf_test, out_test = f_compiled(x)
        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(leaf.grad, leaf_test.grad)

    # https://github.com/pytorch/pytorch/issues/113263
    def test_unpack_hooks_dont_run_during_tracing(self):
        # [FUNCTION DOC] Regression test for issue 113263: under saved_tensors_hooks, compiling
        # with aot_eager must call the pack hook once during the forward and not call the unpack
        # hook until backward, where it runs once. Counts are tracked with nonlocal counters.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product.
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        pack_count = 0
        unpack_count = 0

        def pack_hook(x):
            # [FUNCTION DOC] HELPER: Pack hook that increments the nonlocal pack counter and
            # returns the tensor unchanged.
            nonlocal pack_count
            pack_count += 1
            return x

        # unpack hook shouldn't run during compilation, while we trace the forward
        def unpack_hook(x):
            # [FUNCTION DOC] HELPER: Unpack hook that increments the nonlocal unpack counter and
            # returns the tensor unchanged.
            nonlocal unpack_count
            unpack_count += 1
            return x

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f_compiled(x, y)
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            out_test.sum().backward()
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    # https://github.com/pytorch/pytorch/issues/113263
    def test_unpack_hooks_can_be_disabled(self):
        # [FUNCTION DOC] Regression test for issue 113263: a compiled (aot_eager) product whose
        # forward and backward are run inside a context that disables saved tensor hooks must
        # still work.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product.
            return x * y

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
            out_test = f_compiled(x, y)
            out_test.sum().backward()

    # https://github.com/pytorch/pytorch/issues/113263
    def test_disabling_unpack_hooks_within_compiled_region(self):
        # [FUNCTION DOC] Regression test for issue 113263: a compiled function that disables saved
        # tensor hooks inside a nested call must run forward and backward without error.
        def g(z):
            # [FUNCTION DOC] HELPER: Adds 5 to the input inside a context disabling saved
            # tensor hooks.
            with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
                return z + 5

        def f(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the inputs and passes the product to g.
            z = x * y
            return g(z)

        f_compiled = torch.compile(f, backend="aot_eager")

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)
        out_test = f_compiled(x, y)
        out_test.sum().backward()

    # See https://github.com/pytorch/pytorch/issues/97745
    def test_gan_repro_trying_to_backward_through_the_graph_a_second_time(self):
        # [FUNCTION DOC] Regression test for issue 97745 (GAN pattern): a function that
        # backpropagates through one loss, then through a second loss using a detached
        # intermediate, must produce the same gradients when compiled with aot_eager as in eager
        # without a second-backward error.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Computes a loss from a matmul with ones, runs backward, then
            # computes a second loss from the detached intermediate matmul with b and runs
            # backward; returns None.
            c = torch.ones(2, 2)
            d = torch.ones(2, 2)
            e = torch.matmul(a, c)
            g_loss = torch.abs(e - d).mean()
            g_loss.backward()
            fake_d_pred = torch.matmul(b, e.detach())
            d_loss = fake_d_pred.mean()
            d_loss.backward()

        a_ref = torch.randn(2, 2, requires_grad=True)
        b_ref = torch.randn(2, 2, requires_grad=True)
        out_ref = f(a_ref, b_ref)

        a_test = a_ref.detach().clone().requires_grad_(True)
        b_test = b_ref.detach().clone().requires_grad_(True)
        out_test = torch.compile(f, backend="aot_eager")(a_test, b_test)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a_ref.grad, a_test.grad)
        self.assertEqual(b_ref.grad, b_test.grad)

    # https://github.com/pytorch/pytorch/issues/111603
    def test_tuple_enum_as_key_dict(self):
        # [FUNCTION DOC] Regression test for issue 111603: a compiled (eager backend) module that
        # indexes an input dictionary using an Enum member key must run twice and match the
        # uncompiled module.
        class MyEnum(Enum):
            # [CLASS DOC] HELPER: Enum with a single member A whose value is a string.
            A = "a"

        class SomeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 1 to 1 linear layer whose forward applies the
            # layer to the dictionary entry keyed by the enum member.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the linear layer to the tensor stored under the
                # enum key in the input mapping.
                return self.linear(x[MyEnum.A])

        x = {MyEnum.A: torch.rand(8, 1)}
        model_pytorch = SomeModel()
        model = torch.compile(model_pytorch, backend="eager")
        # Executing twice works
        model(x)
        y = model(x)
        self.assertEqual(y, model_pytorch(x))

    def test_embedding_backward_broadcasting_decomp(self):
        # [FUNCTION DOC] Checks that the embedding dense backward op with padding index and
        # frequency scaling gives the same result under aot_eager as in eager for float16
        # gradients and int64 indices.
        def f(grad_output, indices):
            # [FUNCTION DOC] HELPER: Calls the aten embedding dense backward op with 10 weights,
            # padding index 1 and gradient scaling by frequency enabled.
            num_weights = 10
            padding_idx = 1
            scale_grad_by_freq = True
            return torch.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        f_compiled = torch.compile(f, backend="aot_eager")

        grad_output = torch.ones(2, 4, 3, dtype=torch.float16)
        indices = torch.ones(2, 4, dtype=torch.int64)

        out_ref = f(grad_output, indices)
        out_test = f_compiled(grad_output, indices)

        self.assertEqual(out_ref, out_test)

    def test_reformer_eval(self):
        # [FUNCTION DOC] Runs the Reformer repro with fullgraph under no_grad and expects a single
        # frame and 10 ops.
        with torch.no_grad():
            cnt = self._reformer(nopython=True)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 10)

    def test_reformer_train(self):
        # [FUNCTION DOC] Runs the Reformer repro without fullgraph under enable_grad and expects
        # one frame and 10 ops.
        with torch.enable_grad():
            cnt = self._reformer(nopython=False)
        self.assertExpectedInline(cnt.frame_count, """1""")
        self.assertExpectedInline(cnt.op_count, """10""")

    def test_longformer_chunk(self):
        # [FUNCTION DOC] Compares optimized longformer chunking against eager on two input shapes,
        # each run twice, and checks expected frame count (2) and op counts that depend on
        # static-by-default and automatic dynamic shapes.
        input1 = torch.randn([1, 4096, 1])
        input2 = torch.randn([12, 4096, 64])
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        fn = longformer_chunk
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))

        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """8""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """19""")

    def test_hf_t5_forward(self):
        # [FUNCTION DOC] Compares an optimize_assert compiled PartialT5 against eager on a 1
        # by 2048 by 512 input and expects one frame and 11 ops in both static and dynamic
        # default modes.
        input = torch.randn([1, 2048, 512])
        model = PartialT5()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_module_in_skipfiles(self):
        # [FUNCTION DOC] Compiles a plain Linear module with fullgraph and expects one frame and
        # one op, showing modules in skipped files are still traced.
        model = nn.Linear(10, 10)
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(model, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_function_in_skipfiles(self):
        # [FUNCTION DOC] Compiles torch.sin directly with fullgraph and expects one frame and
        # one op.
        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(torch.sin, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_slicing_dynamic_shape(self):
        # [FUNCTION DOC] Checks that slicing a tensor with an index read from a tensor input
        # compiles to one frame with two ops and gives the right result for a first-element index
        # of one (length 7) and a different input with value 4 (length 4).
        def fn(y):
            # [FUNCTION DOC] HELPER: Reads the first element of y as an index, slices a tensor of
            # eight ones from that index, then adds 3 and multiplies by 5.
            x = torch.ones(8)
            idx = y[0]
            out = x[idx:]
            return (out + 3) * 5

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        out = opt_fn(torch.ones(10, dtype=torch.long))
        # idx should be 1 -> slicing off [1:] of 8 elem tensor
        self.assertEqual(list(out.shape), [7])

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 1)

        self.assertEqual(list(opt_fn(torch.tensor([4])).shape), [4])

    def test_slicing_dynamic_shape_setitem(self):
        # [FUNCTION DOC] Checks that a setitem with a tensor-derived slice start compiled with
        # aot_eager gives the same result as eager.
        def fn(input_lengths: torch.Tensor, new_ones_1):
            # [FUNCTION DOC] HELPER: Reads element 3 of the lengths tensor, assigns zero into the
            # row 3 slice starting at that value of the second tensor, and returns it in a
            # one-element tuple.
            getitem_13 = input_lengths[3]
            new_ones_1[(3, slice(getitem_13, None, None))] = 0
            setitem_13 = new_ones_1
            return (setitem_13,)

        x = torch.randn(10).to(dtype=torch.int64)
        y = torch.randn(10, 204)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_integral_inplace_div_aot_eager_raises(self):
        # [FUNCTION DOC] Checks that in-place integer division by zero raises a RuntimeError about
        # the result not being castable to the output type when compiled with aot_eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Divides x by y in place and returns x.
            x.div_(y)
            return x

        opt_fn = torch.compile(fn, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError, "can't be cast to the desired output type"
        ):
            opt_fn(torch.tensor([1]), torch.tensor([0]))

    @torch._dynamo.config.patch(error_on_recompile=True)
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_dynamic_shape_disable_duck_size(self):
        # [FUNCTION DOC] With recompilation errors enabled and duck shaping disabled, compiles a
        # module that adds an int to a channels_last tensor with dynamic=True and calls it on two
        # different shapes and ints; passes if no recompile error is raised.
        class TestModel(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds an integer to the tensor.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Calls the base module initializer.
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the input plus the integer value.
                return x + val

        main_model = TestModel().to(memory_format=torch.channels_last)
        opt_model = torch.compile(main_model, backend="eager", dynamic=True)

        x1 = torch.rand(2, 5, 10, 10).to(memory_format=torch.channels_last)
        x2 = torch.rand(2, 5, 4, 8).to(memory_format=torch.channels_last)

        main_model(x1, 4)
        opt_model(x1, 4)

        main_model(x2, 20)
        opt_model(x2, 20)

    def test_chunk_reformer_ff(self):
        # [FUNCTION DOC] Compares optimize_assert compiled ChunkReformerFeedForward against eager
        # and expects one frame and at most 10 ops.
        input = torch.randn([1, 4096, 256])
        model = ChunkReformerFeedForward()
        correct = model(input)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(input), correct))

        self.assertEqual(cnt.frame_count, 1)
        self.assertLessEqual(cnt.op_count, 10)

    # see: https://github.com/pytorch/pytorch/issues/80067
    # NB: When you remove the expectedFailure, don't forget to
    # uncomment/adjust the assertEqual below
    @unittest.expectedFailure
    @torch._dynamo.config.patch(
        fake_tensor_propagation=True, capture_scalar_outputs=True
    )
    def test_maml_item_capture(self):
        # [FUNCTION DOC] Expected failure (issue 80067): MAML finetuning with scalar output
        # capture and fake tensor propagation enabled should match eager over repeated runs and
        # have one of several op counts.
        # KNOWN ISSUE: marked as an expected failure; the frame count assertion is commented out.
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # if torch._dynamo.config.assume_static_by_default:
        #     self.assertExpectedInline(cnt.frame_count, """2""")
        # else:
        #     self.assertExpectedInline(cnt.frame_count, """3""")
        # TODO(jansel): figure out why op count depends on imports
        self.assertIn(cnt.op_count, (36, 35, 34, 29, 28, 27))

    # see: https://github.com/pytorch/pytorch/issues/80067
    @torch._dynamo.config.patch(capture_scalar_outputs=False)
    def test_maml_no_item_capture(self):
        # [FUNCTION DOC] With scalar output capture disabled, runs the compiled PartialMaml model
        # ten times against the eager result and expects two frames when static by default,
        # otherwise three.
        a = torch.randn(5, 1, 28, 28)
        b = torch.zeros(5, dtype=torch.int64)
        c = torch.randn(75, 1, 28, 28)
        d = torch.zeros(75, dtype=torch.int64)
        model = PartialMaml()
        correct = model(a, b, c, d)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt)
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """3""")

    def test_hf_model_output(self):
        # [FUNCTION DOC] Checks four ways of reading a ModelOutput inside a compiled function (key
        # lookup, attribute, to_tuple, integer index), each giving one frame and one op and the
        # right value.
        ex = ModelOutput(a=torch.randn(10), b=torch.randn(10), c=torch.randn(10))

        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns the item under key a plus one.
            return x["a"] + 1

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns the attribute a plus one.
            return x.a + 1

        def fn3(x):
            # [FUNCTION DOC] HELPER: Returns the first element of the tuple form plus one.
            return x.to_tuple()[0] + 1

        def fn4(x):
            # [FUNCTION DOC] HELPER: Returns the element at integer index zero plus one.
            return x[0] + 1

        cnt = torch._dynamo.testing.CompileCounter()
        for fn in (fn1, fn2, fn3, fn4):
            cnt.clear()
            opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
            self.assertTrue(same(opt_fn(ex), ex.a + 1))
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(cnt.op_count, 1)

    def test_create_rand_mask_from_inputs(self):
        # [FUNCTION DOC] Compares the compiled BigBird random mask helper against eager on a fixed
        # list of arguments; expects one frame and an op count in the range 8 to 10 when static,
        # else 11.
        args = [
            torch.randn([1, 64, 64]),
            torch.randn([1, 64, 64]),
            torch.zeros([1, 12, 62, 3], dtype=torch.int64),
            12,
            3,
            1,
            4096,
            64,
        ]
        correct = create_rand_mask_from_inputs(*args)
        fn = create_rand_mask_from_inputs

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnt.frame_count, 1)
        # Op count may vary between 8-10 in static mode.
        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(cnt.op_count, range(8, 11))
        else:
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_rng_state(self):
        # [FUNCTION DOC] Checks that Dynamo handles saving and restoring the RNG state in one
        # function (two frames, two ops, identical before and after values), and that dynamo
        # export of the same function raises an Unsupported exception.
        def fn():
            # [FUNCTION DOC] HELPER: Gets the RNG state, draws 1000 random numbers, restores the
            # state, draws again and returns both draws.
            state = torch.get_rng_state()
            before = torch.rand(1000)
            torch.set_rng_state(state)
            after = torch.rand(1000)
            return before, after

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)

        before, after = opt_fn()
        self.assertTrue(same(before, after))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)  # rand, rand
        try:
            _, _ = torch._dynamo.export(fn)()
            # See https://github.com/pytorch/pytorch/pull/87490
            self.fail("unexpected export success")
        except torch._dynamo.exc.Unsupported:
            pass

    def test_threading_local(self):
        # [FUNCTION DOC] Compiles with fullgraph a function that reads a tensor attribute stored
        # on a threading.local object and concatenates it with the input; checks equality with
        # eager and one frame.
        import threading

        foo = threading.local()
        foo.x = torch.rand(1)

        def f(x):
            # [FUNCTION DOC] HELPER: Concatenates the input with the tensor stored on the
            # thread-local object.
            return torch.cat([x, foo.x])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)

        inp = torch.ones(1)
        out = f(inp)
        opt_out = opt_f(inp)
        self.assertEqual(opt_out, out)
        self.assertEqual(cnt.frame_count, 1)

    def test_threading_local_dunder_dict_mutation(self):
        # [FUNCTION DOC] Compiles with fullgraph a function that deletes an attribute and adds
        # another through the dunder dict of a thread-local object; checks the result, that
        # the old attribute is gone, the new attribute holds the value from the eager run, and
        # one frame.
        import threading

        foo = threading.local()

        def f(x):
            # [FUNCTION DOC] HELPER: Removes the old entry and sets the new entry in the
            # thread-local dictionary to the input plus 2, returning the new value times 3.
            del foo.__dict__["old"]
            foo.__dict__["new"] = x + 2
            return foo.new * 3

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)

        inp = torch.ones(1)
        foo.old = torch.zeros(1)
        expected = f(inp)
        expected_new = foo.new
        self.assertFalse(hasattr(foo, "old"))

        foo.old = torch.zeros(1)
        del foo.new
        actual = opt_f(inp)
        self.assertEqual(actual, expected)
        self.assertFalse(hasattr(foo, "old"))
        self.assertEqual(foo.new, expected_new)
        self.assertEqual(cnt.frame_count, 1)

    def test_seq_append_list(self):
        # [FUNCTION DOC] Compiles a SequentialAppendList model that mutates an input list; checks
        # the result and list contents match eager, the returned list is the same object passed
        # in, and there is one frame with 5 ops.
        x = torch.randn(4, 10)
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        # this one is tricky because it mutates the list provided as an input
        l1 = [x]
        l2 = [x]
        correct, _ = model(x, l1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        result, l3 = opt_model(x, l2)
        self.assertTrue(same(result, correct))
        self.assertTrue(same(l1, l2))
        self.assertIs(l2, l3)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 5)

    def test_batch_norm_act(self):
        # [FUNCTION DOC] Runs BatchNormAct2d in eval mode: when specialize_int is off just
        # compares compiled and eager results; otherwise expects optimize_assert to compile it in
        # one frame with two ops.
        a = torch.randn(5, 1, 28, 28)
        model = BatchNormAct2d(1).eval()
        correct = model(a)
        cnt = torch._dynamo.testing.CompileCounter()
        if not torch._dynamo.config.specialize_int:
            # _local_scalar_dense causes graph break w 0-dim tensor
            opt_model = torch.compile(model, backend=cnt)
            self.assertTrue(same(opt_model(a), correct))
            return

        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        self.assertTrue(same(opt_model(a), correct))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_get_parameter_dtype(self):
        # [FUNCTION DOC] Checks that a compiled function calling get_parameter_dtype on a module
        # yields a float32 result with one frame and two ops.
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        def fn(model, x):
            # [FUNCTION DOC] HELPER: Adds a random vector of the model parameter dtype on the
            # input device to x.
            return x + torch.randn(
                10, dtype=get_parameter_dtype(model), device=x.device
            )

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(model, torch.randn(10)).dtype, torch.float32)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_nn_parameter(self):
        # [FUNCTION DOC] Checks that creating an nn.Parameter inside a compiled function keeps its
        # Parameter type, both inside (via a test assertion) and in the returned value.
        def test_fn():
            # [FUNCTION DOC] HELPER: Creates a Parameter, asserts via the test case it is a
            # Parameter and returns it.
            a = torch.nn.Parameter(torch.randn(5, 5))
            # Checks that TensorVariable stores the type information correctly
            self.assertTrue(isinstance(a, torch.nn.Parameter))
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        out = opt_test_fn()
        self.assertTrue(isinstance(out, torch.nn.Parameter))

    def test_Size(self):
        # [FUNCTION DOC] Checks that a torch.Size constructed in a compiled function remains
        # a torch.Size, using both a plain assert and a test case assertion that causes a
        # graph break.
        def test_fn():
            # [FUNCTION DOC] HELPER: Builds a random tensor and a Size, asserts the type with
            # assert and with assertIsInstance, returns the tensor.
            a = torch.randn(4)
            x = torch.Size([1, 2, 3])
            # Checks that SizeVariable return torch.Size object
            assert isinstance(x, torch.Size)  # noqa: S101
            # Causes graph breaks and checks reconstruction of SizeVariable
            # object
            self.assertIsInstance(x, torch.Size)
            return a

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        opt_test_fn()

    # See https://github.com/pytorch/pytorch/issues/100067
    def test_copy_weird_strides(self):
        # [FUNCTION DOC] Regression test for issue 100067: assignment into a slice of a zero
        # buffer followed by view, transpose and in-place add on weirdly strided float16 input
        # must give the same result under aot_eager_decomp_partition as in eager.
        # This test requires inductor's copy() decomp to preserve strides properly.
        def test_fn(a):
            # [FUNCTION DOC] HELPER: Writes the input into a slice of a zeros buffer, views and
            # transposes it, adds one in place and returns the result.
            b = torch.zeros(48, 4, 256, 513)
            b[:, 0, 1:256, 1:256] = a
            c = b.view(4, 12, 1024, 513)
            d = c.transpose(2, 1)
            d.add_(1)
            return d

        sh, st, dt, dev, rg = (
            (48, 255, 255),
            (787968, 513, 1),
            torch.float16,
            "cpu",
            True,
        )
        a = rand_strided(sh, st, dt, dev).requires_grad_(rg)
        compiled_f = torch.compile(test_fn, backend="aot_eager_decomp_partition")
        out1 = test_fn(a)
        out2 = compiled_f(a)
        self.assertEqual(out1, out2)

    def test_indexing_with_list(self):
        # [FUNCTION DOC] Compiles a function that checks, for several index lists containing None
        # and ints, that indexing a numpy view and a tensor give the same shape; just needs to run
        # without error.
        def test_fn():
            # [FUNCTION DOC] HELPER: Defines run_test, runs it for two index cases on an arange
            # tensor and returns a random tensor.
            def run_test(tensor, *idx):
                # [FUNCTION DOC] HELPER: Asserts that indexing the numpy view of a tensor with the
                # given indices yields the same shape as indexing the tensor.
                npt = tensor.numpy()
                assert npt[idx].shape == tensor[idx].shape  # noqa: S101

            x = torch.arange(0, 10)
            cases = [
                [None, None],
                [1, None],
            ]

            for case in cases:
                run_test(x, *case)

            return torch.randn(4)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_test_fn = torch.compile(test_fn, backend=cnt)
        opt_test_fn()

    def test_foreach_decomp_arg_names(self):
        # [FUNCTION DOC] Regression test for issue 138698: foreach pow and in-place foreach lerp
        # must compile with fullgraph and eager when called with keyword arguments such as self
        # and exponent.
        # https://github.com/pytorch/pytorch/issues/138698

        @torch.compile(fullgraph=True, backend="eager")
        def foreach_pow(**kwargs):
            # [FUNCTION DOC] HELPER: Compiled function that forwards keyword arguments to the
            # foreach pow op.
            return torch._foreach_pow(**kwargs)

        foreach_pow(self=[torch.ones(2, 2, device="cpu")], exponent=2.7)

        @torch.compile(fullgraph=True, backend="eager")
        def foreach_lerp_(**kwargs):
            # [FUNCTION DOC] HELPER: Compiled function that forwards keyword arguments to the
            # in-place foreach lerp op.
            return torch._foreach_lerp_(**kwargs)

        foreach_lerp_(
            self=[torch.ones(2, 2, device="cpu")],
            tensors1=[torch.ones(2, 2, device="cpu")],
            weights=[torch.ones(2, 2, device="cpu")],
        )

    def test_reformer_min_chunk_len(self):
        # [FUNCTION DOC] Compares optimize_assert compiled use of _get_min_chunk_len inside a
        # function against 64; expects one frame and 3 or 4 ops depending on dynamic shape config.
        def fn(cfg):
            # [FUNCTION DOC] HELPER: Fills a tensor of 10 elements with the minimum chunk length
            # computed from config and returns the first element.
            t = torch.empty(10)
            t.fill_(_get_min_chunk_len(cfg))
            return t[0]

        cfg = DummyConfig()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertEqual(opt_fn(cfg), 64)
        # With unspec int, maximum computation is preserved
        self.assertExpectedInline(cnt.frame_count, """1""")
        if torch._dynamo.config.automatic_dynamic_shapes:
            if not torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(cnt.op_count, """4""")
            else:
                self.assertExpectedInline(cnt.op_count, """3""")
        else:
            self.assertExpectedInline(cnt.op_count, """3""")

    def test_reformer_sorting(self):
        # [FUNCTION DOC] Compares the compiled stable-bucket sorting helper against eager on an
        # int64 zero input; expects one frame and 14 ops when static by default else 16.
        x = torch.zeros([1, 12, 4096], dtype=torch.int64)
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        self.assertTrue(same(opt_fn(x), correct))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """14""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """16""")

    def test_recursive_map(self):
        # [FUNCTION DOC] Regression test (torchdynamo issue 132): a compiled function recursing
        # through nested dictionaries must run in one frame with 4 ops.
        # https://github.com/pytorch/torchdynamo/issues/132
        def _recursive_map(struct, batch_dim=0):
            # [FUNCTION DOC] HELPER: Recursively walks a dictionary and reassigns each non-None
            # entry to itself, recursing into nested dictionaries.
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = v

        def toy_example(a, b, v):
            # [FUNCTION DOC] HELPER: Computes a normalized tensor from a, multiplies by b and, if
            # v is given, walks it with the recursive helper.
            x = a / (torch.abs(a) + 1)
            if v is not None:
                _recursive_map(v)
            return x * b

        cnt = torch._dynamo.testing.CompileCounter()
        opt_toy_example = torch.compile(toy_example, backend=cnt)
        opt_toy_example(
            torch.randn(10),
            torch.randn(10),
            {"layer0": {"memory_keys": torch.randn(10)}},
        )
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)

    def test_issue114171(self):
        # [FUNCTION DOC] Regression test for issue 114171: a compiled bound method of a module
        # indexing a ModuleList with an integer argument must compile with fullgraph, dynamic
        # shapes and eager backend (previously failed).
        device = torch.device("cpu")

        def fcnn(in_dim, out_dim, hidden_dim, activation=torch.nn.GELU):
            # [FUNCTION DOC] HELPER: Builds a Sequential of a linear layer, an activation instance
            # and another linear layer, on the CPU device.
            layers = [
                torch.nn.Linear(in_dim, hidden_dim, device=device),
                activation(),
                torch.nn.Linear(hidden_dim, out_dim, device=device),
            ]
            return torch.nn.Sequential(*layers)

        class testmodel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleList of four MLPs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the list of four MLP networks.
                super().__init__()
                self.interaction_networks = torch.nn.ModuleList(
                    [fcnn(262, 1174, 400) for _ in range(4)]
                )

            def interact(self, x, cycle):
                # [FUNCTION DOC] HELPER: Applies the network selected by the cycle index to x.
                return self.interaction_networks[cycle](x)

        model = testmodel()
        forward_aot = torch.compile(
            model.interact, fullgraph=True, dynamic=True, backend="eager"
        )

        x = torch.rand([111, 262], device=device)
        forward_aot(x, 2)  # previously failed

    def test_issue175(self):
        # [FUNCTION DOC] Regression test for issue 175: a compiled TransformerEncoderLayer with
        # fullgraph must compile once (one frame) with 12 ops when called twice.
        n_heads = 2
        d_model = 64
        model = TransformerEncoderLayer(d_model, n_heads)
        inp = torch.randn(1, d_model)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)
        opt_model(inp)
        opt_model(inp)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(12, cnt.op_count)

    def test_exec_import(self):
        # [FUNCTION DOC] Checks that code run with exec to import a module inside a compiled
        # function does not leak the import into the globals seen by other functions (the second
        # function still hits a NameError), in eager and with the eager backend.
        def fn1():
            # [FUNCTION DOC] HELPER: Executes an import statement of math through exec.
            exec("import math")

        def fn2():
            # [FUNCTION DOC] HELPER: Returns False if math is usable, True if a NameError
            # is raised.
            try:
                math.sqrt(4)
                return False
            except NameError:
                return True

        def fn3():
            # [FUNCTION DOC] HELPER: Calls fn1 then returns the result of fn2.
            fn1()
            return fn2()

        self.assertTrue(fn3())
        opt_fn3 = torch.compile(fn3, backend="eager")
        self.assertTrue(opt_fn3())

    def test_exec_wildcard_import(self):
        # [FUNCTION DOC] Checks that a wildcard import executed through exec in one function
        # is not carried over into the globals of another, with matching results between eager
        # and compiled.
        # Test that globals are not carried over from frame to frame
        def fn1():
            # [FUNCTION DOC] HELPER: Executes a wildcard import from torch through exec.
            exec("from torch import *")

        def fn2():
            # [FUNCTION DOC] HELPER: Builds a zeros tensor and adds the loop index five times.
            x = torch.zeros(4)
            for i in range(5):
                x = x + i
            return x

        def fn3():
            # [FUNCTION DOC] HELPER: Calls fn1 then returns the result of fn2.
            fn1()
            return fn2()

        ref = fn3()
        opt_fn3 = torch.compile(fn3, backend="eager")
        res = opt_fn3()
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_inst(self):
        # [FUNCTION DOC] Checks that a function using enable_grad, which calls a callee that graph
        # breaks via a print statement (so inlining fails), computes the same input gradient as
        # eager when the compiled function is run under no_grad.
        def reversible(x):
            # [FUNCTION DOC] HELPER: Prints a message to force a graph break and returns the sine
            # of the cosine of the input.
            print("Hello world")  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            # [FUNCTION DOC] HELPER: Under enable_grad computes a chain of sine, reversible and
            # sigmoid, runs backward and returns the gradient of x.
            with torch.enable_grad():
                a = torch.sin(x)
                b = reversible(a)
                c = torch.sigmoid(b)
                c.sum().backward()
                return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch.compile(fn, backend="eager")
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_with_on_graph_break_nested(self):
        # [FUNCTION DOC] Like the previous test but with nested no_grad and enable_grad context
        # managers around a callee that triggers an explicit graph break; compiled (eager backend)
        # gradient under no_grad must match eager.
        def reversible(x):
            # [FUNCTION DOC] HELPER: Forces a graph break with the Dynamo graph_break call and
            # returns the sine of the cosine of the input.
            torch._dynamo.graph_break()  # Cause graph break so inline fails
            return torch.sin(torch.cos(x))

        def fn(x):
            # [FUNCTION DOC] HELPER: Inside nested no_grad and enable_grad blocks, computes
            # sine, the reversible callee and sigmoid, runs backward on the sum and returns the
            # input gradient.
            # nested context manager failed previously
            with torch.no_grad():
                with torch.enable_grad():
                    a = torch.sin(x)
                    b = reversible(a)
                    c = torch.sigmoid(b)
                    c.sum().backward()
                    return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch.compile(fn, backend="eager")
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    # https://github.com/pytorch/torchdynamo/issues/1446
    def test_grad_mode_carrying_correct_state_after_graph_break(self):
        # [FUNCTION DOC] Regression test for torchdynamo issue 1446: after a graph break inside a
        # no_grad block, both outputs (computed before and after the break) must not require grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Under no_grad computes a product, prints to cause a graph
            # break, then computes a sum; returns both.
            with torch.no_grad():
                y = x * 3
                print("Break")
                z = x + 2
            return y, z

        x = torch.randn(3, requires_grad=True)
        opt_fn = torch.compile(fn, backend="eager")
        y, z = opt_fn(x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_basic_tensor_subclass_constructor(self):
        # [FUNCTION DOC] Regression test: constructing a plain torch.Tensor subclass from a tensor
        # inside a function compiled with fullgraph and dynamic shapes must work, return the
        # subclass type and match eager values. Covers a lazy variable tracker being realized
        # before reading the instance dictionary.
        # Tests that basic Tensor subclass construction works with torch.compile
        # Regression test for LazyVariableTracker not being realized before
        # accessing __dict__ in TensorWithTFOverrideVariable.from_tensor_var
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Empty subclass of torch.Tensor.
            pass

        def f(x):
            # [FUNCTION DOC] HELPER: Wraps the input in the subclass and returns its
            # absolute value.
            y = MyTensor(x)
            return torch.abs(y)

        x = torch.randn(4)
        eager_out = f(x)
        compiled_f = torch.compile(f, backend="eager", fullgraph=True, dynamic=True)
        compiled_out = compiled_f(x)

        torch.testing.assert_close(
            torch.as_tensor(eager_out),
            torch.as_tensor(compiled_out),
        )
        self.assertIsInstance(compiled_out, MyTensor)

    def test_locals_traced_correctly_under_compile(self):
        # [FUNCTION DOC] Checks that code asking for the dictionary of local variables to test
        # whether names assigned only on some paths exist gives the same result when compiled with
        # the eager backend as in eager, for a three-dimensional input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Flattens a 3D input to 2D and records batch size, sequence
            # length and hidden dim; adds one; if the recorded names appear in the locals mapping,
            # views the result back to 3D.
            if x.dim() > 2:
                batch_size, seq_len, hidden_dim = x.shape
                x = x.view(-1, hidden_dim)

            x = x + 1

            if "batch_size" in locals() and "seq_len" in locals():
                x = x.view(batch_size, seq_len, -1)
            return x

        x = torch.randn(2, 3, 4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_vars_traced_correctly_under_compile(self):
        # [FUNCTION DOC] Same as the preceding test but queries the vars builtin for the names
        # instead of the locals builtin; compiled and eager results must match.
        def fn(x):
            # [FUNCTION DOC] HELPER: Flattens a 3D input, adds one and, if the recorded names are
            # present in the vars mapping, views the result back to 3D.
            if x.dim() > 2:
                batch_size, seq_len, hidden_dim = x.shape
                x = x.view(-1, hidden_dim)

            x = x + 1

            if "batch_size" in vars() and "seq_len" in vars():
                x = x.view(batch_size, seq_len, -1)
            return x

        x = torch.randn(2, 3, 4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_abc_setattr(self):
        # [FUNCTION DOC] Checks Dynamo bails out correctly on a custom setattr in an ABC-based
        # module subclass with an overridden method calling super; runs the compiled module
        # once and asserts the Dynamo frame counters recorded at least one ok frame and one
        # total frame.
        # tests that we correctly bail out of __setattr__ calls

        # TODO: does not ensure ABC classes are correctly inferred as ClassVariables
        # (doesn't test the fix for 'super()')

        class BaseModule(torch.nn.Module, ABC):
            # [CLASS DOC] HELPER: Abstract-base module with a method adding one to its input.
            def blah(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        class Derived(BaseModule):
            # [CLASS DOC] HELPER: Module subclass with a custom attribute setter that forwards to
            # the base, a forward that sets an attribute and calls blah, and an overridden blah
            # that calls the base class.
            def __setattr__(self, name, value) -> None:
                # [FUNCTION DOC] HELPER: Delegates attribute assignment to the base
                # class implementation.
                super().__setattr__(name, value)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets the attribute foo to zero (expected to cause a graph
                # break per the comment) and returns blah applied to the input.
                # expect a graph break on __setattr__
                self.foo = 0
                return self.blah(x)

            def blah(self, x):
                # [FUNCTION DOC] HELPER: Calls the base class blah.
                return super().blah(x)

        x = torch.randn(3, requires_grad=True)
        mod = Derived()
        opt_mod = torch.compile(mod, backend="eager")
        opt_mod(x)

        # Not sure what this test is testing. It was earlier graph breaking on
        # __dict__, so the counter >= 2. With __dict__ support, there is no
        # graph break.
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 1)

    @torch._dynamo.config.patch("suppress_errors", True)
    def test_guard_fail_tensor_bool(self):
        # [FUNCTION DOC] With suppress_errors enabled, runs a function (excluded from Dynamo
        # recursion by the disable decorator) that builds boolean tensors of several shapes and
        # compares torch.where against a numpy-based expected value; then runs the compiled
        # version too. Ensures guard failures with tensor bool conditions do not crash.
        @torch._dynamo.disable(recursive=False)
        def fn():
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that, for each x and y pair from
            # scalar-and-tensor tuples, builds a random boolean condition, computes the dtype with
            # result_type and asserts via check_equal that torch.where matches the numpy result in
            # both argument orders.
            condition_shape = (5, 5)
            dtypes = (torch.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = [
                torch.empty(shape, dtype=dtype).fill_(17)
                for shape, dtype in itertools.product(shapes, dtypes)
            ]

            x_vals = (5.0, *tensors)
            y_vals = (6.0, *tensors)

            @torch._dynamo.disable
            def get_expected(condition, x, y):
                # [FUNCTION DOC] HELPER: Dynamo-disabled helper that converts tensors to numpy,
                # applies numpy where with the condition and converts back to the common dtype (a
                # variable of the enclosing function).
                x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                return torch.from_numpy(
                    np.where(condition.cpu().numpy(), x_np, y_np)
                ).to(common_dtype)

            for x, y in zip(x_vals, y_vals):
                condition = torch.empty(*condition_shape, dtype=torch.bool).bernoulli_()
                common_dtype = torch.result_type(x, y)

                def check_equal(condition, x, y):
                    # [FUNCTION DOC] HELPER: Asserts that the expected where result and
                    # torch.where are close.
                    # NumPy aggressively promotes to double, hence cast to output to correct dtype
                    expected = get_expected(condition, x, y)
                    result = torch.where(condition, x, y)
                    assert torch.allclose(expected, result)  # noqa: S101

                check_equal(condition, x, y)
                check_equal(condition, y, x)

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_guard_fail_nested_tuple(self):
        # [FUNCTION DOC] Checks that guard failure on a changed nested tuple structure of tensor
        # inputs does not crash: the same compiled function is called with a tuple containing a
        # nested tuple and then with a flat tuple, and the results are equal.
        def fn(args):
            # [FUNCTION DOC] HELPER: Returns a scalar tensor of ones and the first
            # element doubled.
            return torch.ones(()), args[0] * 2

        # This adds a tensor check on args[1][0] and args[1][1]
        args1 = (torch.ones(1), (torch.ones(1), torch.ones(1)))
        args2 = (torch.ones(1), torch.ones(1))
        opt_fn = torch.compile(fn, backend="eager")
        ref = opt_fn(args1)
        res = opt_fn(args2)

        self.assertTrue(same(ref, res))

    def test_nullcontext1(self):
        # [FUNCTION DOC] Checks that a nullcontext instance passed as a function argument works as
        # a with-block context manager under fullgraph compile; result must equal sine cosine sine
        # of the input.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Compiled function applying sin, cos inside the with-block of
            # the given context object, then sin.
            x = x.sin()
            with ctx:
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext()), y.sin().cos().sin()))

    def test_nullcontext2(self):
        # [FUNCTION DOC] Same as nullcontext1 but the context argument is the nullcontext class
        # which the function calls to create an instance.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Compiled function applying sin, cos inside a with-block on a
            # context created by calling the argument, then sin.
            x = x.sin()
            with ctx():
                x = x.cos()
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext), y.sin().cos().sin()))

    def test_no_grad_inline(self):
        # [FUNCTION DOC] Checks that a function decorated with no_grad can be inlined into a
        # fullgraph compiled caller and gives sin followed by cos.
        @torch.no_grad()
        def a(x):
            # [FUNCTION DOC] HELPER: No_grad decorated function returning the sine of the input.
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def b(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the cosine of the result of a.
            return a(x).cos()

        y = torch.randn(10)
        self.assertTrue(same(b(y), y.sin().cos()))

    @skipIfWindows(
        msg="torch._dynamo.exc.TorchRuntimeError: Failed running call_function <class 'torch.LongTensor'>(*(FakeTensor(..., size=(10,), dtype=torch.int32),), **{}):"
    )
    def test_longtensor_list(self):
        # [FUNCTION DOC] For partition sizes 0, 5 and 10, compares compiled and eager results of
        # building a LongTensor from a list mixing numpy values and tensors, with seeding so
        # random lists match; skipped on Windows. Runs the compiled function once first so
        # one-time random calls do not shift the seeded stream.
        for partition in [0, 5, 10]:

            @torch._dynamo.disable
            def rand_gen():
                # [FUNCTION DOC] HELPER: Dynamo-disabled generator of a list of ten random ints in
                # 5 to 10 where the first partition entries are numpy array elements and the rest
                # are tensors.
                rand_vals = [random.randint(5, 10) for _ in range(10)]
                # List of tensors mixed with np.arrays
                return list(np.array(rand_vals[:partition])) + [
                    torch.tensor(val) for val in rand_vals[partition:]
                ]

            def fn(x):
                # [FUNCTION DOC] HELPER: Builds a LongTensor from the random list and multiplies
                # it by the input.
                random_list = rand_gen()
                z = torch.LongTensor(random_list)
                return x * z

            x = torch.ones(10) * 2

            random.seed(0)
            ref0 = fn(x)
            ref1 = fn(x)

            opt_fn = torch.compile(fn, backend="eager")
            # Especially for internal usage, there are many calls to random functions
            # on first compile, e.g., from various library initializations. Run once
            # to get that out of the way before resetting the seed:
            opt_fn(x)

            random.seed(0)
            res0 = opt_fn(x)
            res1 = opt_fn(x)

            self.assertTrue(same(ref0, res0))
            self.assertTrue(same(ref1, res1))

    def test_primtorch(self):
        # [FUNCTION DOC] Checks that a compiled function calling the prim refs abs on a tensor
        # runs (graph breaks allowed).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the refs absolute value on x without returning it.
            torch._refs.abs(x)

        fn(torch.randn(3))

    @unittest.expectedFailure
    # inline_call [('inline in skipfiles: bind ...python3.10/inspect.py', 1)]
    def test_primtorch_no_graph_break(self):
        # [FUNCTION DOC] Expected failure: the same function as the previous test but compiled
        # with fullgraph.
        # KNOWN ISSUE: marked as an expected failure because inlining an inspect bind call from
        # skipped files currently fails per the comment.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the refs absolute value on x without returning it.
            torch._refs.abs(x)

        fn(torch.randn(3))

    def test_torch_tensor_ops_no_graph_break(self):
        # [FUNCTION DOC] Checks that calling the unbound tensor method abs_ on a tensor compiles
        # with fullgraph without a graph break.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the in-place absolute value through the Tensor class
            # on x.
            torch.Tensor.abs_(x)

        fn(torch.randn(3))

    @unittest.skipIf(
        not isinstance(torch.ops.aten.abs, torch._ops.OpOverloadPacket),
        "old pt doesn't work",
    )
    def test_torch_ops_aten(self):
        # [FUNCTION DOC] Skipped on old versions where aten abs is not an OpOverloadPacket. Checks
        # that an aten op that is not in the default list (absolute) can be called inside a
        # function compiled with fullgraph.
        # Picked an op that doesn't show up in the default list
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the aten absolute op applied to x.
            return torch.ops.aten.absolute(x)

        fn(torch.randn(3))

    def test_hf_gelu_inline(self):
        # [FUNCTION DOC] Checks a module constructed inside a compiled function (HuggingFace style
        # GELU activation storing a function as attribute) works with fullgraph and matches gelu;
        # also checks a module instance can be returned from a compiled function, still has the
        # stored function, and had its initializer run (buffers attribute exists).
        class GELUActivation(nn.Module):
            # [CLASS DOC] HELPER: Module storing the functional gelu in an attribute and applying
            # it in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer and stores the functional gelu
                # as act.
                super().__init__()
                self.act = nn.functional.gelu

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the stored activation to the input.
                return self.act(input)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that constructs the module and applies it
            # to x.
            return GELUActivation()(x)

        y = torch.randn(10)
        self.assertTrue(same(fn(y), nn.functional.gelu(y)))

        @torch.compile(backend="eager", fullgraph=True)
        def fn_returns(x):
            # [FUNCTION DOC] HELPER: Compiled function returning a new module instance and x
            # plus one.
            return GELUActivation(), x + 1

        act, _ = fn_returns(y)
        self.assertIsInstance(act, GELUActivation)
        self.assertIs(act.act, nn.functional.gelu)
        self.assertTrue(hasattr(act, "_buffers"))  # check that __init__ got called

    def test_dropout_inline(self):
        # [FUNCTION DOC] Checks that creating and applying a Dropout module inside a
        # compiled function gives the same values as the functional dropout when the RNG is
        # seeded identically.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies a newly constructed dropout module with
            # probability 0.1.
            return torch.nn.Dropout(0.1)(x)

        y = torch.randn(10)
        torch.manual_seed(1337)
        ref = nn.functional.dropout(y, 0.1)
        torch.manual_seed(1337)
        res = fn(y)
        self.assertTrue(same(ref, res))

    def test_dropout_eval_setattr_parameter(self):
        # [FUNCTION DOC] Checks that an eval-mode (training false) dropout of a parameter assigned
        # to a new attribute inside forward works under fullgraph compile and that the attribute
        # afterward is the very same parameter object, and the output is x plus the weight.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 5 by 5 parameter; forward applies non-training
            # dropout to the weight, stores it as attribute foo and returns x plus the weight.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weight parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(5, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Stores the non-training dropout of the weight in foo and
                # returns x plus the weight.
                w = torch.nn.functional.dropout(self.weight, training=False)
                self.foo = w
                return x + self.weight

        mod = Mod()
        x = torch.randn(5, 5)
        mod(x)
        mod.compile(backend="eager", fullgraph=True)
        self.assertTrue(same(mod(x), x + mod.weight))
        self.assertIs(mod.foo, mod.weight)

    def test_setitem_boolean_mask_diff(self):
        # [FUNCTION DOC] Checks that assigning values through a boolean mask index on a cloned
        # tensor that requires grad compiles under aot_eager without error.
        def fn(x, b, y):
            # [FUNCTION DOC] HELPER: Clones x, assigns y at the mask positions and returns it.
            x = x.clone()
            x[b] = y
            return x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_setitem_tuple_boolean_mask_diff(self):
        # [FUNCTION DOC] Checks that assignment through a tuple index containing a slice and a
        # boolean mask on a cloned tensor requiring grad compiles under aot_eager without error.
        def fn(x, b, y):
            # [FUNCTION DOC] HELPER: Clones x, assigns y at the slice and mask positions of the
            # second dimension, and returns it.
            x = x.clone()
            x[:, b] = y
            return x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(8, 4, requires_grad=True)
        b = torch.tensor([True, False, True, False])
        y = torch.randn(2, requires_grad=True)
        opt_fn(x, b, y)

    def test_torch_tensor_ops(self):
        # [FUNCTION DOC] Checks that calling the unbound tensor method abs_ (in place absolute
        # value) on a tensor under fullgraph gives the same result as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the in-place absolute value through the Tensor class.
            return torch.Tensor.abs_(x)

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        y = fn(x)
        y_ = opt_fn(x)
        self.assertTrue(same(y, y_))

    def test_guard_ordering_shape_fail(self):
        # [FUNCTION DOC] Checks guard ordering: after a compiled method of a module has a shape
        # guard from a tensor call, calling it again with an int must not crash guard evaluation
        # with a missing shape attribute error.
        # If a function which takes a tensor has an inner function which
        # is compiled and generates a guard on its shape,
        # they are evaluated in the wrong order. So if on a subsequent call
        # an int is passed instead of a tensor, guard evaluation will crash
        # with a "no attribute: shape" error
        m = MockModule()
        opt_m = torch.compile(m, backend="eager")
        opt_m.fn(torch.ones((5, 5)))
        opt_m.fn(-3)

    def test_tensor_isinstance_tuple(self):
        # [FUNCTION DOC] Checks that isinstance with a tuple of types including int and Tensor is
        # handled inside a compiled function that also formats a message; nothing is raised.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Creates a tensor, and if it is not an instance of the tuple
            # of int or Tensor builds a message and raises ValueError; otherwise returns True.
            t = torch.ones(5, 5)
            if not isinstance(t, (int, torch.Tensor)):
                msg = str.format(
                    "{0} is not an instance of {1}",
                    type(t),
                    (int, torch.Tensor),
                )
                raise ValueError(msg)
            return True

        fn()

    def test_isinstance_dtype(self):
        # [FUNCTION DOC] Checks that isinstance on a dtype value compiles with fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Evaluates the dtype isinstance check and returns x.
            isinstance(torch.bfloat16, torch.dtype)
            return x

        fn(torch.randn(3))

    def test_isinstance_storage(self):
        # [FUNCTION DOC] Checks that creating a BoolStorage from a bytearray buffer and checking
        # its type works inside a compiled function (graph breaks allowed).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a bytearray, creates a BoolStorage from the buffer
            # using big endian, asserts its type and returns x.
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            bools = torch.BoolStorage.from_buffer(f, "big")
            assert isinstance(bools, torch.BoolStorage)  # noqa: S101
            return x

        fn(torch.randn(3))

    def test_issue111522(self):
        # [FUNCTION DOC] Regression test for issue 111522: a fullgraph compiled attribute read
        # specializes on a class attribute; after the attribute is deleted from the class, calling
        # again must raise the Dynamo Unsupported error.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus the attribute a of y.
            return x + y.a

        class A:
            # [CLASS DOC] HELPER: Class with a class attribute a equal to 2.
            a = 2

        self.assertEqual(f(torch.zeros(2), A()), torch.full([2], 2.0))

        del A.a

        # graph break on missing attr
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f(torch.zeros(2), A())

    def test_sort_out(self):
        # [FUNCTION DOC] Checks that sort with out tuple of 0-dim tensors resizes them to
        # contiguous stride one outputs in both eager and compiled (eager backend) runs.
        dtype = torch.float32
        device = "cpu"

        def fn():
            # [FUNCTION DOC] HELPER: Sorts a slice of a random tensor into preallocated value and
            # index outputs, then asserts both outputs have stride of one element.
            tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
            values1 = torch.tensor(0, dtype=dtype, device=device)
            indices1 = torch.tensor(0, dtype=torch.long, device=device)
            torch.sort(tensor, out=(values1, indices1))
            self.assertEqual(values1.stride(), (1,))
            self.assertEqual(indices1.stride(), (1,))

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_sort_out2(self):
        # [FUNCTION DOC] Checks sort with out tensors that are module buffers compiles with the
        # eager backend and returns the same values, including the buffers, as eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two buffers (float values and int64 indices) that
            # forward fills through sort with out and returns x plus one with both buffers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the sorted values buffer and the indices buffer,
                # both 4 by 4.
                super().__init__()
                self.sorted = torch.nn.Buffer(torch.ones(4, 4))
                self.indices = torch.nn.Buffer(torch.ones(4, 4, dtype=torch.long))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sorts x into the buffers and returns x plus one with the
                # two buffers.
                torch.sort(x, out=(self.sorted, self.indices))
                return (x + 1, self.sorted, self.indices)

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_sigmoid_out(self):
        # [FUNCTION DOC] Checks that sigmoid with a 0-dim out tensor resizes it to 15 elements in
        # eager and when compiled.
        dtype = torch.float32
        device = "cpu"

        def fn():
            # [FUNCTION DOC] HELPER: Applies sigmoid with an out tensor and asserts the out tensor
            # has 15 elements.
            inp = torch.randn((3, 5), dtype=dtype, device=device)
            out1 = torch.tensor(0, dtype=dtype, device=device)
            torch.sigmoid(inp, out=out1)
            self.assertEqual(out1.numel(), 15)

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_sigmoid_out2(self):
        # [FUNCTION DOC] Checks sigmoid with out set to a module buffer compiled with the eager
        # backend matches eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 4 by 4 buffer; forward writes sigmoid of x into
            # the buffer and returns x plus the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the buffer.
                super().__init__()
                self.base = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Writes sigmoid of x into the buffer and returns x plus
                # the buffer.
                torch.sigmoid(x, out=self.base)
                return x + self.base

        x = torch.randn(4, 4)
        m = MyModule()
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(same(ref, res))

    def test_out_root_cell_shape_change(self):
        # [FUNCTION DOC] Checks that an out tensor captured in a closure cell and resized by an op
        # inside a nested function reports the new shape (3 by 5).
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Creates an empty out tensor and returns the result of a
            # nested function.
            out = torch.empty(0)

            def run():
                # [FUNCTION DOC] HELPER: Applies sigmoid of a 3 by 5 zeros tensor into the
                # captured out tensor and returns the size of out.
                x = torch.zeros(3, 5)
                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_nested_cell_shape_change(self):
        # [FUNCTION DOC] Same as the previous test but the out tensor is created inside the nested
        # function and is also captured by an inner function so it becomes a nested cell.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Returns the result of the nested function.
            def run():
                # [FUNCTION DOC] HELPER: Creates input and out tensors, defines a capturing inner
                # function to force a cell, applies sigmoid with out and returns the out size.
                x = torch.zeros(3, 5)
                out = torch.empty(0)

                def capture():
                    # [FUNCTION DOC] HELPER: Returns the captured out tensor (exists only to force
                    # a cell).
                    return out  # Force `out` to be a nested cell

                torch.sigmoid(x, out=out)
                return out.size()

            return run()

        res = fn()
        self.assertEqual((3, 5), res)

    def test_out_root_cell_tuple_shape_change(self):
        # [FUNCTION DOC] Checks that sort with a tuple of two out tensors held in closure cells
        # resizes both to 3 by 5.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Creates two empty out tensors and returns the result of the
            # nested function.
            out1 = torch.empty(0)
            out2 = torch.empty(0, dtype=torch.long)

            def run():
                # [FUNCTION DOC] HELPER: Sorts a 3 by 5 zeros tensor into the two captured out
                # tensors and returns both sizes.
                x = torch.zeros(3, 5)
                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_out_nested_cell_tuple_shape_change(self):
        # [FUNCTION DOC] Same as the previous test but the two out tensors are created in the
        # nested function and captured by an inner function.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Returns the result of the nested function.
            def run():
                # [FUNCTION DOC] HELPER: Creates input and the two out tensors, defines a
                # capturing function, sorts into the outs and returns both sizes.
                x = torch.zeros(3, 5)
                out1 = torch.empty(0)
                out2 = torch.empty(0, dtype=torch.long)

                def capture():
                    # [FUNCTION DOC] HELPER: Returns the two captured out tensors (exists only to
                    # force cells).
                    # Force `out1` and `out2` to be nested cells
                    return out1, out2

                torch.sort(x, out=(out1, out2))
                return out1.size(), out2.size()

            return run()

        res = fn()
        self.assertEqual(((3, 5), (3, 5)), res)

    def test_slice_into_list_mutable(self):
        # [FUNCTION DOC] Compiles with fullgraph a module that takes a slice of a list input and
        # assigns tensors into the slice repeatedly in a loop; expects a single frame.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward slices the list, loops ten times assigning
            # a fresh positive random tensor to the first slot, and returns the slice.
            def forward(self, listy):
                # [FUNCTION DOC] HELPER: Slices the list from 3 to 5, assigns tensors into its
                # first element ten times and returns the slice.
                x = listy[3:5]
                for _ in range(10):
                    z = torch.abs(torch.randn(10)) + 1
                    x[0] = z
                return x

        m = Mod()
        listy = [torch.randn(10)] * 10

        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt, fullgraph=True)
        opt_m.forward(listy)

        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_issue111918(self):
        # [FUNCTION DOC] Regression test for issue 111918 with scalar output capture: with dynamic
        # shapes a data-dependent branch on item gives two frames and four ops; after a reset,
        # compiling with fullgraph must raise Dynamo UserError.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, reads a scalar with item and returns double or
            # triple of x depending on whether the scalar exceeds 2.
            x = x + 1
            y = x.item()
            if y > 2:
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)

        torch._dynamo.reset()
        fn = torch.compile(fn, fullgraph=True, backend="eager")
        with self.assertRaises(torch._dynamo.exc.UserError):
            fn(x)

    def test_vdd_duplicate_error(self):
        # [FUNCTION DOC] Compiles with fullgraph a function reading a tensor out of a dictionary
        # held in a user-defined sample object twice (cos and sigmoid) and compares with eager,
        # covering a duplicate-variable error.
        def fn(a, dt):
            # [FUNCTION DOC] HELPER: Reads the same dictionary entry twice, applying cos and
            # sigmoid, and sin to a; returns the sum of the three.
            keys = list(dt._jt_dict.keys())
            p = torch.cos(dt._jt_dict[keys[0]]._value)
            q = torch.sin(a)
            r = torch.sigmoid(dt._jt_dict[keys[0]]._value)
            return p + q + r

        class Value:
            # [CLASS DOC] HELPER: Holds a random 4-element tensor in the attribute _value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the random tensor.
                self._value = torch.randn(4)

        class Sample:
            # [CLASS DOC] HELPER: Holds a dictionary with one key POSITION_ID mapping to a Value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dictionary and inserts a Value.
                self._jt_dict = {}
                self._jt_dict["POSITION_ID"] = Value()

        a = torch.randn(4)
        sample = Sample()

        ref = fn(a, sample)

        optimized_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = optimized_fn(a, sample)

        self.assertTrue(same(ref, res))

    def test_specialized_stride(self):
        # [FUNCTION DOC] Checks that the stride of a step-2 slice of an empty tensor is the same
        # in eager and in a compiled function.
        def f():
            # [FUNCTION DOC] HELPER: Slices every second element of an empty 4-element tensor and
            # returns its stride.
            e = torch.empty(4)
            x = e[::2]
            return x.stride()

        self.assertEqual(f(), torch.compile(f, backend="eager")())

    def test_out_none(self):
        # [FUNCTION DOC] Regression test for issue 92814: functional normalize with out passed as
        # None must work compiled.
        # https://github.com/pytorch/pytorch/issues/92814
        def fn(input):
            # [FUNCTION DOC] HELPER: Normalizes along dim 0 with out set to None.
            return torch.nn.functional.normalize(input, dim=0, out=None)

        x = torch.rand([1])
        self.assertEqual(fn(x), torch.compile(fn, backend="eager")(x))

    def test_multi_import(self):
        # [FUNCTION DOC] Skipped without detectron2. Checks that a compiled function importing
        # several names from a detectron2 module inside it and testing them for None returns input
        # plus one.
        if not has_detectron2():
            raise unittest.SkipTest("requires detectron2")

        @torch.compile(backend="eager", fullgraph=True)
        def to_bitmasks(boxes):
            # [FUNCTION DOC] HELPER: Imports two names from the detectron2 mask ops module and
            # returns boxes plus one if both are available.
            from detectron2.layers.mask_ops import (
                _paste_masks_tensor_shape,
                paste_masks_in_image,
            )

            if (
                paste_masks_in_image is not None
                and _paste_masks_tensor_shape is not None
            ):
                return boxes + 1

        self.assertTrue((to_bitmasks(torch.zeros(10)) == torch.ones(10)).all())

    def test_multi_dot_import(self):
        # [FUNCTION DOC] Checks a function that imports torch.fx locally and symbolically traces
        # another function compiles in one frame.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Imports torch.fx, symbolically traces fn1 and returns
            # x doubled.
            import torch.fx

            _ = torch.fx.symbolic_trace(fn1)
            return x * 2

        x = torch.randn(10)
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import(self):
        # [FUNCTION DOC] Checks that a relative import of a name inside a function compiled with
        # fullgraph works; falls back to an absolute import of the utils module when relative
        # import is not possible. Expects one frame.
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                # [FUNCTION DOC] HELPER: Imports the testing tensor from the relative utils module
                # and multiplies x by two and by the tensor.
                from .utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:

            def fn(x):
                # [FUNCTION DOC] HELPER: Fallback that imports the testing tensor from the
                # top-level utils module and multiplies x by two and by it.
                from utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        x = torch.randn(10, device="cpu")
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_relative_import_no_modulename(self):
        # [FUNCTION DOC] Same as the previous test but imports the whole utils module (relative or
        # absolute) rather than a name from it.
        try:
            from . import utils as _  # noqa: F401

            def fn(x):
                # [FUNCTION DOC] HELPER: Imports the utils module relatively and multiplies x by
                # two and by its testing tensor.
                from . import utils

                return x * 2 * utils.tensor_for_import_testing

        except ImportError:

            def fn(x):
                # [FUNCTION DOC] HELPER: Fallback that imports the utils module absolutely and
                # multiplies x by two and by its testing tensor.
                import utils

                return x * 2 * utils.tensor_for_import_testing

        x = torch.randn(10, device="cpu")
        fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_bigbird_unsqueeze_inplace(self):
        # [FUNCTION DOC] Checks BigBird-style in-place unsqueeze on a clone, followed by
        # concatenation and view, matches eager under aot_eager.
        def fn(reshape_2):
            # [FUNCTION DOC] HELPER: Clones the input, unsqueezes it in place along dim 2,
            # concatenates the single tensor along dim 2, views it as 2, 12, 64 and a free last
            # dim, and returns it in a tuple.
            view_2 = reshape_2.clone()
            view_2.unsqueeze_(2)
            cat_11 = torch.cat([view_2], dim=2)
            view_13 = cat_11.view((2, 12, 64, -1))
            return (view_13,)

        x = torch.randn(2, 12, 64, 64, requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_issue1466_size_aot_autograd(self):
        # [FUNCTION DOC] Regression test for issue 1466: a size computed before a graph break
        # (print) and used after it in a view must work with aot_eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles x, saves its size, prints to cause a graph break,
            # then views the doubled tensor to the saved size and adds one.
            # do a tensor op and a size compute
            y = x * 2
            x_size = x.size()
            # trigger a graph break
            print("arf")
            # use the tensor op and size compute
            z = y.view(x_size) + 1
            return z

        x = torch.randn(2, 3, requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_ellipsis(self):
        # [FUNCTION DOC] Checks a module that applies a layer norm and then indexes with a tuple
        # including slices and Ellipsis before a linear layer compiles with fullgraph and
        # matches eager.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a layer norm over 256 features and a 256 to 256
            # linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layer norm and the linear layer.
                super().__init__()
                self.lnorm = torch.nn.LayerNorm(
                    (256,), eps=1e-06, elementwise_affine=True
                )
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=256, bias=True
                )

            def forward(self, cat_10):
                # [FUNCTION DOC] HELPER: Layer norms the input, takes the first position along dim
                # 1 using a tuple index with Ellipsis, applies the linear layer and returns a
                # one-element tuple.
                lnorm = self.lnorm(cat_10)
                getitem_64 = lnorm[
                    (slice(None, None, None), slice(0, 1, None), Ellipsis)
                ]
                linear = self.linear(getitem_64)
                return (linear,)

        args = [torch.randn(2, 197, 256)]

        mod = Repro()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)

        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_reinplacing(self):
        # [FUNCTION DOC] Checks that a module with two embedding layers added to an input compiles
        # under aot_eager_decomp_partition (reinplacing) with random strided inputs and gives the
        # same outputs and gradients as eager through same_two_models.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two 1024 by 768 embedding layers for x and
            # y positions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two embedding layers.
                super().__init__()
                self.self_layoutlm_embeddings_x_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )
                self.self_layoutlm_embeddings_y_position_embeddings = (
                    torch.nn.Embedding(1024, 768)
                )

            def forward(self, getitem_1, getitem_2, add):
                # [FUNCTION DOC] HELPER: Looks up two embeddings, adds both to the third input and
                # returns the sum in a tuple.
                self_layoutlm_embeddings_x_position_embeddings = (
                    self.self_layoutlm_embeddings_x_position_embeddings(getitem_1)
                )
                self_layoutlm_embeddings_y_position_embeddings = (
                    self.self_layoutlm_embeddings_y_position_embeddings(getitem_2)
                )
                add_1 = add + self_layoutlm_embeddings_x_position_embeddings
                add_2 = add_1 + self_layoutlm_embeddings_y_position_embeddings
                return (add_2,)

        mod = MockModule()
        opt_mod = torch.compile(mod, backend="aot_eager_decomp_partition")

        args = [
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512, 768), (393216, 768, 1), torch.float32, "cpu", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_optimized_deepcopy(self):
        # [FUNCTION DOC] Regression test for PR 88629: deep-copying support for an optimized
        # module; compares the module and its compiled version on an input using same_two_models.
        # See https://github.com/pytorch/pytorch/pull/88629
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer from 2 to 3 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.fc = torch.nn.Linear(in_features=2, out_features=3, bias=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.fc(x)

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager")
        args = [torch.randn(1, 2)]
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_class_member(self):
        # [FUNCTION DOC] Checks that class-level attributes (an int and a tensor) and an instance
        # attribute of a module are read correctly under fullgraph compile.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with class attributes a (4) and b (ones tensor) and
            # instance attribute c (4); forward returns cos of x plus a, b and c.
            a = 4
            b = torch.ones(3, 4)

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer and sets c to 4.
                super().__init__()
                self.c = 4

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns cos of x plus the three attributes.
                return x.cos() + self.a + self.b + self.c

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_named_buffers(self):
        # [FUNCTION DOC] Checks that iterating over named_buffers of a module inside a fullgraph
        # compiled forward sums the buffers correctly.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two buffers of ones; forward adds the sum of all
            # buffers to the cos of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two buffers.
                super().__init__()
                self.x = torch.nn.Buffer(torch.ones(3))
                self.y = torch.nn.Buffer(torch.ones(3))

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Sums all buffers found by named_buffers and returns cos
                # of inp plus that sum.
                res = 0
                for _, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        args = (torch.randn(3, 4),)
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_requires_grad_guards_with_grad_mode1(self):
        # [FUNCTION DOC] Checks that after compiling an inference graph under no_grad for a
        # function branching on requires_grad, calling with a detached input recompiles correctly
        # and keeps the result values and requires_grad flag equal to eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if x requires grad else x plus two.
            if x.requires_grad:
                return x + 1
            else:
                return x + 2

        x = torch.ones(2, requires_grad=True)

        f_compiled = torch.compile(f, backend="eager")
        with torch.no_grad():
            # compile an inference graph
            f_compiled(x)

        # Test: we should fail guards and recompile (even though it's still an inference graph)
        out_ref = f(x.detach())
        out = f_compiled(x.detach())

        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_requires_grad_guards_with_grad_mode2(self):
        # [FUNCTION DOC] Checks that a linear module compiled once under no_grad is recompiled
        # when later called with grad enabled, giving the same output and requires_grad flag
        # as eager.
        x = torch.ones(2, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        m = torch.nn.Linear(2, 2)
        m_compiled = torch.compile(m, backend="eager")

        with torch.no_grad():
            # compile an inference graph
            m_compiled(x)

        # Test: we should fail guards and recompile a training graph
        out_ref = m(x_ref)
        out = m_compiled(x)
        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_is_symbolic_tracing(self):
        # [FUNCTION DOC] Checks that a fx-tracing detection helper can be called inside a
        # fullgraph compiled function (no graph break) and gives the non-tracing branch result.
        # Ensure no graph break here
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns double of x if FX tracing is detected, else four
            # times x.
            if is_fx_tracing_test():
                return x * 2
            return x * 4

        a = torch.randn(4)
        ref = fn(a)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(a)
        self.assertTrue(same(ref, res))

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/184324")
    def test_tokenization(self):
        # [FUNCTION DOC] Skipped on ROCm (issue 184324). Checks that a dict-like user class
        # (copied from tokenization code) with a custom attribute getter works when constructed
        # and indexed in a compiled function.
        from collections import UserDict

        class BatchEncoding(UserDict):
            # [CLASS DOC] HELPER: UserDict subclass copied from tokenization that exposes
            # dictionary keys as attributes.
            """
            Copied from tokenization
            """

            def __init__(
                self,
                data,
            ):
                # [FUNCTION DOC] HELPER: Calls the UserDict initializer with the data.
                super().__init__(data)

            def __getattr__(self, item: str):
                # [FUNCTION DOC] HELPER: Returns the entry named item in the underlying data;
                # converts a KeyError into AttributeError.
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        def tokenization(x):
            # [FUNCTION DOC] HELPER: Wraps x in a BatchEncoding under key key and reads it back.
            encoding = BatchEncoding({"key": x})
            return encoding["key"]

        opt_fn = torch.compile(tokenization, backend="eager")
        x = torch.rand((1, 4))
        ref = tokenization(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_modules(self):
        # [FUNCTION DOC] Compiles with fullgraph a module iterating over its own modules (the
        # module and its linear child, so two iterations) accumulating the linear output; expects
        # 5 ops and one frame.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 4 to 3 linear layer; forward loops once per
            # submodule of itself adding the layer output to zeros.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Adds the linear output to a 3 by 3 zeros tensor once per
                # entry of modules and returns it.
                res = torch.zeros(3, 3)
                for _ in self.modules():
                    res += self.fc(inp)
                return res

        mod = Foo()
        args = (torch.ones(3, 4),)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt, fullgraph=True)
        self.assertTrue(same(mod(*args), opt_mod(*args)))
        self.assertEqual(cnt.op_count, 5)
        self.assertEqual(cnt.frame_count, 1)

    def test_omegaconf_listconfig_iter(self):
        # [FUNCTION DOC] Checks that iterating a ListConfig stand-in (omegaconf style custom
        # iterator) while accumulating values in place compiles with fullgraph and matches eager.
        obj = ListConfig()
        x = torch.zeros(2)

        def fn():
            # [FUNCTION DOC] HELPER: Adds each item of the iterated object to y in place and
            # returns y.
            y = x
            for i in obj:
                y += i
            return y

        expected = fn()
        actual = torch.compile(fn, fullgraph=True, backend="eager")()
        self.assertEqual(actual, expected)

    def test_user_defined_iter(self):
        # [FUNCTION DOC] Checks that a user-defined iterator class (counting 1 to 3) can be used
        # in a for loop inside a fullgraph compiled function; the sum is 6.
        class MyIter:
            # [CLASS DOC] HELPER: Iterator that returns 1, 2, 3 on successive calls and then
            # raises StopIteration.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the counter to zero.
                self.i = 0

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns the iterator itself.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Increments and returns the counter while it is below 3,
                # otherwise raises StopIteration.
                if self.i < 3:
                    self.i += 1
                    return self.i
                raise StopIteration

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds each iterated value to x in place and returns x.
            for i in MyIter():
                x += i
            return x

        self.assertEqual(fn(torch.zeros(1)), torch.full([1], 6.0))

    def test_stop_iteration_reconstruct(self):
        # [FUNCTION DOC] Checks that a StopIteration exception object returned from a compiled
        # function is reconstructed with the same arguments.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x and a StopIteration holding 1, 2, 3.
            return x.sin(), StopIteration(1, 2, 3)

        _, res = fn(torch.ones(1))
        self.assertEqual(str(res), str(StopIteration(1, 2, 3)))

    def test_tensor_data_kwarg(self):
        # [FUNCTION DOC] Regression test for issue 96278: torch.tensor called with the data
        # keyword must compile in one frame with fullgraph.
        # https://github.com/pytorch/pytorch/issues/96278
        def f():
            # [FUNCTION DOC] HELPER: Returns a tensor built from a nested list given via the
            # data keyword.
            return torch.tensor(data=[[1.0, -1.0]])

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(), opt_fn()))
        self.assertEqual(cnt.frame_count, 1)

    def test_for_loop_graph_break(self):
        # [FUNCTION DOC] Checks that a loop of 100 iterations containing a graph break compiles in
        # one frame with one op (graph break inside loop is handled by continuation functions).
        def inner(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Loops 100 times calling inner and an explicit graph break and
            # returns x.
            for _ in range(100):
                inner(x)
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_for_loop_graph_break_before(self):
        # [FUNCTION DOC] Checks the loop backedge is computed correctly when a graph break occurs
        # before the loop; one frame and 100 ops are expected.
        # Checks that the backedge is calculated correctly
        def inner(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Breaks the graph, then calls inner 100 times and returns x.
            torch._dynamo.graph_break()
            for _ in range(100):
                inner(x)
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 100)

    def test_avoid_dupe_specialization(self):
        # [FUNCTION DOC] Checks that the same tensor passed twice to a compiled function
        # (aot_eager) does not cause incorrect specialization on aliasing, for both requires_grad
        # settings, comparing against eager for same and different arguments.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs times one.
            return (x + y) * 1

        opt_f = torch.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            y = torch.randn(4, requires_grad=b)
            self.assertEqual(f(x, x), opt_f(x, x))
            self.assertEqual(f(x, y), opt_f(x, y))

    def test_validate_model_kwargs(self):
        # [FUNCTION DOC] Compiles with fullgraph a function taking keyword arguments that
        # validates them against another function's signature and then calls it; expects the
        # result to match eager, one frame and 3 ops.
        cnt = CompileCounter()

        def f1(a, b):
            # [FUNCTION DOC] HELPER: Returns sine of a plus cosine of b.
            return torch.sin(a) + torch.cos(b)

        @torch.compile(backend=cnt, fullgraph=True)
        def f2(**kwargs):
            # [FUNCTION DOC] HELPER: Compiled function that validates the keyword arguments
            # against f1 and returns f1 called with them.
            _validate_model_kwargs(f1, kwargs)
            return f1(**kwargs)

        x = torch.randn(10)
        y = torch.randn(10)

        self.assertEqual(f2(a=x, b=y), f1(x, y))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_swin_base_tensor_attr(self):
        # [FUNCTION DOC] Checks that a plain tensor attribute (not parameter or buffer) of a
        # module is used correctly: same_two_models comparison plus one more call of the
        # compiled module.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a random 3-element tensor attribute t that is
            # neither a parameter nor a buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the plain tensor attribute.
                super().__init__()
                # NB: not a parameter or buffer
                self.t = torch.randn(3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the concatenation of t with itself to x.
                return x + torch.cat((self.t, self.t))

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager")
        args = [torch.randn(6)]
        self.assertTrue(same_two_models(mod, opt_mod, args))
        opt_mod(*args)

    def test_pointless_graph_removal(self):
        # [FUNCTION DOC] Checks that a graph break inside a no_grad block does not leave useless
        # graphs: one frame and 3 ops are expected for the function.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Under no_grad, forces a graph break and returns x plus one.
            with torch.no_grad():
                torch._dynamo.graph_break()
                return x + 1

        fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_output_aliases_intermediate(self):
        # [FUNCTION DOC] Checks that outputs aliasing an intermediate (a view and the intermediate
        # itself) keep values, requires_grad flags and aliasing (in-place edit through one output
        # shows in the other) under aot_eager for requires_grad true and false.
        def f(x):
            # [FUNCTION DOC] HELPER: Doubles x, returns a flattened view of it and the
            # intermediate itself.
            intermediate = x.mul(2)
            return intermediate.view(-1), intermediate

        opt_f = torch.compile(f, backend="aot_eager")

        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            out = f(x)
            out_test = opt_f(x)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])
            self.assertEqual(out[0].requires_grad, out_test[0].requires_grad)
            self.assertEqual(out[1].requires_grad, out_test[1].requires_grad)
            # test that the aliasing relationship of outputs is preserved
            out[0].mul_(2)
            out_test[0].mul_(2)
            self.assertEqual(out[0], out_test[0])
            self.assertEqual(out[1], out_test[1])

    def test_while_loop_graph_break(self):
        # [FUNCTION DOC] Repro of tacotron2 cache-size recompilation: a while loop with a graph
        # break inside the body compiles into one frame with a single op.
        # Repro of tacotron2 cache_size_recompilation
        def inner(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Loops from 20 down to above 10 applying inner and an explicit
            # graph break each time; returns x.
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_nested_while_loop_graph_break(self):
        # [FUNCTION DOC] Like the while loop graph break test but the graph break is inside a
        # doubly nested call (inner loop inside inner inside the outer loop); expects one frame
        # and one op.
        def inner_loop(x):
            # [FUNCTION DOC] HELPER: Loops three times incrementing x in place with a graph break
            # each pass; returns x.
            i = 3
            while i > 0:
                i -= 1
                x += 1
                torch._dynamo.graph_break()
            return x

        def inner(x):
            # [FUNCTION DOC] HELPER: Runs inner_loop on x and returns the sine of x.
            inner_loop(x)
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Loops ten times applying inner and a graph break; returns x.
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_while_loop_graph_break_inside_call_function(self):
        # [FUNCTION DOC] Repro of the HuggingFace parameter dtype loop: only the inner frame whose
        # loop contains a graph break is skipped; the outer function compiles into two frames and
        # two ops.
        # Repro of huggingface graph break inside loop in `get_parameter_dtype`.
        # Skip only the inner frame that has loop that contains graph break.
        def inner(x):
            # [FUNCTION DOC] HELPER: Loops three times incrementing x in place with a graph break;
            # returns x.
            for _ in range(3):
                x += 1
                torch._dynamo.graph_break()
            return x

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 2 to x in place, calls inner and adds 3 in place;
            # returns x.
            x += 2
            inner(x)
            x += 3
            return x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)
        opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

    def test_exception_in_dynamo_handling(self):
        # [FUNCTION DOC] Checks that an exception raised in a callee while a generator-based
        # context manager is active inside compiled code is delivered to the context manager
        # handler (nonlocal flag set to True) and does not propagate (PR 96488).
        hit_handler = False

        # See https://github.com/pytorch/pytorch/pull/96488
        @contextlib.contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: Context manager generator that catches RuntimeError around
            # its yield and sets the nonlocal hit_handler flag.
            try:
                yield
            except RuntimeError:
                nonlocal hit_handler
                hit_handler = True

        @torch.compile(backend="eager")
        def f():
            # [FUNCTION DOC] HELPER: Compiled function that calls h inside the context manager.
            with ctx():
                h()

        def h():
            # [FUNCTION DOC] HELPER: Raises RuntimeError with message boof.
            raise RuntimeError("boof")

        # Should not error
        f()
        self.assertTrue(hit_handler)

    def test_generator_dealloc(self):
        # [FUNCTION DOC] Regression test for PR 96488: clearing the list that holds a generator
        # from a Dynamo-disabled frame deallocates the generator while the eval frame handler is
        # active; compiling the function must not emit unraisable exceptions and must still
        # capture the addition (one op). Delicate test per its comments.
        # See https://github.com/pytorch/pytorch/pull/96488
        #
        # NB: yes, [(...)] is intentional, this is a list containing a
        # generator
        generator_box = [(x for x in [1, 2, 3])]

        counter = torch._dynamo.testing.CompileCounter()

        def g(x):
            # [FUNCTION DOC] HELPER: Returns x plus 2.
            return x + 2

        # TODO: This test is pretty delicate.  To test if it's actually doing
        # anything, rebuild eval_frame.c with '#define TORCHDYNAMO_DEBUG 1'
        # and then look at the logs for:
        #
        # TRACE[_custom_eval_frame:650] begin <genexpr> test_repros.py 2276 -1 0 0
        # TRACE[_custom_eval_frame:664] throw <genexpr>
        #
        # This means we're actually hitting the relevant codepath

        # NB: Make sure we don't actually Dynamo this frame; if we do Dynamo
        # this frame, Dynamo actually DOES understand list.clear and will
        # arrange for the generator deallocation to happen when the eval frame
        # handler is disabled, which will prevent the bug from happening (we
        # specifically want to trigger the generator deallocation WHILE the
        # dynamo eval frame handler is active), as that will cause the
        # generator to become exhausted and trigger the throw_flag == TRUE
        # case.
        @torch._dynamo.disable(recursive=False)
        def f(x):
            # [FUNCTION DOC] HELPER: Dynamo-disabled (non-recursive) function that clears the list
            # holding the generator and then calls g.
            generator_box.clear()
            return g(x)

        self.assertNoUnraisable(
            lambda: torch.compile(f, backend=counter)(torch.randn(3))
        )

        # Make sure the x + 2 is captured (a previous incorrect implementation
        # of this fix would have disabled the eval frame callback, which means
        # g wouldn't get traced
        self.assertEqual(counter.op_count, 1)

    def test_error_return_without_exception_set(self):
        # [FUNCTION DOC] Regression test for issue 93781: compiled function that computes
        # the type of a generator expression must not leave an error set without
        # exception (assertNoUnraisable).
        # https://github.com/pytorch/pytorch/issues/93781
        @torch.compile(backend="eager")
        def f():
            # [FUNCTION DOC] HELPER: Computes the type of an empty generator expression.
            _generator_type = type(_ for _ in ())

        self.assertNoUnraisable(f)

    def common_merge_criteria_processor_list(self, list_cls, fullgraph):
        # [FUNCTION DOC] HELPER: Shared body for merge criteria tests. Compiles a function that
        # merges two callable list subclasses and calls the result; first with an empty custom
        # list (one frame, 2 ops, same result), then with a one-element custom list (the first
        # list is extended to 3 entries, 3 ops, one frame). The fullgraph option is a parameter.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=fullgraph)
        def f(x, left, right):
            # [FUNCTION DOC] HELPER: Compiled function merging the left and right lists with the
            # merge helper and calling the combined list on x.
            combined = _merge_criteria_processor_list(left, right)
            return combined(x)

        l1 = list_cls([torch.nn.ReLU(), torch.nn.Sigmoid()])
        l2 = list_cls([])
        input = torch.randn(16)
        result = f(input, l1, l2)
        self.assertEqual(result, l1(input))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        cnt.clear()
        l3 = list_cls([torch.nn.SiLU()])
        expected = l3(l1(input))
        result = f(input, l1, l3)
        self.assertEqual(len(l1), 3)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)

    def test_merge_criteria_processor_list1(self):
        # [FUNCTION DOC] Runs the shared merge criteria test with CustomList1 and no fullgraph.
        self.common_merge_criteria_processor_list(CustomList1, False)

    def test_merge_criteria_processor_list2(self):
        # [FUNCTION DOC] Runs the shared merge criteria test with CustomList2 and fullgraph.
        self.common_merge_criteria_processor_list(CustomList2, True)

    def test_restricted_list_subclass1(self):
        # [FUNCTION DOC] Compiles with fullgraph a function using a list subclass: extend, append,
        # pop of the first element and a custom method; checks the sum of the result equals x + y
        # + 20 and 3 ops are produced.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a CustomList2, extends and appends entries, pops the
            # first, appends ten times its length and returns the builtin sum.
            l = CustomList2()
            l.extend([True])
            l.append(a)
            l.extend([b])
            l.pop(0)
            l.append(l.length_times_10())
            return sum(l)

        x = torch.randn(10)
        y = torch.randn(10)
        self.assertEqual(fn(x, y), x + y + 20)
        self.assertEqual(cnt.op_count, 3)

    def test_restricted_list_subclass2(self):
        # [FUNCTION DOC] Checks that constructing two list subclass instances and extending one
        # with the other works under fullgraph and the returned object keeps the subclass type,
        # length 2 and the expected contents.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Creates two CustomList2 objects holding a+1 and b+2, extends
            # the first by the second and returns it.
            l1 = CustomList2([a + 1])
            l2 = CustomList2([b + 2])
            l1.extend(l2)
            return l1

        x = torch.randn(10)
        y = torch.randn(10)
        z = fn(x, y)
        self.assertEqual(type(z), CustomList2)
        self.assertEqual(len(z), 2)
        self.assertEqual(z.length_times_10(), 20)
        self.assertEqual(list(z), [x + 1, y + 2])

    def test_restricted_list_subclass3(self):
        # [FUNCTION DOC] Checks mutation of a list subclass input passed twice (aliasing) under
        # fullgraph: extend with itself, append twice and append once; verifies the final length 7
        # and each element or identity.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a: CustomList2, b: CustomList2):
            # [FUNCTION DOC] HELPER: Extends a by b, appends b's third element plus one twice
            # using the custom method, appends b's fourth element plus two and returns b.
            a.extend(b)
            a.append_twice(b[2] + 1)
            a.append(b[3] + 2)
            return b

        x = torch.randn(10)
        y = torch.randn(10)
        l = CustomList2([x, y])
        self.assertIs(fn(l, l), l)
        self.assertEqual(len(l), 7)
        self.assertIs(l[0], x)
        self.assertIs(l[1], y)
        self.assertIs(l[2], x)
        self.assertIs(l[3], y)
        self.assertEqual(l[4], x + 1)
        self.assertIs(l[5], l[4])
        self.assertEqual(l[6], y + 2)

    def test_rewrite_assert_with_msg(self):
        # [FUNCTION DOC] Checks that an assert statement with a message on a tensor value compiles
        # with fullgraph (6 ops, one frame) and that export also works with it.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes sine, asserts the first element equals 3 with a
            # message, and returns cosine plus sine.
            b = x.sin()
            assert x[0] == 3, "First dim need to be 3"  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        cnt = torch._dynamo.testing.CompileCounter()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_list_aliasing(self):
        # [FUNCTION DOC] Checks that appending to an input list inside a fullgraph compiled
        # function mutates the same list object: returned list is the input, has length 2 with the
        # sine of the first element appended; one frame and one op.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a):
            # [FUNCTION DOC] HELPER: Appends the sine of the first element to the list and
            # returns it.
            a.append(torch.sin(a[0]))
            return a

        x = torch.randn(10)
        l = [x]
        self.assertIs(fn(l), l)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], x)
        self.assertEqual(l[1], torch.sin(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_not_rewrite_assert_for_other_errors(self):
        # [FUNCTION DOC] Checks that a plain conditional raising ValueError (not an assert) keeps
        # its error type and message when compiled.
        def f(x):
            # [FUNCTION DOC] HELPER: Raises ValueError if the sum is greater than 3, otherwise
            # returns cosine plus sine.
            b = x.sin()
            if not x.sum() <= 3:
                raise ValueError("input sum needs to be 3")
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        opt_fn = torch.compile(f, backend="eager")
        with self.assertRaisesRegex(ValueError, "input sum needs to be 3"):
            opt_fn(*args)

    def test_rewrite_assert_dont_change_bytecode(self):
        # [FUNCTION DOC] Checks that an assert with an f-string message inside a no_grad block
        # compiles and matches eager without changing bytecode handling.
        def fn(x):
            # [FUNCTION DOC] HELPER: Under no_grad asserts the max is below 5 with a formatted
            # message, then returns sine of x.
            with torch.no_grad():
                assert x.max() < 5, f"invalid max {x.max()}"  # noqa: S101
                x = torch.sin(x)
            return x

        x = torch.ones(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_rewrite_assert_without_msg(self):
        # [FUNCTION DOC] Checks that an assert without a message exports correctly and the
        # exported module raises RuntimeError mentioning assertion error for failing input.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes sine, asserts the first element equals 3 and returns
            # cosine plus sine.
            b = x.sin()
            assert x[0] == 3  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported(torch.Tensor([5, 6, 7]))

    def test_rewrite_assert_with_non_string_msg(self):
        # [FUNCTION DOC] Checks that an assert with a non-string message (the tensor) raises
        # AssertionError mentioning the tensor and records a graph break named for assert with
        # non-string message exactly once in the counters; otherwise the test fails.
        def f(x):
            # [FUNCTION DOC] HELPER: Asserts the first element equals 2 with the tensor as
            # message, then returns cosine plus sine.
            b = x.sin()
            assert x[0] == 2, x  # noqa: S101
            return x.cos() + b

        torch._dynamo.utils.counters.clear()
        args = torch.Tensor([3, 4, 5])
        opt_f = torch.compile(f, backend="eager")
        with self.assertRaisesRegex(AssertionError, "tensor"):
            opt_f(args)
        for gb, cnt in torch._dynamo.utils.counters["graph_break"].items():
            if "assert with non-string message" in gb:
                self.assertEqual(cnt, 1)
                break
        else:
            # graph break not found
            self.assertTrue(False)

    def test_rewrite_assert_noop(self):
        # [FUNCTION DOC] Checks that trivial asserts (constant true and dtype comparison) export
        # correctly and compile with fullgraph into 3 ops and one frame, i.e. no assertion op
        # appears in the graph.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes sine, asserts True and that the dtype is float32,
            # returns cosine plus sine.
            b = x.sin()
            assert True  # noqa: S101
            assert x.dtype == torch.float32  # noqa: S101
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(*args), opt_f(*args)))
        # torch._assert shouldn't be in the graph
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(cnt.frame_count, 1)

        exported, _ = torch._dynamo.export(f)(torch.Tensor([4, 4, 5]))
        self.assertTrue(same(exported(*args), f(*args)))

    def test_size_typematch(self):
        # [FUNCTION DOC] Checks that isinstance against torch.Size specializes: a Size and a plain
        # tuple input each give the correct branch result and cause two frames in total.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns y plus 1 if x is a torch.Size, otherwise y plus 2.
            if isinstance(x, torch.Size):
                return y + 1
            else:
                return y + 2

        y = torch.zeros(1)
        x1 = torch.Size((3,))
        x2 = (3,)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertTrue(same(f(x1, y), opt_f(x1, y)))
        self.assertTrue(same(f(x2, y), opt_f(x2, y)))
        self.assertEqual(cnt.frame_count, 2)

    def test_hf_classinstantier(self):
        # [FUNCTION DOC] Checks a HuggingFace ClassInstantier style ordered dictionary that
        # instantiates module classes on item access, used with fullgraph compile for relu (with
        # kwargs) and tanh keys.
        # hf activations.py
        class ClassInstantier(collections.OrderedDict):
            # [CLASS DOC] HELPER: Ordered dictionary whose getitem builds a module by calling the
            # stored class with its stored kwargs (or none).
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Looks up the entry, splits it into a class and kwargs if
                # it is a tuple, and returns the class instantiated with the kwargs.
                content = super().__getitem__(key)
                cls, kwargs = content if isinstance(content, tuple) else (content, {})
                return cls(**kwargs)

        ACT2CLS = ClassInstantier(
            {
                "relu": (nn.ReLU, {"inplace": False}),
                "tanh": nn.Tanh,
            }
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, act):
            # [FUNCTION DOC] HELPER: Instantiates the activation for the given name and applies it
            # to x.
            return ACT2CLS[act](x)

        y = torch.randn(10)
        self.assertTrue(same(f(y, "tanh"), torch.tanh(y)))
        self.assertTrue(same(f(y, "relu"), torch.relu(y)))

    def test_ephemeral_module(self):
        # [FUNCTION DOC] Checks that a module defined locally and instantiated inside a compiled
        # function (HuggingFace squared relu) matches eager under fullgraph.
        # hf activations.py
        class ReLUSquaredActivation(nn.Module):
            # [CLASS DOC] HELPER: Module computing the square of relu of its input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies relu and squares the result.
                relu_applied = torch.nn.functional.relu(input)
                squared = torch.square(relu_applied)
                return squared

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Adds 0.2, applies a freshly created squared relu module and
            # adds 1.
            x = x + 0.2
            x = ReLUSquaredActivation()(x)
            x = x + 1
            return x

        y = torch.randn(10)
        self.assertTrue(same(f(y), ReLUSquaredActivation()(y + 0.2) + 1))

    def test_inplace_unsqueeze_input(self):
        # [FUNCTION DOC] Checks that in-place unsqueeze on a graph input is reflected in the
        # caller: backend sees the last tensor input with size 1, 3, 4; the returned output and
        # the original input both have that size.
        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Custom backend that asserts the last tensor example input has
            # shape 1, 3, 4 and returns the graph module unchanged.
            tensor_inputs = [x for x in example_inputs if isinstance(x, torch.Tensor)]
            self.assertEqual(tensor_inputs[-1].size(), torch.Size([1, 3, 4]))
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Unsqueezes x in place at dim 0 and returns x plus one.
            x.unsqueeze_(0)
            return x + 1

        inputs = [torch.randn(3, 4)]
        self.assertEqual(fn(*inputs).size(), torch.Size([1, 3, 4]))
        self.assertEqual(inputs[0].size(), torch.Size([1, 3, 4]))

    def test_batchnorm_e2e(self):
        # [FUNCTION DOC] Checks batch norm plus conv plus relu through aot_eager_decomp_partition
        # over three iterations (running stats change each time): outputs, gradients, parameters
        # and buffers must match the eager reference model.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a batch norm 2d layer of 64 channels and a 3 by 3
            # convolution with padding and no bias; forward returns relu of conv of batch norm in
            # a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the batch norm and convolution layers.
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                )
                self.conv1 = torch.nn.Conv2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies batch norm, convolution and relu and returns the
                # result in a one-element tuple.
                x1 = self.bn(x)
                x2 = self.conv1(x1)
                out = torch.nn.functional.relu(x2)
                return (out,)

        torch.manual_seed(1337)

        m_ref = Repro()
        m_test = deepcopy(m_ref)

        @torch.compile(backend="aot_eager_decomp_partition")
        def compiled_fn(x):
            # [FUNCTION DOC] HELPER: Compiled function calling the test copy of the model.
            return m_test(x)

        x_ref = torch.randn(2, 64, 32, 32, requires_grad=True)
        x_test = x_ref.clone()

        # Loop multiple times: each iteration the running_mean/var on batchnorm will update,
        # which changes the output of the next iteration
        for _ in range(3):
            ref = m_ref(x_ref)
            res = compiled_fn(x_test)

            self.assertTrue(same(ref, res))

            for r in ref:
                if r.requires_grad:
                    r.sum().backward()
            for r in res:
                if r.requires_grad:
                    r.sum().backward()

            for param_ref, param_test in zip(m_ref.parameters(), m_test.parameters()):
                self.assertTrue(same(param_ref, param_test))
            # Assert running_mean/var
            for buffer_ref, buffer_test in zip(m_ref.buffers(), m_test.buffers()):
                self.assertTrue(same(buffer_ref, buffer_test))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_right_side(self):
        # [FUNCTION DOC] With dynamic shapes by default, exports a function creating ones of size
        # five times the first dim at one shape and checks the graph gives the right shape on
        # another input.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns a ones tensor whose length is 5 times the
            # first dimension.
            return torch.ones(5 * x.shape[0])

        inp = torch.randn(6, 5)

        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))
        self.assertEqual(gm(inp).shape, f(inp).shape)

    @torch._dynamo.config.patch("specialize_int", False)
    def test_maybe_multiply_symint(self):
        # [FUNCTION DOC] With int specialization disabled (issue 97346), compiles through a custom
        # AOTAutograd backend a function adding two tensors with an integer alpha, using dynamic
        # shapes; checks results for alpha 5 and 6.
        # https://github.com/pytorch/pytorch/issues/97346
        from torch._functorch.aot_autograd import aot_module_simplified

        def my_aot_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that runs aot_module_simplified with a forward
            # compiler that returns the graph module forward method.
            def my_compiler(gm, example_inputs):
                # [FUNCTION DOC] HELPER: Forward compiler returning the forward function of the
                # graph module.
                return gm.forward

            # Invoke AOTAutograd
            return aot_module_simplified(gm, example_inputs, fw_compiler=my_compiler)

        def my_example(t1, t2, d):
            # [FUNCTION DOC] HELPER: Adds t1 and t2 scaled by d.
            out = torch.add(t1, t2, alpha=d)
            return out

        compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

        t1 = torch.arange(3, dtype=torch.float32).requires_grad_(True)
        t2 = torch.arange(3, dtype=torch.float32).requires_grad_(True)

        ra = compiled_fn(t1, t2, 5)
        self.assertEqual(ra, torch.tensor([0.0, 6.0, 12.0]))

        ra = compiled_fn(t1, t2, 6)
        self.assertEqual(ra, torch.tensor([0.0, 7.0, 14.0]))

    def test_build_map_unpack_with_call(self):
        # [FUNCTION DOC] Checks that building keyword arguments by merging two dictionaries and
        # unpacking them in a call compiles with fullgraph and gives sine plus 15.
        def forward_with_cond_scale(x, t, cond_scale, self_cond, other1, other2):
            # [FUNCTION DOC] HELPER: Returns the sum of sine of x and all the other
            # scalar arguments.
            return x.sin() + t + cond_scale + self_cond + other1 + other2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds two dictionaries, merges them and calls
            # forward_with_cond_scale with positional and keyword arguments plus the
            # merged dictionary.
            d1 = dict(other1=5)
            d2 = dict(other2=4)
            text_cond = {**d1, **d2}
            return forward_with_cond_scale(x, 1, cond_scale=2, self_cond=3, **text_cond)

        self.assertTrue(same(fn(torch.ones(4)), torch.ones(4).sin() + 15))

    @torch._dynamo.config.patch(verbose=True)
    def test_graph_break_unsupported_fake(self):
        # [FUNCTION DOC] With verbose Dynamo config, a custom op that has no fake implementation
        # causes a graph break; expects 2 ops and 2 frames.
        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x):
            # [FUNCTION DOC] HELPER: Calls the test_sample custom op foo on x plus one and
            # adds one.
            return torch.ops.test_sample.foo(x + 1) + 1

        f(torch.randn(3, device="cpu"))

        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 2)

    def test_delattr(self):
        # [FUNCTION DOC] Checks that deleting attributes (existing, newly added and on a newly
        # created object) works under fullgraph compile, including the attribute-exists test
        # after delete.
        class MyObj:
            # [CLASS DOC] HELPER: Simple object with attributes a and b.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores a and b.
                self.a = a
                self.b = b

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Deletes a from the input object, adds then deletes c, creates
            # a new object with x+2 and x+3 and deletes its b; returns x+1 if a still exists else
            # the new object.
            del obj.a
            obj.c = x + 1
            del obj.c
            tmp = MyObj(x + 2, x + 3)
            del tmp.b
            if hasattr(obj, "a"):
                return x + 1
            return tmp

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        obj2 = fn(x, obj1)
        self.assertFalse(hasattr(obj1, "a"))
        self.assertFalse(hasattr(obj1, "c"))
        self.assertFalse(hasattr(obj2, "b"))
        self.assertEqual(obj1.b.item(), 0)
        self.assertEqual(obj2.a.item(), 2)

    def test_delattr_return(self):
        # [FUNCTION DOC] Checks a user-defined attribute deletion hook: deleting val sets a flag
        # instead of removing the attribute, for two instances; under fullgraph the results are
        # (input plus one, True, True, True, True).
        class MyObject:
            # [CLASS DOC] HELPER: Object with val and a flag that records whether deletion of val
            # was attempted.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores val and sets the deletion flag to False.
                self.val = val
                self.deletion_attempted = False

            def __delattr__(self, attr):
                # [FUNCTION DOC] HELPER: When asked to delete val only sets the flag; other
                # attributes are deleted by the base class.
                if attr == "val":
                    self.deletion_attempted = True
                else:
                    super().__delattr__(attr)

        @torch.compile(fullgraph=True, backend="eager")
        def test_delattr(input_tensor):
            # [FUNCTION DOC] HELPER: Compiled function creating two MyObject instances, deleting
            # val on both, and returning x+1 with their val existence flags and deletion flags.
            instance_a = MyObject(1)
            instance_b = MyObject(2)
            del instance_a.val
            del instance_b.val
            exists_a = hasattr(instance_a, "val")
            exists_b = hasattr(instance_b, "val")
            deletion_attempted_a = instance_a.deletion_attempted
            deletion_attempted_b = instance_b.deletion_attempted
            return (
                input_tensor + 1,
                exists_a,
                exists_b,
                deletion_attempted_a,
                deletion_attempted_b,
            )

        result = test_delattr(torch.ones(1))
        self.assertEqual(result[0], torch.tensor([2.0]))
        self.assertEqual(result[1:], (True, True, True, True))

    def test_delattr_raises(self):
        # [FUNCTION DOC] Checks that deleting an attribute and then reading it inside a compiled
        # function raises AttributeError.
        class MyObj:
            # [CLASS DOC] HELPER: Simple object with attributes a and b.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores a and b.
                self.a = a
                self.b = b

        @torch.compile(backend="eager")
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Deletes a, adds one to x, then reads a (raises).
            del obj.a
            x = x + 1
            obj.a  # will raise
            return x

        x = torch.zeros([])
        obj1 = MyObj(x, x)
        self.assertRaises(AttributeError, lambda: fn(x, obj1))

    def test_delsubscr(self):
        # [FUNCTION DOC] Checks deleting a dictionary key inside a compiled function removes it
        # from the input dictionary and later reads work.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Deletes key a from the input dictionary and returns the b
            # entry plus one.
            del x["a"]
            y = x["b"] + 1
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        result = fn(x)
        self.assertFalse(hasattr(x, "a"))
        self.assertEqual(result.item(), 2)

    def test_delsubscr_raises(self):
        # [FUNCTION DOC] Checks that deleting a dictionary key and then reading it inside a
        # compiled function raises KeyError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Deletes key a then reads it, which should raise KeyError.
            del x["a"]
            y = x["a"] + 1  # should raise KeyError
            return y

        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        self.assertRaises(KeyError, lambda: fn(x))

    def test_attached_attribute_in_dir(self):
        # [FUNCTION DOC] Checks that attributes attached to a compiled module are visible in dir.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear and relu layers.
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then relu.
                return self.relu(self.linear(x))

        mod = torch.compile(MyModule(), backend="eager")
        mod.is_compiled = True
        self.assertTrue("is_compiled" in dir(mod))

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_shapes_implicit_guard(self):
        # [FUNCTION DOC] With automatic dynamic shapes off, compiles a function whose size query
        # depends on a shape value and checks that only one frame is used.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies x by the size at the dim given by its first
            # dimension, sums y over a dim equal to its first dim (result unused) and returns
            # the product.
            y = x * x.size(x.shape[0])
            torch.sum(y, [y.shape[0]])
            return y

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        opt_fn(torch.randn(3, 1, 1, 1, 1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dalle2_maybe(self):
        # [FUNCTION DOC] Checks the maybe wrapper (dalle2 pattern) can be called inside a
        # fullgraph compiled function with a function argument.
        def normalize(x):
            # [FUNCTION DOC] HELPER: Returns cos of x.
            return x.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, normalize_img):
            # [FUNCTION DOC] HELPER: Takes the sine and applies the maybe wrapped function to it.
            lowres_cond_img = x.sin()
            lowres_cond_img = maybe(normalize_img)(lowres_cond_img)
            return lowres_cond_img

        self.assertEqual(fn(torch.ones([]), normalize), torch.ones([]).sin().cos())

    def test_functools_wraps(self):
        # [FUNCTION DOC] Checks that a function created with functools.wraps inside a compiled
        # function and returned keeps the wrapped name and works when called afterward.
        def cool_name(x):
            # [FUNCTION DOC] HELPER: Returns sine of x.
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the cosine, defines a wrapper that applies cool_name to
            # it with wraps metadata and returns the wrapper.
            y = x.cos()

            @functools.wraps(cool_name)
            def uncool_name():
                # [FUNCTION DOC] HELPER: Calls cool_name on the captured tensor.
                return cool_name(y)

            return uncool_name

        result = fn(torch.ones([]))
        self.assertEqual(result.__name__, "cool_name")
        self.assertEqual(result(), torch.ones([]).cos().sin())

    def test_dynamic_shapes_float_guard(self):
        # [FUNCTION DOC] Checks that a dropout probability computed from a shape under dynamic
        # specialization compiles in one frame.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies dropout with probability equal to the first dim
            # divided by 6.
            return torch.nn.functional.dropout(x, x.shape[0] / 6)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cnt, fullgraph=True)
        opt_fn(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensor_item(self):
        # [FUNCTION DOC] With scalar output capture, exports a function using item at aten level
        # and checks the exported graph matches eager for two different scalar values.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Reads a scalar from y and adds it to the sum of x.
            val = y.item()
            return x.sum() + val

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
            torch.tensor(1),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(1)),
            gm(torch.zeros(6, 4), torch.tensor(1)),
        )
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(2)),
            gm(torch.zeros(6, 4), torch.tensor(2)),
        )

    def test_dataclass_init_with_default_factory_with_inputs(self):
        # [FUNCTION DOC] Checks a dataclass with a default factory field that is initialized from
        # an input list works under fullgraph and the stored lists match eager for three inputs of
        # different content and length.
        @dataclasses.dataclass
        class DClass:
            # [CLASS DOC] HELPER: Dataclass with a list-default field sharding_contexts and an int
            # field a with default 1.
            sharding_contexts: Any = dataclasses.field(default_factory=list)
            a: int = 1

        def fn(x, inp_list):
            # [FUNCTION DOC] HELPER: Creates the dataclass from the list, appends the sine of x
            # plus the field a to its list and returns it.
            d = DClass(inp_list)
            d.sharding_contexts.append(x.sin() + d.a)
            return d

        x = torch.randn(4)
        inp_list1 = [1, 2, 3]
        inp_list2 = [2, 3, 4]
        inp_list3 = [1, 2]
        ref1 = fn(x, inp_list1)
        ref2 = fn(x, inp_list2)
        ref3 = fn(x, inp_list3)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")

        opt_ret1 = opt_fn(x, inp_list1)
        opt_ret2 = opt_fn(x, inp_list2)
        opt_ret3 = opt_fn(x, inp_list3)
        self.assertEqual(ref1.sharding_contexts, opt_ret1.sharding_contexts)
        self.assertEqual(ref2.sharding_contexts, opt_ret2.sharding_contexts)
        self.assertEqual(ref3.sharding_contexts, opt_ret3.sharding_contexts)

    def test_list_index(self):
        # [FUNCTION DOC] Checks the index method on a list, tuple, torch.Size, deque and a
        # namedtuple for no, one and two bounds arguments, compiled with fullgraph; each should
        # find the value 3 at position 2.
        for i, list_type in enumerate(
            (
                list,
                tuple,
                torch.Size,
                collections.deque,
                namedtuple("FourElems", "one two three four", defaults=[0, 0, 0, 0]),
            )
        ):
            torch._dynamo.reset()
            for index in ([], [2], [0, 3]):

                def f(t):
                    # [FUNCTION DOC] HELPER: Builds the sequence of the current type (namedtuple
                    # built by positional args), calls its index method for 3 with the bounds
                    # arguments and returns the input plus the index.
                    if i == 4:  # namedtuple
                        xs = list_type(1, 2, 3, 4)
                    else:
                        xs = list_type([1, 2, 3, 4])
                    res = xs.index(3, *index)
                    return t + res

                res = torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

                self.assertEqual(res, torch.tensor([2.0]))

    def test_list_index_not_found(self):
        # [FUNCTION DOC] Checks that index of a missing item raises the Dynamo Unsupported error
        # under fullgraph.
        def f(t):
            # [FUNCTION DOC] HELPER: Looks up a missing string in a list and adds the result.
            xs = ["bar", "foo", "baz", "buzz"]
            res = xs.index("non-existent")
            return t + res

        # Raising ValueError from item not found is unsupported
        with self.assertRaises(
            torch._dynamo.exc.Unsupported,
        ):
            torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

    def test_list_index_tensor_unsupported(self):
        # [FUNCTION DOC] Checks that index on a list of tensors raises Unsupported mentioning
        # data-dependent branching for each combination of bounds.
        for index in ([], [2], [0, 3]):

            def f(t):
                # [FUNCTION DOC] HELPER: Builds a list of four one-element tensors, looks up a
                # tensor with the index method and adds.
                xs = [torch.tensor([i]) for i in range(4)]
                res = xs.index(torch.tensor([2]), *index)
                return t + res

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "Data-dependent branching",
            ):
                torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(1))

    def test_hf_xsoftmax_inference(self):
        # [FUNCTION DOC] Compares a compiled function using the XSoftmax custom function (in
        # inference) with eager.
        def fn(input, mask):
            # [FUNCTION DOC] HELPER: Applies XSoftmax to x plus one with the mask along dim 1 and
            # adds 2.
            return XSoftmax.apply(input + 1, mask, 1) + 2

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)

        inputs = [
            torch.randn(4, 10),
            torch.randn(4, 10) < 0,
        ]
        expected = fn(*inputs)
        actual = fn_opt(*inputs)
        self.assertTrue(same(actual, expected))

    @mock.patch("torch._dynamo.config.guard_nn_modules", True)
    def test_hf_xsoftmax_training(self):
        # [FUNCTION DOC] With nn module guarding on, compiles XSoftmax in training mode: forward
        # gives 2 ops in one frame; backward results match; the backward is not traced (no frames,
        # no ops, no frame or graph break counters).
        from torch._dynamo.utils import counters

        counters.clear()

        def fn(input, mask):
            # [FUNCTION DOC] HELPER: Applies XSoftmax with the mask along dim 1.
            return XSoftmax.apply(input, mask, 1)

        cnt = torch._dynamo.testing.CompileCounter()
        fn_opt = torch.compile(fn, backend=cnt, fullgraph=False)

        torch.manual_seed(1234)
        inputs1 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]
        torch.manual_seed(1234)
        inputs2 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]

        expected = fn(*inputs1)
        actual = fn_opt(*inputs2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(cnt.frame_count, 1)
        cnt.clear()
        counters.clear()

        expected.sum().backward()
        actual.sum().backward()
        self.assertTrue(same(inputs1[0].grad, inputs2[0].grad))

        # currently we don't capture the backwards frame
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(cnt.op_count, 0)
        self.assertEqual(dict(counters["frames"]), {})
        self.assertEqual(dict(counters["graph_break"]), {})

    def test_autograd_function_graph_break(self):
        # [FUNCTION DOC] Checks that an autograd Function whose forward graph breaks works
        # compiled (eager backend): value matches sine and the gradient from autograd.grad
        # matches cosine.
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing sine, with a forced graph break in
            # forward, saving the input for backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Breaks the graph, saves x on ctx and returns sine of x.
                torch._dynamo.graph_break()
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gx):
                # [FUNCTION DOC] HELPER: Returns the incoming gradient times cosine of the
                # saved tensor.
                (x,) = ctx.saved_tensors
                return gx * x.cos()

        x = torch.randn([], requires_grad=True)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MySin to x.
            return MySin.apply(x)

        y = fn(x)
        self.assertEqual(y, x.sin())

        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_jit_trace_errors(self):
        # [FUNCTION DOC] Checks that jit trace of a function compiled with Dynamo
        # raises RuntimeError.
        @torch.compile(backend="eager", dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        with self.assertRaises(RuntimeError):
            torch.jit.trace(f, torch.randn(3))

    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        # [FUNCTION DOC] With dynamic shapes by default, exports a function splitting along dim 0
        # into chunks of half the first dim and checks the exported graph on a different shape.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first chunk of splitting x with chunk size equal
            # to half its first dimension.
            return torch.split(x, x.shape[0] // 2, dim=0)[0]

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
        )

        self.assertEqual(f(torch.ones(8, 4)), gm(torch.ones(8, 4)))

    @skipIfWindows(
        msg="TODO: (xuhancn) fix, AssertionError: tensor([[0.1000, 0.1000, 0.1000,  ..., 0.1000, 0.1000, 0.1000],"
    )
    def test_optim_state_references_cleared(self):
        # [FUNCTION DOC] Skipped on Windows. Checks that after compiling an optimizer step and
        # discarding the optimizer, the weak reference to its square_avg state is dead, i.e.
        # compiled code does not keep optimizer state alive.
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        state_ref = 0

        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            # [FUNCTION DOC] HELPER: Calls optimizer step.
            optimizer.step()

        compiled_opt_step = torch.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            # [FUNCTION DOC] HELPER: Zeroes grads, runs the model on x, backpropagates the sum and
            # runs the compiled optimizer step.
            optimizer.zero_grad()
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # Picked "square_avg" arbitrarily to check that
        # optimizer state tensors are deallocated
        state_ref = weakref.ref(
            optimizer.state[optimizer.param_groups[0]["params"][0]]["square_avg"]
        )
        optimizer = None

        self.assertIsNone(state_ref())

    def test_grad_references_cleared(self):
        # [FUNCTION DOC] Checks that gradients are released after zero_grad with set-to-none
        # even when the optimizer step was compiled: a weak reference to the gradient is
        # dead afterward.
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        def opt_step():
            # [FUNCTION DOC] HELPER: Calls optimizer step.
            optimizer.step()

        compiled_opt_step = torch.compile(opt_step, backend="eager")

        def compiled_model_step(x):
            # [FUNCTION DOC] HELPER: Zeroes grads (set to none), runs the model, backpropagates
            # the sum and runs the compiled optimizer step.
            optimizer.zero_grad(True)
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)
        param_grad_ref = weakref.ref(next(iter(model.parameters())).grad)
        optimizer.zero_grad(True)
        self.assertIsNone(param_grad_ref())

    def test_batch_encoding_clone_inputs(self):
        # [FUNCTION DOC] Checks that clone_inputs of a dict subclass with a custom attribute
        # getter returns an object that is not a plain dict (the subclass type is kept).
        class BatchEncoding(dict):
            # [CLASS DOC] HELPER: Dict subclass copied from test_tokenization with an attribute
            # getter reading from a data attribute.
            """
            Copied from test_tokenization
            """

            def __init__(
                self,
                data,
            ):
                # [FUNCTION DOC] HELPER: Calls the dict initializer with data.
                super().__init__(data)

            def __getattr__(self, item: str):
                # [FUNCTION DOC] HELPER: Returns the entry in the data attribute; converts
                # KeyError to AttributeError.
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        encoding = BatchEncoding({"key": torch.rand((1, 4))})
        cloned_encoding = torch._dynamo.utils.clone_inputs(encoding)
        self.assertTrue(type(cloned_encoding) is not dict)

    def test_iadd_graph_break(self):
        # [FUNCTION DOC] Checks that in-place addition on a tuple (building a new tuple) compiles
        # with fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Starts with an empty tuple, adds a one-element tuple holding
            # the sine of x in place and returns the tuple.
            a = ()
            x = torch.sin(x)
            a += (x,)
            return a

        x = torch.randn(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_odict_get_item_index_name(self):
        # [FUNCTION DOC] Checks that dictionary indexing with a dtype-like key objects (float and
        # numpy float16) works inside a compiled function.
        d = {float: torch.float32, np.float16: torch.float16}

        @torch.compile(backend="eager")
        def f(x, y1, y2):
            # [FUNCTION DOC] HELPER: Builds two zero tensors with dtypes looked up in the
            # outer dictionary.
            return torch.zeros(5, dtype=d[y1]), torch.zeros(5, dtype=d[y2])

        f(torch.zeros(4), float, np.float16)

    def test_dedup_global(self):
        # [FUNCTION DOC] Checks that a global CPU tensor referenced twice in a compiled function
        # is deduplicated and the result matches eager.
        @torch.compile(backend="eager")
        def f():
            # [FUNCTION DOC] HELPER: Returns the global tensor added to itself.
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    def test_randint_out_dynamic(self):
        # [FUNCTION DOC] Checks randint with an out tensor can be called under compile with two
        # different out sizes (dynamic shape change).
        def randint_fn(high, size, out):
            # [FUNCTION DOC] HELPER: Calls randint with high, size and out.
            return torch.randint(high, size, out=out)

        opt_model = torch.compile(randint_fn, backend="eager")

        out1 = torch.empty(10, dtype=torch.int32)
        opt_model(17, (10,), out1)

        out2 = torch.empty(12, dtype=torch.int32)
        opt_model(17, (12,), out2)

    def test_list_self_reference(self):
        # [FUNCTION DOC] Regression test for issue 100150: a compiled function reading from a
        # self-referencing list global must not hang or fail.
        # Issue - https://github.com/pytorch/pytorch/issues/100150
        root = []
        root[:] = [root, root, None, None]

        @torch.compile(fullgraph=False, backend="eager")
        def test_bug():
            # [FUNCTION DOC] HELPER: Returns the first element of the global
            # self-referencing list.
            return root[0]

        test_bug()

    def test_hf_bigbird_unsqueeze(self):
        # [FUNCTION DOC] Mimics a BigBird pattern: two batched matmul helper calls (each preceded
        # by a graph break) with in-place unsqueeze, an add and a concatenation; compiled with a
        # frame counter, expects two frames.
        # KNOWN ISSUE: the helper ignores its parameters and reads variables of the enclosing
        # test scope.
        def torch_bmm_nd(inp_1, inp_2, ndim=None):
            # [FUNCTION DOC] HELPER: Forces a graph break and returns the batched matmul of the
            # enclosing-scope tensors inp1 and inp2 rather than its own two parameters (the ndim
            # argument is also unused).
            torch._dynamo.graph_break()
            return torch.bmm(inp1, inp2)

        def fn(inp1, inp2, inp3, inp4, c):
            # [FUNCTION DOC] HELPER: Calls the matmul helper twice, unsqueezes both results in
            # place at dim 2, doubles the first, sums them and concatenates both with c along dim
            # 2; returns the concatenation and the sum.
            a = torch_bmm_nd(inp1, inp2, 4)
            a.unsqueeze_(2)
            a = a * 2

            b = torch_bmm_nd(inp3, inp4, 4)
            b.unsqueeze_(2)
            l = a + b

            out = torch.cat([a, b, c], dim=2)
            return out, l

        inp1 = torch.rand(1, 64, 448)
        inp2 = torch.rand(1, 448, 64)
        inp3 = torch.rand(1, 64, 448)
        inp4 = torch.rand(1, 448, 64)
        c = torch.rand(1, 64, 1, 64)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(inp1, inp2, inp3, inp4, c)
        self.assertEqual(cnt.frame_count, 2)

    def test_torch_variable_type(self):
        # [FUNCTION DOC] Checks a torchvision-style type check helper (isinstance for types, call
        # for predicates) behaves the same compiled as in eager for a tensor and the Tensor class.
        # from torchvision
        def check_type(obj, types_or_checks):
            # [FUNCTION DOC] HELPER: Returns True when obj matches any entry in the list, using
            # isinstance for type entries and calling the entry otherwise; returns False if
            # none match.
            for type_or_check in types_or_checks:
                if (
                    isinstance(obj, type_or_check)
                    if isinstance(type_or_check, type)
                    else type_or_check(obj)
                ):
                    return True
            return False

        opt_check_type = torch.compile(check_type, backend="eager")
        ref = check_type(torch.randn(4), [torch.Tensor])
        res = opt_check_type(torch.randn(4), [torch.Tensor])
        self.assertEqual(ref, res)

    # Test for https://github.com/pytorch/pytorch/issues/103132
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_inference_mode_dynamic_shapes(self):
        # [FUNCTION DOC] Regression test for issue 103132: with dynamic shapes by default, a
        # compiled (aot_eager, dynamic) module doing a 3D matmul runs under inference_mode
        # without error.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the matmul of its input
            # with itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, param):
                # [FUNCTION DOC] HELPER: Returns the matmul of param with itself.
                z = torch.matmul(param, param)
                return z

        model = Repro()
        # Need a 3d tensor to actually cause the error:
        # we go down a path of the C++ matmul decomp that calls sizes().
        inp = torch.randn(4, 4, 4, requires_grad=True)
        model = torch.compile(model, backend="aot_eager", dynamic=True)
        with torch.inference_mode():
            model(inp)

    def test_kwargs_out_list_variable(self):
        # [FUNCTION DOC] Checks calling torch.frexp with a keyword dictionary whose out entry is
        # a list of two preallocated tensors; the compiled module returns the expected mantissa
        # and exponent.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls frexp with the dictionary as
            # keyword arguments.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, param):
                # [FUNCTION DOC] HELPER: Returns the result of frexp called with the keyword
                # arguments in param.
                z = torch.frexp(**param)
                return z

        model = Repro()
        params = {"input": torch.tensor([[0.0, 1, 2, 4]])}
        params["out"] = [
            torch.empty(0, dtype=torch.float32),  # mantissa
            torch.empty(0, dtype=torch.int32),  # exponent
        ]

        model = torch.compile(model, backend="eager")
        mantissa, exponent = model(params)
        ref_mantissa = torch.tensor([[0.0000, 0.5000, 0.5000, 0.5000]])
        ref_exponent = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        self.assertEqual(ref_mantissa, mantissa)
        self.assertEqual(ref_exponent, exponent)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_aot_autograd(self):
        # [FUNCTION DOC] With scalar output capture, compiles split_with_sizes using sizes from a
        # tensor with aot_eager and fullgraph; result must match eager.
        def fn(result, split_sizes):
            # [FUNCTION DOC] HELPER: Converts the sizes tensor to a Python list and splits the
            # first tensor with those sizes.
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        actual = torch.compile(fn, fullgraph=True, backend="aot_eager")(*example_inputs)
        expected = fn(*example_inputs)
        self.assertEqual(actual, expected)

    def test_unspecialized_nn_module_with_torch_variable_attribute(self):
        # [FUNCTION DOC] Checks a module with forced dynamic (unspecialized) behavior holding a
        # torch function as attribute and called with keyword arguments including a string;
        # compiled with fullgraph it must match eager (the attribute must be treated as a torch
        # function, not traced through).
        """
        In this case self.fn = something that should be a TorchVariable.
        When it's not a TorchVariable, dynamo tries to trace through and fails.
        This makes sure that the self.fn is handled as a TorchVariable.
        """

        class UserModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module marked to be treated as an unspecialized nn module,
            # storing a callable and calling it with its keyword arguments in forward.
            torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Calls the base initializer and stores fn.
                super().__init__()
                self.fn = fn

            def forward(self, **inp):
                # [FUNCTION DOC] HELPER: Returns the stored function called with the
                # keyword arguments.
                return self.fn(**inp)

        inputs = {
            "input": torch.randn([2, 9]).uniform_(0, 1),
            "target": torch.randn([2, 9]).uniform_(0, 1),
            "reduction": "mean",
        }

        mod = UserModule(torch.nn.functional.binary_cross_entropy)
        ref = mod(**inputs)
        res = torch.compile(mod, backend="eager", fullgraph=True)(**inputs)
        self.assertEqual(ref, res)

    def test_string_format(self):
        # [FUNCTION DOC] Checks that calling the format method of a global string with a keyword
        # argument is evaluated at compile time under fullgraph, selecting the sine branch.
        s = "temp{i}"

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of x if the formatted string equals temp4, else
            # cos of x.
            if s.format(i=4) == "temp4":
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(4)
        self.assertEqual(fn(x), torch.sin(x))

    @unittest.skip("Fails with incorrect result with fullgraph constraints")
    def test_int_format(self):
        # [FUNCTION DOC] Skipped test (reason: incorrect result with fullgraph constraints). Would
        # compare the builtin format of an int to binary under fullgraph and non-dynamic compile.
        def fn(num: int):
            # [FUNCTION DOC] HELPER: Returns the binary string format of the integer.
            return format(num, "b")

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        self.assertEqual(fn(10), opt_fn(10))

    # Repro of torch._dynamo.exc.InternalTorchDynamoError: 'NoneType' object has no attribute 'guards'
    # due to bad empty list handling
    def test_empty_list_contains_with_jump(self):
        # [FUNCTION DOC] Repro for an InternalTorchDynamoError caused by bad empty list
        # handling: a membership test of a tensor in an empty list followed by a branch compiles
        # into one frame.
        def fn(x, l):
            # [FUNCTION DOC] HELPER: Returns cos of x if x is a member of the list l, otherwise
            # sin of x.
            if x in l:
                return x.cos()
            return x.sin()

        counter = CompileCounter()
        torch.compile(fn, backend=counter)(torch.randn([2, 2]), [])
        self.assertEqual(counter.frame_count, 1)

    def test_get_type_hints(self):
        # [FUNCTION DOC] Checks that calling typing.get_type_hints on a user class inside a
        # fullgraph compiled function works and the result matches eager.
        class Foo:
            # [CLASS DOC] HELPER: Empty class used as a type hints target.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls get_type_hints on Foo (result ignored) and returns sin
            # of x.
            typing.get_type_hints(Foo, include_extras=True)
            return torch.sin(x)

        x = torch.randn(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_graph_break_on_jit_isinstance(self):
        # [FUNCTION DOC] Checks that torch.jit.isinstance with a typing List annotation in a
        # compiled function falls back (graph break) and gives the same result as eager.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x doubled if torch.jit.isinstance says x is a list of
            # strings, else x.
            if torch.jit.isinstance(x, typing.List[str]):  # noqa: UP006
                return x * 2
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_graph_break_on_jit_isinstance_pep585(self):
        # [FUNCTION DOC] Same as the previous test but with a builtin generic list annotation.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x doubled if torch.jit.isinstance says x is a list of
            # strings (builtin generic), else x.
            if torch.jit.isinstance(x, list[str]):
                return x * 2
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand(4)
        self.assertTrue(same(fn(x), opt_fn(x)))

    def test_add_sub_alpha_out(self):
        # [FUNCTION DOC] Checks that add and sub with an alpha keyword and an out tensor, with
        # tensor and Python scalar inputs, give the same out values when compiled dynamically
        # as eager.
        test_cases = (
            (torch.randn(2, 3, 4), 1, 2, torch.zeros(2, 3, 4)),
            (2, 1.1, 0.4, torch.tensor(0.0)),
        )
        for op in [torch.add, torch.sub]:
            for inp, other, alpha, out in test_cases:
                compiled_fn = torch.compile(op, dynamic=True, backend="eager")
                eager_out = out.clone()
                compiled_out = out.clone()
                op(inp, other, alpha=alpha, out=eager_out)
                compiled_fn(inp, other, alpha=alpha, out=compiled_out)
                self.assertTrue(same(eager_out, compiled_out))

    def test_negative_shape_guard(self):
        # [FUNCTION DOC] Checks a guard on tensor size comparison against a tuple with dynamic
        # shapes: two inputs with different shapes cause two frames and match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns cos of x if its size differs from 5, 1, 2, 3,
            # otherwise sin of x.
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, dynamic=True)

        x = torch.ones(5, 1, 3, 4)
        x2 = torch.ones(5, 1, 2, 3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_deferred_runtime_asserts(self):
        # [FUNCTION DOC] With scalar output capture, a torch check on a scalar read from a tensor
        # inside a fullgraph compiled function: passes for 3 and raises RuntimeError for -2.
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Reads a scalar from x, checks it is non-negative via torch
            # check and returns x doubled or tripled by branch.
            y = x.item()
            torch._check(y >= 0)
            if y >= 0:
                return x * 2
            else:
                return x * 3

        f(torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: f(torch.tensor([-2])))

    def test_addr_alpha_beta_out(self):
        # [FUNCTION DOC] Checks addr with alpha, beta and out tensor gives the same result
        # compiled dynamically as eager.
        inp = torch.randn(2, 3)
        vec1 = torch.randn(2)
        vec2 = torch.randn(3)
        alpha = 2
        beta = 5

        out = torch.zeros(2, 3)
        compile_out = torch.zeros(2, 3)

        torch.addr(inp, vec1, vec2, alpha=alpha, beta=beta, out=out)
        compiled_fn = torch.compile(torch.addr, dynamic=True, backend="eager")
        compiled_fn(inp, vec1, vec2, alpha=alpha, beta=beta, out=compile_out)
        self.assertTrue(same(out, compile_out))

    def test_setattr_requires_grad_graph_breaks(self):
        # [FUNCTION DOC] For a frame counter backend, eager and aot_eager, checks that setting
        # requires_grad on an input inside a function causes a graph break (two frames for the
        # counter backend), with matching outputs, backward and final tensor values.
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes x plus 4, sets requires_grad on x and returns x
            # times that value.
            z = x + 4
            x.requires_grad = True
            y = x * z
            return y

        for backend in ["count", "eager", "aot_eager"]:
            if backend == "count":
                backend = CompileCounter()
            opt_fn = torch.compile(fn, backend=backend)

            eager = torch.zeros(5)
            compiled = eager.clone()

            out_eager = fn(eager)
            out_opt = opt_fn(compiled)

            self.assertEqual(out_eager, out_opt)

            out_eager.sum().backward()
            out_opt.sum().backward()

            self.assertEqual(eager, compiled)
            if isinstance(backend, CompileCounter):
                self.assertEqual(backend.frame_count, 2)  # graph breaks

    def test_dynamic_shapes_double_not_equal(self):
        # [FUNCTION DOC] Regression test for issue 113393: size comparison with a tuple inside a
        # compiled function gives the same result as eager for inputs matching and not matching.
        # https://github.com/pytorch/pytorch/issues/113393
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns cos of x if its size differs from 5, 1, 2, 3,
            # otherwise sin of x.
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            return x.sin()

        opt_fn = torch.compile(fn, backend="eager")

        x = torch.ones(5, 1, 2, 3)
        x2 = torch.ones(5, 1, 3, 4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))

    def test_user_defined_object_callable(self):
        # [FUNCTION DOC] Regression test for issue 114019: instantiating a user-defined
        # callable class inside a compiled function (object has no source) and calling it works
        # with fullgraph.
        # https://github.com/pytorch/pytorch/issues/114019
        class MyCallable:
            # [CLASS DOC] HELPER: Class whose call adds one to its argument.
            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a callable object inside the graph and calls it on x.
            # Create in graph - will not have source
            return MyCallable()(x)

        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn_opt(torch.zeros(1)), fn(torch.zeros(1)))

    @torch._dynamo.config.patch(log_compilation_metrics=True)
    def test_many_views_with_mutation(self):
        # [FUNCTION DOC] With compilation metrics logging on, compiles with aot_eager a function
        # mutating a view of a list of 32 overlapping strided views and checks the total guard
        # counts (including shape env guards) stay under 8 per input, i.e. not quadratic.
        # When symbolic storage offsets were added in #113734, tensors_definitely_do_not_overlap
        # began adding shape guards - a quadratic amount relative to the number of inputs.
        # Test this configuration, and test that a reasonable number of guards are added.
        # Note, when dynamic shapes are turned on, this test fails and we still get quadratic guards.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies relu in place to the first list element and returns
            # the sum of the concatenation of all.
            x[0].relu_()
            return torch.cat(x).sum()

        AMT = 32
        src = torch.rand(16 * (AMT + 1))

        x = [src.as_strided((4, 4), (4, 1), 3 + 16 * i) for i in range(AMT)]

        torch._dynamo.reset()
        torch._dynamo.utils.clear_compilation_metrics()

        torch.compile(fn, backend="aot_eager")(x)

        all_metrics = torch._dynamo.utils.get_compilation_metrics()

        total_guards = sum(metric.guard_count for metric in all_metrics)
        self.assertLess(total_guards, AMT * 8)

        total_shape_env_guards = sum(
            metric.shape_env_guard_count for metric in all_metrics
        )
        self.assertLess(total_shape_env_guards, AMT * 8)

    # https://github.com/pytorch/pytorch/issues/118799
    def test_subclass_graph_output_repro(self):
        # [FUNCTION DOC] Regression test for issue 118799: a function allowed in graph that builds
        # a TwoTensor subclass and a view of it returns correctly under aot_eager.
        @torch._dynamo.allow_in_graph
        def to_subclass(x):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function returning a TwoTensor made from two
            # clones of x.
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            # [FUNCTION DOC] HELPER: Builds the subclass from x and returns its flattened view.
            tmp_subclass = to_subclass(x)
            return tmp_subclass.view(-1)

        x = torch.ones(2)
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(out_ref, out_test)

    def test_numpy_tobytes_no_error(self):
        # [FUNCTION DOC] Checks that calling tobytes on a numpy array between two in-place
        # increments compiles into two frames and gives the same bytes as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one in place, converts to bytes, adds one in place again
            # and returns the bytes.
            x += 1
            z = x.tobytes()
            x += 1
            return z

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_arg, arg = np.array([1, 2]), np.array([1, 2])
        self.assertEqual(opt_fn(opt_arg), fn(arg))
        self.assertEqual(cnt.frame_count, 2)

    def test_numpy_not_ndarray_recompiles(self):
        # [FUNCTION DOC] Checks that a function taking different argument types (numpy array,
        # nothing, int, string) recompiles for each, giving frame counts 1, 2, 3 and 4 and
        # matching eager.
        import torch

        def fn(x=None):
            # [FUNCTION DOC] HELPER: Builds a numpy ones array of size 3, 6 or 9 depending on
            # whether x is None, int or str (otherwise uses x itself) and returns its square.
            if x is None:
                x = np.ones(3)
            elif isinstance(x, int):
                x = np.ones(6)
            elif isinstance(x, str):
                x = np.ones(9)
            return x**2

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)

        x = np.zeros((2, 2))

        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(opt_fn(), fn())
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(10), fn(10))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(opt_fn("10"), fn("10"))
        self.assertEqual(cnt.frame_count, 4)

    @parametrize(
        "backend",
        ["eager", "aot_eager", "inductor"],
    )
    @parametrize(
        "func_name",
        ["func1", "func2", "func3"],
    )
    def test_tensor_set_data(self, backend, func_name):
        # [FUNCTION DOC] Parametrized over backend (eager, aot_eager, inductor) and function name;
        # regression test for issue 113030 about assigning to the data attribute of tensors. Runs
        # the compiled function five times against eager, comparing inputs, outputs and gradients,
        # and expects the compiled function to stay at one frame. Returns early for the in-place
        # variant on non-eager backends.
        # https://github.com/pytorch/pytorch/issues/113030
        def func1(x, y):
            # [FUNCTION DOC] HELPER: Sets x data to y, adds one to x in place and returns x.
            x.data = y
            x.add_(1)
            return x

        def func2(x, y):
            # [FUNCTION DOC] HELPER: Sets x data to y, then sets y data to an empty tensor and
            # returns x.
            x.data = y
            y.data = torch.zeros([0])
            return x

        def func3(x, y):
            # [FUNCTION DOC] HELPER: Keeps a reference to x, sets x data to y, empties y data and
            # returns a tensor telling whether x is still the same object.
            z = x
            x.data = y
            y.data = torch.zeros([0])
            return torch.tensor(x is z)

        funcs = {"func1": func1, "func2": func2, "func3": func3}
        func = funcs[func_name]

        if backend != "eager" and func is func1:
            # add_ not working w/ aot_autograd?
            return

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        compiled_fn = torch.compile(func, backend=cnt, fullgraph=True)
        requires_grad = func is not func1
        for _ in range(5):
            # Inputs
            eager_a = torch.ones([6], requires_grad=requires_grad)
            compiled_a = torch.ones([6], requires_grad=requires_grad)

            eager_b = torch.ones([6], requires_grad=requires_grad)
            compiled_b = torch.ones([6], requires_grad=requires_grad)

            # Eager
            out_eager = func(eager_a, eager_b)
            # Compiled
            out_compiled = compiled_fn(compiled_a, compiled_b)
            self.assertEqual(eager_a, compiled_a)
            self.assertEqual(eager_b, compiled_b)
            self.assertTrue(torch.equal(out_eager, out_compiled))

            # func1 hits a leaf Variable that requires grad is being used in an in-place operation
            if requires_grad:
                bwd_inp_eager = torch.randn([6])
                bwd_inp_compiled = torch.clone(bwd_inp_eager)
                eager_a.backward(bwd_inp_eager)
                compiled_a.backward(bwd_inp_compiled)
                self.assertEqual(eager_a.grad, compiled_a.grad)

        # Prove guarding works - we run the compiled_fn 5 times
        # frame_count should stay at 1.
        self.assertEqual(cnt.frame_count, 1)

    def test_tensor_set_data_mismatched_dtype(self):
        # [FUNCTION DOC] Checks that assigning data of a different dtype (converted to bfloat16)
        # to a tensor behaves the same compiled (eager backend) as in eager for the target, its
        # data and the source.
        def func(x, y):
            # [FUNCTION DOC] HELPER: Sets x data to y converted to bfloat16.
            x.data = y.to(dtype=torch.bfloat16)

        x1 = torch.tensor([], dtype=torch.float32)
        x2 = torch.tensor([], dtype=torch.float32)
        y1 = torch.tensor([1, 2, 3], dtype=torch.float32)
        y2 = torch.tensor([1, 2, 3], dtype=torch.float32)
        func(x1, y1)
        torch.compile(func, backend="eager")(x2, y2)
        self.assertEqual(x1, x2)
        self.assertEqual(x1.data, x2.data)
        self.assertEqual(y1, y2)

    @requires_cuda
    def test_tensor_set_data_cross_device(self):
        # [FUNCTION DOC] Requires CUDA. Checks that moving the data of a CPU tensor to cuda via
        # data assignment matches eager under fullgraph and the tensor keeps the same device.
        def func(x):
            # [FUNCTION DOC] HELPER: Replaces x data with its cuda copy and returns x plus one.
            x.data = x.data.to("cuda")
            return x + 1

        x_eager = torch.randn(4, device="cpu")
        x_compiled = x_eager.clone()

        out_eager = func(x_eager)
        out_compiled = torch.compile(func, backend="eager", fullgraph=True)(x_compiled)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(x_eager.device, x_compiled.device)

    @requires_cuda
    def test_tensor_set_data_cross_device_shape_mismatch_graphbreaks(self):
        # [FUNCTION DOC] Requires CUDA. Checks that replacing data with a cuda tensor of a
        # different shape raises the Dynamo Unsupported error under fullgraph.
        def func(x):
            # [FUNCTION DOC] HELPER: Sets x data to a new cuda random tensor of size 8 and returns
            # x plus one.
            x.data = torch.randn(8, device="cuda")
            return x + 1

        x = torch.randn(4, device="cpu")
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(func, backend="eager", fullgraph=True)(x)

    @requires_cuda
    def test_tensor_set_data_cross_device_placeholder_metadata(self):
        # [FUNCTION DOC] Requires CUDA. Checks that in the recorded graph, placeholder example
        # values that are tensors are on cpu after a data assignment moving the data to cuda.
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        def func(x):
            # [FUNCTION DOC] HELPER: Replaces x data with its cuda copy and returns x plus one.
            x.data = x.data.to("cuda")
            return x + 1

        x = torch.randn(4, device="cpu")
        torch.compile(func, backend=backend, fullgraph=True)(x)

        gm = backend.graphs[0]
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                ev = node.meta.get("example_value")
                if isinstance(ev, torch.Tensor):
                    self.assertEqual(ev.device.type, "cpu")

    def test_user_ctor_ctx_manager(self):
        # [FUNCTION DOC] Checks that creating a user context manager object inside a fullgraph
        # compiled function (without entering it) compiles into one frame.
        class UserCtxManager:
            # [CLASS DOC] HELPER: Context manager class whose enter returns 1 and whose exit
            # does nothing.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns 1.
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                # [FUNCTION DOC] HELPER: Does nothing and does not suppress exceptions.
                pass

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Creates the context manager (unused) and returns x squared.
            ucm = UserCtxManager()  # noqa: F841
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.rand([2, 2])
        opt_fn(x, x)
        self.assertExpectedInline(cnt.frame_count, """1""")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_arange_in_bounds(self):
        # [FUNCTION DOC] With scalar output capture (issue 113002), a module building an arange up
        # to an unbacked maximum and a padding mask compiles with fullgraph into one frame.
        # see https://github.com/pytorch/pytorch/issues/113002
        class PaddingNet(nn.Module):
            # [CLASS DOC] HELPER: Module that builds a float mask of shape batch, max length, 1
            # from a lengths tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, lengths):
                # [FUNCTION DOC] HELPER: Takes the maximum length as a scalar, compares an arange
                # against the lengths column, casts to float32 and adds a trailing dim.
                max_seq_len = lengths.max().item()
                row_vector = torch.arange(0, max_seq_len, 1)
                matrix = torch.unsqueeze(lengths, dim=-1)
                mask = row_vector < matrix
                mask = mask.type(torch.float32)
                mask_3d_btd = mask[:, :, None]
                return mask_3d_btd

        model = PaddingNet()
        lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(model, backend=cnt, fullgraph=True)
        opt_fn(lengths)
        self.assertEqual(cnt.frame_count, 1)

    def test_overlapping_inputs_with_dynamic_shapes_error(self):
        # [FUNCTION DOC] Intended to check that overlapping inputs with dynamic shapes raise
        # an error.
        # KNOWN ISSUE: the code that builds the overlapping views and calls the compiled function
        # is inside the body of the compiled function itself, so the test never calls it and
        # asserts nothing.
        @torch.compile(backend="aot_eager")
        def fn(a, b, c, d, e, f):
            # [FUNCTION DOC] HELPER: Compiled function (aot_eager) that doubles all six arguments
            # in place; its body also (mistakenly) contains the view setup and the recursive calls
            # meant for the test body.
            a.mul_(2)
            b.mul_(2)
            c.mul_(2)
            d.mul_(2)
            e.mul_(2)
            f.mul_(2)

            base = torch.ones(2, 20)
            a = base[:, 0:2]
            b = base[:, 2:4]
            c = base[:, 4:6]
            d = base[:, 6:8]
            e = base[:, 8:10]
            f = base[:, 10:12]
            f2 = base[:, 10:14]
            fn(a, b, c, d, e, f)
            with self.assertRaisesRegex(
                AssertionError, "is being compiled with dynamic shapes"
            ):
                fn(a, b, c, d, e, f2)

    def test_user_ctor_ctx_manager_custom_init(self):
        # [FUNCTION DOC] Checks that a user context manager whose constructor mutates an argument
        # list compiles under fullgraph into one frame and matches eager.
        class UserCtxManager:
            # [CLASS DOC] HELPER: Context manager class whose constructor sets the first element
            # of its list argument to 10.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Sets x first element to 10.
                x[0] = 10

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns 1.
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Creates the context manager with y (mutating it) and returns
            # x times the first element of y.
            ucm = UserCtxManager(y)  # noqa: F841
            return x * y[0]

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, [5]), fn(x, [5]))
        self.assertExpectedInline(cnt.frame_count, """1""")

    def test_user_ctor_ctx_manager_custom_init_graph_break(self):
        # [FUNCTION DOC] Checks that a context manager constructor with side effects on an input
        # list (increment of a counter) is run once per call even with a graph break: counter ends
        # at 2 then 12; frame count 2 when static by default, else 1.
        counter = [0]

        class UserCtxManager:
            # [CLASS DOC] HELPER: Context manager class whose constructor increments the first
            # element of its list argument.
            def __init__(self, k):
                # [FUNCTION DOC] HELPER: Increments the first element of k.
                k[0] += 1

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns 1.
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        def fn(x, counter):
            # [FUNCTION DOC] HELPER: Squares x, creates the context manager with the counter and
            # squares again.
            x = x * x
            ucm = UserCtxManager(counter)  # noqa: F841
            return x * x

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.rand([2, 2])
        self.assertEqual(opt_fn(x, counter), fn(x, counter))
        self.assertEqual(counter[0], 2)
        for _ in range(10):
            opt_fn(x, counter)
        self.assertEqual(counter[0], 12)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_many_overlapping_inputs_does_not_explode_guards(self):
        # [FUNCTION DOC] Checks that many aliased inputs (100 splits of one tensor) do not explode
        # guard counts: using an AOT backend hook, the number of AOT autograd guards stays under
        # 5000, shape env guards are zero, and a second call does not recompile.
        from torch._dynamo.backends.common import aot_autograd

        # Before, this was (9702, 0)
        num_shape_guards = None
        num_aot_guards = None
        num_compiles = 0

        def guard_count_backend(gm, *args):
            # [FUNCTION DOC] HELPER: Backend forward compiler that records the shape env guard
            # count and AOT autograd guard count into nonlocal variables, increments a compile
            # counter and returns the graph module.
            nonlocal num_shape_guards
            nonlocal num_aot_guards
            nonlocal num_compiles
            num_shape_guards = len(
                torch._guards.TracingContext.try_get().fake_mode.shape_env.guards
            )
            num_aot_guards = len(
                torch._guards.TracingContext.try_get().guards_context.aotautograd_guards
            )
            num_compiles += 1
            return gm

        aot_guard_counter = aot_autograd(fw_compiler=guard_count_backend)

        @torch.compile(backend=aot_guard_counter, dynamic=True)
        def f(*args):
            # [FUNCTION DOC] HELPER: Compiled function with dynamic shapes that adds one in place
            # to every argument.
            for a in args:
                a.add_(1)

        x = torch.ones(1000, requires_grad=True)
        args = x.split(10)

        with torch.no_grad():
            f(*args)
        # In this example, there were 4950 guards (roughly (# tensors) ^ 2 // 2),
        # because every pair of aliased inputs needs a guard.
        self.assertTrue(num_aot_guards < 5000)
        # But there are no dynamic shape guards.
        self.assertEqual(num_shape_guards, 0)
        # don't recompile
        with torch.no_grad():
            f(*args)
        self.assertEqual(num_compiles, 1)

    def test_issue134451(self):
        # [FUNCTION DOC] Regression test for issue 134451: an IntEnum with classmethod-property
        # accessors returning slices used for slice assignment in a module forward must compile
        # with fullgraph and match eager; skipped if eager fails (Python version dependent).
        class BoundingBox2DIndex(IntEnum):
            # [CLASS DOC] HELPER: Integer enum with five members for bounding box fields and
            # classmethod properties exposing indices or slices.
            _X = 0
            _Y = 1
            _HEADING = 2
            _LENGTH = 3
            _WIDTH = 4

            @classmethod
            def size(cls):
                # [FUNCTION DOC] HELPER: Class method returning the constant 5.
                return 5

            @classmethod
            @property
            def X(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the X member.
                return cls._X

            @classmethod
            @property
            def Y(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the Y member.
                return cls._Y

            @classmethod
            @property
            def HEADING(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the HEADING member.
                return cls._HEADING

            @classmethod
            @property
            def LENGTH(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the LENGTH member.
                return cls._LENGTH

            @classmethod
            @property
            def WIDTH(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the WIDTH member.
                return cls._WIDTH

            @classmethod
            @property
            def POINT(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the slice covering X
                # and Y.
                # assumes X, Y have subsequent indices
                return slice(cls._X, cls._Y + 1)

            @classmethod
            @property
            def STATE_SE2(cls):
                # [FUNCTION DOC] HELPER: Class-level property returning the slice covering X, Y
                # and HEADING.
                # assumes X, Y, HEADING have subsequent indices
                return slice(cls._X, cls._HEADING + 1)

        class SimpleModel(nn.Module):
            # [CLASS DOC] HELPER: Module with an MLP outputting one value per bounding box field.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds a Sequential of linear 10 to 20, relu and linear
                # 20 to the enum size.
                super().__init__()
                self._mlp_states = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, BoundingBox2DIndex.size()),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the MLP, then replaces the point slice with tanh
                # times 32 and the heading entry with tanh times pi, in place; returns the result.
                agent_states = self._mlp_states(x)
                agent_states[..., BoundingBox2DIndex.POINT] = (
                    agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
                )
                agent_states[..., BoundingBox2DIndex.HEADING] = (
                    agent_states[..., BoundingBox2DIndex.HEADING].tanh() * torch.pi
                )
                return agent_states

        model = SimpleModel().eval()
        input_tensor = torch.randn(1, 10, dtype=torch.float32)
        opt = torch.compile(model.eval(), backend="eager", fullgraph=True)
        try:
            expected = model(input_tensor)
        except Exception as e:
            raise unittest.SkipTest(
                "eager failed, requires Python between 3.9 and 3.12"
            ) from e
        actual = opt(input_tensor)
        self.assertEqual(actual, expected)

    def test_invalid_seq_unpack(self):
        # [FUNCTION DOC] Checks that unpacking a 3-tuple into two names inside a compiled function
        # raises ValueError.
        def myfn(arg):
            # [FUNCTION DOC] HELPER: Unpacks the argument into two names.
            (a, b) = arg

        def fn():
            # [FUNCTION DOC] HELPER: Calls myfn with a 3-tuple.
            return myfn((1, 2, 3))

        try:
            torch.compile(fn, backend="eager")()
        except ValueError:
            pass
        else:
            self.fail("expected exception")

    def test_udf_classes_reconstruction(self):
        # [FUNCTION DOC] Checks that a class looked up through a closure/outer variable is
        # re-guarded: the compiled function uses IncByOne at first and, after the variable is
        # rebound to IncByTwo, recompiles and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Instantiates the class held in the enclosing variable T with
            # 5 and returns its x plus the input.
            o = T(5)
            return o.x + x

        opt_fn = torch.compile(fn, backend="eager")
        T = IncByOne

        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should recompile
        T = IncByTwo
        self.assertEqual(fn(x), opt_fn(x))

    def test_contains_range_constprop(self):
        # [FUNCTION DOC] Checks that a membership test of an int in a range is constant folded and
        # the compiled result matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if 3 is in range of 10 (always), else x
            # plus 2.
            # dynamo should const prop to False
            if 3 in range(10):
                return x + 1
            else:
                return x + 2

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    # https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_base_with_mutation_works(self):
        # [FUNCTION DOC] Regression for issue 104505: as_strided on a base tensor followed by an
        # in-place add works under aot_eager and the base is updated as in eager.
        def foo(a):
            # [FUNCTION DOC] HELPER: Takes a strided view of two elements, adds one in place and
            # returns the base.
            f = a.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        out_ref = foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        out = f_compiled(a)
        self.assertEqual(out_ref, out)
        self.assertEqual(a_ref, a)

    # https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_existing_view_banned(self):
        # [FUNCTION DOC] Regression for issue 104505: mutating through as_strided applied to an
        # existing view (diagonal) raises RuntimeError mentioning a mutation on a view chain of
        # length 2 under aot_eager.
        def foo(a):
            # [FUNCTION DOC] HELPER: Takes the diagonal view, then an as_strided view of it, adds
            # one in place and returns the base.
            e = a.diagonal()
            f = e.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        a = torch.randn(2, 4)
        a_ref = a.clone()
        foo(a_ref)
        f_compiled = torch.compile(foo, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError,
            "encountered a mutation on a view chain of length 2, where view 1 was an as_strided",
        ):
            f_compiled(a)
        # See https://github.com/pytorch/pytorch/issues/161010

    # https://github.com/pytorch/pytorch/issues/185888
    @parametrize("backend", ["eager", "inductor"])
    def test_as_strided_inplace_internal_tensor_metadata(self, backend):
        # [FUNCTION DOC] Parametrized on backend (eager, inductor). Regression for issue 185888:
        # in-place as_strided on an internal tensor must give the same values, sizes, strides and
        # dims as eager when compiled with fullgraph and dynamic shapes.
        def fn():
            # [FUNCTION DOC] HELPER: Creates arange 4, applies in-place as_strided to 2 by 2,
            # takes a max (observer) and returns the view, sizes, shape, dim, strides of both and
            # the observer.
            x = torch.arange(4.0)
            y = x.as_strided_((2, 2), (2, 1))
            observer = torch.max(y)
            return (
                y,
                x.size(),
                x.shape,
                x.dim(),
                y.size(),
                x.stride(),
                y.stride(),
                observer,
            )

        eager = fn()
        compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        actual = compiled()

        self.assertEqual(actual[0], eager[0])
        self.assertEqual(tuple(actual[1]), tuple(eager[1]))
        self.assertEqual(tuple(actual[2]), tuple(eager[2]))
        self.assertEqual(actual[3], eager[3])
        self.assertEqual(tuple(actual[4]), tuple(eager[4]))
        self.assertEqual(tuple(actual[5]), tuple(eager[5]))
        self.assertEqual(tuple(actual[6]), tuple(eager[6]))
        self.assertEqual(actual[7], eager[7])

    # https://github.com/pytorch/pytorch/issues/185888
    @parametrize("backend", ["eager", "inductor"])
    def test_as_strided_inplace_internal_tensor_symbolic_metadata(self, backend):
        # [FUNCTION DOC] Parametrized on backend (eager, inductor). Regression for issue 185888
        # with a symbolic size: in-place as_strided with a size from an input must give the same
        # sizes, strides and contiguity as eager.
        def fn(inp):
            # [FUNCTION DOC] HELPER: Creates arange 16, applies in-place as_strided with the
            # size of the input dim 0 and stride 2, and returns sizes, strides and contiguity of
            # both tensors.
            x = torch.arange(16.0)
            n = inp.size(0)
            y = x.as_strided_((n,), (2,))
            return (
                x.size(),
                y.size(),
                x.stride(),
                y.stride(),
                x.is_contiguous(),
                y.is_contiguous(),
            )

        inp = torch.empty(5)
        eager = fn(inp)
        compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        actual = compiled(inp)

        self.assertEqual(tuple(actual[0]), tuple(eager[0]))
        self.assertEqual(tuple(actual[1]), tuple(eager[1]))
        self.assertEqual(tuple(actual[2]), tuple(eager[2]))
        self.assertEqual(tuple(actual[3]), tuple(eager[3]))
        self.assertEqual(actual[4], eager[4])
        self.assertEqual(actual[5], eager[5])

    # Extension of https://github.com/pytorch/pytorch/issues/161010
    # in the non memory dense case
    def test_clone_not_memory_dense(self):
        # [FUNCTION DOC] Extension of issue 161010 for non memory-dense tensors: clone of a
        # transposed strided slice must have stride 1, 4 in eager and under eager, aot_eager and
        # inductor compile.
        def foo() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Clones a strided slice of a transposed random tensor and
            # returns the clone.
            x = torch.randn(10, 8).t()[::2, ::2]
            y = x.clone()
            return y

        y = foo()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Reference eager implementation should have stride (1, 4)",
        )
        y = torch.compile(foo, backend="eager")()
        self.assertEqual(
            y.stride(), (1, 4), "Compile with eager backend should have stride (1, 4)"
        )
        y = torch.compile(foo, backend="aot_eager")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with aot_eager backend should have stride (1, 4)",
        )
        y = torch.compile(foo, backend="inductor")()
        self.assertEqual(
            y.stride(),
            (1, 4),
            "Compile with inductor backend should have stride (1, 4)",
        )

    # https://github.com/pytorch/pytorch/issues/146598
    @unittest.expectedFailure
    def test_lru_cache_tracing(self):
        # [FUNCTION DOC] Expected failure (issue 146598): a function wrapped with an lru cache
        # should run its body only once for repeated calls with the same tensor when compiled.
        # KNOWN ISSUE: marked as an expected failure since lru_cache tracing is not handled
        # that way.
        from functools import lru_cache

        counter = 0

        @lru_cache
        def cached_fn(x):
            # [FUNCTION DOC] HELPER: Cached function that counts calls in a nonlocal counter and
            # returns x plus one.
            nonlocal counter
            counter += 1
            return x + 1

        compiled_fn = torch.compile(cached_fn, backend="eager")

        t = torch.randn(2, 2)
        result1 = compiled_fn(t)
        self.assertEqual(counter, 1)

        result2 = compiled_fn(t)
        self.assertEqual(counter, 1)
        self.assertEqual(result1, result2)

    def test_dont_aggressively_write_assert(self):
        # [FUNCTION DOC] Checks that assertions in a dynamic-shape function are not aggressively
        # converted: the recorded graph contains only the tensor-value assert (async assert) and
        # the cos, shape env lower bounds are 4 and 2, and a failing assert on shape graph-breaks
        # so the eager AssertionError is raised.
        record_graph = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(dynamic=True, backend=record_graph)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function asserting shape and value conditions then
            # returning cos of x.
            assert x.shape[0] > 3  # noqa: S101
            assert x[0].sum() > 0  # noqa: S101
            assert 1 % (x.shape[0] // 2) != 0  # noqa: S101
            assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0  # noqa: S101
            return x.cos()

        f(torch.ones(6, 4))
        graph = record_graph.graphs[0]
        # It is bit annoying that we generate useless statements for
        # shape guards, but DCE should be able to remove them since t
        # there is no backed assert on them. The reason this is ok is
        # because dynamo will only skip the assert statement, but not
        # the instructions before it.

        # The code generation can non-deterministically use either form
        generated_code = str(graph.code).strip().replace(".gt(0)", " > 0")
        self.assertExpectedInline(
            generated_code,
            """\
def forward(self, L_x_ : torch.Tensor, s77 : torch.SymInt, s27 : torch.SymInt):
    l_x_ = L_x_
    getitem = l_x_[0]
    sum_1 = getitem.sum();  getitem = None
    gt = sum_1 > 0;  sum_1 = None
    _assert_async = torch._assert_async(gt, 'assertion error');  gt = _assert_async = None
    cos = l_x_.cos();  l_x_ = None
    return (cos,)""",
        )
        for node in graph.graph.nodes:
            if "example_value" in node.meta and isinstance(
                node.meta["example_value"], torch._subclasses.fake_tensor.FakeTensor
            ):
                shape_env = node.meta["example_value"].fake_mode.shape_env
                lower_ranges = [val.lower for val in shape_env.var_to_range.values()]
                self.assertTrue(lower_ranges == [4, 2])

        @torch.compile(dynamic=True, backend=record_graph)
        def f_fail(x):
            # [FUNCTION DOC] HELPER: Compiled function asserting that the first dimension is below
            # 3 (fails for the input used).
            assert x.shape[0] < 3  # noqa: S101

        # We graph-break here, so the failure should be eager
        with self.assertRaisesRegex(AssertionError, ""):
            f_fail(torch.ones(6, 4))

    def test_detectron2_instances_cat(self):
        # [FUNCTION DOC] Detectron2 Instances cat pattern: defines an Instances class with field
        # storage and a static cat concatenating fields (tensors, lists or objects with cat),
        # compiles a call on three instances with fullgraph and checks type and instance
        # dictionary equal eager.
        class Instances:
            # [CLASS DOC] HELPER: Simplified detectron2 Instances: stores an image size and a
            # dictionary of named fields set through attribute assignment.
            def __init__(self, image_size: tuple[int, int], **kwargs: Any):
                # [FUNCTION DOC] HELPER: Stores the image size, creates the field dictionary and
                # sets each keyword argument as a field.
                self._image_size = image_size
                self._fields: dict[str, Any] = {}
                for k, v in kwargs.items():
                    self.set(k, v)

            @property
            def image_size(self) -> tuple[int, int]:
                # [FUNCTION DOC] HELPER: Property returning the stored image size.
                return self._image_size

            def __setattr__(self, name: str, val: Any) -> None:
                # [FUNCTION DOC] HELPER: Attribute setter that stores names starting with
                # underscore normally and treats other names as fields.
                if name.startswith("_"):
                    super().__setattr__(name, val)
                else:
                    self.set(name, val)

            def __getattr__(self, name: str) -> Any:
                # [FUNCTION DOC] HELPER: Attribute getter that returns the field with the given
                # name or raises AttributeError if absent.
                if name == "_fields" or name not in self._fields:
                    raise AttributeError(
                        f"Cannot find field '{name}' in the given Instances!"
                    )
                return self._fields[name]

            def __len__(self) -> int:
                # [FUNCTION DOC] HELPER: Returns the length of the first field (via its length
                # method) or raises NotImplementedError if there are none.
                for v in self._fields.values():
                    # use __len__ because len() has to be int and is not friendly to tracing
                    return v.__len__()
                raise NotImplementedError("Empty Instances does not support __len__!")

            def set(self, name: str, value: Any) -> None:
                # [FUNCTION DOC] HELPER: Adds a field after checking its length matches the
                # existing fields (asserting) and stores it.
                with warnings.catch_warnings(record=True):
                    data_len = len(value)
                if len(self._fields):
                    assert len(self) == data_len, (  # noqa: S101
                        f"Adding a field of length {data_len} to a Instances of length {len(self)}"
                    )
                self._fields[name] = value

            def get(self, name: str) -> Any:
                # [FUNCTION DOC] HELPER: Returns the named field.
                return self._fields[name]

            @staticmethod
            def cat(instance_lists: list["Instances"]) -> "Instances":
                # [FUNCTION DOC] HELPER: Static method concatenating a list of Instances: asserts
                # types and non-emptiness, returns the single instance unchanged, checks image
                # sizes are equal when not tensors, and builds a new Instances by concatenating
                # tensors, chaining lists or calling a type's cat; raises ValueError for
                # unsupported types.
                assert all(isinstance(i, Instances) for i in instance_lists)  # noqa: S101
                assert len(instance_lists) > 0  # noqa: S101
                if len(instance_lists) == 1:
                    return instance_lists[0]

                image_size = instance_lists[0].image_size
                if not isinstance(
                    image_size, torch.Tensor
                ):  # could be a tensor in tracing
                    for i in instance_lists[1:]:
                        assert i.image_size == image_size  # noqa: S101
                ret = Instances(image_size)
                for k in instance_lists[0]._fields:
                    values = [i.get(k) for i in instance_lists]
                    v0 = values[0]
                    if isinstance(v0, torch.Tensor):
                        values = torch.cat(values, dim=0)
                    elif isinstance(v0, list):
                        values = list(itertools.chain(*values))
                    elif hasattr(type(v0), "cat"):
                        values = type(v0).cat(values)
                    else:
                        raise ValueError(
                            f"Unsupported type {type(v0)} for concatenation"
                        )
                    ret.set(k, values)
                return ret

        instances = [
            Instances((16, 16), a=torch.randn(16, 16), b=torch.randn(16, 16))
            for _ in range(3)
        ]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(instances):
            # [FUNCTION DOC] HELPER: Compiled function returning the concatenation of the
            # instances through the first instance.
            return instances[0].cat(instances)

        actual = fn(instances)
        expected = instances[0].cat(instances)
        self.assertEqual(type(actual), type(expected))
        self.assertEqual(actual.__dict__, expected.__dict__)

    def test_weakref_construction(self):
        # [FUNCTION DOC] Checks that creating a weak reference to a tensor and dereferencing it
        # inside a fullgraph compiled function works.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Makes a weak reference to x, dereferences it and multiplies
            # by y.
            x_weak = weakref.ref(x)
            return x_weak() * y

        x = torch.randn(4)
        y = torch.randn(4)

        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_weakref(self):
        # [FUNCTION DOC] Checks that a weak reference passed as an argument can be dereferenced
        # and compared by identity with a tensor in a fullgraph compiled function.
        def fn(x_weak, weight, y):
            # [FUNCTION DOC] HELPER: Returns sine of y if the weak reference is set and its
            # referent is not weight, otherwise cosine of y.
            if x_weak is not None and x_weak() is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)

    # https://github.com/pytorch/pytorch/issues/159258
    def test_weakref_proxy(self):
        # [FUNCTION DOC] Regression for issue 159258: a model method that reads through a weakref
        # proxy to an object compiles with fullgraph and returns the referenced tensor.
        class DummyTrainer:
            # [CLASS DOC] HELPER: Simple object holding a tensor in the attribute foo.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x as foo.
                self.foo = x

        class DummyModel(nn.Module):
            # [CLASS DOC] HELPER: Module holding a trainer reference (initially None) with a
            # method returning its foo.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer and sets trainer to None.
                super().__init__()
                self.trainer = None

            def foo(self):
                # [FUNCTION DOC] HELPER: Returns the foo attribute of the trainer.
                return self.trainer.foo

        x = torch.randn(4)
        model = DummyModel()
        trainer = DummyTrainer(x)
        model.trainer = weakref.proxy(trainer)
        compiled_foo = torch.compile(model.foo, backend="eager", fullgraph=True)
        self.assertEqual(compiled_foo(), x)

    def test_weakref_reconstruct(self):
        # [FUNCTION DOC] Checks that dereferencing a weak reference before a graph break and using
        # the referent after it works: compiled with a frame counter, the result equals eager and
        # exactly two frames are compiled.
        def fn(x_weak, weight, y):
            # [FUNCTION DOC] HELPER: Takes sine of y, dereferences the weak reference, forces a
            # graph break, then returns sine or cosine of y depending on whether the referent
            # is weight.
            y = torch.sin(y)
            referent = x_weak()
            torch._dynamo.graph_break()
            if referent is not weight:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        y = torch.randn(4)
        x_weak = weakref.ref(weight)

        ref = fn(x_weak, weight, y)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        res = opt_fn(x_weak, weight, y)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 2)

    def test_return_weakref(self):
        # [FUNCTION DOC] Checks that a weak reference created inside a fullgraph compiled function
        # and returned alongside its referent dereferences to the same value as in eager.
        def f(t):
            # [FUNCTION DOC] HELPER: Doubles t, creates a weak reference to the result and returns
            # the reference and the tensor.
            t = t * 2
            wr = weakref.ref(t)
            return wr, t

        ref_t = torch.randn(2, 2, requires_grad=True)
        ref_y = f(ref_t)

        t = ref_t.detach().clone().requires_grad_()
        y = torch.compile(f, backend="eager", fullgraph=True)(t)
        self.assertEqual(ref_y[0](), y[0]())

    def test_weakref_del(self):
        # [FUNCTION DOC] Checks a function reading a weak reference compiled with fullgraph gives
        # the same result as eager before and after the referent is deleted and garbage collected
        # (sine when alive, cosine when dead).
        def fn(x_weak, y):
            # [FUNCTION DOC] HELPER: Dereferences the weak reference; returns sine of y if the
            # referent is alive, otherwise cosine of y.
            x = x_weak()
            if x is not None:
                return torch.sin(y)
            return torch.cos(y)

        weight = torch.randn(4)
        x_weak = weakref.ref(weight)
        y = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

        del weight
        gc.collect()
        ref = fn(x_weak, y)
        res = opt_fn(x_weak, y)
        self.assertEqual(ref, res)

    # The programming model around (weak)references is that we DO NOT guarantee
    # any behavior that depends on deallocation order. We do guarantee "eventual consistency",
    # that is, after the torch.compile'd function is finished running (including any graph breaks),
    # refcount semantics will match eager's.
    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: False is not true")
    def test_weakref_callback(self):
        # [FUNCTION DOC] Skipped on Windows. Checks weakref callback behavior with the eager
        # backend: a callback for a weakref created in a compiled region is itself run compiled
        # (is_compiling is true) and is called; a callback firing after the region when the
        # referent is deleted runs uncompiled and is called; a weakref and referent both dropped
        # in one region must not call its callback. Per the preceding comment only eventual
        # consistency is guaranteed.
        called1 = False

        def callback1(ref):
            # [FUNCTION DOC] HELPER: Weakref callback that sets the nonlocal called1 flag and
            # raises RuntimeError if not running under compilation.
            nonlocal called1
            called1 = True
            if not torch.compiler.is_compiling():
                raise RuntimeError("callback1 expected to be compiled")

        # weakref callbacks that should be called in the compiled region will be compiled.
        # But the exact place in the compiled code that the callback is made is undefined.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that computes x plus one, creates a weak
            # reference with callback1, forces a graph break and returns the reference.
            y = x + 1
            ref = weakref.ref(y, callback1)
            torch._dynamo.graph_break()
            return ref

        fn(torch.ones(3))
        self.assertTrue(called1)

        called2 = False

        def callback2(ref):
            # [FUNCTION DOC] HELPER: Weakref callback that sets the nonlocal called2 flag and
            # raises RuntimeError if running under compilation.
            nonlocal called2
            called2 = True
            if torch.compiler.is_compiling():
                raise RuntimeError("callback2 expected to not be compiled")

        # weakref callbacks that fire outside the compiled region work
        @torch.compile(backend="eager")
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled function that computes x plus one, creates a
            # weak reference with callback2, forces a graph break and returns the tensor and
            # the reference.
            y = x + 1
            ref = weakref.ref(y, callback2)
            torch._dynamo.graph_break()
            return y, ref

        y, _ = gn(torch.ones(3))
        del y
        self.assertTrue(called2)

        def callback3(ref):
            # [FUNCTION DOC] HELPER: Weakref callback that always raises RuntimeError (it must not
            # be called).
            raise RuntimeError("callback3 should not be called")

        # The callback will NOT be called if both the weakref and the referrent are
        # deleted in the same compiled region (graph breaks act like a "memory sync"
        # and thus make things tricky - the callback is actually expected to be called).
        # This test does NOT mean that this behavior is part of the (weak)ref programming
        # model, but rather reminds us that this is an intentionally allowed weakref-Dynamo behavior.
        @torch.compile(backend="eager")
        def hn(x):
            # [FUNCTION DOC] HELPER: Compiled function that computes x plus one and creates a weak
            # reference with callback3 whose result is discarded.
            y = x + 1
            _ = weakref.ref(y, callback3)

        hn(torch.ones(3))

    #     @torch._functorch.config.patch(
    #         recompute_views=True,
    #     )
    #     def test_storage_resize_forward_full_graph(self):
    #         class TestModule(torch.nn.Module):
    #             def __init__(self) -> None:
    #                 super().__init__()
    #                 self.param = torch.nn.Parameter(torch.randn(4, 4))

    #             def forward(self, x):
    #                 self.param.untyped_storage().resize_(
    #                     self.param.numel() * self.param.itemsize
    #                 )
    #                 with torch.no_grad():
    #                     torch._foreach_copy_([self.param], [x])
    #                 out = torch.matmul(self.param, self.param)
    #                 self.param.untyped_storage().resize_(0)
    #                 return out

    #         def post_accumulate_grad_hook(param):
    #             param.untyped_storage().resize_(0)

    #         # Beginning of backward, resize and put data into the param
    #         def pre_backward_hook(module, grad) -> None:
    #             module.param.untyped_storage().resize_(
    #                 self.param.numel() * self.param.itemsize
    #             )
    #             with torch.no_grad():
    #                 # simulates loading data into param from allgather
    #                 module.param.fill_(2)

    #         def post_forward_hook(module, args, output):
    #             output.register_hook(functools.partial(pre_backward_hook, module))

    #         x = torch.randn(4, 4)

    #         mod_ref = TestModule()
    #         mod_test = deepcopy(mod_ref)

    #         # Start the param off with zero storage size to mimic fsdp
    #         mod_ref.param.untyped_storage().resize_(0)
    #         mod_test.param.untyped_storage().resize_(0)

    #         # Resize storage at beginning of backward
    #         # Free storage at end of backward
    #         mod_ref.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_ref.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)
    #         mod_test.register_forward_hook(post_forward_hook, prepend=False)
    #         mod_test.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)

    #         mod_test = torch.compile(mod_test, backend=aot_graph_capture_backend)

    #         out_ref = mod_ref(x)
    #         out_test = mod_test(x)
    #         self.assertExpectedInline(
    #             str(fw_graph[0].code.strip()),
    #             """\
    # def forward(self, primals_1, primals_2):
    #     _foreach_copy = torch.ops.aten._foreach_copy.default([primals_1], [primals_2]);  primals_1 = primals_2 = None
    #     getitem = _foreach_copy[0];  _foreach_copy = None
    #     mm = torch.ops.aten.mm.default(getitem, getitem)
    #     return [mm, getitem]""",
    #         )
    #         self.assertEqual(out_ref, out_test)

    def test_super_in_staticmethod(self):
        # [FUNCTION DOC] Checks that using zero-argument super inside a static method
        # raises RuntimeError mentioning super, both in eager and when called through a
        # compiled function.
        class A:
            # [CLASS DOC] HELPER: Class with a static method that uses zero-argument
            # super (invalid).
            @staticmethod
            def foo():
                # [FUNCTION DOC] HELPER: Static method returning the result of zero-argument super
                # init, which raises RuntimeError since no class cell is bound.
                return super().__init__()

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the static method foo on the object.
            return obj.foo()

        obj = A()

        with self.assertRaisesRegex(RuntimeError, r"super\(\)"):
            fn(obj)

        with self.assertRaisesRegex(RuntimeError, r"super\(\)"):
            torch.compile(backend="eager")(fn)(obj)

    def test_super_staticmethod(self):
        # [FUNCTION DOC] Checks super with explicit class arguments inside a static method of a
        # child class, compiled with fullgraph, matches eager.
        class Parent:
            # [CLASS DOC] HELPER: Class with a static method greet returning 5.
            @staticmethod
            def greet():
                # [FUNCTION DOC] HELPER: Returns the constant 5.
                return 5

        class Child(Parent):
            # [CLASS DOC] HELPER: Subclass of Parent whose static greet multiplies by the parent
            # greet obtained via super with explicit arguments.
            @staticmethod
            def greet(x):
                # [FUNCTION DOC] HELPER: Returns x times the parent static greet value.
                return x * super(Child, Child).greet()

        child = Child()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls greet on the child instance with x.
            return child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod(self):
        # [FUNCTION DOC] Checks super calls to a class method from an instance method of a
        # subclass of a subclass (class method returns different values per class), compiled with
        # fullgraph, match eager.
        class Parent:
            # [CLASS DOC] HELPER: Class with a class method greet returning 4, 3 or 5 depending on
            # whether the class is Parent, Child or GrandChild (2 otherwise).
            @classmethod
            def greet(cls):
                # [FUNCTION DOC] HELPER: Returns a constant chosen by the class on which it
                # is called.
                if cls == Parent:
                    return 4
                if cls == Child:
                    return 3
                if cls == GrandChild:
                    return 5
                return 2

        class Child(Parent):
            # [CLASS DOC] HELPER: Subclass whose instance method greet multiplies x by the parent
            # class method value.
            def greet(self, x):
                # [FUNCTION DOC] HELPER: Returns x times the super class method result.
                return x * super().greet()

        class GrandChild(Child):
            # [CLASS DOC] HELPER: Empty subclass of Child.
            pass

        grand_child = GrandChild()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls greet on the grandchild instance with x.
            return grand_child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_classmethod_inheritance(self):
        # [FUNCTION DOC] Checks chained super calls through class methods in three levels, where
        # the attribute A is only defined on the most derived class, compiled with fullgraph
        # matches eager.
        class GrandParent:
            # [CLASS DOC] HELPER: Base class whose class method multiplies x by the class
            # attribute A.
            @classmethod
            def greet(cls, x):
                # [FUNCTION DOC] HELPER: Returns A of the calling class times x.
                return cls.A * x

        class Parent(GrandParent):
            # [CLASS DOC] HELPER: Middle class whose class method delegates to super.
            @classmethod
            def greet(cls, x):
                # [FUNCTION DOC] HELPER: Returns the super greet applied to x.
                return super().greet(x)

        class Child(Parent):
            # [CLASS DOC] HELPER: Most derived class defining A as 5 and delegating to super.
            A = 5

            @classmethod
            def greet(cls, x):
                # [FUNCTION DOC] HELPER: Returns the super greet applied to x.
                return super().greet(x)

        child = Child()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls greet on the child instance with x.
            return child.greet(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_super_diamond(self):
        # [FUNCTION DOC] Checks multiple inheritance with super calls (class B inherits from an
        # empty class and from A): construction and a method that multiplies the attributes by x
        # under fullgraph match eager.
        class A:
            # [CLASS DOC] HELPER: Class whose initializer calls super and sets a to 5.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls super init and sets a to 5.
                super().__init__()
                self.a = 5

        class Nothing:
            # [CLASS DOC] HELPER: Empty class used in the base list.
            pass

        class B(Nothing, A):
            # [CLASS DOC] HELPER: Class deriving from Nothing and A that sets b to 10 and has a
            # method multiplying a, b and x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls super init (reaching A) and sets b to 10.
                super().__init__()
                self.b = 10

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns a times b times x.
                return self.a * self.b * x

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a B instance and returns run on x.
            b = B()
            return b.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_vc_bumped_in_inference_graph(self):
        # [FUNCTION DOC] Checks that an in-place multiply on an input inside a compiled function
        # (eager backend) bumps the input version counter.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies x by 2 in place and returns it.
            return x.mul_(2)

        x = torch.randn(4)
        vc_before = x._version
        f(x)
        vc_after = x._version
        self.assertTrue(vc_after > vc_before)

    def test_nn_module_callable(self):
        # [FUNCTION DOC] Checks that the builtin callable on an nn.Module instance returns True
        # inside a fullgraph compiled function.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module returning sine of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of x.
                return x.sin()

        def f(m):
            # [FUNCTION DOC] HELPER: Returns whether the given module is callable.
            return callable(m)

        res = torch.compile(f, fullgraph=True, backend="eager")(M())
        self.assertTrue(res)

    def test_stk_sdd_is_transposed(self):
        # [FUNCTION DOC] Checks that a custom autograd Function that inspects whether saved
        # tensors are transposed in backward yields the same gradients when applied inside a
        # fullgraph compiled function as in eager; the second input is transposed so strides
        # are preserved.
        def _is_transposed(x):
            # [FUNCTION DOC] HELPER: Returns True when the tensor is not contiguous and has stride
            # 1 in dim 0 and size of dim 0 in dim 1.
            return (
                not x.is_contiguous()
                and x.stride()[0] == 1
                and x.stride()[1] == x.size()[0]
            )

        class SDD(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function whose forward returns a tensor of ones like
            # lhs and whose backward returns gradient fill values depending on whether each saved
            # input is transposed (1.0 if transposed, else 2.0) and None for extra outputs.
            @staticmethod
            def forward(ctx, lhs, rhs):
                # [FUNCTION DOC] HELPER: Saves both inputs and returns ones with the shape and
                # dtype of lhs.
                ctx.save_for_backward(lhs, rhs)
                out = torch.full_like(lhs, 1.0, dtype=lhs.dtype, device=lhs.device)
                return out

            @staticmethod
            def backward(ctx, dy):
                # [FUNCTION DOC] HELPER: Computes a constant gradient filled with 1.0 or 2.0 for
                # each input that needs a gradient depending on transposition; returns those and
                # two None values.
                saved_tensors = ctx.saved_tensors
                lhs, rhs = saved_tensors[:2]
                trans_a = _is_transposed(lhs)
                trans_b = _is_transposed(rhs)
                dlhs = None
                if ctx.needs_input_grad[0]:
                    dlhs = torch.full_like(lhs, 1.0 if trans_a else 2.0)
                drhs = None
                if ctx.needs_input_grad[1]:
                    drhs = torch.full_like(rhs, 1.0 if trans_b else 2.0)
                return dlhs, drhs, None, None

        x1 = torch.randn((8, 8), requires_grad=True)
        y1 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)
        x2 = torch.randn((8, 8), requires_grad=True)
        y2 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)

        SDD.apply(x1, y1).sum().backward()

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function applying SDD to the globals x2 and y2.
            return SDD.apply(x2, y2)

        fn().sum().backward()

        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(y1.grad, y2.grad)

    def test_partially_initialized_module_property(self):
        # [FUNCTION DOC] Checks that a module whose initializer reads properties before its
        # attributes are fully set can be constructed inside a fullgraph compiled function: the
        # result has type Matrix, foo equal to 200 and data of shape 10 by 20.
        class Matrix(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing data and setting foo to ten times the
            # property blocking.
            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Calls the base initializer, stores data in the private
                # attribute and sets foo from the property blocking.
                super().__init__()
                self._data = data
                self.foo = 10 * self.blocking

            @property
            def data(self):
                # [FUNCTION DOC] HELPER: Property returning the stored data.
                return self._data

            @property
            def blocking(self):
                # [FUNCTION DOC] HELPER: Property returning the size of dim 1 of data.
                return self.data.shape[1]

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Creates and returns a Matrix with a random 10 by 20 tensor.
            return Matrix(torch.randn(10, 20))

        v = fn()
        self.assertEqual(v.foo, 200)
        self.assertEqual(v.data.shape, (10, 20))
        self.assertEqual(type(v), Matrix)

    def test_classmethod_with_slots(self):
        # [FUNCTION DOC] Checks that an instance of a class with slots and a class method used
        # inside an instance method, created inside a fullgraph function, matches eager.
        class Mock:
            # [CLASS DOC] HELPER: Class using slots with one attribute and a class method
            # returning 3.
            __slots__ = ("_a",)

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the slot value to 2.
                self._a = 2

            @classmethod
            def _m(cls):
                # [FUNCTION DOC] HELPER: Class method returning the constant 3.
                return 3

            def run(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x times the slot value times the class
                # method result.
                return torch.sin(x) * self._a * self._m()

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a Mock and returns run on x.
            mock = Mock()
            return mock.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_nn_parametrize(self):
        # [FUNCTION DOC] Checks parametrization from torch.nn.utils.parametrize: after registering
        # a sine parametrization, the compiled module graph contains sine and matches eager; after
        # removing it the graph (recompiled) no longer contains sine.
        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with a 10 by 10 parameter whose forward returns the
            # parameter matmul x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter matrix multiplied by x.
                return self.param @ x

        class Parametrization(torch.nn.Module):
            # [CLASS DOC] HELPER: Parametrization module returning sine of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x.
                return torch.sin(x)

        m = Module()
        torch.nn.utils.parametrize.register_parametrization(
            m, "param", Parametrization()
        )

        sin_found = False

        def backend(gm, _):
            # [FUNCTION DOC] HELPER: Backend that sets the nonlocal sin_found flag when any graph
            # node targets the sine function, and returns the graph module.
            nonlocal sin_found
            for node in gm.graph.nodes:
                if node.target is torch.sin:
                    sin_found = True
            return gm

        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        inp = torch.randn(10, 10)
        self.assertEqual(m(inp), opt_m(inp))
        self.assertTrue(sin_found)

        torch.nn.utils.parametrize.remove_parametrizations(m, "param")
        sin_found = False
        self.assertEqual(m(inp), opt_m(inp))
        self.assertFalse(sin_found)

    def test_nn_module_property_closure(self):
        # [FUNCTION DOC] Checks a property of a module that reads a closure variable, used in
        # forward, compiled with fullgraph matches eager.
        x = torch.randn(10, 10)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a property y (ones plus the closure tensor x) used
            # in forward as a matmul operand.
            @property
            def y(self):
                # [FUNCTION DOC] HELPER: Property returning ones of 10 by 10 plus the enclosing
                # tensor x.
                return torch.ones(10, 10) + x

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x matmul the property y.
                return x @ self.y

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the module on x.
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        inp = torch.randn(10, 10)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_global_fn_mutation(self):
        # [FUNCTION DOC] Checks that rebinding a module-level function after compilation triggers
        # a recompile: the compiled function uses global_fn, which is then replaced by a cosine
        # function through a global statement, and results still match eager. Side effect: leaves
        # global_fn replaced for the rest of the module.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns global_fn of x plus y.
            return global_fn(x) + y

        x = torch.ones(1)
        y = torch.ones(1)

        opt = torch.compile(foo, fullgraph=True, backend="eager")
        self.assertEqual(opt(x, y), foo(x, y))

        # Change global_fn
        global global_fn

        def new_fn(x):
            # [FUNCTION DOC] HELPER: Replacement function returning cosine of x.
            return torch.cos(x)

        global_fn = new_fn
        self.assertEqual(opt(x, y), foo(x, y))

    # ref https://github.com/pytorch/pytorch/issues/123974
    def test_list_reverse(self):
        # [FUNCTION DOC] Regression for issue 123974: reversing a list of tensors in place and
        # using both orders in matmuls compiles with fullgraph and matches eager.
        def ladder(x):
            # [FUNCTION DOC] HELPER: Builds three weight matrices of sizes derived from the last
            # dim, multiplies x through them, reverses the list in place, then multiplies by the
            # transposes; asserts the last dim exceeds 2.
            trail = x.size(-1)
            assert trail > 2  # noqa: S101
            weights = []
            for s in [trail, trail - 1, trail - 2]:
                weights.append(torch.ones(s, s - 1))

            for w in weights:
                x = x @ w

            weights.reverse()

            for w in weights:
                x = x @ w.t()

            return x

        data = torch.randn(3, 4)
        opt_ladder = torch.compile(ladder, fullgraph=True, backend="eager")
        self.assertEqual(opt_ladder(data), ladder(data))

    def test_trace_functional_tensor_with(self):
        # [FUNCTION DOC] Checks that compiling a function that sets a functional tensor input and
        # mutates a view, under FunctionalTensorMode with fake tensors, raises RuntimeError saying
        # the tensor has frozen storage.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def f(a, tmp):
            # [FUNCTION DOC] HELPER: Takes a flat view of a, then under no_grad sets a to tmp
            # storage and multiplies the view in place; returns a plus tmp.
            a_view = a.view(-1)
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        fake_mode = FakeTensorMode()
        with FunctionalTensorMode():
            inp = torch.ones(3, 3, requires_grad=True)
            inp = fake_mode.from_tensor(inp, static_shapes=True)
            inp = FunctionalTensor.to_functional(inp)

            tmp = torch.ones(3, 3, requires_grad=True)
            tmp = fake_mode.from_tensor(tmp, static_shapes=True)
            tmp = FunctionalTensor.to_functional(tmp)

            opt_f = torch.compile(f, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError, "cannot mutate tensors with frozen storage"
            ):
                opt_f(inp, tmp)

    def test_const_dict_keyerror(self):
        # [FUNCTION DOC] Checks that catching a KeyError for a missing key of a global empty dict
        # inside a compiled function behaves the same as eager.
        d = {}

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads key 0 of the dictionary, using 1 if KeyError is raised,
            # and adds it to x.
            try:
                y = d[0]
            except KeyError:
                y = 1
            return x + y

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_nonconst_issubclass(self):
        # [FUNCTION DOC] Checks that issubclass with a non-constant class (the class of a numpy
        # array argument) is handled by the compiler.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns 1 if the class of x is a subclass of ndarray, else 0.
            if issubclass(x.__class__, np.ndarray):
                return 1
            return 0

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(np.ones([3, 3]))

    def test_issue126128(self):
        # [FUNCTION DOC] Regression for issue 126128: compiles two matmul-and-sum functions, the
        # first under a fresh cache context, with the eager backend; only checks nothing fails.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of random 1 by 10 and
            # 10 by 1 tensors.
            x = torch.randn(1, 10)
            y = torch.randn(10, 1)
            return torch.mm(x, y).sum()

        def fn2():
            # [FUNCTION DOC] HELPER: Returns the sum of the matrix product of random 10 by 100 and
            # 100 by 10 tensors.
            x = torch.randn(10, 100)
            y = torch.randn(100, 10)
            return torch.mm(x, y).sum()

        with fresh_cache():
            torch.compile(fn, backend="eager")()

        torch.compile(fn2, backend="eager")()

    def test_jit_script_defaults(self):
        # [FUNCTION DOC] Checks that a TorchScript function with a default argument value stored
        # as a module attribute can be called without the default inside a fullgraph compile and
        # matches eager.
        @torch.jit.script
        def fast_cos(x, c: float = 2.0):
            # [FUNCTION DOC] HELPER: Scripted function returning cosine of x times the scalar c,
            # default 2.0.
            return torch.cos(x) * c

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the scripted function as an attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the scripted function.
                super().__init__()
                self.fast_cos = fast_cos

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the scripted function on x with defaults.
                return self.fast_cos(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(mod(x), opt_mod(x))

    def test_enum(self):
        # [FUNCTION DOC] Checks a string-based enum with a custom lookup failure hook:
        # constructing a member from a value and comparing with another member compiles under
        # fullgraph; sine is chosen.
        class ExplicitEnum(str, Enum):
            # [CLASS DOC] HELPER: String enum base whose missing-value hook raises ValueError
            # listing valid values.
            @classmethod
            def _missing_(cls, value):
                # [FUNCTION DOC] HELPER: Raises ValueError naming the invalid value and the
                # valid keys.
                raise ValueError(
                    f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
                )

        class PaddingStrategy(ExplicitEnum):
            # [CLASS DOC] HELPER: Padding strategy enum with members longest, max_length
            # and do_not_pad.
            LONGEST = "longest"
            MAX_LENGTH = "max_length"
            DO_NOT_PAD = "do_not_pad"

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the enum member from a string value; returns sine of x
            # if it equals the longest member, else cosine.
            a = PaddingStrategy("longest")
            if a == PaddingStrategy.LONGEST:
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_hasattr_builtin(self):
        # [FUNCTION DOC] Checks that getattr with a default on the type of an argument (including
        # a non-class value 0) and a class attribute are handled under fullgraph.
        class MyClass:
            # [CLASS DOC] HELPER: Class with an int class attribute foo equal to 1.
            foo: int = 1

        def func(x, m):
            # [FUNCTION DOC] HELPER: Returns x plus the class attribute foo if the type of m has a
            # truthy foo, else x.
            if getattr(type(m), "foo", 0):
                return x + MyClass.foo
            return x

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        m = MyClass()
        x = torch.zeros(())
        self.assertEqual(func(x, m), opt_func(x, m))
        self.assertEqual(func(x, 0), opt_func(x, 0))

    def test_grad(self):
        # [FUNCTION DOC] Checks that writes to the grad and _grad attributes of tensors inside a
        # compiled function are visible via the other name and are applied to the real tensors;
        # returned data and resulting grads equal eager.
        # Write to `grad` or `_grad` should reflective in reading from the other,
        # and should be codegen-ed.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets private grad of x to y plus 1 and grad of y to x plus 2;
            # returns the data of x.grad and y._grad.
            x._grad = y + 1
            y.grad = x + 2
            return x.grad.data, y._grad.data

        x0 = torch.randn(4, requires_grad=True)
        y0 = torch.randn(4, requires_grad=True)
        x1 = x0.clone()
        y1 = y0.clone()
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x0, y0), opt_fn(x1, y1))
        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(y0.grad, y1.grad)

    def test_nn_module_stack_bc(self):
        # [FUNCTION DOC] Checks that module stack metadata on graph nodes contains no strings with
        # _module in the name for a plain compile and for a module tagged and marked dynamic
        # through the GenerationTracker (Pippy-like use).
        from torch._dynamo.mutation_guard import GenerationTracker

        def compiler(gm, *args):
            # [FUNCTION DOC] HELPER: Backend that gathers module stack metadata from graph
            # nodes, flattens it, asserts no string contains the text _module, and returns the
            # forward function.
            module_stacks = [
                node.meta.get("nn_module_stack", None) for node in gm.graph.nodes
            ]
            module_stacks, _ = pytree.tree_flatten(module_stacks)
            module_stacks = [x for x in module_stacks if isinstance(x, str)]
            for stack in module_stacks:
                self.assertTrue("_module" not in stack)
            return gm.forward

        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 2 by 2 linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two submodules whose outputs are added.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two submodules.
                super().__init__()
                self.submod1 = SubMod()
                self.submod2 = SubMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns submod1 of x plus submod2 of x.
                return self.submod1(x) + self.submod2(x)

        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

        # an example similar to Pippy usecase
        mod = Mod()
        GenerationTracker.tag(mod.submod1)
        GenerationTracker.mark_class_dynamic(type(mod.submod1))
        mod = Mod()
        opt_mod = torch.compile(mod, backend=compiler)
        opt_mod(torch.randn(2, 2))

    def test_is_make_fx_tracing(self):
        # [FUNCTION DOC] Checks that a private activation helper that detects make_fx tracing can
        # be called inside a fullgraph compiled function.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the make_fx tracing detection helper and returns sine
            # of x.
            torch.nn.modules.activation._is_make_fx_tracing()
            return torch.sin(x)

        fn(torch.rand(4))

    def test_export_vs_dynamo_for_multiheadattention(self):
        # [FUNCTION DOC] Regression for issue 164062: under no_grad in eval mode, neither Dynamo
        # compile nor export of a MultiheadAttention should use the native fast path op; the
        # recorded graphs contain zero such nodes.
        # More details at https://github.com/pytorch/pytorch/issues/164062

        # Ensure that both dynamo and export do not take the fast path.
        with torch.no_grad():
            inp = torch.randn(1, 2, 64)
            mha = nn.MultiheadAttention(64, 2, dropout=0.1, batch_first=True)
            mha.eval()

            backend = EagerAndRecordGraphs()
            mha_compile = torch.compile(mha, backend=backend, fullgraph=True)
            mha_compile(inp, inp, inp)
            torch.compiler.reset()

            mha_export = torch._dynamo.export(mha)(inp, inp, inp)

            compile_nodes = backend.graphs[0].graph.find_nodes(
                op="call_function", target=torch._native_multi_head_attention
            )
            export_nodes = mha_export.graph_module.graph.find_nodes(
                op="call_function", target=torch._native_multi_head_attention
            )
            self.assertEqual(len(compile_nodes), 0)
            self.assertEqual(len(export_nodes), 0)

    def test_multiheadattention_tracing_slowpath_matches_fastpath_layout(self):
        # [FUNCTION DOC] Checks that MultiheadAttention compiled with fullgraph in eval no_grad
        # mode takes the slow path (no native fast-path nodes) yet gives the same result as eager
        # when followed by a view.
        class MHAWithView(nn.Module):
            # [CLASS DOC] HELPER: Module with batch-first multihead attention of width 64 and 8
            # heads whose forward flattens the attention output to rows of the hidden dim.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets hidden_dim to 64 and creates the attention layer.
                super().__init__()
                self.hidden_dim = 64
                self.attention = nn.MultiheadAttention(
                    self.hidden_dim, 8, batch_first=True
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs self-attention on x and views the output as rows
                # of hidden_dim.
                attn_output, _ = self.attention(x, x, x)
                return attn_output.view(-1, self.hidden_dim)

        with torch.no_grad():
            model = MHAWithView().eval()
            x = torch.randn(4, 32, model.hidden_dim)
            eager = model(x)

            backend = EagerAndRecordGraphs()
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            compiled = compiled_model(x)

            compile_nodes = backend.graphs[0].graph.find_nodes(
                op="call_function", target=torch._native_multi_head_attention
            )
            self.assertEqual(compiled, eager)
            self.assertEqual(len(compile_nodes), 0)

    def test_negative_floor_div_solve(self):
        # [FUNCTION DOC] Checks that indexing a tensor with an integer floor-divided index of a
        # mutated attribute works across two compiled calls (the second used to fail).
        class CompiledClass(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor of 1 to 10 and integer counter t starting at
            # 5; forward indexes the tensor by t floor-divided by 12, stores the item in attribute
            # num, increments t and returns it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the tensor and counter.
                super().__init__()
                self.nums = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                self.t = 5

            def forward(self):
                # [FUNCTION DOC] HELPER: Stores the indexed element in num, increments t and
                # returns num.
                self.num = self.nums[self.t // 12]
                self.t += 1
                return self.num

        m = CompiledClass()
        m = torch.compile(m, backend="eager")

        # the first call works
        m()
        # the second call causes a failure
        m()

    # https://github.com/pytorch/pytorch/issues/121621
    def test_tensor_random(self):
        # [FUNCTION DOC] Regression for issue 121621: calling the random_ method with various
        # argument forms (keywords from/to, mixed positional and keyword, all positional) in a
        # compiled function works.
        def random_op(tensor, args, kwargs):
            # [FUNCTION DOC] HELPER: Calls random_ on the tensor with the given positional and
            # keyword arguments and returns it.
            res = tensor.random_(*args, **kwargs)
            return res

        random_op = torch.compile(random_op, backend="eager")
        tensor = torch.randn([2, 3])
        random_op(tensor, [], {"from": -10, "to": 10})
        random_op(tensor, [-10], {"to": 10})
        random_op(tensor, [-10, 10], {})

    # https://github.com/pytorch/pytorch/issues/131019
    def test_tensor_uniform(self):
        # [FUNCTION DOC] Regression for issue 131019: calling uniform_ with various argument forms
        # in a compiled function works.
        def uniform_op(tensor, args, kwargs):
            # [FUNCTION DOC] HELPER: Calls uniform_ on the tensor with the given positional and
            # keyword arguments and returns it.
            res = tensor.uniform_(*args, **kwargs)
            return res

        uniform_op = torch.compile(uniform_op, backend="eager")
        tensor = torch.randn([2, 3])
        uniform_op(tensor, [], {"from": -10, "to": 10})
        uniform_op(tensor, [-10], {"to": 10})
        uniform_op(tensor, [-10, 10], {})

    def test_data_attr_mutation_after_saved_for_bw(self):
        # [FUNCTION DOC] Checks that a mutation of the data attribute of an input after it was
        # saved for backward gives the same output and gradient under aot_eager as eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes sine of x, doubles the data of x in place and
            # returns the sine.
            out = x.sin()
            x.data.mul_(2)
            return out

        x = torch.randn(4, requires_grad=True)
        x_test = x.detach().clone().requires_grad_(True)

        out = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x_test)
        self.assertEqual(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        self.assertEqual(x.grad, x_test.grad)

    # https://github.com/pytorch/pytorch/issues/128072
    def test_map_with_multiple_args(self):
        # [FUNCTION DOC] Regression for issue 128072: builtin map with two iterables of equal or
        # unequal length inside a fullgraph compiled function returns the same type and values
        # as eager.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of products of the first two elements of
            # two sequences.
            return a[0] * b[0] + a[1] * b[1]

        def gen_inps(len_x, len_y):
            # [FUNCTION DOC] HELPER: Builds two lists of random tensors with the given lengths.
            x = [torch.randn(5) for _ in range(len_x)]
            y = [torch.randn(5) for _ in range(len_y)]
            return x, y

        def g(x, y):
            # [FUNCTION DOC] HELPER: Returns the map of f over x and y.
            return map(f, x, y)

        opt_g = torch.compile(g, fullgraph=True, backend="eager")

        inps = gen_inps(3, 3)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

        inps = gen_inps(3, 5)
        self.assertEqual(type(g(*inps)), type(opt_g(*inps)))
        self.assertEqual(tuple(g(*inps)), tuple(opt_g(*inps)))

    def test_staticmethod_allow_in_graph(self):
        # [FUNCTION DOC] Checks that a static method decorated with allow_in_graph that internally
        # graph-breaks is not inlined by Dynamo (otherwise the fullgraph compile would error).
        class MyClass:
            # [CLASS DOC] HELPER: Class with an int class attribute i equal to 3 and two
            # static methods.
            i = 3

            @staticmethod
            def foo_inner(x):
                # [FUNCTION DOC] HELPER: Returns x multiplied by the class attribute i.
                return torch.mul(x, MyClass.i)

            # if dynamo inlines with fullgraph, will error
            # verify that dynamo doesn't inline
            @staticmethod
            @torch._dynamo.allow_in_graph
            def foo1(x):
                # [FUNCTION DOC] HELPER: Allowed-in-graph static method that forces a graph break
                # and calls foo_inner.
                torch._dynamo.graph_break()
                return MyClass.foo_inner(x)

        @torch.compile(backend="eager", fullgraph=True)
        def f_bad(x):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function calling foo1.
            return MyClass.foo1(x)

        f_bad(torch.ones(2, 2))

    def test_guard_with_tuple_mutation(self):
        # [FUNCTION DOC] Checks guards on a mutable object held inside a tuple inside a
        # dictionary: after mutating the attribute through the dictionary, the compiled function
        # (eager backend) matches eager again.
        class Foo:
            # [CLASS DOC] HELPER: Object with an attribute x equal to 10.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets x to 10.
                self.x = 10

        foo = Foo()
        d = {
            "a": 2,
            "b": (foo,),
        }

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Returns x times d at key a times attribute x of the first
            # tuple member in d at key b.
            return x * d["a"] * d["b"][0].x

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp, d), opt_fn(inp, d))
        d["b"][0].x = 12
        self.assertEqual(fn(inp, d), opt_fn(inp, d))

    def test_compile_complex_conj(self):
        # [FUNCTION DOC] Checks complex multiplication by an imaginary scalar on a complex view of
        # a real tensor under aot_eager gives the same value and the same real gradient as eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies x by the imaginary constant 2j.
            return torch.mul(x, 2j)

        x_ref = torch.randn(4, 2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)

        out_ref = f(torch.view_as_complex(x_ref))
        out_test = torch.compile(f, backend="aot_eager")(torch.view_as_complex(x_test))
        self.assertEqual(out_ref, out_test)

        torch.view_as_real(out_ref).sum().backward()
        torch.view_as_real(out_test).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)

    def test_compile_complex_tensor_constant_signed_zero(self):
        # [FUNCTION DOC] Checks that a complex tensor constant created in a compiled function
        # keeps signed zeros (sign of the imaginary part of cosine) and the same string form
        # as eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Builds a complex tensor constant and returns its cosine and
            # its string form.
            y = torch.tensor([1e28 + 2j, -1e-28j])
            return y.cos(), str(y)

        x = torch.tensor([1e28 + 2j, -1e-28j])
        expected_cos, expected_str = f(x)
        actual_cos, actual_str = torch.compile(f, backend="eager")(x)

        self.assertEqual(actual_cos, expected_cos)
        self.assertEqual(
            torch.signbit(actual_cos.imag), torch.signbit(expected_cos.imag)
        )
        self.assertEqual(actual_str, expected_str)

    @unittest.skipIf(
        not SM70OrLater,
        "Triton only supports devices of CUDA capability >= 7.0",
    )
    def test_add_complex_conj(self):
        # [FUNCTION DOC] Skipped unless the GPU supports Triton (SM70 or later). Checks that
        # adding a complex tensor to its conjugate with the eager backend gives a complex64 result
        # equal to twice the real part.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus its conjugate.
            return x + x.conj()

        x = torch.randn(4, dtype=torch.complex64, requires_grad=True)
        out = torch.compile(f, backend="eager")(x)
        expected_complex = (2 * x.real).to(dtype=out.dtype)

        self.assertTrue(out.dtype == torch.complex64)
        self.assertEqual(out, expected_complex)

    # https://github.com/pytorch/pytorch/issues/132200
    def test_partitioner_cse_respects_mutation_boundaries(self):
        # [FUNCTION DOC] Regression for issue 132200: skipped (returns) unless the fsdp set op
        # exists. Checks that the partitioner CSE does not merge identical sine ops across an fsdp
        # mutation of the input: outputs before the copy use the original values and those after
        # use x plus one.
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            # [FUNCTION DOC] HELPER: Takes two sines before and two after an fsdp copy of x plus
            # one into x, returning all four and the square of l.
            # z0 and z1 can be CSEd
            z0 = x.sin()
            z1 = x.sin()
            y = x + 1
            torch.ops.fsdp.copy_.default(x, y)
            # z3 and z3 can be CSEd with each other,
            # but *not* with z0/z1 (they cross a mutation boundary)
            z2 = x.sin()
            z3 = x.sin()
            return z0, z1, z2, z3, l**2

        x = torch.randn(3)
        x_clone = x.clone()
        l = torch.randn(3, requires_grad=True)
        z0, z1, z2, z3, _ = f(x, l)

        # the partitioner runs CSE. We expect that of the 4 sin() ops above:
        # - the first 2 are CSE'd
        # - the last 2 are CSE'd
        # - the set_() op in the middle is a mutation barrier, preventing CSE
        self.assertEqual(z0, (x_clone).sin())
        self.assertEqual(z1, (x_clone).sin())
        self.assertEqual(z2, (x_clone + 1).sin())
        self.assertEqual(z3, (x_clone + 1).sin())

    # https://github.com/pytorch/pytorch/issues/132197
    def test_fsdp_set_input_mutation_applied_when_input_gets_no_gradients(self):
        # [FUNCTION DOC] Regression for issue 132197: skipped (returns) unless the fsdp set op
        # exists. Checks that an input mutation done via the fsdp copy op is applied when the
        # input receives no gradient, comparing results and mutated inputs of two compilations.
        set_available = hasattr(torch.ops, "fsdp") and hasattr(torch.ops.fsdp, "set_")
        if not set_available:
            return

        @torch.compile(backend="aot_eager_decomp_partition")
        def f(x, l):
            # [FUNCTION DOC] HELPER: Takes sine, copies x plus one into x through the fsdp op,
            # takes sine again and returns it with the square of l.
            z = x.sin()  # noqa: F841
            y = x + 1
            # graph input has its storage mutated
            torch.ops.fsdp.copy_.default(x, y)
            z2 = x.sin()
            return z2, l**2

        x = torch.randn(3)
        x_test = x.clone()
        l = torch.randn(3, requires_grad=True)
        result, _ = f(x, l)
        result_test, _ = torch.compile(f, backend="aot_eager_decomp_partition")(
            x_test, l
        )

        self.assertEqual(result, result_test)
        self.assertEqual(x, x_test)

    # https://github.com/pytorch/pytorch/issues/167009
    def test_inbuilt_nn_module_forward_after_hook_graph_break(self):
        # [FUNCTION DOC] Regression for issue 167009: a forward pre-hook that is excluded from
        # Dynamo causes a graph break on a built-in linear module; the linear forward must still
        # be traced afterward (output shape 3 by 5, one recorded graph, readable code includes the
        # linear op).
        # When a hook causes a graph break on an inbuilt nn.Module, the module's
        # forward should still be traced after the graph break.

        @torch._dynamo.disable
        def my_hook(module, inp):
            # [FUNCTION DOC] HELPER: Dynamo-disabled forward pre-hook returning the
            # inputs unchanged.
            return inp

        class Wrapper(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a given module and calling it.
            def __init__(self, lin):
                # [FUNCTION DOC] HELPER: Stores the wrapped module.
                super().__init__()
                self.lin = lin

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the wrapped module.
                return self.lin(x)

        lin = nn.Linear(10, 5)
        lin.register_forward_pre_hook(my_hook)
        model = Wrapper(lin)

        backend = EagerAndRecordGraphs()
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=backend)
        output = compiled(torch.randn(3, 10))

        self.assertEqual(output.shape, torch.Size([3, 5]))
        self.assertEqual(len(backend.graphs), 1)
        graph_code = backend.graphs[0].print_readable(print_output=False)
        self.assertIn("torch._C._nn.linear", graph_code)

    def test_aot_autograd_runtime_wrapper_prologue_profiled(self):
        # [FUNCTION DOC] Checks that profiling a compiled (aot_eager) linear module, with and
        # without grad, records an AOTDispatcher runtime wrapper prologue event that ends before
        # the last event starts.
        # Names for prologue profiling event
        prologue_name = "AOTDispatcher Runtime Wrapper Prologue"

        # Simple linear op to compile
        mod = torch.nn.Linear(4, 4)
        opt_mod = torch.compile(mod, backend="aot_eager")
        x = torch.randn(4, 4)

        # Run this test with grad and no-grad to test both boolean cases trace_joint
        for c in [contextlib.nullcontext, torch.no_grad]:
            # Run compiled op with profiling
            with c():
                # warmup before profiling
                opt_mod(x)
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    opt_mod(x)

            # Make sure events are populated then find prologue event and last start time
            events = prof.events()
            self.assertTrue(events is not None)

            prologue_event = None
            last_start_time = 0
            for event in events:
                if hasattr(event, "name") and prologue_name in event.name:
                    prologue_event = event
                if event.time_range.start > last_start_time:
                    last_start_time = event.time_range.start

            # Make sure prologue event exist
            self.assertTrue(prologue_event is not None)

            # Make sure there is at least one other event (compiled function) that starts
            # after prologue starts
            self.assertLess(prologue_event.time_range.end, last_start_time)

    def test_changing_stride(self):
        # [FUNCTION DOC] Checks that calling a compiled multiply with a view of changing stride
        # and a fixed second argument compiles at most two frames over three shapes.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product.
            return x * y

        for i in range(1, 4):
            x = torch.randn(4, i)

            # create a view for i > 1
            if i == 1:
                x1 = x
            else:
                x1 = x[:, 0:1]

            y = torch.randn(4, 1)
            print(x1.shape, y.shape)
            fn(x1, y)

        self.assertTrue(cnt.frame_count <= 2)

    def test_unsqueeze_mul_strides(self):
        # [FUNCTION DOC] Checks that an unbacked marked input in a multiply with an unsqueezed
        # operand does not crash (previously gave wrong strides); no assertion is needed.
        # This is a case where we had an input that was marked unbacked:
        # size=[2, u0], stride=[1, 1] which is bad. We want it to actually
        # be size=[2, u0], stride=[u0, 1]. See more in the issue below:
        # https://github.com/pytorch/pytorch/issues/142024

        @torch.compile(backend="eager", fullgraph=True)
        def fn(aot6_sub_58, aot6_mul_170):
            # [FUNCTION DOC] HELPER: Unsqueezes the second input at dim 1 with the aten op and
            # multiplies with the first using the aten op.
            aot6_unsqueeze_14 = torch.ops.aten.unsqueeze.default(aot6_mul_170, 1)
            return torch.ops.aten.mul.Tensor(aot6_sub_58, aot6_unsqueeze_14)

        aot6_sub_58 = torch.randn(2, 1)
        torch._dynamo.decorators.mark_unbacked(aot6_sub_58, 1)
        aot6_mul_170 = torch.randn(2)

        # No assert necessary since this used to crash.
        fn(aot6_sub_58, aot6_mul_170)

    def test_optimized_module_training(self):
        # [FUNCTION DOC] Checks that train and eval mode changes propagate between a module and
        # its compiled wrapper in both directions.
        mod = torch.nn.Linear(3, 3)
        mod.eval()

        opt_mod = torch.compile(mod, backend="eager")
        self.assertFalse(opt_mod.training)

        opt_mod.train()
        self.assertTrue(opt_mod.training)
        self.assertTrue(mod.training)

        mod.eval()
        self.assertFalse(opt_mod.training)

    def test_optimized_module_patched_init(self):
        # [FUNCTION DOC] Regression for issue 138157 (deepspeed pattern): a module whose class
        # initializer is patched with a wrapper that records an attribute on the instance compiles
        # and matches eager.
        # A regression test for #138157, and the pattern acame from deepspeed.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x multiplied by 5.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x multiplied by 5.0.
                return x.mul(5.0)

        def patch_init(init):
            # [FUNCTION DOC] HELPER: Returns a wrapper around an initializer that sets a marker
            # attribute on the module the first time and then calls the original initializer.
            @functools.wraps(init)
            def wrapper(module, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Sets the _ds_child_entered attribute if absent and calls
                # the original initializer with the same arguments.
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    module._ds_child_entered = True
                init(module, *args, **kwargs)

            return wrapper

        def patch_init_for_class(cls):
            # [FUNCTION DOC] HELPER: If the class defines its own initializer, saves it as
            # _old_init and replaces it with the patched version.
            if "__init__" in cls.__dict__:
                cls._old_init = cls.__init__
                cls.__init__ = patch_init(cls.__init__)

        patch_init_for_class(MyModule)
        mod = MyModule()
        opt_mod = torch.compile(mod, backend="eager")

        x = torch.rand(10)
        ref = mod(x)
        res = opt_mod(x)

        self.assertEqual(ref, res)

    def test_os_fspath(self):
        # [FUNCTION DOC] Checks that os.fspath on a string inside a fullgraph compiled function
        # is supported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls os.fspath on a path string and returns sine of x.
            os.fspath(".")
            return torch.sin(x)

        fn(torch.randn(4))

    # https://github.com/pytorch/pytorch/issues/189925
    def test_io_text_encoding(self):
        # [FUNCTION DOC] Regression for issue 189925: the low-level io text_encoding function,
        # called with an explicit encoding and with None inside a fullgraph compiled function,
        # returns the same values as outside compile.
        import _io

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls text_encoding with utf-8 and with None and returns sine
            # of x with both results.
            enc_explicit = _io.text_encoding("utf-8")
            enc_default = _io.text_encoding(None)
            return torch.sin(x), enc_explicit, enc_default

        x = torch.randn(4)
        _, enc_explicit, enc_default = fn(x)
        self.assertEqual(enc_explicit, _io.text_encoding("utf-8"))
        self.assertEqual(enc_default, _io.text_encoding(None))

    # https://github.com/pytorch/pytorch/issues/88813
    def test_return_value_duplication_tensor(self) -> None:
        # [FUNCTION DOC] Regression for issue 88813: a function returning two separate products
        # of the same input must keep the two outputs in different storages in eager and in
        # aot_eager compile.
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns two separately computed doubled copies of the input.
            return val * 2, val * 2

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    # https://github.com/pytorch/pytorch/issues/114344
    def test_return_value_duplication_mixed_grad(self) -> None:
        # [FUNCTION DOC] Regression for issue 114344: one output computed under no_grad and
        # one with grad must keep different requires_grad flags matching eager after
        # aot_eager compile.
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns val plus one computed under no_grad and val plus one
            # computed normally.
            with torch.no_grad():
                out0 = val + 1
            out1 = val + 1
            return out0, out1

        x = torch.randn(2, requires_grad=True)

        with torch.enable_grad():
            expect = fn(x)
            actual = torch.compile(fn, backend="aot_eager")(x)

            self.assertEqual(expect[0].requires_grad, actual[0].requires_grad)
            self.assertEqual(expect[1].requires_grad, actual[1].requires_grad)

    # https://github.com/pytorch/pytorch/pull/134726#discussion_r1738774371
    def test_return_value_duplication_scalar(self) -> None:
        # [FUNCTION DOC] Regression from PR 134726 discussion: two scalar elements taken from
        # two separately computed tensors must not share storage in eager or compiled
        # (aot_eager) results.
        def fn(val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Computes two doubled copies and returns the first element
            # of each.
            x, y = val * 2, val * 2
            return x[0], y[0]

        x = torch.randn(2, requires_grad=True)

        expect = fn(x)
        self.assertNotEqual(
            expect[0].untyped_storage().data_ptr(),
            expect[1].untyped_storage().data_ptr(),
        )

        actual = torch.compile(fn, backend="aot_eager")(x)
        self.assertNotEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_torch_compile_in_compile_frame(self):
        # [FUNCTION DOC] Checks that calling torch.compile on an inner function from inside a
        # function that is itself compiled with fullgraph works and gives the eager result.
        def gn(x, c=None):
            # [FUNCTION DOC] HELPER: Multiplies x by c, defaulting c to 2 when None.
            if c is None:
                c = 2
            return c * x

        def outer_func(x):
            # [FUNCTION DOC] HELPER: Compiles gn and calls it on x.
            return torch.compile(gn, backend="eager")(x)

        compile_outer = torch.compile(outer_func, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = outer_func(x)
        res = compile_outer(x)
        self.assertEqual(ref, res)

    @unittest.skipIf(not HAS_MSGSPEC, "missing msgspec package")
    def test_c_defined_metaclass(self):
        # [FUNCTION DOC] Skipped without msgspec. Checks that a msgspec Struct (class with a
        # C-defined metaclass) can be instantiated and read inside a compiled function.
        class User(msgspec.Struct):
            # [CLASS DOC] HELPER: msgspec Struct with a name string and a value int.
            """A new type describing a User"""

            name: str
            value: int

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a User with name alice and value 10 and returns x
            # times its value.
            u = User("alice", 10)
            return x * u.value

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x), opt_fn(x))

    @unittest.skipIf(not HAS_OMEGACONF, "missing omegaconf package")
    def test_omegaconf_dictconfig(self):
        # [FUNCTION DOC] Skipped without omegaconf. Checks a function reading nested omegaconf
        # DictConfig values and writing an entry into the config instance dictionary compiles with
        # fullgraph; results match eager and the deep-copied config has baz equal to 4.
        def fn(cfg, x):
            # [FUNCTION DOC] HELPER: Reads nested config values, multiplies them with x, writes
            # baz into the config instance dictionary and returns the product times baz.
            a = cfg["foo"].a * x
            b = cfg.bar["b"] * a
            cfg.__dict__["baz"] = 4
            return b * cfg.baz

        config = OmegaConf.create({"foo": {"a": 3}, "bar": {"b": 5}})

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        fn(config, x)
        cloned_config = copy.deepcopy(config)
        opt_fn(cloned_config, x)

        self.assertEqual(fn(config, x), opt_fn(config, x))
        self.assertEqual(cloned_config.baz, 4)

    @unittest.skipIf(not HAS_OMEGACONF, "missing omegaconf package")
    def test_omegaconf_listconfig_contains(self):
        # [FUNCTION DOC] Skipped without omegaconf. Checks the membership test on an omegaconf
        # ListConfig inside a fullgraph compiled function matches eager.
        def fn(cfg, x):
            # [FUNCTION DOC] HELPER: Returns sine of x if 1 is in the config, else cosine of x.
            if 1 in cfg:
                return torch.sin(x)
            return torch.cos(x)

        config = OmegaConf.create([1, 2, 3, {"key": "value"}])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(config, x), opt_fn(config, x))

    # https://github.com/pytorch/pytorch/issues/136257
    def test_overwriting_params(self):
        # [FUNCTION DOC] Regression for issue 136257: a module whose per-module parameter
        # dictionaries were replaced with an OrderedDict subclass (ZeRO style) compiles with
        # fullgraph and runs under no_grad.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two 2 by 2 linear layers applied in sequence.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1 then fc2.
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        class ZeROOrderedDict(collections.OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass remembering a parent module and returning
            # entries unchanged on item access (None passes through).
            def __init__(self, parent_module=None, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the base initializer with the remaining arguments
                # and stores the parent module.
                """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

                Args:
                    parent_module (``collections.OrderedDict``): the collection to replace
                """

                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Gets the entry from the base class and returns it
                # (including None).
                param = super().__getitem__(key)

                # Params can be registered as None (e.g., bias)
                if param is None:
                    return param

                # do something here
                return param

        def inject_parameters(module, cls):
            # [FUNCTION DOC] HELPER: For every submodule of the given module, creates a new
            # parameter dictionary of the given class, copies all entries from the existing
            # parameter dictionary and replaces it on the module.
            for module in module.modules():  # noqa: B020
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=module)
                else:
                    new_param = cls()

                for key, param in module._parameters.items():
                    new_param[key] = param
                module._parameters = new_param

        model = M()

        inject_parameters(model, ZeROOrderedDict)

        model = torch.compile(model, backend="eager", fullgraph=True)

        x = torch.ones(2)
        with torch.no_grad():
            model(x)

    def test_typed_dict(self):
        # [FUNCTION DOC] Checks a TypedDict class with a data field constructed inside a fullgraph
        # compiled function, read and updated, matches eager.
        class LlavaImagePixelInputs(TypedDict):
            # [CLASS DOC] HELPER: TypedDict with a literal type field and a tensor data field.
            type: Literal["pixel_values"]
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Constructs the typed dict, multiplies x by its data, assigns
            # 3 to data and returns the product times 3.
            obj = LlavaImagePixelInputs(type=int, data=y)
            out = x * obj["data"]
            obj["data"] = 3
            return out * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    def test_typed_dict_total(self):
        # [FUNCTION DOC] Checks a TypedDict constructed with a keyword named total inside a
        # compiled function matches eager.
        class LlavaImagePixelInputs(TypedDict):
            # [CLASS DOC] HELPER: TypedDict with a literal type field and a tensor data field.
            type: Literal["pixel_values"]
            data: torch.Tensor
            """Shape: `(batch_size, num_channels, height, width)`"""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Constructs the typed dict with data and a total keyword and
            # returns x times its data.
            obj = LlavaImagePixelInputs(data=y, total=False)
            return x * obj["data"]

        x, y = torch.randn(4), torch.randn(4)
        ref = fn(x, y)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

    @skipIfPy312  # listcomp bytecode is optimized
    @skipIfWindows(msg="TODO: (xuhancn) fix, AssertionError: Scalars are not equal!")
    def test_listcomp(self):
        # [FUNCTION DOC] Skipped on Python 3.12 and on Windows. Checks that a list comprehension
        # inside a method excluded from recursion (disabled non-recursively) is fully compiled:
        # one frame and 8 ops, equal to eager.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer count of 4 and a forward summing i times
            # cosine of x for each i.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the count attribute.
                super().__init__()
                self._num = 4

            @torch._dynamo.disable(recursive=False)
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Non-recursively Dynamo-disabled forward that returns the
                # sum of i times cosine of x over the counts.
                values = [i * torch.cos(x) for i in range(self._num)]
                return sum(values)

        mod = Module()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the module on x.
            return mod(x)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 1)
        # Ensure that the listcomp is fully compiled
        self.assertEqual(cnt.op_count, 8)

    # https://github.com/pytorch/pytorch/issues/140266
    def test_distributions_subclass(self):
        # [FUNCTION DOC] Regression for issue 140266: a subclass of the Categorical distribution
        # constructed, sampled and queried inside a fullgraph compiled function, run twice.
        import torch
        from torch.distributions import Categorical

        class SubCateg(Categorical):
            # [CLASS DOC] HELPER: Empty subclass of the Categorical distribution.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def make_dist_and_execute(t, d):
            # [FUNCTION DOC] HELPER: Builds a distribution of the given class from logits and
            # returns the log probability of a sample plus probabilities plus logits.
            categ = d(logits=t)
            a = categ.log_prob(categ.sample()) + categ.probs + categ.logits
            return a

        for _ in range(2):
            make_dist_and_execute(torch.randn(10), SubCateg)

    def test_bitwise_print_precedence(self):
        # [FUNCTION DOC] Checks that a bitwise-or on a symbolic size inside a torch check with
        # floor compiles under dynamic shapes with fullgraph.
        import math

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Checks that floor of (size or 3) times 4 equals 12 and
            # returns sine of x.
            torch._check(math.floor((x.size(0) | 3) * 4) == 12)
            return x.sin()

        f(torch.randn(2))

    def test_tensor_split_within_device_cm(self):
        # [FUNCTION DOC] Checks that tensor split with fullgraph works both at module level and
        # inside a CPU device context manager, with equal results.
        @torch.compile(fullgraph=True, backend="eager")
        def split(x):
            # [FUNCTION DOC] HELPER: Splits x into chunks of 4 along dim 0.
            return x.split(4, 0)

        x = torch.zeros(12)
        res = split(x)

        with torch.device("cpu"):
            self.assertEqual(res, split(x))

    def test_method_overriding(self):
        # [FUNCTION DOC] Checks a submodule that holds a bound method of its parent module as an
        # attribute and calls it; compiled with fullgraph matches eager.
        class DilateConv(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a function and multiplying its call result by
            # sine of x.
            def __init__(
                self,
                dilate_func=None,
            ):
                # [FUNCTION DOC] HELPER: Calls the base initializer and stores the function.
                super().__init__()
                self.dilate_func = dilate_func

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stored function result times sine of x.
                return self.dilate_func() * torch.sin(x)

        class MainModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a child DilateConv constructed with its own bound
            # method and an attribute a of 4.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the child with its bound method and sets a to 4.
                super().__init__()
                self.mod = DilateConv(self.dilate_func)
                self.a = 4

            def dilate_func(self):
                # [FUNCTION DOC] HELPER: Returns the attribute a.
                return self.a

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the child module.
                return self.mod(x)

        mod = MainModule()

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

    def test_symnode_is_op(self):
        # [FUNCTION DOC] Checks that an identity comparison of a symbolic size with a list is
        # false at compile time under dynamic shapes (the else branch is taken).
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            # [FUNCTION DOC] HELPER: Returns x plus one if the size of dim 0 is the list xs, else
            # x doubled.
            if x.size(0) is xs:
                return x + 1
            else:
                return x * 2

        t = torch.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t * 2, res)

    def test_compile_copy__int_overload(self):
        # [FUNCTION DOC] Checks that copy_ with an integer source on a tensor works under
        # aot_eager with fullgraph and gives ones.
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Copies the integer 1 into x and returns x.
            return x.copy_(1)

        t = torch.zeros(2)
        res = f(t)
        self.assertEqual(torch.ones_like(t), res)

    def test_symnode_is_not_op(self):
        # [FUNCTION DOC] Checks that a negated identity comparison of a symbolic size with a list
        # is true at compile time under dynamic shapes.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, xs):
            # [FUNCTION DOC] HELPER: Returns x plus one if the size of dim 0 is not the list xs,
            # else x doubled.
            if x.size(0) is not xs:
                return x + 1
            else:
                return x * 2

        t = torch.randn(2)
        res = f(t, [1, 2])
        self.assertEqual(t + 1, res)

    def test_symint_bitwise(self):
        # [FUNCTION DOC] Checks bitwise or, and, and shifts on a symbolic size, combined with
        # comparisons and modulo, compile with dynamic shapes and fullgraph and match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies in-place style bitwise or, shift and and operations
            # to the first dim, returns a boolean combination and a modulo result.
            z = x.shape[0]
            z |= z >> 1
            z |= z << 1
            z &= z | (z > 1)
            y = (z > 1) | (z <= 1)
            # test composition with non-bitwise ops
            z = (z | z) % 6
            return y, z

        opt_fn = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_bitwise_op_guard(self):
        # [FUNCTION DOC] Checks that guards involving bitwise or and bitwise and of two symbolic
        # sizes can be evaluated, compiled with dynamic shapes and fullgraph.
        # attempt evaluating a guard with BitwiseFn_bitwise_[and/or]
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one if (size0 or size1) exceeds 4 and returns x plus one
            # if (size0 and size1) exceeds 2, else x minus one.
            if x.shape[0] | x.shape[1] > 4:
                x = x + 1
            if x.shape[0] & x.shape[1] > 2:
                return x + 1
            return x - 1

        opt_fn = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        inp = torch.randn(3, 3)
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_ones_out_dynamic(self):
        # [FUNCTION DOC] Checks ones with an out tensor and a tuple size, called with two
        # different shapes, works compiled (eager backend).
        def ones_fn(size, out):
            # [FUNCTION DOC] HELPER: Fills the out tensor with ones of the given size.
            return torch.ones(size, out=out)

        opt_model = torch.compile(ones_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_zeros_out_dynamic(self):
        # [FUNCTION DOC] Checks zeros with an out tensor and a tuple size, called with two
        # different shapes, works compiled.
        def zeros_fn(size, out):
            # [FUNCTION DOC] HELPER: Fills the out tensor with zeros of the given size.
            return torch.zeros(size, out=out)

        opt_model = torch.compile(zeros_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_empty_out_dynamic(self):
        # [FUNCTION DOC] Checks empty with an out tensor and a tuple size, called with two
        # different shapes, works compiled.
        def empty_fn(size, out):
            # [FUNCTION DOC] HELPER: Calls empty with the given size and out tensor.
            return torch.empty(size, out=out)

        opt_model = torch.compile(empty_fn, backend="eager")

        out1 = torch.empty(2, 3)
        opt_model((2, 3), out1)

        out2 = torch.empty(3, 4)
        opt_model((3, 4), out2)

    def test_dataclass_in_module(self):
        # [FUNCTION DOC] Checks that a dataclass held as a module attribute with a float field
        # used to create a tensor in forward works compiled (eager backend) and matches eager.
        @dataclasses.dataclass
        class MyData:
            # [CLASS DOC] HELPER: Dataclass with one float field value.
            value: float

        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Module holding a MyData with value 3.14 and adding it as a
            # tensor in forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the dataclass attribute.
                super().__init__()
                self.my_data = MyData(value=3.14)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus a tensor made from the dataclass value.
                # Make sure to use the scalar 'value' correctly in tensor operations
                value_tensor = torch.tensor(self.my_data.value)
                return x + value_tensor

        model = MyModel()
        inputs = torch.randn(2, 2)
        expected = model(inputs)
        compiled_model = torch.compile(model, backend="eager")
        actual = compiled_model(inputs)
        self.assertEqual(actual, expected)

    def test_no_tracing_into_eval_frame(self):
        # [FUNCTION DOC] Checks that Dynamo does not trace into nested calls triggered by the
        # eval-frame setter: the setter is replaced by a function that graph breaks and calls the
        # original, and a fullgraph compiled function still succeeds.
        # test that dynamo doesn't trace into nested calls from eval_frame
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        orig_fn = torch._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for the eval frame setter that forces a graph
            # break and then calls the original setter with all arguments.
            torch._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("torch._dynamo.eval_frame._maybe_set_eval_frame", bad):
            fn(torch.ones(3))

    @torch._dynamo.config.patch(raise_on_ctx_manager_usage=False)
    def test_no_tracing_into_eval_frame_ctx_manager(self):
        # [FUNCTION DOC] With context-manager-usage errors disabled, checks the same property as
        # the previous test when using the optimize_assert context manager: the eval-frame setter
        # that graph breaks must not be traced.
        # Test that dynamo doesn't trace into nested calls from eval_frame
        # when using a context manager.
        # Even though we don't officially support Dynamo context managers, we still
        # have tests that use them, so we should still make sure the eval_frame callback
        # is set at the correct places in these cases.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        orig_fn = torch._dynamo.eval_frame._maybe_set_eval_frame

        def bad(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement setter that forces a graph break and then calls
            # the original.
            torch._dynamo.graph_break()
            return orig_fn(*args, **kwargs)

        with mock.patch("torch._dynamo.eval_frame._maybe_set_eval_frame", bad):
            with torch._dynamo.optimize_assert("eager"):
                fn(torch.ones(3))

    @torch._dynamo.config.patch(allow_empty_graphs=True)
    @parametrize("fullgraph", [True, False])
    def test_empty_graph_nested_calls(self, fullgraph):
        # [FUNCTION DOC] With empty graphs allowed and parametrized on fullgraph, checks that
        # nested identity functions compile as a single top-level frame: after clearing the
        # thread-local state, the traced frame count is 1 after the first call, 1 after a
        # same-shape call, and 2 after a new shape.
        def k(x):
            # [FUNCTION DOC] HELPER: Returns x unchanged.
            return x

        def g(x):
            # [FUNCTION DOC] HELPER: Calls k.
            return k(x)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls g.
            return g(x)

        # TODO clear this on all tests
        torch._dynamo.eval_frame.clear_dynamo_tls()

        opt_f = torch.compile(f, backend="eager", fullgraph=fullgraph, dynamic=False)
        opt_f(torch.randn(3))
        # we should not be compiling g or h as top-level functions
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # no recompilation
        opt_f(torch.randn(3))
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 1)
        # recompilation
        opt_f(torch.randn(4))
        self.assertEqual(len(torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos), 2)

    def test_torchname(self):
        # [FUNCTION DOC] Checks that the typename helper on a typing object gives the same result
        # compiled and in eager.
        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the torch type name string of the object.
            return torch.typename(obj)

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(typing.Any), opt_fn(typing.Any))

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
    )
    def test_compilation_metrics_on_error(self):
        # [FUNCTION DOC] With a recompile limit of 1 and failure on hitting it, checks compilation
        # metrics: after the first call one metrics entry exists with config recorded and no
        # recompile reason; after a second input triggering a recompile, FailOnRecompileLimitHit
        # is raised and a second entry with a recompile reason exists. Clears metrics at the start
        # and end.
        torch._dynamo.utils.clear_compilation_metrics()

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function branching on the number of elements (100 or
            # 10000) to force a recompile for a differently sized input.
            # force a recompile in a way friendly to test_dynamic_shapes
            if x.numel() == 100:
                return x.sum()
            elif x.numel() == 10000:
                return x.sum()

        x = torch.randn(10, 10)
        y = torch.randn(100, 100)
        metrics = torch._dynamo.utils._compilation_metrics
        self.assertEqual(len(metrics), 0)

        fn(x)
        self.assertTrue(metrics is torch._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 1)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is None)

        with self.assertRaises(torch._dynamo.exc.FailOnRecompileLimitHit):
            fn(y)
        self.assertTrue(metrics is torch._dynamo.utils._compilation_metrics)
        self.assertEqual(len(metrics), 2)
        latest_metrics = metrics[-1]
        self.assertTrue(latest_metrics.dynamo_config is not None)
        self.assertTrue(latest_metrics.recompile_reason is not None)

        torch._dynamo.utils.clear_compilation_metrics()

    def test_incompatible_configs(self):
        # [FUNCTION DOC] Checks config validation: the combinations of suppress errors and fail on
        # recompile limit hit that are compatible do not raise when compiling, but enabling both
        # raises AssertionError.
        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=False
        ):
            torch.compile(lambda: None, backend="eager")

        with torch._dynamo.config.patch(
            suppress_errors=True, fail_on_recompile_limit_hit=False
        ):
            torch.compile(lambda: None, backend="eager")

        with torch._dynamo.config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True
        ):
            torch.compile(lambda: None, backend="eager")

        with (
            torch._dynamo.config.patch(
                suppress_errors=True, fail_on_recompile_limit_hit=True
            ),
            self.assertRaises(AssertionError),
        ):
            torch.compile(lambda: None, backend="eager")

    def test_str_isalnum(self):
        # [FUNCTION DOC] Checks that the unbound str method isalnum can be called on a string in a
        # fullgraph compiled function.
        def f(x, c):
            # [FUNCTION DOC] HELPER: Calls the string isalnum method (result unused) and returns
            # sine of x.
            str.isalnum(c)
            return x.sin()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3)
        c = "foobar"
        self.assertEqual(f(x, c), opt_f(x, c))

    def test_nn_param_freevar_codegen(self):
        # [FUNCTION DOC] Checks that parameters created inside forward from free tensors
        # (convolution weight and bias) and eval mode switching are code-generated
        # correctly, comparing a functools wrapped module call compiled with fullgraph to
        # eager under no_grad.
        class Model2(nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, batch norm and two plain tensors,
            # which in forward are wrapped as parameters and assigned to the convolution before
            # applying conv, batch norm and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and two random tensors for weight
                # and bias.
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assigns new Parameters to the convolution weight and bias
                # (bias without grad), sets the convolution to eval, then applies conv, batch norm
                # and relu.
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that forwards all arguments to the module.
            return func(*args, **kwargs)

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(wrapper, backend="eager", fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    def test_amp_foreach_fake_impl(self):
        # [FUNCTION DOC] Checks the AMP foreach non-finite check and unscale in place op gives the
        # same result under aot_eager (fake implementation) as eager.
        inv_scale = torch.full((1,), 0.25)
        found_inf = torch.full((1,), 0.0)
        grads = [torch.ones(10), torch.ones(10)]

        def f():
            # [FUNCTION DOC] HELPER: Runs the in-place unscale and finite check on the global
            # gradient list with the found-inf and inverse scale tensors and returns the result.
            res = torch._amp_foreach_non_finite_check_and_unscale_(
                grads, found_inf, inv_scale
            )
            return res

        ref = f()
        res = torch.compile(f, backend="aot_eager")()
        self.assertEqual(ref, res)

    def test_guard_tag_safe_tensor_metadata_segfault(self):
        # [FUNCTION DOC] Regression for issue 180741: a model with a submodule containing a numpy
        # float64 attribute that is accessed via a lambda guard accessor must compile and run
        # without segfault; checks output shape 2 by 10.
        # Regression test for https://github.com/pytorch/pytorch/issues/180741
        # When a submodule becomes a tag-safe root, the recording pass stashes
        # tensor pointers found during traversal.  PythonLambdaGuardAccessor
        # (from ___from_numpy on the np.float64 attribute) creates a temporary
        # tensor with refcount 1, which is freed after Py_DECREF.  The stashed
        # raw pointer then dangles and check_tensor_metadata_fast segfaults.
        import numpy as np

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and a numpy float64 scale attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer and sets scale to 8.0.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.scale = np.float64(8.0)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear output divided by the scale.
                return self.linear(x) / self.scale

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleList with one Layer applied in sequence.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the module list.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies every layer in order.
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Model()
        compiled = torch.compile(model, backend="eager")
        out = compiled(torch.randn(2, 10))
        self.assertEqual(out.shape, torch.Size([2, 10]))

    def test_deleted_compile_wrapper_segfault(self):
        # [FUNCTION DOC] Checks that resetting Dynamo between creating two compiled wrappers
        # of the same function does not cause a crash (possible segfault when the first
        # wrapper is deleted).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        # This calls cached_backend.clear() which removes any strong references
        # to the callback
        torch._dynamo.reset()
        opt_fn(torch.randn(3))
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3))  # possible segfault due to first opt_fn deletion

    def test_delete_local_error(self):
        # [FUNCTION DOC] Checks that using a local variable after deleting it in a fullgraph
        # compiled function raises the Dynamo Unsupported error.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Deletes y after computing it and then uses it.
            y = x + 1
            del y
            z = y + 1  # noqa: F821
            return z

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(torch.ones(3))

    def test_nanmean_out(self):
        # [FUNCTION DOC] Checks nanmean with out into a scalar tensor under fullgraph gives the
        # same out value as eager.
        def f(x, out):
            # [FUNCTION DOC] HELPER: Calls nanmean on x writing to out.
            torch.nanmean(x, out=out)

        x = torch.randn(4)
        out_ref = torch.tensor(0.0)
        out_res = torch.tensor(0.0)

        f(x, out_ref)
        torch.compile(f, backend="eager", fullgraph=True)(x, out_res)
        self.assertEqual(out_ref, out_res)

    def test_gather_out_dynamic_shapes(self):
        # [FUNCTION DOC] Checks gather with out and dynamic shapes over two widths gives the same
        # out values and return as eager.
        def f(x, index, out):
            # [FUNCTION DOC] HELPER: Gathers along dim 1 with the index into out and returns out.
            torch.gather(x, 1, index, sparse_grad=False, out=out)
            return out

        opt_f = torch.compile(f, backend="eager", fullgraph=True, dynamic=True)
        for width in (2, 3):
            x = torch.randn(1, width)
            index = torch.randint(width, (1, 1))
            out_ref = torch.empty(1, 1)
            out_res = torch.empty(1, 1)

            f(x, index, out_ref)
            res = opt_f(x, index, out_res)
            self.assertEqual(out_ref, out_res)
            self.assertEqual(out_ref, res)

    @skipIfNotPy312
    def test_sys_monitoring(self):
        # [FUNCTION DOC] Skipped unless Python 3.12. Checks that sys.monitoring callbacks
        # still fire while Dynamo runs (inside the symbolic convert run function) and on the
        # compiled graph call, using a registered tool id and restoring events, callback and
        # tool id afterwards.
        found_dynamo = False
        found_compiled_graph = False
        compiled_graph = None

        def backend(gm, _):
            # [FUNCTION DOC] HELPER: Backend recording the graph module in a nonlocal variable and
            # returning it.
            nonlocal compiled_graph
            compiled_graph = gm
            return gm

        def callback(code, offset):
            # [FUNCTION DOC] HELPER: Monitoring callback that forces a graph break, sets
            # found_dynamo when called for the symbolic convert run code and sets
            # found_compiled_graph when called for the compiled graph call code.
            nonlocal found_dynamo
            nonlocal found_compiled_graph
            torch._dynamo.graph_break()
            if (
                code
                is torch._dynamo.symbolic_convert.InstructionTranslator.run.__code__
            ):
                found_dynamo = True
            elif compiled_graph and code is compiled_graph.__call__.__code__:
                found_compiled_graph = True

        tool_id = 0
        sys.monitoring.use_tool_id(tool_id, "test")
        old_events = sys.monitoring.get_events(tool_id)
        old_callback = sys.monitoring.register_callback(
            tool_id, sys.monitoring.events.PY_START, callback
        )
        sys.monitoring.set_events(tool_id, sys.monitoring.events.PY_START)
        try:

            @torch.compile(backend=backend, fullgraph=True)
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled function returning x plus one.
                return x + 1

            fn(torch.ones(3))
            # sys.monitoring should still run in Python dynamo
            self.assertTrue(found_dynamo)
            # sys.monitoring should still run on the compiled graph
            self.assertTrue(found_compiled_graph)
        finally:
            sys.monitoring.set_events(tool_id, old_events)
            sys.monitoring.register_callback(
                tool_id, sys.monitoring.events.PY_START, old_callback
            )
            sys.monitoring.free_tool_id(tool_id)

    def test_312_local_cell_overlap(self):
        # [FUNCTION DOC] Checks (Python 3.12 cell/local overlap) that a list comprehension whose
        # variable has the same name as a later nonlocal cell variable, after a graph break, gives
        # the same result compiled and eager.
        keys = range(10)
        allowed = [0, 1, 2, 3]

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, breaks the graph, builds a list of allowed keys,
            # defines an inner function declaring the variable nonlocal and returns x plus the
            # first key.
            x = x + 1
            torch._dynamo.graph_break()
            key = [key for key in keys if key in allowed]

            def inner():
                # [FUNCTION DOC] HELPER: Inner function that only declares key nonlocal and
                # does nothing.
                nonlocal key

            return x + key[0]

        self.assertEqual(
            fn(torch.ones(3)), torch.compile(fn, backend="eager")(torch.ones(3))
        )

    def test_cells_unsupported_step_exception(self):
        # [FUNCTION DOC] Regression check for stack handling: after an unsupported step graph
        # break, raising an exception inside a context manager block must propagate AssertionError
        # correctly even though cells are on the stack.
        # This error happened because:
        #  - we were generating cells into a list on the stack
        #  - we encountered an unsupported step, resulting in a step graph break
        #  - we encounter an exception, which pops the stack until it reaches a certain length;
        #    the presence of the list of cells then messes things up.

        cell = 0

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 3, triggers a step-unsupported break, enters a null
            # context, prints the free variable and raises AssertionError.
            x = x + 1 + 2
            torch._dynamo.step_unsupported()
            with contextlib.nullcontext():
                print(cell)
                raise AssertionError

        with self.assertRaises(AssertionError):
            fn(torch.ones(3))

    def test_unbind_copy_out(self):
        # [FUNCTION DOC] Checks unbind_copy with a tuple of out tensors under fullgraph gives the
        # same outputs as eager.
        def f(eye, out):
            # [FUNCTION DOC] HELPER: Calls unbind_copy on the matrix with out.
            torch.unbind_copy(eye, out=out)

        eye = torch.eye(3)
        out_ref = (torch.zeros(3), torch.zeros(3), torch.zeros(3))
        out_res = (torch.zeros(3), torch.zeros(3), torch.zeros(3))

        f(eye, out_ref)
        torch.compile(f, backend="eager", fullgraph=True)(eye, out_res)
        self.assertEqual(out_ref, out_res)

    def test_setitem_tensor_prop(self):
        # [FUNCTION DOC] Checks that assigning a row from a tensor requiring grad into a plain
        # tensor makes it require grad before a custom autograd function; gradients of the source
        # match between eager and compiled.
        # Using the composite implicit of the forward would be incorrect
        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing x matmul x transposed whose backward
            # passes the gradient through unchanged.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns x matmul its transpose.
                return torch.matmul(x, x.t())

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Returns grad_out unchanged.
                return grad_out

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Copies the first row of y into x and applies MyFn to x.
            x[0] = y[0]
            return MyFn.apply(x)

        def inputs():
            # [FUNCTION DOC] HELPER: Seeds the RNG and returns a random 10 by 10 tensor without
            # grad and one with grad.
            torch.manual_seed(123)
            x = torch.randn(10, 10)
            y = torch.randn(10, 10, requires_grad=True)
            return x, y

        x1, y1 = inputs()
        fn(x1, y1).sum().backward()
        self.assertTrue(x1.requires_grad)

        x2, y2 = inputs()
        torch.compile(fn, backend="eager")(x2, y2).sum().backward()
        self.assertTrue(x2.requires_grad)

        self.assertEqual(y1.grad, y2.grad)

    def test_nn_parameter_ctor_graph_breaks(self):
        # [FUNCTION DOC] Checks that constructing a Parameter inside a compiled function causes
        # exactly one graph break and the explain output reason matches the expected message text,
        # which names the Parameter constructor, hints and a link to the graph break page.
        def fn():
            # [FUNCTION DOC] HELPER: Creates a Parameter of ones and returns it doubled.
            param = torch.nn.Parameter(torch.ones(10))
            return param * 2

        self.maxDiff = None
        eb = ExplainWithBackend("eager")
        optimized_fn = torch.compile(fn, backend=eb)
        _ = optimized_fn()
        explain_output = eb.output()
        self.assertEqual(explain_output.graph_break_count, 1)
        expected_msg = (
            "Attempted to use `torch.nn.Parameter()` constructor with Dynamo\n"
            "  Explanation: Dynamo does not support this\n"
            "  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.\n"
            "  Hint: If this is not possible, turn `graph_break_on_nn_param_ctor` off\n"
            "  Hint: It may be possible to write Dynamo tracing rules for this code. "
            "Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.\n\n"
            "  Developer debug context: \n\n"
            " For more details about this graph break, please visit: "
            "https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0264.html"
        )
        self.assertEqual(explain_output.break_reasons[0].reason, expected_msg)

    @parametrize("backend", ["eager", "inductor"])
    def test_issue164247(self, backend: str):
        # [FUNCTION DOC] Parametrized on backend (eager, inductor); regression for issue 164247. A
        # model that uses a captured tensor in a mask function for flex attention and calls both a
        # compiled and an uncompiled flex attention must compile with fullgraph as one graph and
        # one frame; inductor is expected to raise RuntimeError (known issue 157612) and is
        # skipped under dynamic-shape wrappers.
        if backend == "inductor" and torch._dynamo.config.dynamic_shapes:
            raise unittest.SkipTest(
                "Skip only in dynamic-shapes wrapper (known issue #157612)"
            )

        class MixedFakeModeModel(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward builds a block mask
            # from a data dependent mask function and calls flex attention twice (once compiled).
            def __init__(self, dim=64):
                # [FUNCTION DOC] HELPER: Stores the dimension and creates a 64 by 64 linear layer.
                super().__init__()
                self.dim = dim
                self.lin = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, builds a block mask that
                # thresholds a captured sum tensor, creates detached query, key and value views,
                # runs a compiled flex attention and then an uncompiled one, and returns the
                # last output.
                batch_size, seq_len, _ = x.shape

                # Process input first - this creates fake tensors in export's fake mode
                processed = self.lin(x)

                # Create some computation that depends on processed tensor
                intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

                def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                    # [FUNCTION DOC] HELPER: Mask function returning whether kv index does not
                    # exceed q index and the captured threshold is positive.
                    threshold = intermediate[
                        batch_idx, q_idx % seq_len
                    ]  # Access the captured tensor
                    return (kv_idx <= q_idx) & (threshold > 0)

                block_mask = create_block_mask(
                    mask_mod=dynamic_mask_function,
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=x.device,
                    _compile=False,
                )
                q = processed.view(batch_size, 1, seq_len, self.dim).detach()
                k = processed.view(batch_size, 1, seq_len, self.dim).detach()
                v = processed.view(batch_size, 1, seq_len, self.dim).detach()

                out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)  # noqa: UNSPECIFIED_BACKEND
                out = flex_attention(q, k, v, block_mask=block_mask)

                return out

        backend_counter = CompileCounterWithBackend(backend)
        model = MixedFakeModeModel()
        compiled = torch.compile(model, backend=backend_counter, fullgraph=True)

        if backend == "inductor":
            # A known InductorError Issue https://github.com/pytorch/pytorch/issues/157612
            with self.assertRaises(RuntimeError):
                compiled(torch.randn(2, 128, 64))
        else:
            compiled(torch.randn(2, 128, 64))

        # One graph, so no graph breaks
        self.assertEqual(backend_counter.frame_count, 1)
        self.assertEqual(len(backend_counter.graphs), 1)

    def test_flex_attention_non_strict_unbacked_sequence_length(self):
        # [FUNCTION DOC] Checks flex attention in non-strict tracing with a fake tensor mode, an
        # unbacked sequence length (hinted to 128) and a block mask built from kv blocks produces
        # output with the symbolic shape.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        seq = shape_env.create_unbacked_symint()
        shape_env._set_unbacked_var_to_hint_override(seq, 128)

        with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
            q = torch.empty((1, 1, seq, 8), device="cpu")
            k = torch.empty((1, 1, seq, 8), device="cpu")
            v = torch.empty((1, 1, seq, 8), device="cpu")
            kv_num_blocks = torch.ones((1, 1, 1), dtype=torch.int32)
            kv_indices = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
            block_mask = BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                BLOCK_SIZE=128,
                seq_lengths=(seq, seq),
            )

            with (
                torch.compiler._non_strict_tracing_context(),
                torch._dynamo.config.patch(force_compile_during_fx_trace=True),
            ):
                out = flex_attention(q, k, v, block_mask=block_mask)

        self.assertEqual(out.shape, (1, 1, seq, 8))

    # https://github.com/pytorch/pytorch/issues/164990
    def test_guard_same_frame_fail_message(self):
        # [FUNCTION DOC] Regression for issue 164990: patches the guard check function compiler so
        # that the installed guard check always returns False; compiling then must raise
        # AssertionError whose message says the guard failed on the same frame.
        import torch._dynamo.guards as g

        # deterministically fail check on the same frame to verify error message correctness
        # the other example of fail might be datetime.now() until patched - see issue #164990
        compile_check_fn = g.CheckFunctionManager.compile_check_fn

        def wrapper(self, builder, sorted_guards, guard_fail_fn):
            # [FUNCTION DOC] HELPER: Replacement for compile_check_fn that calls the original and
            # then overrides the guard manager check with a function always returning False.
            compile_check_fn(self, builder, sorted_guards, guard_fail_fn)

            def check(x):
                # [FUNCTION DOC] HELPER: Guard check replacement that returns False for any input.
                return False

            self.guard_manager.check = check

        with mock.patch.object(g.CheckFunctionManager, "compile_check_fn", new=wrapper):

            class Model(nn.Module):
                # [CLASS DOC] HELPER: Module returning x plus one.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns x plus one.
                    return x + 1

            model = Model()
            x = torch.randn(5)

            with self.assertRaises(AssertionError) as e:
                torch.compile(model, backend="eager")(x)

        msg = str(e.exception)
        self.assertIn(
            "Guard failed on the same frame it was created. This is a bug - please create an issue."
            "Guard fail reason: ",
            msg,
        )

    @xfailIfS390X
    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit(self):
        # [FUNCTION DOC] Skipped on s390x failures and unless Python is 3.12 or 3.13. Checks that
        # raising the Dynamo recursion limit allows compiling a function recursing 500 times:
        # first with a low recursion limit eager raises RecursionError, then with a high python
        # limit compile still fails, then after setting the Dynamo limit it matches eager.
        # Restores both limits in a finally block.
        old_recursion_limit = sys.getrecursionlimit()
        old_dynamo_recursion_limit = torch._dynamo.get_recursion_limit()
        try:

            def fn(x, n):
                # [FUNCTION DOC] HELPER: Recursive function that adds one n times.
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            sys.setrecursionlimit(100)

            with self.assertRaises(RecursionError):
                fn(torch.ones(3), 500)

            sys.setrecursionlimit(1000)

            fn(torch.ones(3), 500)
            opt_fn = torch.compile(fn, backend="eager", dynamic=False)
            sys.setrecursionlimit(20000)
            with self.assertRaises(Exception):
                opt_fn(torch.ones(3), 500)

            torch._dynamo.set_recursion_limit(20000)
            self.assertEqual(fn(torch.ones(3), 500), opt_fn(torch.ones(3), 500))
        finally:
            torch._dynamo.set_recursion_limit(old_dynamo_recursion_limit)
            sys.setrecursionlimit(old_recursion_limit)

    @unittest.skipIf(
        sys.version_info < (3, 12) or sys.version_info >= (3, 14),
        "only 3.12, 3.13 affected by c recursion limit",
    )
    def test_dynamo_set_recursion_limit_usage(self):
        # [FUNCTION DOC] Skipped unless Python is 3.12 or 3.13. Checks the Dynamo recursion limit
        # API: set and get a value of 500, a RuntimeWarning about the new C recursion limit when
        # compiling a recursive function, and ValueError when setting 0 (the limit stays 500).
        # Restores the original limit.
        old_dynamo_recursion_limit = torch._dynamo.get_recursion_limit()
        try:
            torch._dynamo.set_recursion_limit(500)
            self.assertEqual(torch._dynamo.get_recursion_limit(), 500)

            @torch.compile(backend="eager", dynamic=False)
            def fn(x, n):
                # [FUNCTION DOC] HELPER: Compiled recursive function adding one n times.
                if n == 0:
                    return x
                return fn(x, n - 1) + 1

            # a limit of 500 should be lower than the default limit
            with self.assertWarnsRegex(RuntimeWarning, "new c_recursion limit"):
                fn(torch.ones(3), 5)

            with self.assertRaisesRegex(ValueError, "recursion limit"):
                torch._dynamo.set_recursion_limit(0)

            self.assertEqual(torch._dynamo.get_recursion_limit(), 500)
        finally:
            torch._dynamo.set_recursion_limit(old_dynamo_recursion_limit)

    @expectedFailureDynamic
    def test_dynamo_default_lru_cache_behavior(self):
        # [FUNCTION DOC] Expected failure under dynamic shapes. Checks the default LRU behavior of
        # the Dynamo cache: after compiling a static then a dynamic graph, rerunning the first
        # input moves the dynamic entry to the front of the cache entry list.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning x plus 10.
            return x + 10

        torch._dynamo.reset()
        if torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        ):
            raise AssertionError("Expected no cache entries after reset")

        # Step 1: Compile a static shapes graph
        x = torch.randn(10, 10)
        fn(x)
        a = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(a), 1)
        static_shapes_cache_entry = a[0]

        # Step 2: Compile a dynamic shapes graph
        y = torch.randn(20, 20)
        fn(y)
        b = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(b), 2)
        self.assertEqual(b[1], static_shapes_cache_entry)
        dynamic_shapes_cache_entry = b[0]

        # Step 3: Run with Step 1's inputs
        # LRU cache will match against dynamic shape graph first
        fn(x)
        c = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
            fn._torchdynamo_orig_callable.__code__
        )
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], dynamic_shapes_cache_entry)
        self.assertEqual(c[1], static_shapes_cache_entry)

    @expectedFailureDynamic
    def test_dynamo_disable_lru_cache_behavior(self):
        # [FUNCTION DOC] Expected failure under dynamic shapes. Checks that with the LRU cache
        # disabled, the static entry stays first in the cache entry list after reruns.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning x plus 10.
            return x + 10

        def run():
            # [FUNCTION DOC] HELPER: Resets Dynamo, compiles a static graph for a 10 by 10 input
            # and a dynamic graph for a 20 by 20 input, and asserts the cache entry order (static
            # first, dynamic second) is unchanged after rerunning the first input; the test body
            # disables the LRU cache around this call and restores it.
            torch._dynamo.reset()
            if torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            ):
                raise AssertionError("Expected no cache entries after reset")

            # Step 1: Compile a static shapes graph
            x = torch.randn(10, 10)
            fn(x)
            a = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(a), 1)
            static_shapes_cache_entry = a[0]

            # Step 2: Compile a dynamic shapes graph
            y = torch.randn(20, 20)
            fn(y)
            b = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(b), 2)
            self.assertEqual(b[0], static_shapes_cache_entry)
            dynamic_shapes_cache_entry = b[1]

            # Step 3: Run with Step 1's inputs
            # LRU cache is disabled, we should still have static entry first
            fn(x)
            c = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(
                fn._torchdynamo_orig_callable.__code__
            )
            self.assertEqual(len(c), 2)
            self.assertEqual(c[0], static_shapes_cache_entry)
            self.assertEqual(c[1], dynamic_shapes_cache_entry)

        try:
            torch._C._dynamo.eval_frame._set_lru_cache(False)
            run()
        finally:
            torch._C._dynamo.eval_frame._set_lru_cache(True)

    def test_patch_track_step_called_skipped(self):
        # [FUNCTION DOC] Regression test: the LR scheduler step-tracking patch helper must be
        # skipped by Dynamo even when the scheduler file is not in the forced skip set. Clears
        # that set, compiles a function that creates a StepLR scheduler with fullgraph in one
        # frame, then restores the set.
        # Regression test for patch_track_step_called being ignored by dynamo
        # We need to clear FORCE_SKIP_FILES to test that the function name check
        # properly ignores patch_track_step_called even when lr_scheduler.py is not
        # in FORCE_SKIP_FILES
        import torch._dynamo.trace_rules as trace_rules

        old_force_skip_files = trace_rules.FORCE_SKIP_FILES
        try:
            trace_rules.FORCE_SKIP_FILES = set()

            cnt = CompileCounter()

            @torch.compile(backend=cnt, fullgraph=True)
            def fn(x, optimizer):
                # [FUNCTION DOC] HELPER: Compiled function that creates a StepLR scheduler (step
                # size 1) for the optimizer and returns x doubled together with the scheduler.
                # Create an LR scheduler which internally calls patch_track_step_called
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                return x * 2, scheduler

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            x = torch.randn(10, 10)

            result, _ = fn(x, optimizer)
            expected = x * 2
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
        finally:
            trace_rules.FORCE_SKIP_FILES = old_force_skip_files

    @parametrize("set_type", [set, frozenset], name_fn=lambda t: t.__name__)
    def test_set_doesnt_recompile_with_ac(self, set_type):
        # [FUNCTION DOC] Parametrized over set and frozenset. Checks that selective activation
        # checkpointing with a policy based on a set of ops does not trigger a recompile (errors
        # on recompile enabled).
        import torch

        with torch._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from torch.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Checkpoint policy returning must-save for ops in the
                # given set and prefer-recompute otherwise.
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                # [FUNCTION DOC] HELPER: Returns exp of sine of x matmul x.
                return torch.mm(x, x).sin().exp()

            @torch.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                # [FUNCTION DOC] HELPER: Compiled function running checkpoint of g with the given
                # policy contexts and non-reentrant mode.
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = torch.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([torch.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, set_type([torch.ops.aten.mm.default])),
                ),
            )

    # https://github.com/pytorch/pytorch/issues/151296
    def test_select_scatter_mixed_dtype(self):
        # [FUNCTION DOC] Regression for issue 151296: select_scatter of an int64 source into a
        # float tensor works under the eager backend and matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that builds an int64 one-element source tensor and
            # select-scatters it into the input at dim 1, index 0.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates the integer source and returns the
                # select_scatter result.
                src = torch.tensor([0])
                out = torch.select_scatter(x, src, 1, 0)
                return out

        model = Model()
        x = torch.randn(1, 10)
        inputs = [x]

        compiled_model = torch.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    # https://github.com/pytorch/pytorch/issues/152307
    def test_fp8_qdq_repeated_modules_no_recompile(self):
        # [FUNCTION DOC] Regression for issue 152307: three stacked FP8 quantize-dequantize linear
        # blocks, each with its own buffers, compiled once under no_grad with recompiles treated
        # as errors: neither the first nor the second call may recompile, and both match eager.
        class FP8QDQLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear layer with a float8 quantized weight buffer, its scale, a
            # bias buffer, and a fixed input scale; forward dequantizes the weight, quantizes and
            # dequantizes the input and applies functional linear.
            def __init__(self, in_features, out_features):
                # [FUNCTION DOC] HELPER: Builds a positive random weight, computes its scale from
                # the amax and float8 max, stores the clamped float8 weight, weight scale and bias
                # as buffers and sets the input scale.
                super().__init__()
                weight = torch.randn(out_features, in_features).abs().add(0.01)
                scale = weight.amax() / torch.finfo(torch.float8_e4m3fn).max
                self.register_buffer(
                    "qweight",
                    torch.clamp(
                        weight / scale,
                        torch.finfo(torch.float8_e4m3fn).min,
                        torch.finfo(torch.float8_e4m3fn).max,
                    ).to(torch.float8_e4m3fn),
                )
                self.register_buffer("weight_scale", scale)
                self.register_buffer("bias", torch.randn(out_features))
                self.scale = 1 / torch.finfo(torch.float8_e4m3fn).max

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Dequantizes the stored weight, quantizes and then
                # dequantizes x per tensor using the float8 range, and returns linear of that with
                # the weight and bias.
                from torch.ao.quantization.fx._decomposed import (
                    dequantize_per_tensor,
                    quantize_per_tensor,
                )

                weight = dequantize_per_tensor(
                    self.qweight,
                    self.weight_scale,
                    0,
                    torch.finfo(torch.float8_e4m3fn).min,
                    torch.finfo(torch.float8_e4m3fn).max,
                    self.qweight.dtype,
                    out_dtype=torch.float,
                )
                qx = quantize_per_tensor(
                    x,
                    self.scale,
                    0,
                    torch.finfo(torch.float8_e4m3fn).min,
                    torch.finfo(torch.float8_e4m3fn).max,
                    torch.float8_e4m3fn,
                )
                dx = dequantize_per_tensor(
                    qx,
                    self.scale,
                    0,
                    torch.finfo(torch.float8_e4m3fn).min,
                    torch.finfo(torch.float8_e4m3fn).max,
                    qx.dtype,
                    out_dtype=torch.float,
                )
                return torch.nn.functional.linear(dx, weight, self.bias)

        class Block(torch.nn.Module):
            # [CLASS DOC] HELPER: Block holding one FP8 quantized linear layer followed by relu.
            def __init__(self, in_features, out_features):
                # [FUNCTION DOC] HELPER: Creates the quantized linear layer.
                super().__init__()
                self.linear = FP8QDQLinear(in_features, out_features)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns relu of the linear layer output.
                return torch.relu(self.linear(input))

        model = torch.nn.Sequential(
            Block(13, 512),
            Block(512, 256),
            Block(256, 128),
        ).eval()
        x = torch.randn(8, 13)

        with torch.no_grad(), torch._dynamo.config.patch(error_on_recompile=True):
            expected = model(x)
            opt_model = torch.compile(model, backend="eager")
            result = opt_model(x)
            result_second_call = opt_model(x)

        self.assertEqual(result, expected)
        self.assertEqual(result_second_call, expected)

    # https://github.com/pytorch/pytorch/issues/144080
    def test_pad_sequence_mixed_dtype_padding_value(self):
        # [FUNCTION DOC] Regression for issue 144080: pad_sequence with a mixed dtype list (float
        # or complex first sequence plus an int32 sequence) and a float padding value compiles
        # with inductor and fullgraph and matches eager; the complex-codegen warning is ignored.
        class RNNPadSequence(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the padded stack of the given sequences with
            # the given padding value.
            def forward(self, sequences, padding_value):
                # [FUNCTION DOC] HELPER: Pads the sequences using the given value.
                return torch.nn.utils.rnn.pad_sequence(
                    sequences, padding_value=padding_value
                )

        for first_sequence in (
            torch.tensor([0, 0.4]),
            torch.tensor([0, 0.4 + 0j]),
        ):
            model = RNNPadSequence()
            compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
            inputs = ([first_sequence, torch.tensor([0], dtype=torch.int32)], -0.7)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Torchinductor does not support code generation for complex operators.",
                    UserWarning,
                )
                compiled_result = compiled_model(*inputs)

            self.assertEqual(model(*inputs), compiled_result)

    # https://github.com/pytorch/pytorch/issues/162374
    def test_pad_packed_sequence_graph_break_after_packed_gru(self):
        # [FUNCTION DOC] Regression for issue 162374: an embedding, GRU on a packed sequence,
        # pad_packed_sequence and attention head compiles with the eager backend (graph break
        # around padding) and matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with embedding, GRU, attention linear, projection and a
            # vector parameter; its forward packs padded embeddings by lengths, runs the GRU,
            # unpacks, computes attention weights over time and projects the context.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the embedding, GRU, two linear layers and the
                # parameter vector.
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.rnn = torch.nn.GRU(4, 6)
                self.attn = torch.nn.Linear(6, 6)
                self.proj = torch.nn.Linear(6, 8)
                self.v = torch.nn.Parameter(torch.rand(6))

            def forward(self, input_seq, input_lengths):
                # [FUNCTION DOC] HELPER: Embeds the sequence, packs it with CPU lengths
                # (unsorted), runs the GRU, pads the output, computes tanh attention energies with
                # the vector, softmaxes over time, forms a context via batched matmul and returns
                # the projection.
                embedded = self.embedding(input_seq)
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    embedded, input_lengths.cpu(), enforce_sorted=False
                )
                outputs_packed, _ = self.rnn(packed)
                outputs_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    outputs_packed
                )
                energy = torch.sum(
                    torch.tanh(self.attn(outputs_padded)) * self.v.view(1, 1, -1),
                    dim=2,
                )
                attn_weights = torch.nn.functional.softmax(
                    energy.transpose(0, 1), dim=1
                )
                context = (
                    attn_weights.unsqueeze(1)
                    .bmm(outputs_padded.transpose(0, 1))
                    .squeeze(1)
                )
                return self.proj(context)

        model = Model().eval()
        input_seq = torch.arange(10, dtype=torch.long).view(5, 2)
        input_lengths = torch.tensor([3, 5], dtype=torch.int64)

        expected = model(input_seq, input_lengths)
        actual = torch.compile(model, backend="eager")(input_seq, input_lengths)

        self.assertEqual(expected, actual)

    # https://github.com/pytorch/pytorch/issues/162374
    def test_pad_packed_sequence_fullgraph_unsupported(self):
        # [FUNCTION DOC] Regression for issue 162374: pad_packed_sequence on a packed sequence
        # input under fullgraph raises the Dynamo Unsupported error naming pad_packed_sequence.
        def fn(sequence):
            # [FUNCTION DOC] HELPER: Unpacks the padded output and lengths and adds the lengths as
            # a broadcast tensor to the output.
            output, lengths = torch.nn.utils.rnn.pad_packed_sequence(sequence)
            return output + lengths.to(output.dtype).view(1, -1, 1)

        x = torch.randn(5, 2, 4)
        lengths = torch.tensor([3, 5], dtype=torch.int64)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, enforce_sorted=False
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "pad_packed_sequence"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(packed)

    def test_autograd_function_ctx_stash_no_vc_check(self):
        # [FUNCTION DOC] Checks that tensors stashed directly on ctx in a custom autograd Function
        # (not through save_for_backward) skip version counter checks: after two forwards, a first
        # backward mutates the stashed tensors and the second backward must not error. Also checks
        # the forward graph output descriptors (plain output, saved-for-backward outputs and
        # no-version-check saved outputs, with an extra symbolic saved value when shapes are
        # dynamic) against expected text.
        # Test that tensors stashed directly on ctx (e.g., ctx.x = x) in an
        # autograd.Function don't trigger version counter checks, while tensors
        # saved via save_for_backward do.
        class MutatingFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function that stashes two inputs on ctx, saves three
            # via save_for_backward, returns z plus one, and in backward mutates the stashed
            # tensors in place and returns a gradient for z only.
            @staticmethod
            def forward(ctx, a, b, c, x, y, z):
                # [FUNCTION DOC] HELPER: Stashes b and y on ctx, saves a, c, x for backward and
                # returns z plus one.
                # Stash b and y directly on ctx (no VC check)
                ctx.b = b
                ctx.y = y
                # Save a, c, x via save_for_backward (with VC check)
                ctx.save_for_backward(a, c, x)
                return z + 1

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Reads saved and stashed tensors, multiplies the stashed
                # ones in place by 2 and 3, and returns None for five inputs plus grad_output plus
                # 2 plus a, c and x for z.
                a, c, x = ctx.saved_tensors
                b = ctx.b
                y = ctx.y
                # Mutate the stashed tensors in backward
                # This would fail with VC check if they went through save_for_backward
                b.mul_(2)
                y.mul_(3)
                return None, None, None, None, None, grad_output + 2 + a + c + x

        def my_func(*args):
            # [FUNCTION DOC] HELPER: Applies MutatingFunction to all arguments.
            return MutatingFunction.apply(*args)

        compiled_func = torch.compile(my_func, backend=aot_graph_capture_backend)

        # Create tensors - only z requires grad
        a = torch.zeros(4, requires_grad=False)
        b = torch.zeros(4, requires_grad=False)
        c = torch.zeros(4, requires_grad=False)
        x = torch.zeros(4, requires_grad=False)
        y = torch.zeros(4, requires_grad=False)
        z1 = torch.randn(4, requires_grad=True)
        z2 = torch.randn(4, requires_grad=True)

        # Two forward calls that save b and y
        out1 = compiled_func(a, b, c, x, y, z1)
        out2 = compiled_func(a, b, c, x, y, z2)

        # First backward mutates b and y
        out1.sum().backward()

        # Second backward should NOT error even though b and y were mutated
        # because they were stashed on ctx, not saved via save_for_backward
        out2.sum().backward()
        # If we got here without error, the test passed
        # Also, assert that the AOTAutograd output descriptors on the fw graph show up
        # Of 5 total activations, 2 of them are smuggled through ctx without VC checks
        # (b and y via ctx.b = b, ctx.y = y) while 3 are saved via save_for_backward
        # (a, c, x via ctx.save_for_backward(a, c, x))
        # In dynamic shapes mode, there's also a symint saved for backward.
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)""",
            )
        else:
            self.assertExpectedInline(
                "\n".join(
                    [
                        str(x)
                        for x in fw_graph[0]
                        .graph.find_nodes(op="output")[0]
                        .meta["desc"]
                    ]
                ),
                """\
PlainAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=0)
SavedForBackwardsAOTOutput(idx=1)
SavedForBackwardsAOTOutput(idx=2)
SavedForBackwardsNoVcCheckAOTOutput(idx=3)
SavedForBackwardsNoVcCheckAOTOutput(idx=4)
SavedForBackwardsAOTOutput(idx=5)""",
            )

    def test_move_tensor_subclass_parameter_after_compile(self):
        # [FUNCTION DOC] Checks that moving a tensor subclass parameter to the same device (cpu)
        # after compile only clears weakrefs on the parameter and does not cause a recompile
        # (recompiles treated as errors).
        aten = torch.ops.aten

        class Subclass(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass holding an inner data tensor, with torch
            # function special casing functional linear and torch dispatch supporting only to_copy
            # and detach.
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Creates a wrapper subclass instance with the shape, dtype
                # and device of data.
                return torch.Tensor._make_wrapper_subclass(
                    cls, data.shape, dtype=data.dtype, device=data.device
                )

            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Stores the inner data tensor.
                self._data = data

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string with the class name and the inner data.
                return f"{self.__class__.__name__}(data={self._data})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns the inner tensor attribute names and empty
                # context for the subclass protocol.
                return ["_data"], []

            @classmethod
            def __tensor_unflatten__(cls, inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Rebuilds the subclass from the inner tensors mapping.
                return cls(inner_tensors["_data"])

            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: For functional linear, calls it with the inner data of
                # the weight; otherwise calls the function with subclass torch function disabled.
                if func == torch.nn.functional.linear:
                    return func(args[0], args[1]._data, *args[2:])

                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **(kwargs or dict()))

            def __torch_dispatch__(self, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: For to_copy and detach, unwraps subclass args, calls the
                # op and rewraps the result; raises NotImplementedError for other ops.
                if func in (aten._to_copy.default, aten.detach.default):
                    args = [x._data if isinstance(x, Subclass) else x for x in args]
                    out = func(*args, **kwargs)
                    return Subclass(out)

                raise NotImplementedError(f"{func=}")

        # Compile on CPU
        linear = torch.nn.Linear(2, 2)
        linear.weight = torch.nn.Parameter(Subclass(linear.weight.detach()))
        linear.compile()
        linear(torch.randn(1, 2))

        # Check that weakrefs are cleared after compile
        t1 = linear.weight
        self.assertEqual(len(weakref.getweakrefs(t1)), 0)

        # Already on CPU, so should just clear weakrefs
        linear.cpu()

        # Check that there is no recompile
        with torch._dynamo.config.patch(error_on_recompile=True):
            linear(torch.randn(1, 2, device="cpu"))

    def test_nested_compile_in_tensor_subclass_handling(self):
        # [FUNCTION DOC] Checks that a nested compile call (aot_eager dequantize function) inside
        # the torch function handler of a tensor subclass works when the module with a subclass
        # weight is compiled with aot_eager; result equals the manual dequantized embedding.
        def dequantize_impl(int4_data, scales):
            # [FUNCTION DOC] HELPER: Unpacks int4 values from int8 data by shifts, scales them and
            # returns the flattened result in the scale dtype.
            int8_data = torch.stack([int4_data << 4 >> 4, int4_data >> 4], dim=-1)
            fp32_data = int8_data.float().view(*scales.shape, -1) * scales.unsqueeze(-1)
            return fp32_data.flatten(-2).to(scales.dtype)

        dequantize = torch.compile(dequantize_impl, backend="aot_eager")

        class MyEmbedding(nn.Module):
            # [CLASS DOC] HELPER: Module with an embedding weight parameter applying
            # functional embedding.
            def __init__(self, num_embeds, embed_dim):
                # [FUNCTION DOC] HELPER: Creates the random embedding weight parameter.
                super().__init__()
                self.weight = nn.Parameter(torch.randn(num_embeds, embed_dim))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the functional embedding of x with the weight.
                return F.embedding(x, self.weight)

        class Int4Tensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass storing packed int4 data and scales,
            # whose torch function turns functional embedding into a compiled dequantize of
            # embedded data and scales.
            @staticmethod
            def __new__(cls, int4_data, scales):
                # [FUNCTION DOC] HELPER: Creates a wrapper subclass whose last dimension is twice
                # that of the int4 data, with the dtype and device of the scales.
                shape = int4_data.shape
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    shape[:-1] + (shape[-1] * 2,),
                    dtype=scales.dtype,
                    device=scales.device,
                )

            def __init__(self, int4_data, scales):
                # [FUNCTION DOC] HELPER: Stores the int4 data and scales.
                self.int4_data = int4_data
                self.scales = scales

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns the inner tensor attribute names and
                # empty context.
                return ["int4_data", "scales"], []

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string with shape and device.
                return f"Int4Tensor(shape={tuple(self.shape)}, device={self.device})"

            @classmethod
            def __tensor_unflatten__(
                cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
            ):
                # [FUNCTION DOC] HELPER: Rebuilds the subclass from the inner tensors mapping.
                return cls(tensor_data_dict["int4_data"], tensor_data_dict["scales"])

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: For functional embedding, embeds the indices in the int4
                # data and scales and dequantizes; otherwise runs the function with subclass torch
                # function disabled.
                kwargs = kwargs or {}
                if func is F.embedding:
                    input = args[0]
                    weight = args[1]
                    return dequantize(
                        F.embedding(input, weight.int4_data),
                        F.embedding(input, weight.scales),
                    )
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Supports detach by rewrapping the same inner tensors;
                # raises NotImplementedError for other ops.
                kwargs = kwargs or {}
                if func is torch.ops.aten.detach.default:
                    x = args[0]
                    return Int4Tensor(x.int4_data, x.scales)
                raise NotImplementedError(func)

        indices = torch.tensor([3, 17])
        int4_data = torch.randint(-128, 127, size=(100, 16), dtype=torch.int8)
        scales = torch.randn(100, 1)
        embedding = MyEmbedding(100, 32)
        embedding.weight = nn.Parameter(Int4Tensor(int4_data, scales))

        compiled = torch.compile(embedding, backend="aot_eager")
        result = compiled(indices)
        expected = dequantize_impl(
            F.embedding(indices, int4_data), F.embedding(indices, scales)
        )

        self.assertEqual(result, expected)

    def test_nested_compile_allowed_in_custom_aot_fw_compiler(self):
        # [FUNCTION DOC] Checks that torch.compile of an inner function is allowed inside a custom
        # AOT forward compiler: the inner compiled function records exactly one graph.
        from functorch.compile import aot_function

        inner_backend = EagerAndRecordGraphs()

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Returns sine of x plus 1.
            return x.sin() + 1

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that compiles inner_fn with fullgraph on
            # random input using the recording backend, then returns the graph module.
            torch.compile(inner_fn, backend=inner_backend, fullgraph=True)(
                torch.randn(4)
            )
            return gm

        def f(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x plus 1.
            return x.cos() + 1

        compiled = aot_function(f, fw_compiler=fw_compiler)
        compiled(torch.randn(4))

        self.assertEqual(len(inner_backend.graphs), 1)

    def test_property_setter_with_dict_get_176608(self):
        # [FUNCTION DOC] Regression for issue 176608: a property with a setter that reads through
        # the instance dictionary getter, mutated by a method inside a compiled function and read
        # through a Dynamo-disabled len, gives 5.
        """
        Test that property setters work correctly with __dict__.get() in compiled functions.
        Regression test for https://github.com/pytorch/pytorch/issues/176608
        """
        from torch.compiler import disable

        class Container:
            # [CLASS DOC] HELPER: Container with a private value and a property _len backed by it,
            # an add method and a length method.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the private _len_value to 0.
                self._len_value = 0

            @property
            def _len(self):
                # [FUNCTION DOC] HELPER: Property getter reading _len_value from the instance
                # dictionary with default 0.
                # Using __dict__.get instead of self._len_value triggers the bug
                return self.__dict__.get("_len_value", 0)

            @_len.setter
            def _len(self, value):
                # [FUNCTION DOC] HELPER: Property setter storing the value in _len_value.
                self._len_value = value

            def add(self, n):
                # [FUNCTION DOC] HELPER: Increases the length property by n.
                self._len = self._len + n

            @disable()
            def __len__(self):
                # [FUNCTION DOC] HELPER: Dynamo-disabled length returning the length property.
                return self._len

        c = Container()

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Adds the first dimension of x to the container and returns
            # its length.
            c.add(x.shape[0])  # mutates c._len_value via property setter
            return len(c)  # reads c._len via property getter -> __dict__.get

        result = f(torch.randn(5))
        self.assertEqual(result, 5)

    def test_one_hot_bounds_check_compiled(self):
        # [FUNCTION DOC] Regression for issue 144211: a compiled one_hot must raise RuntimeError
        # for out-of-range indices (too large and negative) instead of silently returning wrong
        # results, and still match eager for valid input. Dynamo is reset between checks.
        # https://github.com/pytorch/pytorch/issues/144211
        # torch.compile(one_hot) should raise on out-of-bounds indices,
        # not silently produce wrong results.
        one_hot = torch.compile(torch.nn.functional.one_hot, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a = torch.arange(0, 5) % 3  # [0, 1, 2, 0, 1]
        with self.assertRaises(RuntimeError):
            one_hot(a, 1)

        torch._dynamo.reset()
        with self.assertRaises(RuntimeError):
            one_hot(torch.tensor([-1, 0, 1]), 3)

        torch._dynamo.reset()
        expected = torch.nn.functional.one_hot(a, 3)
        self.assertEqual(one_hot(a, 3), expected)

    def test_issue183886_istft_length_pads_in_fake_tensor(self):
        # [FUNCTION DOC] Regression for issue 183886: istft with a requested length larger than
        # the signal pads the output in fake tensor mode; compiled with fullgraph, results and
        # shape 3 by 352 match eager.
        n_fft = 64
        hop_length = 32
        length = 352

        def fn(spec, window):
            # [FUNCTION DOC] HELPER: Runs the inverse STFT of the spectrogram with the enclosing
            # n_fft, hop length, window and length.
            return torch.istft(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=length,
            )

        spec = torch.view_as_complex(torch.randn(3, n_fft // 2 + 1, 10, 2))
        window = torch.hann_window(n_fft)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = fn(spec, window)
            actual = torch.compile(fn, backend="eager", fullgraph=True)(spec, window)

        self.assertEqual(actual, expected)
        self.assertEqual(actual.shape, (3, length))

    def test_istft_compile_dynamic_length_clamp(self):
        # [FUNCTION DOC] Regression check that istft with dynamic shapes keeps the length versus
        # signal clamp symbolic: both a short (50 frame) and a long (200 frame) spectrogram share
        # one graph (one frame) and match eager.
        # Regression for the sym_min/sym_max clamp in the istft ref. Under
        # dynamic shapes the (requested length vs signal size) clamp must stay
        # symbolic: the builtin min would install a guard and recompile when the
        # relationship flips (or raise GuardOnDataDependentSymNode for unbacked
        # symints). sym_min/sym_max carry it symbolically, so a single graph
        # serves both the "length exceeds signal" and "length within signal"
        # cases.
        n_fft = 1024
        hop_length = 512
        length = 60000

        def fn(spec, window):
            # [FUNCTION DOC] HELPER: Runs the inverse STFT of the spectrogram with the enclosing
            # n_fft, hop length, window and length.
            return torch.istft(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=length,
            )

        window = torch.hann_window(n_fft)
        # 50 frames -> signal shorter than length (tail is padded);
        # 200 frames -> signal longer than length (clamped/sliced).
        spec_short = torch.view_as_complex(torch.randn(4, n_fft // 2 + 1, 50, 2))
        spec_long = torch.view_as_complex(torch.randn(4, n_fft // 2 + 1, 200, 2))

        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt, fullgraph=True, dynamic=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(compiled(spec_short, window), fn(spec_short, window))
            self.assertEqual(compiled(spec_long, window), fn(spec_long, window))

        # Both shapes share a single graph: the symbolic clamp installs no guard
        # that would force a recompile when the length/signal relationship flips.
        self.assertEqual(cnt.frame_count, 1)

    @unittest.expectedFailure
    def test_method_dunder_dict_setitem(self):
        # [FUNCTION DOC] Expected failure: setting an entry in the dunder dict of a bound test
        # method inside a fullgraph compiled function should work and be visible afterward.
        # KNOWN ISSUE: marked as an expected failure; method dict access currently raises
        # AttributeError per the comments.
        # Reproducer for: getattr(obj, method_name).__dict__['key'] = value
        # method.__dict__ is handled specially by CPython at C level (no
        # tp_dictoffset, no Python-visible descriptor), which caused
        # object.__getattribute__(method, "__dict__") to raise AttributeError.

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that writes slow_test into the dictionary
            # of the running test method and returns sine of x.
            getattr(self, self._testMethodName).__dict__["slow_test"] = True
            return x.sin()

        x = torch.randn(2)
        _ = fn(x)
        self.assertTrue(getattr(self, self._testMethodName).__dict__.get("slow_test"))

    # https://github.com/pytorch/pytorch/issues/190171
    @parametrize(
        "kind",
        ["module_method", "plain_method", "classmethod", "staticmethod", "function"],
    )
    def test_getattr_on_compiled_method(self, kind):
        # [FUNCTION DOC] Parametrized over kinds of callables (module method, plain method, class
        # method, static method, free function). Regression for issue 190171: wrapping a bound
        # method with torch.compile, then reading its name and qualified name inside another
        # fullgraph compiled function, matches eager.
        # torch.compile(obj.meth) stores the bound method in
        # _torchdynamo_inline. A method owns no __dict__ and forwards lookups to
        # __func__, so materializing its __dict__ used to raise AttributeError.
        # staticmethod/function are controls: those are plain functions.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a method returning its input.
            def meth(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        class Plain:
            # [CLASS DOC] HELPER: Plain class with an instance method, a class method and a static
            # method each returning the input.
            def meth(self, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

            @classmethod
            def cls_meth(cls, x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

            @staticmethod
            def stat(x):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        def free_fn(x):
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        target = {
            "module_method": Mod().meth,
            "plain_method": Plain().meth,
            "classmethod": Plain.cls_meth,
            "staticmethod": Plain().stat,
            "function": free_fn,
        }[kind]
        wrapped = torch.compile(target, backend="eager")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one with the name and qualified name of the
            # compiled wrapper.
            return x + 1, wrapped.__name__, wrapped.__qualname__

        x = torch.zeros(1)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(expected, actual)

    # https://github.com/pytorch/pytorch/issues/190171
    def test_getfullargspec_on_dynamo_ctx_method(self):
        # [FUNCTION DOC] Regression for issue 190171 (pytorch-lightning pattern): after rebinding
        # a step method to a Dynamo-wrapped version, inspecting its signature inside the traced
        # region gives the same answer as outside.
        # What pytorch-lightning does: rebind a step method to a dynamo-wrapped
        # version, then introspect its signature from inside the traced region.
        traced = []

        def takes_param(fn, name):
            # [FUNCTION DOC] HELPER: Unwraps a function with a wrapped attribute and returns
            # whether the given name is among its positional argument names.
            if hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
            return name in inspect.getfullargspec(fn).args

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose step method records the signature inspection result
            # in an outer list and doubles the input; forward calls step.
            def step(self, x, dataloader_iter=None):
                # [FUNCTION DOC] HELPER: Appends the inspection result for dataloader_iter to the
                # traced list and returns x doubled.
                traced.append(takes_param(self.step, "dataloader_iter"))
                return x * 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls step with x.
                return self.step(x)

        model = Model()
        compiled = torch.compile(model, backend="eager")
        model.step = compiled.dynamo_ctx(model.step)

        expected = takes_param(model.step, "dataloader_iter")
        compiled(torch.randn(4))
        self.assertEqual(traced, [expected])

    @parametrize("kind", ["compile", "lru_cache", "script_if_tracing"])
    def test_wrapper_function_identity(self, kind):
        # [FUNCTION DOC] Parametrized over compile, lru_cache and script_if_tracing wrappers.
        # Checks that the identity comparison between a wrapped function and the original is False
        # in both orders when evaluated inside a fullgraph compiled function.
        # A wrapper is not the function it wraps. WrapperUserFunctionVariable
        # stands for the wrapper, so identity must compare against it and not
        # against the inline target reached via attr_to_trace.
        from torch.jit import _script_if_tracing

        def g(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        wrapped = {
            "compile": lambda: torch.compile(g, backend="eager"),
            "lru_cache": lambda: functools.lru_cache(g),
            "script_if_tracing": lambda: _script_if_tracing(g),
        }[kind]()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one and the identity comparisons of the
            # wrapper and the original function in both orders.
            return x + 1, (wrapped is g), (g is wrapped)

        x = torch.zeros(1)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(expected, actual)

    def test_elementwise_dtypes_constant_fold(self):
        # [FUNCTION DOC] Checks that the elementwise dtype promotion helper with the default
        # promotion kind is constant folded in a fullgraph compile: a float tensor stays float32.
        from torch._prims_common import (
            elementwise_dtypes,
            ELEMENTWISE_TYPE_PROMOTION_KIND,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Gets the computed dtype for x with default promotion and
            # converts x to it.
            dt, _ = elementwise_dtypes(
                x, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            )
            return x.to(dt)

        result = fn(torch.randn(3))
        self.assertEqual(result.dtype, torch.float32)

    def test_elementwise_dtypes_int_to_float(self):
        # [FUNCTION DOC] Checks the dtype helper with int to float promotion yields float32 for an
        # integer tensor under fullgraph.
        from torch._prims_common import (
            elementwise_dtypes,
            ELEMENTWISE_TYPE_PROMOTION_KIND,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Gets the computed dtype for x with int to float promotion and
            # converts x to it.
            dt, _ = elementwise_dtypes(
                x, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            return x.to(dt)

        result = fn(torch.randint(0, 10, (3,)))
        self.assertEqual(result.dtype, torch.float32)

    def test_elementwise_dtypes_multi_args(self):
        # [FUNCTION DOC] Checks the dtype helper with two arguments of float16 and float32 yields
        # float32 under fullgraph.
        from torch._prims_common import (
            elementwise_dtypes,
            ELEMENTWISE_TYPE_PROMOTION_KIND,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Gets the computed dtype for x and y with default promotion
            # and converts x to it.
            dt, _ = elementwise_dtypes(
                x, y, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            )
            return x.to(dt)

        result = fn(
            torch.randn(3, dtype=torch.float16),
            torch.randn(3, dtype=torch.float32),
        )
        self.assertEqual(result.dtype, torch.float32)

    def test_empty_out_shape_mismatch_dynamic(self):
        # [FUNCTION DOC] Checks empty with an out tensor of mismatched size is resized to the
        # requested size when compiled with dynamic shapes, matches eager; a compatible out is
        # unchanged and a zero-element out is resized.
        def f(size, out):
            # [FUNCTION DOC] HELPER: Calls empty with the size, out and float32 dtype.
            return torch.empty(size, out=out, dtype=torch.float32)

        size = [2, 3]

        # Mismatched out shape should be resized to match requested size
        out_wrong = torch.empty([1])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            eager_result = f(size, out_wrong.clone())

        cf = torch.compile(f, dynamic=True)  # noqa: UNSPECIFIED_BACKEND
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            compiled_result = cf(size, out_wrong.clone())

        self.assertEqual(compiled_result.shape, eager_result.shape)

        # Compatible out should pass through unchanged
        compiled_ok = cf(size, torch.empty([2, 3]))
        self.assertEqual(compiled_ok.shape, torch.Size([2, 3]))

        # Zero-element out should be resized
        compiled_zero = cf(size, torch.empty([0]))
        self.assertEqual(compiled_zero.shape, torch.Size([2, 3]))

    def test_udf_class_source(self):
        # [FUNCTION DOC] Checks that creating an instance through the type of another instance
        # inside a fullgraph compiled function is supported.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a Foo, creates another through its type (unused) and
            # returns cosine of x.
            foo = Foo()
            bar = type(foo)()  # noqa: F841
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_tensor_size_hasattr(self):
        # [FUNCTION DOC] Checks that hasattr on a tensor for size and stride is evaluated at
        # compile time under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies x by 2 and by 3 if the tensor has size and stride
            # attributes and then by 5.
            if hasattr(x, "size"):
                x = x * 2
            if hasattr(x, "stride"):
                x = x * 3
            return x * 5

        x = torch.ones(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dual_tensor_input_graph_breaks(self):
        # [FUNCTION DOC] Checks that a forward-mode dual tensor input is not compiled (frame count
        # 0 with a counting backend and correct tangent) and under fullgraph raises Unsupported
        # mentioning dual tensor input.
        import torch.autograd.forward_ad as fwAD

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of squares of x.
            return (x**2).sum()

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.tensor([0.1, 0.2, 0.3])
        v = torch.ones(3)
        with fwAD.dual_level():
            dual = fwAD.make_dual(x, v)
            expected = fwAD.unpack_dual(fn(dual)).tangent
            out = torch.compile(fn, backend=cnt)(dual)
            self.assertEqual(fwAD.unpack_dual(out).tangent, expected)
        self.assertEqual(cnt.frame_count, 0)

        torch._dynamo.reset()
        with fwAD.dual_level():
            dual = fwAD.make_dual(x, v)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "dual tensor input"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(dual)

    def test_swap_tensors_after_discarded_attempt(self):
        # [FUNCTION DOC] Regression for issue 186796: after compiles that discard an earlier
        # attempt (graph break restart, skipped frame, scalar tensorify restart), parameters
        # can still be swapped with swap_tensors because observational weakrefs were cleaned
        # up; for the tensorify case records show both fake mode describers emptied on each
        # discarded attempt.
        # Issue #186796: a discarded restart/skip attempt fakifies the real
        # params and builds guards on them, leaving weakrefs on the real params
        # (guard TensorWeakRef, tensor_to_context WeakIdRef, fake-mode describer
        # WeakIdRef) that block torch.utils.swap_tensors after compile.
        # _cleanup_output_graph must drop those weakrefs.
        def assert_swappable(param):
            # [FUNCTION DOC] HELPER: Swaps the given parameter with a zero parameter of the same
            # shape; raises if a weakref is left on it.
            # Raises "Cannot swap ... has weakref" if any observational weakref
            # is left on the real param.
            torch.utils.swap_tensors(param, nn.Parameter(torch.zeros_like(param.data)))

        # Axis 1: SpeculationRestartAnalysis - a graph break after the param is
        # fakified discards the first attempt.
        torch._dynamo.reset()

        class RestartMod(nn.Module):
            # [CLASS DOC] HELPER: Module with a 4 by 4 parameter whose forward multiplies the
            # input, graph breaks and adds one.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Matmul with the weight, explicit graph break, then
                # adds one.
                y = x @ self.weight
                torch._dynamo.graph_break()
                return y + 1

        m1 = RestartMod()
        torch.compile(m1, backend="eager")(torch.randn(2, 4))
        assert_swappable(m1.weight)

        # Axis 2: SkipFrame - reading the param but tracing no ops skips the
        # frame after the param is fakified.
        torch._dynamo.reset()

        class SkipMod(nn.Module):
            # [CLASS DOC] HELPER: Module with a 4 by 4 parameter whose forward only returns a
            # shape entry (traces no ops).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the first dimension of the weight.
                return self.weight.shape[0]

        m2 = SkipMod()
        torch.compile(m2, backend="eager")(torch.randn(2, 4))
        assert_swappable(m2.weight)

        # Axis 3: TensorifyScalarRestartAnalysis (backend-raised restart). By
        # the time cleanup runs, tracing_context.fake_mode has been swapped to a
        # fresh backend fake_mode and the tracing describer holding the real
        # params lives on _old_fake_mode, so cleanup must clear BOTH. We assert
        # both describers are emptied on the discarded attempt: without clearing
        # _old_fake_mode its describer still pins the real-param WeakIdRefs
        # (whether they still block swap depends on non-deterministic GC).
        import math

        from torch._dynamo.output_graph import DynamoTracerOutput

        records = []
        orig_cleanup = DynamoTracerOutput._cleanup_output_graph

        def recording_cleanup(tracer_self):
            # [FUNCTION DOC] HELPER: Replacement cleanup method that calls the original, then (if
            # the output graph is available) records the sizes of the describers of the current
            # and old fake modes in the outer list.
            orig_cleanup(tracer_self)
            og = tracer_self.output_graph_for_cleanup
            if og is None:
                return

            def describer_len(fake_mode):
                # [FUNCTION DOC] HELPER: Returns None for a missing fake mode, otherwise the
                # number of tensors plus storages tracked by the meta converter describer.
                if fake_mode is None:
                    return None
                d = fake_mode.fake_tensor_converter.meta_converter.describer
                return len(d.lookup_tensor) + len(d.lookup_storage)

            old_fake_mode = getattr(og, "_old_fake_mode", None)
            records.append(
                (
                    describer_len(og.tracing_context.fake_mode),
                    describer_len(old_fake_mode),
                )
            )

        class TensorifyMod(nn.Module):
            # [CLASS DOC] HELPER: Module with an 8 by 8 parameter whose forward multiplies a
            # matmul by the floor of the square of a float input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns floor of y squared times the matmul of x and
                # the weight.
                return math.floor(y**2) * (x @ self.weight)

        torch._dynamo.reset()
        m3 = TensorifyMod()
        x = torch.randn(4, 8)
        # specialize_float=False is required for the scalar-tensorify restart to
        # fire: the dynamic-shapes test variant sets specialize_float=True, which
        # constant-folds the float so no symfloat (and no tensorify restart) is
        # produced. Pin it here so axis 3 deterministically triggers the restart
        # under both the default config and the dynamic-shapes variant.
        with (
            torch._dynamo.config.patch(specialize_float=False),
            mock.patch.object(
                DynamoTracerOutput, "_cleanup_output_graph", recording_cleanup
            ),
        ):
            cm3 = torch.compile(m3, backend="aot_eager")
            cm3(x, 2.0)
            cm3(x, 3.0)  # automatic dynamic -> symfloat -> tensorify restart
            cm3(x, 4.0)

        # At least one discarded attempt must have had _old_fake_mode set (the
        # tensorify restart raised after the backend fake_mode swap).
        self.assertTrue(
            any(old is not None for _, old in records),
            "expected a discarded attempt with _old_fake_mode set",
        )
        # Cleanup must empty BOTH describers on every discarded attempt.
        for cur, old in records:
            if cur is not None:
                self.assertEqual(cur, 0)
            if old is not None:
                self.assertEqual(old, 0)
        assert_swappable(m3.weight)

    def test_grid_sampler_2d_cpu_fallback_captured(self):
        # [FUNCTION DOC] Checks the grid sampler 2D CPU fallback op is captured by Dynamo with its
        # meta kernel: one frame and two ops under fullgraph, result equals eager.
        # torch._grid_sampler_2d_cpu_fallback used to have no meta kernel, so
        # running it under FakeTensorMode hard-errored ("data is not allocated
        # yet") and Dynamo could not capture it. Regression test for the
        # NGB-enabled failure of test_nn.py TestNN.test_grid_sample: with the
        # meta registrations it captures with no graph break.
        def fn(inp, grid):
            # [FUNCTION DOC] HELPER: Calls the grid sampler CPU fallback with zero modes and align
            # corners, and adds one.
            out = torch._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, True)
            return out + 1

        inp = torch.randn(2, 3, 4, 5)
        grid = torch.rand(2, 6, 7, 2) * 2 - 1
        expected = fn(inp, grid)

        cnt = CompileCounter()
        out = torch.compile(fn, backend=cnt, fullgraph=True)(inp, grid)
        self.assertEqual(out, expected)
        # Both the fallback and the add are captured into one graph.
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_grid_sampler_2d_cpu_fallback_backward(self):
        # [FUNCTION DOC] Checks the backward of the grid sampler 2D CPU fallback traces through
        # aot_eager: input and grid gradients match eager.
        # The backward also needs a meta kernel, or AOTAutograd fails to trace
        # it. Grads must match eager.
        def fn(inp, grid):
            # [FUNCTION DOC] HELPER: Returns the sum of the CPU fallback grid sampler output.
            return torch._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, True).sum()

        inp = torch.randn(2, 3, 4, 5)
        grid = torch.rand(2, 6, 7, 2) * 2 - 1

        i_ref, g_ref = inp.clone().requires_grad_(), grid.clone().requires_grad_()
        fn(i_ref, g_ref).backward()

        i_test, g_test = inp.clone().requires_grad_(), grid.clone().requires_grad_()
        torch.compile(fn, backend="aot_eager", fullgraph=True)(
            i_test, g_test
        ).backward()

        self.assertEqual(i_test.grad, i_ref.grad)
        self.assertEqual(g_test.grad, g_ref.grad)

    def test_grid_sampler_2d_cpu_fallback_inductor(self):
        # [FUNCTION DOC] Checks the grid sampler CPU fallback compiled with inductor (fullgraph)
        # gives the same values and strides as eager, including a channels_last input, and that
        # the generated code mentions the op (i.e. a real fallback).
        # The op has no inductor lowering, so once it is capturable inductor
        # must auto-fall-back to the aten op and infer the output layout from
        # the meta kernel. Check values and strides against eager, including a
        # channels_last input (where the meta returning contiguous must match
        # what the real kernel allocates).
        def fn(inp, grid):
            # [FUNCTION DOC] HELPER: Calls the grid sampler CPU fallback with zero modes and
            # align corners.
            return torch._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, True)

        from torch._inductor.utils import run_and_get_code

        grid = torch.rand(2, 6, 7, 2) * 2 - 1
        for inp in (
            torch.randn(2, 3, 4, 5),
            torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last),
        ):
            expected = fn(inp, grid)
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            out, (code,) = run_and_get_code(opt_fn, inp, grid)
            self.assertEqual(out, expected)
            self.assertEqual(out.stride(), expected.stride())
            # fullgraph plus this keep the test from passing via an eager
            # fallback, which would return the same values and strides.
            self.assertIn("_grid_sampler_2d_cpu_fallback", code)

    def test_grid_sampler_2d_cpu_fallback_dynamic_shapes(self):
        # [FUNCTION DOC] Checks the grid sampler CPU fallback with dynamic dims does not recompile
        # when shapes change: one frame for two different sizes with equal results.
        # The meta kernels run check_grid_sampler_common/_2d, so their
        # torch._check calls must stay symint-safe: changing the dynamic dims
        # must not add guards that force a recompile.
        def fn(inp, grid):
            # [FUNCTION DOC] HELPER: Calls the grid sampler CPU fallback with zero modes and
            # align corners.
            return torch._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, True)

        inp = torch.randn(2, 3, 4, 5)
        grid = torch.rand(2, 6, 7, 2) * 2 - 1
        torch._dynamo.mark_dynamic(inp, 2)
        torch._dynamo.mark_dynamic(grid, 1)

        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        self.assertEqual(opt_fn(inp, grid), fn(inp, grid))

        inp2 = torch.randn(2, 3, 9, 5)
        grid2 = torch.rand(2, 11, 7, 2) * 2 - 1
        self.assertEqual(opt_fn(inp2, grid2), fn(inp2, grid2))
        self.assertEqual(cnt.frame_count, 1)


class ReproTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic repro tests (instantiated per device type by the device test
    # machinery at the end of the module).
    @serialTest()
    @onlyAccelerator
    def test_mem_leak_guards(self, device):
        # [FUNCTION DOC] Serial, accelerator-only test. Checks that repeated runs (1000) of a
        # compiled module with a Dynamo-disabled forward do not increase peak device memory beyond
        # the first run (no leak from guards).
        def gn(x0, x):
            # [FUNCTION DOC] HELPER: Returns x0 times x.
            return x0 * x

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose Dynamo-disabled forward applies sine and two calls
            # to gn.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            @torch._dynamo.disable(recursive=False)
            def forward(self, running_x):
                # [FUNCTION DOC] HELPER: Applies sine, then multiplies the result by itself and by
                # the product again, returning it.
                running_x = torch.sin(running_x)
                x = running_x
                x = gn(running_x, running_x)
                x = gn(running_x, x)
                return x

        mod = MyMod().to(device)

        fn = torch.compile(mod, backend="eager")
        x = torch.randn(10, 10, device=device)
        torch.get_device_module(device).reset_peak_memory_stats()

        fn(x)
        peak_mem1 = torch.get_device_module(device).max_memory_allocated()

        for _ in range(1000):
            fn(x)
        peak_mem2 = torch.get_device_module(device).max_memory_allocated()
        self.assertTrue(peak_mem1 == peak_mem2)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3860")
    # test involves custom ops that return unbacked symints
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    # test requires the activation memory budget code to think
    # that j() is banned from recompute
    @torch._functorch.config.patch(activation_memory_budget=0.5)
    def test_partitioner_activation_memory_budget_with_unbacked_symints(self, device):
        # [FUNCTION DOC] Skipped on XPU (issue 3860). With dynamic output shape ops captured and
        # an activation memory budget of 0.5, checks that the partitioner handles custom ops with
        # unbacked symbolic sizes where one op is banned from recompute;
        # aot_eager_decomp_partition output matches eager.
        @torch.library.custom_op("test_partitioner::f", mutates_args=[])
        def f(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning zeros of size 512 by 1 on the input.
            return x.new_zeros(512, 1)

        @f.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of the first custom op returning an empty
            # tensor with a new dynamic (unbacked) size.
            ctx = torch.library.get_ctx()
            s = ctx.new_dynamic_size()
            return torch.empty(s, 1, device=x.device, dtype=x.dtype)

        @torch.library.custom_op("test_partitioner::g", mutates_args=[])
        def g(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op appending the first element of x as a new row.
            return torch.cat([x, x[0].unsqueeze(-1)])

        @g.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of the second custom op, same behavior as
            # the real one.
            return torch.cat([x, x[0].unsqueeze(-1)])

        @torch.library.custom_op("test_partitioner::i", mutates_args=[])
        def i(x: torch.Tensor, sz: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning ones of the given length as a column.
            return torch.ones(sz, 1, dtype=x.dtype, device=x.device)

        @i.register_fake
        def _(x: torch.Tensor, sz: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty column of the
            # given length.
            return torch.empty(sz, 1, dtype=x.dtype, device=x.device)

        @torch.library.custom_op("test_partitioner::j", mutates_args=[])
        def j(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op returning x plus one.
            return x + 1

        @j.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation that checks the length relation between
            # the inputs and returns a reduction (sum) so the partitioner bans recompute.
            sz1 = x.shape[0] - 1
            sz2 = y.numel()
            torch._check(sz1 == sz2)
            # make this a reduction so partitioner bans recompute of it
            return x.sum()

        def f(x, param):
            # [FUNCTION DOC] HELPER: Chains the four custom ops starting from x, using a size
            # derived from the intermediate, and returns sine of the matmul with the parameter
            # times the sum of the last result.
            y = torch.ops.test_partitioner.f(x)
            z = torch.ops.test_partitioner.g(y)
            z2 = torch.ops.test_partitioner.i(x, z.shape[0] - 1)
            z2 = torch.ops.test_partitioner.j(z, z2)
            return torch.matmul(x, param).sin() * z2.sum()

        x = torch.randn(512, 512, device=device)
        param = torch.randn(512, 512, device=device, requires_grad=True)
        out_ref = f(x, param)
        out_test = torch.compile(f, backend="aot_eager_decomp_partition")(x, param)
        self.assertEqual(out_ref, out_test)

    # This test will fail as flip in combination with particular input lengths
    # produces weird results.
    # This is under investigations in
    # https://github.com/pytorch/pytorch/issues/131805
    @unittest.skip("Skip this flip test for the moment. It is under investigation")
    def test_flip_bad_accuracy(self, device):
        # [FUNCTION DOC] Skipped (under investigation, issue 131805): would check that a
        # flip-cumsum-flip graph from a generated module produces correct results.
        import torch
        import torch._dynamo.config
        import torch._functorch.config
        import torch._inductor.config
        import torch._inductor.inductor_prims
        import torch.fx.experimental._config

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Machine-generated module applying reverse, strided slices, adds,
            # concatenations and padding in a prefix-sum pattern to a 1D input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, arg0_1):
                # [FUNCTION DOC] HELPER: Reverses the input and computes a scan with slices, adds,
                # unsqueeze, cat and view ops, then reverses again; returns it in a tuple.
                rev = torch.ops.prims.rev.default(arg0_1, [0])
                arg0_1 = None
                slice_1 = torch.ops.aten.slice.Tensor(rev, 0, 0, -1, 2)
                slice_2 = torch.ops.aten.slice.Tensor(rev, 0, 1, 9223372036854775807, 2)
                add_1 = torch.ops.aten.add.Tensor(slice_1, slice_2)
                slice_1 = slice_2 = None
                slice_3 = torch.ops.aten.slice.Tensor(add_1, 0, 0, -1, 2)
                slice_4 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 1, 9223372036854775807, 2
                )
                add_2 = torch.ops.aten.add.Tensor(slice_3, slice_4)
                slice_3 = slice_4 = None
                slice_5 = torch.ops.aten.slice.Tensor(add_2, 0, 0, -1, 2)
                slice_6 = torch.ops.aten.slice.Tensor(
                    add_2, 0, 1, 9223372036854775807, 2
                )
                add_3 = torch.ops.aten.add.Tensor(slice_5, slice_6)
                slice_5 = slice_6 = None
                slice_9 = torch.ops.aten.slice.Tensor(add_2, 0, 0, 1)
                add_2 = None
                unsqueeze = torch.ops.aten.unsqueeze.default(slice_9, 1)
                slice_9 = None
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_3, 1)
                add_3 = None
                cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], 1)
                unsqueeze = unsqueeze_1 = None
                view = torch.ops.aten.view.default(cat, [2])
                cat = None
                slice_10 = torch.ops.aten.slice.Tensor(view, 0, 0, -1)
                slice_11 = torch.ops.aten.slice.Tensor(
                    add_1, 0, 2, 9223372036854775807, 2
                )
                add_5 = torch.ops.aten.add.Tensor(slice_10, slice_11)
                slice_10 = slice_11 = None
                slice_12 = torch.ops.aten.slice.Tensor(add_1, 0, 0, 1)
                add_1 = None
                cat_1 = torch.ops.aten.cat.default([slice_12, add_5])
                slice_12 = add_5 = None
                unsqueeze_2 = torch.ops.aten.unsqueeze.default(cat_1, 1)
                cat_1 = None
                unsqueeze_3 = torch.ops.aten.unsqueeze.default(view, 1)
                view = None
                cat_2 = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], 1)
                unsqueeze_2 = unsqueeze_3 = None
                view_1 = torch.ops.aten.view.default(cat_2, [4])
                cat_2 = None
                slice_13 = torch.ops.aten.slice.Tensor(
                    rev, 0, 2, 9223372036854775807, 2
                )
                add_6 = torch.ops.aten.add.Tensor(view_1, slice_13)
                slice_13 = None
                slice_14 = torch.ops.aten.slice.Tensor(rev, 0, 0, 1)
                rev = None
                cat_3 = torch.ops.aten.cat.default([slice_14, add_6])
                slice_14 = add_6 = None
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    view_1, [0, 1], 0.0
                )
                view_1 = None
                unsqueeze_4 = torch.ops.aten.unsqueeze.default(cat_3, 1)
                cat_3 = None
                unsqueeze_5 = torch.ops.aten.unsqueeze.default(constant_pad_nd, 1)
                constant_pad_nd = None
                cat_4 = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], 1)
                unsqueeze_4 = unsqueeze_5 = None
                view_2 = torch.ops.aten.view.default(cat_4, [10])
                cat_4 = None
                slice_15 = torch.ops.aten.slice.Tensor(view_2, 0, 0, 9)
                view_2 = None
                rev_1 = torch.ops.prims.rev.default(slice_15, [0])
                slice_15 = None
                return (rev_1,)

        mod = Repro()
        x = torch.arange(9, device=torch.device(device))

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function applying the repro module to x.
            return mod(x)

        out = f(x)
        self.assertEqual(torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0]), out[0])

    # https://github.com/pytorch/pytorch/issues/156580
    @serialTest()
    @onlyAccelerator
    def test_dont_dce_rand(self, device):
        # [FUNCTION DOC] Serial, accelerator-only test for issue 156580: a random-index selection
        # function compiled with eager and aot_eager (fullgraph, fallback random on) must not have
        # its random calls eliminated; the sum matches eager with the same seed.
        # https://github.com/pytorch/pytorch/issues/143431
        def f(image_latent):
            # [FUNCTION DOC] HELPER: Draws random tensors, takes an argsort based index selection
            # and gathers reference frames of the input latent.
            B = 2
            num_ref = 3
            num_tar = 3
            x = torch.rand(B, 12)
            indices = torch.argsort(torch.rand(*x.shape), dim=-1)[
                :, : num_ref + num_tar
            ]
            return image_latent[torch.arange(B).unsqueeze(-1), indices][:, :num_ref]

        # Generate input once to ensure consistency across runs
        torch.manual_seed(54321)
        torch.get_device_module(device).manual_seed_all(54321)
        image_latent = torch.randn((2, 12, 16, 32, 32))

        torch.manual_seed(54321)
        torch.get_device_module(device).manual_seed_all(54321)
        expected = f(image_latent).sum()

        # https://github.com/pytorch/pytorch/issues/147171
        with torch._inductor.config.patch(fallback_random=True):
            for backend in ["eager", "aot_eager"]:
                torch.manual_seed(54321)
                torch.get_device_module(device).manual_seed_all(54321)
                actual = torch.compile(backend=backend, fullgraph=True)(f)(
                    image_latent
                ).sum()
                self.assertEqual(actual, expected)

    # https://github.com/pytorch/pytorch/issues/151670
    @onlyAccelerator
    def test_diagonal_scatter_single_elem_cpu_with_gpu_tensor(self, device):
        # [FUNCTION DOC] Accelerator-only test for issue 151670: diagonal_scatter with a source
        # created on the CPU (default device) and an input on the accelerator must give the same
        # result when compiled with the eager backend as in eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward scatters a CPU vector of ones along the
            # diagonal of its input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a CPU vector of ones sized by the first dimension
                # of x and returns the diagonal scatter of it into x.
                y = torch.ones(x.size(0))
                x = torch.diagonal_scatter(x, y)
                return x

        model = Model()

        x = torch.rand(1, 2)
        inputs = [x]

        model = model.to(device)
        inputs = [x.to(device) for x in inputs]

        compiled_model = torch.compile(model, backend="eager")

        self.assertEqual(model(*inputs), compiled_model(*inputs))

    def test_norm_dtype(self, device):
        # [FUNCTION DOC] Checks that normalizing a strided slice of a float16 tensor under
        # autocast gives the same dtype and values on two eager calls. Note the compile call
        # result is discarded, so Dynamo is not actually exercised.
        # KNOWN ISSUE: the compiled function is never called.
        def foo(_stack0):
            # [FUNCTION DOC] HELPER: Takes the last element along dim 1, normalizes it with p of 2
            # along dim 1 and returns it in a tuple.
            getitem = _stack0[(slice(None, None, None), -1)]
            _stack0 = None
            normalize = torch.nn.functional.normalize(getitem, p=2, dim=1)
            getitem = None
            return (normalize,)

        args = [((2, 50, 256), (1, 256, 1), torch.float16, device, False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        torch.compile(foo, backend="aot_eager_decomp_partition")
        with torch.amp.autocast(device_type=device, enabled=True):
            ref = foo(*args)[0]
            res = foo(*args)[0]
            self.assertEqual(ref.dtype, res.dtype)

            self.assertTrue(same(res, ref))

    def test_guard_default_device(self, device):
        # [FUNCTION DOC] Checks that the Dynamo guards on the default device: with the default
        # device set to the test device a compiled function (optimize decorator) creates a tensor
        # on it and compiles once; after setting the default back to cpu the result is on cpu and
        # a recompile occurs (two frames). Always resets the default device to None.
        try:
            torch.set_default_device(device)

            counter = torch._dynamo.testing.CompileCounter()

            @torch._dynamo.optimize(counter)
            def f():
                # [FUNCTION DOC] HELPER: Creates a random 3-element tensor on the default device
                # and doubles it.
                x = torch.randn(3)
                return x * 2

            self.assertEqual(f().device.type + ":0", device)
            self.assertEqual(counter.frame_count, 1)

            torch.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            torch.set_default_device(None)

    @skipIfHpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self, device):
        # [FUNCTION DOC] Skipped on HPU and when flash attention is unsupported. Checks that the
        # meta implementation of scaled dot product flash backward gives the same shapes and
        # strides as the real kernel for inputs with mixed (transposed and contiguous) strides.
        # in this repro, "grad_out" and "value" are transposed tensors,
        # but "key" and "value" are contiguous
        def gen_inputs(device):
            # [FUNCTION DOC] HELPER: Builds the argument tuple for the flash attention backward
            # call on the given device: two transposed and three contiguous float16 tensors, a
            # log-sum-exp tensor, two None values, sequence lengths 513, dropout 0.0, a False flag
            # and two scalar int64 tensors.
            return (
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1, dtype=torch.int64),
            )

        inps_device = gen_inputs(device)
        inps_meta = gen_inputs("meta")
        (
            out1_ref,
            out2_ref,
            out3_ref,
        ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_device, scale=0.125
        )
        from torch._meta_registrations import meta__scaled_dot_product_flash_backward

        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    @onlyAccelerator
    def test_memleak_when_graph_input_has_tensor_attr(self, device):
        # [FUNCTION DOC] Accelerator-only. Checks that a compiled in-place add on an accelerator
        # tensor that carries an extra attribute holding a tensor (directly, in a list, or in a
        # reference cycle) does not leak accelerator memory: allocated memory returns to the
        # starting value after deleting the tensors.
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one in place to x.
            x.add_(1)

        mem_before = torch.accelerator.memory_allocated()

        x = torch.ones(2, device=device)
        x.foo = torch.zeros(2, device=device)
        f(x)
        del x.foo
        del x
        mem_after = torch.accelerator.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check when non-tensor data structure attribute contains a tensor
        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one in place to x (redefinition for
            # the list attribute case).
            x.add_(1)

        mem_before = torch.accelerator.memory_allocated()
        x = torch.ones(2, device=device)
        x.foo = [torch.zeros(2, device=device) for _ in range(5)]
        f(x)
        del x.foo
        del x
        mem_after = torch.accelerator.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check with tensor refcycle
        @torch.compile(backend="eager")
        def g(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of x and y.
            return x + y

        mem_before = torch.accelerator.memory_allocated()
        x = torch.ones(2, device=device)
        y = torch.zeros(2, device=device)
        x.foo = [y]
        y.foo = [x]
        g(x, y)
        del x.foo
        del y.foo
        del x
        del y
        mem_after = torch.accelerator.memory_allocated()
        self.assertEqual(mem_before, mem_after)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3835")
    def test_sdpa_dynamic_shapes(self, device):
        # [FUNCTION DOC] Skipped on XPU (issue 3835). Checks that scaled dot product attention
        # with a causal flag on a view with symbolic dimensions compiles with dynamic shapes
        # (assume-static off) and matches eager.
        def f(x, s0, s1, s2):
            # [FUNCTION DOC] HELPER: Views x as 2 by s0 by s2 by s0 and applies causal scaled dot
            # product attention with itself.
            q = x.view(2, s0, s2, s0)
            return torch._C._nn.scaled_dot_product_attention(
                q, q, q, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        x = torch.randn(2, 32, 4096, dtype=torch.bfloat16, device=device)
        x_ref = x.clone().detach().requires_grad_()
        s0 = 32
        s1 = 64
        s2 = 128

        f_compiled = torch.compile(f, dynamic=True, backend="eager")

        with torch._dynamo.config.patch(assume_static_by_default=False):
            out_ref = f(x_ref, s0, s1, s2)
            out = f_compiled(x, s0, s1, s2)
            self.assertEqual(out_ref, out)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "requires gpu with fp8 support")
    def test_partitioner_saves_weights_for_bw(self, device):
        # [FUNCTION DOC] Skipped without FP8 support. Checks that the partitioner saves the
        # weights for backward (recompute of the weight scaling happens in backward): with a
        # custom FP8 matmul autograd function inside a compiled linear module, the backward
        # contains 5 copy-conversion ops.
        def mul_tiled(a, *bs):
            # [FUNCTION DOC] HELPER: Reshapes a tensor into tiles for each scale tensor and
            # multiplies by the scale broadcast over tiles, flattening back; returns the
            # scaled tensor.
            for b in bs:
                a = a.unflatten(0, (b.shape[0], -1)).unflatten(-1, (b.shape[-1], -1))
                a = a * b[:, None, :, None]
                a = a.flatten(end_dim=1).flatten(start_dim=-2)
            return a

        def scale(t, amax_t):
            # [FUNCTION DOC] HELPER: Computes a float scale from the amax divided by the e4m3 max
            # and returns the tensor scaled to e4m3 together with the scale.
            max_v = E4M3_MAX_POS
            scale_t = torch.clamp(amax_t.float(), min=1e-12) / max_v
            t_fp8 = mul_tiled(t, scale_t.reciprocal()).to(e4m3_type)
            return t_fp8, scale_t

        def matmul(first, amax_first, second_t, amax_second_t, bias):
            # [FUNCTION DOC] HELPER: FP8 scaled matmul: scales both operands, runs the scaled mm
            # with unit scales, rescales the result with the saved scales (tiled), adds the bias
            # after and returns bfloat16 output.
            first_fp8, scale_first = scale(first, amax_first)
            second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
            post_scales = []
            post_bias = None
            post_scales = [scale_first, scale_second_t.t()]
            scale_first = scale_first.new_ones((1, 1))
            scale_second_t = scale_second_t.t().new_ones((1, 1))
            post_bias, bias = bias, None
            res = torch._scaled_mm(
                first_fp8,
                second_t_fp8.t(),
                scale_a=scale_first,
                scale_b=scale_second_t.t(),
                bias=bias,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            res = mul_tiled(res, *post_scales).to(torch.bfloat16)
            if post_bias is not None:
                res += post_bias
            return res

        @torch.compiler.allow_in_graph
        class Fp8LinearFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function (allowed in graph) for FP8 linear: forward
            # computes amax values, the scaled matmul and stores requires-grad flags and saved
            # tensors; backward computes gradients for input, weight and bias as needed.
            @staticmethod
            def forward(ctx, a, b_t, bias):
                # [FUNCTION DOC] HELPER: Computes amax per row of the input and weight, runs
                # matmul, records which inputs require grad, saves the input, weight and weight
                # amax and returns the output.
                amax_a = a.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                amax_b_t = b_t.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                out = matmul(a, amax_a, b_t, amax_b_t, bias)
                ctx.a_requires_grad = a.requires_grad
                ctx.b_requires_grad = b_t.requires_grad
                ctx.bias_requires_grad = (
                    bias.requires_grad if bias is not None else False
                )
                ctx.save_for_backward(a, b_t, amax_b_t)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Computes the input gradient with an FP8 matmul against
                # the weight (when needed), the weight gradient as grad transposed times the
                # input (when needed) and the bias gradient as a sum (when needed); returns the
                # three gradients.
                a, b_t, amax_b_t = ctx.saved_tensors
                # Workaround for https://github.com/pytorch/pytorch/issues/141881.
                # The partitioner would pre-compute the transposed scaling of the weight
                # in the forward (as it's most efficient, but it actually uses too much
                # memory). We prevent that by making the scaling depend on the gradient
                # in a way that has no effect and will be optimized away later.
                # Care is needed to support tensor parallelism and circumvent bugs.
                #        b_t = b_t + grad_out[:1, :, None].squeeze(0) * 0
                if ctx.a_requires_grad:
                    b = b_t.t().contiguous()
                    amax_grad_out = grad_out.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b_t.t().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b.repeat_interleave(
                        b.shape[0] // amax_b.shape[0], dim=0, output_size=b.shape[0]
                    )
                    grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
                else:
                    grad_a = None
                if ctx.b_requires_grad:
                    grad_b = grad_out.t() @ a
                else:
                    grad_b = None
                if ctx.bias_requires_grad:
                    grad_bias = grad_out.sum(dim=0)
                else:
                    grad_bias = None
                return grad_a, grad_b, grad_bias

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three 64-sized parameters (two matrices and a bias);
            # defined but not used further in this test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the three bfloat16 parameters on the test device.
                super().__init__()
                self.a = torch.nn.Parameter(
                    torch.randn(
                        64, 64, dtype=torch.bfloat16, device=device, requires_grad=True
                    )
                )
                self.b = torch.nn.Parameter(
                    torch.randn(
                        64, 64, dtype=torch.bfloat16, device=device, requires_grad=True
                    )
                )
                self.bias = torch.nn.Parameter(
                    torch.randn(
                        64, dtype=torch.bfloat16, device=device, requires_grad=True
                    )
                )

        class CustomLinear(torch.nn.Linear):
            # [CLASS DOC] HELPER: Linear layer subclass whose forward flattens the input to 2D,
            # applies the FP8 autograd function with the weight and bias and restores the
            # leading shape.
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the FP8 linear output with the leading dimensions
                # of the input restored.
                out = Fp8LinearFn.apply(
                    input.flatten(end_dim=-2), self.weight, self.bias
                )
                out = out.unflatten(0, input.shape[:-1])
                return out

        m = CustomLinear(64, 64, dtype=torch.bfloat16, device=device)
        m = torch.compile(m, backend="aot_eager")

        # simple mode to track how many collective ops we saw in the backward
        class TrackingMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode counting how many times each operator is executed
            # in its ops_counter mapping.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer and creates the
                # counter mapping.
                super().__init__()
                self.ops_counter = defaultdict(int)

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the operator, increments its counter and returns
                # the result.
                if kwargs is None:
                    kwargs = {}
                rs = func(*args, **kwargs)
                self.ops_counter[func] += 1
                return rs

        a = torch.randn(64, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
        out = m(a)
        with TrackingMode() as mode:
            out.sum().backward()
        # If you print out the AOT fw and bw graphs,
        # the main thing to look for is that both weights (primals_1/primals_2)
        # *are* saved for backward, and become back inputs.
        # The easier-to-test thing I'm checking for here is that the recompute
        # on primals_2 happens in the backward. With the recompute,
        # there are 5 _to_copy ops in the backward. Without it, there are 4
        # (aka if you set torch._functorch.config.treat_parameters_as_free_to_save = False)
        self.assertEqual(mode.ops_counter[torch.ops.aten._to_copy.default], 5)

    def test_getattr_return(self):
        # [FUNCTION DOC] Checks that the logic of an inspect-like helper using getattr, isinstance
        # against a tuple of C-level callable types, and a return from an else branch, compiles
        # with fullgraph and matches eager.
        _WrapperDescriptor = type(type.__call__)
        _MethodWrapper = type(all.__call__)
        _ClassMethodWrapper = type(int.__dict__["from_bytes"])

        _NonUserDefinedCallables = (
            _WrapperDescriptor,
            _MethodWrapper,
            _ClassMethodWrapper,
            types.BuiltinFunctionType,
        )

        def _signature_get_user_defined_method(cls, method_name):
            # [FUNCTION DOC] HELPER: Returns the attribute of the class if it exists and is not
            # an instance of the non-user-defined callable types, otherwise returns None (also
            # on AttributeError).
            try:
                meth = getattr(cls, method_name)
            except AttributeError:
                return
            else:
                if not isinstance(meth, _NonUserDefinedCallables):
                    # Once '__signature__' will be added to 'C'-level
                    # callables, this check won't be necessary
                    return meth

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns cosine of x if the call attribute lookup yields None,
            # else sine of x.
            s = _signature_get_user_defined_method(type(torch.nn.Linear), "__call__")
            if s is None:
                return torch.cos(x)

            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_dependent_error_log_no_print(self):
        # [FUNCTION DOC] Regression test for PR 149831: a data-dependent branch error in a
        # fullgraph compile must not print the graph module text to stderr. Replaces sys.stderr
        # during the call and restores it afterward.
        # This is a regression test case for
        # https://github.com/pytorch/pytorch/pull/149831
        from io import StringIO

        capturedOutput = StringIO()
        sys.stderr = capturedOutput

        @torch.compile(fullgraph=True, backend="eager")
        def func(a):
            # [FUNCTION DOC] HELPER: Compiled function branching on whether the sum of a is
            # positive (data dependent).
            if a.sum() > 0:
                return a + 1
            return a + 2

        a = torch.rand(10, 10)
        try:
            func(a)
        except Exception:
            pass
        sys.stderr = sys.__stderr__

        # Make sure we don't _print_ out the graph module.
        output = capturedOutput.getvalue()
        self.assertNotIn("class GraphModule", output)

    def test_deepcopy_constant_tensor_in_aot_bwd(self):
        # [FUNCTION DOC] Checks that a constant tensor used in a custom autograd backward (with a
        # dynamic shape factor) can be deep-copied by AOT autograd with dynamic shapes; gradients
        # match eager.
        class Fn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function adding one in forward and in backward
            # multiplying the gradient by a constant tensor 2 and the batch size.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Returns the gradient times a scalar tensor of 2 times its
                # first dimension.
                return grad_out * torch.tensor(2) * grad_out.shape[0]

        def f(x):
            # [FUNCTION DOC] HELPER: Applies Fn to x.
            return Fn.apply(x)

        x = torch.randn(8, requires_grad=True)
        out = f(x)  # should not raise
        c_out = torch.compile(f, backend="aot_eager", dynamic=True)(x)
        expected = torch.autograd.grad(out.sum(), inputs=(x,))
        actual = torch.autograd.grad(c_out.sum(), inputs=(x,))
        self.assertEqual(expected, actual)

    def test_module_attribute_error(self):
        # [FUNCTION DOC] Checks that accessing a nonexistent torch attribute inside a compiled
        # function raises AttributeError when uncaught and is catchable inside the function
        # (returning x plus one).
        @torch.compile(backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled function accessing the missing torch attribute _bar
            # on x.
            return torch._bar(x)

        @torch.compile(backend="eager")
        def f2(x):
            # [FUNCTION DOC] HELPER: Compiled function accessing the missing attribute inside try
            # and returning x plus one on AttributeError.
            try:
                return torch._bar(x)
            except AttributeError:
                return x + 1

        with self.assertRaises(AttributeError):
            f1(torch.ones(3))

        self.assertEqual(f2(torch.ones(3)), torch.ones(3) + 1)

    def test_named_tuple_vt_clone(self):
        # [FUNCTION DOC] Regression for issue 157945: a module using the named tuple result of SVD
        # with slicing and diag_embed compiles with the eager backend and matches eager.
        # https://github.com/pytorch/pytorch/issues/157945
        class SVDCompressor(nn.Module):
            # [CLASS DOC] HELPER: Module that keeps the leading k singular directions and returns
            # U times the diagonal of S.
            def __init__(self, k=10):
                # [FUNCTION DOC] HELPER: Stores the rank k (default 10).
                super().__init__()
                self.k = k

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes U and S from the SVD, and returns U sliced to k
                # columns times the diag_embed of S sliced to k.
                U, S = torch.linalg.svd(x)[:2]
                reduced = U[:, :, : self.k] @ torch.diag_embed(S[:, : self.k])
                return reduced

        input = torch.randn(4, 8, 6)
        model = SVDCompressor(k=5)

        out1 = model(input.clone())
        out2 = torch.compile(model, backend="eager")(input.clone())
        self.assertEqual(out1, out2)

    @onlyAccelerator
    def test_zero_dim_param_mixed_device_grad(self, device):
        # [FUNCTION DOC] Accelerator-only (issue 160084): zero-dimension CPU parameters with
        # accelerator gradients; a compiled forward with aot_eager and fullgraph yields gradients
        # on the CPU for both parameters.
        # cpu 0-dim params with cuda grads
        # https://github.com/pytorch/pytorch/issues/160084
        class RegressionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two scalar float parameters a and b computing x
            # times a plus b.
            def __init__(self, a=0, b=0):
                # [FUNCTION DOC] HELPER: Creates scalar float parameters a and b from the
                # given numbers.
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times a plus b.
                return x * self.a + self.b

        model = RegressionModel()
        model.forward = torch.compile(
            model.forward, backend="aot_eager", fullgraph=True
        )
        inputs = torch.randn(4, 10).to(device)
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, torch.device("cpu"))
        self.assertEqual(model.b.grad.device, torch.device("cpu"))

    def test_filter_warnings(self):
        # [FUNCTION DOC] Checks that a repeated warning issued within a compiled function (called
        # five times) is emitted exactly once with the right message.
        x = torch.ones(2, 2, requires_grad=True)

        def call_foobar(x):
            # [FUNCTION DOC] HELPER: Issues a warning with the text foobar.
            warnings.warn("foobar")

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function calling the warning helper five times.
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            return call_foobar(x)

        with warnings.catch_warnings(record=True) as w:
            f(x)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), "foobar")

    def test_filter_safe_grad_warning(self):
        # [FUNCTION DOC] Checks that the safe_grad utility suppresses the non-leaf grad access
        # warning only at its own call site: a different call site warns once, and repeating that
        # same call site does not warn again.
        x = torch.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn
        torch._subclasses.meta_utils.safe_grad(y)  # filters out warning

        def unsafe_grad(y):
            # [FUNCTION DOC] HELPER: Returns the grad attribute of y (which triggers a warning for
            # non-leaf tensors).
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_filter_user_warnings(self):
        # [FUNCTION DOC] Checks that warnings from a call site wrapped with the
        # distributed-warning suppressor are filtered, while a different call site warns once and
        # does not warn again on repeat.
        x = torch.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn

        @torch._dynamo.eval_frame.TorchPatcher.suppress_torch_distributed_warnings
        def mute_warn(y):
            # [FUNCTION DOC] HELPER: Decorated by the distributed warnings suppressor; returns the
            # grad attribute of y.
            return y.grad

        mute_warn(y)  # filters out warning

        def unsafe_grad(y):
            # [FUNCTION DOC] HELPER: Returns the grad attribute of y (which triggers a warning for
            # non-leaf tensors).
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_partial_export(self):
        # [FUNCTION DOC] Checks that a module whose call implementation was replaced with a
        # partial that converts inputs to bfloat16 can be captured by the functional export
        # capture and gives the same result as eager.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding two inputs, with a parallelize method that wraps
            # its call implementation to convert arguments to bfloat16.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the base initializer only.
                super().__init__()

            def parallelize(self):
                # [FUNCTION DOC] HELPER: Replaces the call implementation attribute with a partial
                # of a wrapper that casts the first two positional arguments to bfloat16 before
                # calling the original.
                fn = self._call_impl

                def wrapped_fn(fn, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Casts the first two arguments to bfloat16 and calls
                    # the given function.
                    new_args_0 = args[0].to(torch.bfloat16)
                    new_args_1 = args[1].to(torch.bfloat16)
                    return fn(new_args_0, new_args_1)

                fn = functools.partial(wrapped_fn, fn)
                self._call_impl = fn

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a plus b.
                return a + b

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        foo = Foo()
        foo.parallelize()
        x = torch.randn(4, 4, dtype=torch.float32)
        y = torch.randn(4, 4, dtype=torch.float32)
        ref = foo(x, y)
        gm = dynamo_graph_capture_for_export(foo)(x, y)
        res = gm(x, y)
        self.assertEqual(res, ref)

    def test_current_accelerator(self):
        # [FUNCTION DOC] Checks that querying the current accelerator inside a fullgraph compiled
        # function is supported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the current accelerator query (result unused) and
            # returns x plus one.
            torch.accelerator.current_accelerator()
            return x + 1

        self.assertEqual(fn(torch.ones(3)), torch.ones(3) + 1)

    def test_pytree_get_node_type_not_traced(self):
        # [FUNCTION DOC] Checks that the pytree node type helper is not traced into: a compiled
        # function calling it on a list plus doing a tree flatten of a dictionary compiles in one
        # frame with correct values.
        # Test that torch.utils._pytree._get_node_type is not traced into
        # and doesn't cause excessive trace time overhead
        from torch.utils._pytree import _get_node_type

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Asserts the node type of a list is list, flattens a
            # dictionary of the two inputs and returns the sum of the leaves.
            # Call _get_node_type which is used internally by pytree operations
            node_type = _get_node_type([x, y])
            assert node_type is list  # noqa: S101
            # Do some work with pytree structures
            data = {"a": x, "b": y}
            flat, spec = pytree.tree_flatten(data)
            result = flat[0] + flat[1]
            return result

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_get_node_type_with_namedtuple(self):
        # [FUNCTION DOC] Checks that the pytree node type helper handles a namedtuple
        # instance without being traced into, with a tree flatten of the namedtuple; one
        # frame, correct sum.
        # Test that torch.utils._pytree._get_node_type handles namedtuples correctly
        # without being traced into, even when is_namedtuple_class is True
        from collections import namedtuple

        from torch.utils._pytree import _get_node_type

        Point = namedtuple("Point", ["x", "y"])

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a Point namedtuple, asserts its node type is
            # namedtuple, flattens it and returns the sum of the leaves.
            # Create a namedtuple
            point = Point(a, b)
            # Call _get_node_type with a namedtuple instance
            node_type = _get_node_type(point)
            assert node_type is namedtuple  # noqa: S101
            # Use pytree operations with namedtuples
            flat, spec = pytree.tree_flatten(point)
            result = flat[0] + flat[1]
            return result

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_tree_is_leaf_not_traced(self):
        # [FUNCTION DOC] Checks that the pytree is-leaf helper is not traced into for tensors
        # (leaf), lists and dicts (not leaves); compiles in one frame with fullgraph.
        # Test that torch.utils._pytree.tree_is_leaf is not traced into
        # when is_leaf parameter is None (the common case)
        from torch.utils._pytree import tree_is_leaf

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Asserts the leaf status of a tensor (True), a list (False)
            # and a dictionary (False), then returns x plus y.
            # Test with various types
            # Tensors are leaves
            is_leaf_tensor = tree_is_leaf(x)
            assert is_leaf_tensor is True  # noqa: S101

            # Lists are not leaves (they're in SUPPORTED_NODES)
            is_leaf_list = tree_is_leaf([x, y])
            assert is_leaf_list is False  # noqa: S101

            # Dicts are not leaves
            is_leaf_dict = tree_is_leaf({"a": x, "b": y})
            assert is_leaf_dict is False  # noqa: S101

            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_ordered_set_doesnt_recompile_with_ac(self):
        # [FUNCTION DOC] Checks that selective activation checkpointing with a policy whose
        # compute-heavy op collection is an OrderedSet does not recompile when called twice with
        # fresh equivalent partials (recompiles are errors).
        import torch

        with torch._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from torch.utils._ordered_set import OrderedSet
            from torch.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Checkpoint policy returning must-save for ops in the
                # given collection and prefer-recompute otherwise.
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                # [FUNCTION DOC] HELPER: Returns exp of sine of x matmul x.
                return torch.mm(x, x).sin().exp()

            @torch.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                # [FUNCTION DOC] HELPER: Compiled function running checkpoint of g with the given
                # policy contexts and non-reentrant mode.
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = torch.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([torch.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([torch.ops.aten.mm.default])),
                ),
            )

    def test_mro_source_cache_includes_attr_name(self):
        # [FUNCTION DOC] Checks that the cache for attribute sources over a class hierarchy
        # includes the attribute name: two class attributes with the same small int value must be
        # guarded separately. After compiling once (one frame), changing the second attribute on
        # the base class recompiles (two frames) and gives the new result 43.
        # Base -> Mid -> A hierarchy: two class attributes with the same
        # interned integer value share the same id().  The mro_source_cache
        # must include the attribute name in its key; otherwise the second
        # lookup returns the first attribute's source, installing a guard
        # on the wrong key and missing mutations to the second attribute.
        class Base:
            # [CLASS DOC] HELPER: Base class with two class attributes x and y both equal to 1.
            x = 1
            y = 1

        class Mid(Base):
            # [CLASS DOC] HELPER: Empty intermediate subclass of Base.
            pass

        class A(Mid):
            # [CLASS DOC] HELPER: Empty subclass of Mid whose instances are used as the
            # compiled argument.
            pass

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(obj, t):
            # [FUNCTION DOC] HELPER: Returns t times attribute x plus t times attribute y of obj.
            return t * obj.x + t * obj.y

        obj = A()
        t = torch.tensor([1.0])
        result = fn(obj, t)
        self.assertEqual(result, torch.tensor([2.0]))
        self.assertEqual(cnt.frame_count, 1)

        # Changing y on Base must trigger recompilation.
        Base.y = 42
        result = fn(obj, t)
        self.assertEqual(result, torch.tensor([43.0]))
        self.assertEqual(cnt.frame_count, 2)

    def test_pytree_tree_is_leaf_with_namedtuple(self):
        # [FUNCTION DOC] Checks the pytree is-leaf helper with a namedtuple (not a leaf) and a
        # tensor (leaf) in a fullgraph compile of one frame.
        # Test that torch.utils._pytree.tree_is_leaf handles namedtuples correctly
        from collections import namedtuple

        from torch.utils._pytree import tree_is_leaf

        Point = namedtuple("Point", ["x", "y"])

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a Point namedtuple, asserts it is not a leaf and that
            # a is a leaf, and returns a plus b.
            # Namedtuples are not leaves (they're in SUPPORTED_NODES)
            point = Point(a, b)
            is_leaf_namedtuple = tree_is_leaf(point)
            assert is_leaf_namedtuple is False  # noqa: S101

            # But individual tensors are leaves
            is_leaf_tensor = tree_is_leaf(a)
            assert is_leaf_tensor is True  # noqa: S101

            return a + b

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_data_attr_mutation_with_noop_add(self):
        # [FUNCTION DOC] Regression test: removal of no-op additions must not alias the returned
        # value to an input that is later mutated by data assignment; compiled (inductor) result
        # for boolean tensors matches eager.
        # Regression test: remove_no_ops incorrectly eliminated add(x, 0) -> x
        # when x was subsequently mutated by set_, causing the return value to
        # alias the mutated input instead of being an independent copy.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Sets a data to b, then sets b data to zeros and returns a
            # plus b.
            a.data = b
            b.data = torch.zeros_like(b)
            return a + b

        a = torch.tensor([True, False, True, False])
        b = torch.tensor([False, False, True, True])
        a_ = a.clone()
        b_ = b.clone()
        cfunc = torch.compile(fn, backend="inductor")
        res1 = fn(a, b)
        res2 = cfunc(a_, b_)
        self.assertEqual(res1, res2)

    def test_custom_op_mutation_with_noop_add(self):
        # [FUNCTION DOC] Checks that a custom op that mutates its first argument after a no-op
        # addition of zeros gives the same result under inductor as eager (the addition result
        # must not alias the mutated input).
        @torch.library.custom_op("test_repros::mutate_tensor", mutates_args={"x"})
        def mutate_tensor(x: torch.Tensor, src: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op (mutating x) copying src into x.
            x.copy_(src)

        def fn(b):
            # [FUNCTION DOC] HELPER: Adds zeros to b, calls the mutating op on b with zeros and
            # returns the earlier result.
            zeros = torch.zeros_like(b)
            result = b + zeros
            mutate_tensor(b, zeros)
            return result

        b = torch.tensor([4.0, 5.0, 6.0])
        b_ = b.clone()
        cfunc = torch.compile(fn, backend="inductor")
        res1 = fn(b)
        res2 = cfunc(b_)
        self.assertEqual(res1, res2)

    def test_getset_descriptor_objclass_identity(self):
        # [FUNCTION DOC] Checks that the objclass attribute of the instance dictionary descriptor
        # of a class is identical to the class in a fullgraph compile.
        # GetSetDescriptor.__objclass__ should preserve identity with the class
        # under torch.compile. This is needed for inspect.getattr_static (and
        # therefore inspect.signature) to work on callable class instances.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        desc = Foo.__dict__["__dict__"]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1.0 if the descriptor objclass is Foo, else x
            # plus 2.0.
            if desc.__objclass__ is Foo:
                return x + 1.0
            return x + 2.0

        result = f(torch.tensor(0.0))
        self.assertEqual(result.item(), 1.0)

    def test_inspect_signature_callable_class(self):
        # [FUNCTION DOC] Checks that signature inspection of a callable class instance works
        # inside a fullgraph compiled function (needed by flex attention): the number of
        # parameters without defaults is 4.
        # inspect.signature should work on callable class instances under
        # torch.compile, needed by flex_attention's _get_mod_type.
        class MyCallable:
            # [CLASS DOC] HELPER: Callable class with a mask-like method of four parameters
            # returning whether q index is at least kv index.
            def __call__(self, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Returns whether q_idx is greater than or equal to kv_idx.
                return q_idx >= kv_idx

        obj = MyCallable()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Counts parameters without a default in the signature of the
            # callable object and adds that count to x.
            sig = inspect.signature(obj)
            num_params = sum(
                1
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            )
            return x + num_params

        result = f(torch.tensor(0.0))
        self.assertEqual(result.item(), 4.0)

    def test_enum_with_class_values(self):
        # [FUNCTION DOC] Checks that an enum whose member value is a class, used to instantiate a
        # metric object inside a fullgraph compiled logger call, is supported.
        from enum import Enum

        class AvgMetric:
            # [CLASS DOC] HELPER: Running-sum metric with a sum and a count.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets sum to None and count to 0.
                self.sum = None
                self.count = 0

            def append(self, x):
                # [FUNCTION DOC] HELPER: Adds x to the running sum (or clones x for the first
                # item) and increments the count.
                if self.count > 0:
                    self.sum = self.sum + x
                else:
                    self.sum = x.clone()
                self.count += 1

        class GlobalReduction(Enum):
            # [CLASS DOC] HELPER: Enum with member AVG whose value is the AvgMetric class.
            AVG = AvgMetric

        class ScalarLogger:
            # [CLASS DOC] HELPER: Logger holding a mapping of metric objects.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates an empty metrics mapping.
                self.metrics = {}

            def log(self, key, value, global_reduction):
                # [FUNCTION DOC] HELPER: Creates the metric by instantiating the enum value if the
                # key is new, then appends the value to it.
                if key not in self.metrics:
                    self.metrics[key] = global_reduction.value()
                self.metrics[key].append(value)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(logger, x):
            # [FUNCTION DOC] HELPER: Logs x under key test with the AVG reduction and returns x
            # plus 1.
            logger.log("test", x, GlobalReduction.AVG)
            return x + 1

        logger = ScalarLogger()
        fn(logger, torch.tensor(1.0))

    def test_class_attr_mutation_recompiles(self):
        # [FUNCTION DOC] Checks that mutating a class attribute used by a compiled function
        # triggers a recompile: results 4.0 and 40.0 with frame counts 1 then 2.
        class GlobalState:
            # [CLASS DOC] HELPER: Class with a float attribute factor.
            factor = 1.0

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x times the global state factor.
            return x * GlobalState.factor

        x = torch.tensor([4.0])

        GlobalState.factor = 1.0
        result1 = fn(x)
        self.assertEqual(result1, torch.tensor([4.0]))
        self.assertEqual(cnt.frame_count, 1)

        GlobalState.factor = 10.0
        result2 = fn(x)
        self.assertEqual(result2, torch.tensor([40.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfHpu
    def test_deterministic_pad_replicate_compile(self, device):
        # [FUNCTION DOC] Skipped on HPU. Checks that under deterministic mode the replicate
        # padding 1D module compiled with aot_eager and fullgraph gives the same output and
        # gradient as eager.
        from torch.testing._internal.common_utils import DeterministicGuard

        pad = torch.nn.ReplicationPad1d(2).to(device)
        compiled_pad = torch.compile(pad, backend="aot_eager", fullgraph=True)
        x = torch.randn(3, 3, device=device, requires_grad=True)
        with DeterministicGuard(True):
            ref = pad(x)
            res = compiled_pad(x)
            self.assertEqual(ref, res)
            grad = torch.autograd.grad(res.sum(), x)
            ref_grad = torch.autograd.grad(ref.sum(), x)
            self.assertEqual(grad, ref_grad)

    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flex_attention_guard_on_constant_func_defaults(self, device):
        # [FUNCTION DOC] Skipped on ROCm or without flash attention. Checks that Dynamo guards on
        # the default arguments of mask functions: a flex attention chunk loop whose mask
        # functions share code but have different default offsets must recompile per chunk, so
        # merging chunk outputs matches a reference causal attention within tolerance. Skips
        # without triton.
        """
        Dynamo must guard on mask_mod.__defaults__ so that when a
        compiled function is re-invoked with a new BlockMask whose
        mask_mod has the same __code__ but different __defaults__,
        Dynamo recompiles instead of reusing the stale first graph.
        """
        from torch.utils._triton import has_triton

        if not has_triton():
            self.skipTest("requires triton")

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def flex_chunk(q, k, v, block_mask, scale):
            # [FUNCTION DOC] HELPER: Compiled function running flex attention with a block mask
            # and scale and returning the output and the log-sum-exp.
            out, aux = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
            return out, aux.lse

        def merge(out, lse, new_out, new_lse):
            # [FUNCTION DOC] HELPER: Merges two partial attention results using a numerically
            # stable log-sum-exp combination; returns the merged output and log-sum-exp.
            lse, new_lse = lse.unsqueeze(-1), new_lse.unsqueeze(-1)
            mx = torch.maximum(lse, new_lse)
            e0, e1 = torch.exp(lse - mx), torch.exp(new_lse - mx)
            d = e0 + e1
            return (out * e0 + new_out * e1) / d, (mx + torch.log(d)).squeeze(-1)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def ref_attn(q, k, v, block_mask, scale):
            # [FUNCTION DOC] HELPER: Compiled reference function running flex attention with the
            # given mask and scale.
            return flex_attention(q, k, v, block_mask=block_mask, scale=scale)

        torch.manual_seed(42)
        B, H, S, D = 1, 1, 512, 16
        NUM_CHUNKS = 4
        chunk_size = S // NUM_CHUNKS

        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        scale = D**-0.5

        merged_out = merged_lse = None
        for step in range(NUM_CHUNKS):
            kv_offset = step * chunk_size

            def mask_mod(b, h, q_idx, kv_idx, _offset=kv_offset):
                # [FUNCTION DOC] HELPER: Mask function allowing positions where the query index is
                # at least the key index plus an offset bound as a default argument.
                return q_idx >= kv_idx + _offset

            bm = create_block_mask(
                mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=chunk_size, device=device
            )
            out, lse = flex_chunk(
                q,
                k[:, :, kv_offset : kv_offset + chunk_size],
                v[:, :, kv_offset : kv_offset + chunk_size],
                bm,
                scale,
            )
            if merged_out is None:
                merged_out, merged_lse = out, lse
            else:
                merged_out, merged_lse = merge(merged_out, merged_lse, out, lse)

        def causal(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Plain causal mask function.
            return q_idx >= kv_idx

        ref_bm = create_block_mask(causal, B=B, H=H, Q_LEN=S, KV_LEN=S, device=device)
        ref_out = ref_attn(q, k, v, ref_bm, scale)

        self.assertTrue(
            (merged_out - ref_out).abs().max().item() < 1e-3,
            "flex_attention mask_mod __defaults__ not properly guarded",
        )

    def test_sub_alpha_scalar_repro(self, device):
        # [FUNCTION DOC] Checks that sub with a scalar and an alpha keyword on a float64 tensor
        # compiles with aot_eager.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function returning x minus 1 scaled by alpha 2.
            return x.sub(1, alpha=2)

        f(torch.ones(2, device=device, dtype=torch.float64))

    def test_truthiness_of_symints_no_recompiles(self, device):
        # [FUNCTION DOC] Checks that branching on the truthiness of a symbolic element count under
        # dynamic shapes does not cause recompiles across two sizes (one frame).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if the element count is nonzero, else x
            # plus two.
            numel = x.numel()
            if numel:
                return x + 1
            else:
                return x + 2

        cnt = torch._dynamo.testing.CompileCounter()
        f_compiled = torch.compile(f, backend=cnt, dynamic=True)

        x1 = torch.randn(4, device=device)
        _ = f_compiled(x1)
        x2 = torch.randn(5, device=device)
        _ = f_compiled(x2)

        self.assertEqual(cnt.frame_count, 1)

    def test_megablocks_moe(self, device):
        # [FUNCTION DOC] Skipped without megablocks. Checks a megablocks MoE layer in half
        # precision gives the same output compiled (eager backend) as in eager.
        try:
            from megablocks.layers import moe
            from megablocks.layers.arguments import Arguments
        except ImportError as e:
            raise unittest.SkipTest("requires megablocks") from e
        bs, sl, hs, num_experts, top_k = (16, 1024, 512, 1, 1)
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        moe_mlp = moe.MoE(args)
        moe_mlp.to(device).half()
        x = torch.randn(sl, bs, hs, device=device).half()
        out1, _ = moe_mlp(x)
        out2, _ = torch.compile(moe_mlp, backend="eager")(x)
        self.assertEqual(out1, out2)

    def test_preserve_stride_with_clone(self, device) -> None:
        # [FUNCTION DOC] Checks that cloning with preserve format keeps the strides of a
        # triangular solve result, both in eager and compiled (eager backend), by comparing the
        # strides of the result and of its clone.
        A = torch.rand(5, 5, device=device)
        B = torch.rand(5, 5, device=device)

        def fn(
            src: torch.Tensor, count: torch.Tensor
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            # [FUNCTION DOC] HELPER: Takes the QR decomposition of src, solves a triangular
            # system, clones the solution preserving format and returns the strides of the
            # solution and of the clone.
            Q, R = torch.linalg.qr(src)
            rhs = torch.ones(Q.shape[0], 1, device=src.device)
            a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
            cloned = a.clone(memory_format=torch.preserve_format)
            return a.stride(), cloned.stride()

        a_stride, cloned_stride = fn(A, torch.zeros(1))
        self.assertEqual(
            a_stride,
            cloned_stride,
            lambda msg: f"{msg}\nStrides should match in eager: {a_stride} against {cloned_stride}",
        )

        compiled_a_stride, compiled_cloned_stride = torch.compile(fn, backend="eager")(
            B, torch.zeros(1)
        )
        self.assertEqual(
            compiled_a_stride,
            compiled_cloned_stride,
            lambda msg: f"{msg}\nStrides should match in eager: {compiled_a_stride} against {compiled_cloned_stride}",
        )


class CUDAReproTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] CUDA-only repro tests (each is skipped when CUDA is unavailable).
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @torch._dynamo.config.patch(capture_scalar_outputs=False)
    def test_aot_backward_context_reentry_after_graph_break(self):
        # [FUNCTION DOC] Skipped without CUDA, with scalar capture disabled. Checks that backward
        # through an aot_eager compiled function with a graph break (scalar item call) between two
        # outputs works: gradients equal ones for x and negative sine for y.
        def fn(x, y, scalar):
            # [FUNCTION DOC] HELPER: Copies x to the CPU twice, views one as the other, reads a
            # scalar (causing a graph break), and returns the view and cosine of y.
            cpu = x.cpu()
            other_cpu = x.cpu()
            before_break = cpu.view_as(other_cpu)
            scalar.item()
            after_break = y.cos()
            return before_break, after_break

        x = torch.randn(8, device="cuda", requires_grad=True)
        y = torch.randn(8, device="cuda", requires_grad=True)
        scalar = torch.randn((), device="cuda")

        before_break, after_break = torch.compile(fn, backend="aot_eager")(x, y, scalar)
        loss = before_break.sum().to("cuda") + after_break.sum()
        loss.backward()

        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, -y.detach().sin())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_sync(self):
        # [FUNCTION DOC] Skipped without CUDA. Checks that a CUDA synchronize call inside a
        # compiled function does not break the graph: one frame and the same result as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, synchronizes the CUDA device and doubles.
            y = x + 1
            torch.cuda.synchronize()
            return y * 2

        x = torch.ones(2, device="cuda")
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_torch_cuda_is_initialized(self):
        # [FUNCTION DOC] Skipped without CUDA. Checks that the cuda is_initialized query is
        # evaluated at trace time and is respected after patching it to return False (second call
        # takes the other branch).
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x plus 1 if CUDA is initialized, else x plus 2.
            if torch.cuda.is_initialized():
                return x + 1
            return x + 2

        inp = torch.randn(3)
        self.assertEqual(f(inp), inp + 1)

        with mock.patch("torch.cuda.is_initialized", lambda: False):
            self.assertEqual(f(inp), inp + 2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_graph_metadata_does_not_retain_cuda_fake_constants(self):
        # [FUNCTION DOC] Skipped without CUDA. Checks that compiling a function that creates a
        # CUDA scalar and deep-copies it does not retain the CUDA memory through graph metadata:
        # allocated memory returns to the starting value after clearing caches.
        def f():
            # [FUNCTION DOC] HELPER: Creates a CUDA float scalar tensor and deep-copies it.
            x = torch.tensor(5, dtype=torch.float32, device="cuda")
            copy.deepcopy(x)

        def clear_cuda_memory(*, reset_dynamo):
            # [FUNCTION DOC] HELPER: Optionally resets Dynamo, collects garbage, clears cuBLAS
            # workspaces and empties the CUDA cache.
            if reset_dynamo:
                torch._dynamo.reset()
            gc.collect()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

        clear_cuda_memory(reset_dynamo=True)
        memory_before = torch.cuda.memory_allocated()

        opt_f = torch.compile(f, backend="eager")
        opt_f()
        clear_cuda_memory(reset_dynamo=False)

        self.assertEqual(torch.cuda.memory_allocated(), memory_before)
        # Keep the compiled callable alive through the assertion. Before the
        # fix, it retained the compiled FX graph, whose FakeTensor metadata
        # retained the real CUDA scalar through FakeTensor.constant.
        self.assertIsNotNone(opt_f)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skipIf(not PLATFORM_SUPPORTS_BF16, "requires CUDA bf16 support")
    def test_layer_norm_mixed_dtype_aot_eager_decomp_partition_errors(self):
        # [FUNCTION DOC] Skipped without CUDA or bf16 support (issue 151478). Checks that layer
        # norm with weight or bias dtypes different from the bfloat16 input raises RuntimeError
        # with the expected dtype message in eager and under aot_eager_decomp_partition, for four
        # dtype combinations.
        # https://github.com/pytorch/pytorch/issues/151478
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            device="cuda",
            dtype=torch.bfloat16,
        )

        def check_error(weight_dtype, bias_dtype, error):
            # [FUNCTION DOC] HELPER: Checks that a layer norm on the bfloat16 input with the given
            # weight and bias dtypes raises RuntimeError matching the message, both eagerly and
            # when compiled.
            def forward(input):
                # [FUNCTION DOC] HELPER: Builds ones weight and bias of the given dtypes on the
                # input device and applies layer norm over the last dim with eps 0.1.
                weight = torch.ones(4, device=input.device, dtype=weight_dtype)
                bias = torch.ones(4, device=input.device, dtype=bias_dtype)
                return torch.layer_norm(
                    input,
                    (4,),
                    weight,
                    bias,
                    0.1,
                    torch.backends.cudnn.enabled,
                )

            with self.assertRaisesRegex(RuntimeError, error):
                forward(x)

            compiled_forward = torch.compile(
                forward, backend="aot_eager_decomp_partition"
            )
            with self.assertRaisesRegex(RuntimeError, error):
                compiled_forward(x)

        check_error(
            torch.float32,
            torch.float32,
            "expected scalar type BFloat16 but found Float",
        )
        check_error(
            torch.float16,
            torch.float32,
            "expected scalar type BFloat16 but found Half",
        )
        check_error(
            torch.bfloat16,
            torch.float16,
            "expected scalar type BFloat16 but found Half",
        )
        check_error(
            torch.int64,
            torch.bfloat16,
            "expected scalar type BFloat16 but found Long",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_device_context_matmul_avoids_native_bmm_router_graph_break(self):
        # [FUNCTION DOC] Skipped without CUDA. Checks that a matmul inside a CUDA device
        # context manager, compiled with inductor, does not graph break on trace-unsafe
        # copy-on-write probes or boxed calls: those strings must not appear in the recorded
        # graph break reasons.
        torch._dynamo.utils.counters.clear()

        @torch.compile(backend="inductor", dynamic=False)
        def fn(q, k):
            # [FUNCTION DOC] HELPER: Under the CUDA device context reshapes q into 4D and returns
            # its matmul with k.
            with torch.device("cuda"):
                a = torch.reshape(q, [-1, 8, 1, 32])
                return torch.matmul(a, k)

        q = torch.randn(64, 8 * 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.float16)

        out = fn(q, k)
        torch.cuda.synchronize()
        self.assertEqual(out.shape, (64, 8, 1, 128))

        graph_break_reasons = "\n".join(
            torch._dynamo.utils.counters["graph_break"].keys()
        )
        # The native router should not run trace-unsafe eager predicates here.
        # If it does, the COW probe on the reshaped operand graph-breaks before
        # it can fold or install a guard.
        self.assertNotIn("_is_cow_tensor", graph_break_reasons)
        self.assertNotIn("call_boxed", graph_break_reasons)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skipIf(not dist.is_available(), "test requires distributed")
    # TODO: Remoe this skip once nccl issue if fixed
    @unittest.skip(
        "Failing with ncc update 2.25.1 : https://github.com/pytorch/pytorch/issues/147141"
    )
    def test_ddp_checkpoint(self):
        # [FUNCTION DOC] Skipped (CUDA, distributed, and disabled due to an NCCL issue 147141).
        # Would check that a DDP-wrapped MLP using non-reentrant checkpoint around a compiled
        # forward trains over a dataloader with a single-process NCCL group, restoring environment
        # variables and destroying the process group at the end.
        # https://github.com/pytorch/pytorch/issues/144035
        DIM = 256
        SEQ_LEN = 32

        @torch.compile(backend="eager", fullgraph=True)
        def mlp_forward(x, w1, w2, b1, b2):
            # [FUNCTION DOC] HELPER: Compiled MLP forward with two functional linear layers and
            # relu between them.
            y = F.linear(x, w1, b1)
            y = F.relu(y)
            y = F.linear(y, w2, b2)
            return y

        class MLP(nn.Module):
            # [CLASS DOC] HELPER: MLP module with two weight matrices and two biases whose
            # forward applies the compiled MLP under non-reentrant checkpoint and asserts the
            # result is a tensor.
            def __init__(
                self,
                in_features: int,
                hidden_features: int,
                out_features: int,
            ):
                # [FUNCTION DOC] HELPER: Creates the weight and bias parameters from the input,
                # hidden and output sizes.
                super().__init__()
                self.w_in = nn.Parameter(torch.randn(hidden_features, in_features))
                self.w_out = nn.Parameter(torch.randn(out_features, hidden_features))
                self.b_in = nn.Parameter(torch.randn(hidden_features))
                self.b_out = nn.Parameter(torch.randn(out_features))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the compiled MLP forward under checkpoint with the
                # stored parameters and returns the result.
                result = torch.utils.checkpoint.checkpoint(
                    mlp_forward,
                    x,
                    self.w_in,
                    self.w_out,
                    self.b_in,
                    self.b_out,
                    use_reentrant=False,
                )
                assert isinstance(result, torch.Tensor)  # noqa: S101
                return result

        x = torch.randn(100, SEQ_LEN, DIM)
        y = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        model = MLP(DIM, 4 * DIM, DIM)

        try:
            # required for DDP wrapper initialization
            prior_master_addr = os.environ.get("MASTER_ADDR", None)
            prior_master_port = os.environ.get("MASTER_PORT", None)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
            model = model.to("cuda")
            model = nn.parallel.DistributedDataParallel(model)

            for batch in dataloader:
                x, y = batch
                x = x.to("cuda")
                output = model(x)
                loss = output.sum()
                loss.backward()
        finally:
            dist.destroy_process_group()
            if prior_master_addr:
                os.environ["MASTER_ADDR"] = prior_master_addr
            else:
                del os.environ["MASTER_ADDR"]

            if prior_master_port:
                os.environ["MASTER_PORT"] = prior_master_port
            else:
                del os.environ["MASTER_PORT"]


instantiate_parametrized_tests(ReproTests)

devices = ["cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    ReproTestsDevice, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
