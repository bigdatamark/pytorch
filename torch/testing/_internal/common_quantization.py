# mypy: ignore-errors

# [MODULE DOC] Shared helpers for quantization tests: a node-matching spec for FX graphs,
# training/eval/serialization utilities, skip decorators keyed on quantized engine availability,
# weight-group quantization helpers, the QuantizationTestCase base classes with graph/eager/JIT
# check methods, and a large collection of small toy nn.Module fixtures.
# Import-time side effects: imports quantization, distributed and functorch control-flow modules;
# tries to import the FX quantization and numeric-suite APIs and records success in HAS_FX (False
# on ImportError); probes for torchvision and sets HAS_TORCHVISION and the skip_if_no_torchvision
# marker; creates a module-level default cross entropy loss.

r"""Importing this file includes common utility methods and base classes for
checking quantization api and properties of resulting modules.
"""

import torch
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from functorch.experimental import control_flow
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization import (
    convert,
    default_dynamic_qat_qconfig,
    default_dynamic_qconfig,
    default_dynamic_quant_observer,
    default_embedding_qat_qconfig,
    default_observer,
    default_per_channel_qconfig,
    default_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    default_weight_observer,
    DeQuantStub,
    float_qparams_weight_only_qconfig,
    get_default_qat_qconfig,
    get_default_qat_qconfig_mapping,
    get_default_qconfig,
    get_default_qconfig_mapping,
    PerChannelMinMaxObserver,
    propagate_qconfig_,
    QConfig,
    QConfigMapping,
    quantize,
    quantize_dynamic_jit,
    quantize_jit,
    QuantStub,
    QuantType,
    QuantWrapper,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
)

from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

try:
    from torch.ao.ns.fx.ns_types import NSSingleResultValuesType, NSSubgraph

    # graph mode quantization based on fx
    from torch.ao.quantization.quantize_fx import (
        convert_fx,
        convert_to_reference_fx,
        prepare_fx,
        prepare_qat_fx,
    )
    from torch.fx import GraphModule
    from torch.fx.graph import Node

    HAS_FX = True
except ImportError:
    HAS_FX = False

import copy
import functools
import io
import os

import unittest
from typing import Any
from collections.abc import Callable

import numpy as np
import torch._dynamo as torchdynamo
from torch.testing import FileCheck


class NodeSpec:
    # [CLASS DOC] Hashable (op, target) pair used to describe an FX graph node when checking graph
    # contents. Attributes: op (call_function, call_method or call_module) and target (a function,
    # method name, or module type). Equality and hashing use both fields, so instances can be dict
    # keys; comparing with a non-NodeSpec yields NotImplemented. Factory classmethods build each
    # op kind.
    """Used for checking GraphModule Node"""

    def __init__(self, op, target):
        # [FUNCTION DOC] Stores the node op string and its target.
        """
        op: call_function | call_module
        target:
          for call_function, target would be a function
          for call_module, target would be the type of PyTorch module
        """
        self.op = op
        self.target = target

    @classmethod
    def call_function(cls, target):
        # [FUNCTION DOC] Builds a NodeSpec with op call_function for the given target function.
        return NodeSpec("call_function", target)

    @classmethod
    def call_method(cls, target):
        # [FUNCTION DOC] Builds a NodeSpec with op call_method for the given method name.
        return NodeSpec("call_method", target)

    @classmethod
    def call_module(cls, target):
        # [FUNCTION DOC] Builds a NodeSpec with op call_module for the given module type.
        return NodeSpec("call_module", target)

    def __hash__(self):
        # [FUNCTION DOC] Hash combining op and target.
        return hash((self.op, self.target))

    def __eq__(self, other):
        # [FUNCTION DOC] Returns NotImplemented for non-NodeSpec; otherwise True when both op and
        # target are equal.
        if not isinstance(other, NodeSpec):
            return NotImplemented

        return self.op == other.op and self.target == other.target

    def __repr__(self):
        # [FUNCTION DOC] String form: the op repr, a space, then the target repr.
        return repr(self.op) + " " + repr(self.target)


def get_supported_device_types():
    # [FUNCTION DOC] Returns the list of device type strings to test on: cpu, plus cuda when CUDA
    # is available and the build is not ROCm.
    return (
        ["cpu", "cuda"] if torch.cuda.is_available() and not TEST_WITH_ROCM else ["cpu"]
    )


def test_only_eval_fn(model, calib_data):
    # [FUNCTION DOC] Default calibration function: runs the model once on every entry of
    # calib_data, unpacking each entry as positional arguments. Returns nothing.
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for inp in calib_data:
        model(*inp)


_default_loss_fn = torch.nn.CrossEntropyLoss()


def test_only_train_fn(model, train_data, loss_fn=_default_loss_fn):
    # [FUNCTION DOC] Default training function. Builds an Adam optimizer (learning rate 0.001) and
    # runs 10 epochs over train_data, each batch being an (input, target) pair, doing a standard
    # forward, loss, backward, step. Mutates the model parameters and puts it in train mode.
    # Returns a tuple of accumulated loss, number of correct predictions and total sample count.
    r"""
    Default train function takes a torch.utils.data.Dataset and train the model
    on the dataset
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    for _ in range(10):
        model.train()

        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total


class AverageMeter:
    # [CLASS DOC] Running-statistics tracker. Attributes: name and fmt (format spec used when
    # printing), val (latest value), sum, count and avg (weighted mean). reset zeroes the
    # statistics, update adds a value with a weight, and the string form shows name, value
    # and average.
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        # [FUNCTION DOC] Stores name and format spec and resets the statistics.
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        # [FUNCTION DOC] Sets val, avg, sum and count back to zero.
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # [FUNCTION DOC] Records val as the latest value, adds val weighted by n to the sum, adds n
        # to the count and recomputes the average. Raises ZeroDivisionError if the resulting count
        # is zero.
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # [FUNCTION DOC] Formats name, current value and average using the stored format spec.
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    # [FUNCTION DOC] Computes top-k accuracy in percent without tracking gradients. For each k in
    # topk, returns a one-element tensor with the percentage of samples whose true class is among
    # the k highest-scoring predictions of output (shape batch by classes). Returns a list ordered
    # like topk.
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    # [FUNCTION DOC] Trains the model for at most ntrain_batches batches from data_loader: sets
    # train mode, moves data to device, runs forward, loss, backward and optimizer step, prints a
    # dot per batch, and stops early once the batch count is reached. Returns nothing. The accuracy
    # value computed each batch is discarded.
    model.train()
    for cnt, (image, target) in enumerate(data_loader, start=1):
        print(".", end="")
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy(output, target, topk=(1, 5))
        if cnt >= ntrain_batches:
            return
    return


def ddp_setup(rank, world_size):
    # [FUNCTION DOC] Sets MASTER_ADDR to localhost and MASTER_PORT to a fixed port in the
    # environment and initializes the gloo process group for the given rank and world size. Side
    # effects: mutates os.environ and global distributed state.
    # KNOWN ISSUE: the port is hard coded, so concurrent runs on one host can collide.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup():
    # [FUNCTION DOC] Destroys the default distributed process group.
    dist.destroy_process_group()


def run_ddp(rank, world_size, prepared):
    # [FUNCTION DOC] Runs one DDP training epoch for a prepared QAT model on the given rank: sets
    # up the process group, moves the model to CUDA, wraps it in DistributedDataParallel, trains
    # one batch with SGD, then tears down the group.
    # KNOWN ISSUE: criterion and dataset are undefined names in this module (lint-suppressed), so
    # calling this raises NameError after setup.
    ddp_setup(rank, world_size)
    prepared.cuda()
    prepared = torch.nn.parallel.DistributedDataParallel(prepared, device_ids=[rank])
    prepared.to(rank)
    model_with_ddp = prepared
    optimizer = torch.optim.SGD(model_with_ddp.parameters(), lr=0.0001)
    train_one_epoch(model_with_ddp, criterion, optimizer, dataset, rank, 1)  # noqa: F821
    ddp_cleanup()


def convert_dynamic(module):
    # [FUNCTION DOC] Converts a prepared dynamic-quantization module in place using the default
    # dynamic quantization module mappings. Returns nothing.
    convert(module, get_default_dynamic_quant_module_mappings(), inplace=True)


def prepare_dynamic(model, qconfig_dict=None):
    # [FUNCTION DOC] Attaches qconfigs to the model (and children) in place from qconfig_dict by
    # propagating them; this is the prepare step for dynamic quantization. Returns nothing.
    propagate_qconfig_(model, qconfig_dict)


def _make_conv_test_input(
    batch_size,
    in_channels_per_group,
    input_feature_map_size,
    out_channels_per_group,
    groups,
    kernel_size,
    X_scale,
    X_zero_point,
    W_scale,
    W_zero_point,
    use_bias,
    use_channelwise,
):
    # [FUNCTION DOC] HELPER: Builds random conv test data. Inputs: batch size, per-group channel
    # counts, spatial size tuple, groups, kernel size tuple, activation scale and zero point, lists
    # of weight scales and zero points (extended to out_channels entries), and flags for bias and
    # per-channel weights. Values are small integers so integer kernels do not overflow. Returns a
    # tuple of float input, quantized (quint8) input, float weight, quantized (qint8) weight
    # (per-channel or per-tensor) and the bias or None.
    in_channels = in_channels_per_group * groups
    out_channels = out_channels_per_group * groups

    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        (
            batch_size,
            in_channels,
        )
        + input_feature_map_size,
    )
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8
    )

    W_scale = W_scale * out_channels
    W_zero_point = W_zero_point * out_channels
    # Resize W_scale and W_zero_points arrays equal to out_channels
    W_scale = W_scale[:out_channels]
    W_zero_point = W_zero_point[:out_channels]
    # For testing, we use small values for weights and for activations so that
    # no overflow occurs in vpmaddubsw instruction. If the overflow occurs in
    # qconv implementation and if there is no overflow.
    # In reference we can't exactly match the results with reference.
    # Please see the comment in qconv implementation file
    #   aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
    (W_value_min, W_value_max) = (-5, 5)
    # The operator expects them in the format
    # (out_channels, in_channels/groups,) + kernel_size
    W_init = torch.randint(
        W_value_min,
        W_value_max,
        (
            out_channels,
            in_channels_per_group,
        )
        + kernel_size,
    )
    b_init = torch.randint(0, 10, (out_channels,))

    if use_channelwise:
        W_shape = (-1, 1) + (1,) * len(kernel_size)
        W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
        W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
        W = (
            W_scales_tensor.reshape(*W_shape)
            * (W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
        )
        b = X_scale * W_scales_tensor * b_init.float()
        W_q = torch.quantize_per_channel(
            W,
            W_scales_tensor.double(),
            W_zero_points_tensor.long(),
            0,
            dtype=torch.qint8,
        )
    else:
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        b = X_scale * W_scale[0] * b_init.float()
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8
        )

    return (X, X_q, W, W_q, b if use_bias else None)


def _make_conv_add_extra_input_tensor(scale, zero_point, sizes):
    # [FUNCTION DOC] HELPER: Builds a random extra operand for fused conv-add tests: small integer
    # data of the given sizes, dequantized with scale and zero point. Returns the float tensor and
    # its quint8 quantized version.
    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        sizes,  # Infer the size of tensor to do the add
    )
    X = scale * (X_init - zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=scale, zero_point=zero_point, dtype=torch.quint8
    )
    return X, X_q


def skipIfNoFBGEMM(fn):
    # [FUNCTION DOC] Decorator that skips tests when the FBGEMM engine is unavailable. Applied to a
    # class it marks the class as skipped at decoration time if fbgemm is not a supported quantized
    # engine; applied to a function it returns a wrapper that raises unittest.SkipTest at call time
    # when unavailable.
    reason = "Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support AVX2 or newer."
    if isinstance(fn, type):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoQNNPACK(fn):
    # [FUNCTION DOC] Decorator that skips tests when the QNNPACK engine is unavailable. Applied to
    # a class it marks the class as skipped at decoration time if qnnpack is not a supported
    # quantized engine; applied to a function it returns a wrapper that raises unittest.SkipTest at
    # call time when unavailable.
    reason = "Quantized operations require QNNPACK."
    if isinstance(fn, type):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def withQNNPACKBackend(fn):
    # [FUNCTION DOC] Decorator that skips when QNNPACK is unsupported and otherwise runs the test
    # under the qnnpack quantized engine. Applied to a class it only marks it skipped when
    # qnnpack is unavailable (the engine is not switched for classes); applied to a function it
    # returns a wrapper.
    # TODO(future PR): consider combining with skipIfNoQNNPACK,
    # will require testing of existing callsites
    reason = "Quantized operations require QNNPACK."
    if isinstance(fn, type):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest when qnnpack is unsupported, else
        # runs the test inside the engine override context. Return value is discarded.
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        with override_quantized_engine("qnnpack"):
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNN(fn):
    # [FUNCTION DOC] Decorator that skips tests when the ONEDNN engine is unavailable. Applied to a
    # class it marks the class as skipped at decoration time if onednn is not a supported quantized
    # engine; applied to a function it returns a wrapper that raises unittest.SkipTest at call time
    # when unavailable.
    reason = "Quantized operations require ONEDNN."
    if isinstance(fn, type):
        if "onednn" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if "onednn" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNNBF16(fn):
    # [FUNCTION DOC] Decorator that skips tests when oneDNN bf16 support is missing. Applied to a
    # class it marks the class as skipped at decoration time if the mkldnn bf16 support query
    # returns false; applied to a function it returns a wrapper that raises unittest.SkipTest at
    # call time when unavailable.
    reason = "Quantized operations require BF16 support."
    if isinstance(fn, type):
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoX86(fn):
    # [FUNCTION DOC] Decorator that skips tests when the x86 quantized engine is unavailable.
    # Applied to a class it marks the class as skipped at decoration time if x86 is not a supported
    # quantized engine; applied to a function it returns a wrapper that raises unittest.SkipTest at
    # call time when unavailable.
    reason = "Quantized operations require X86."
    if isinstance(fn, type):
        if "x86" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if "x86" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoDynamoSupport(fn):
    # [FUNCTION DOC] Decorator that skips tests when Dynamo is unsupported on the platform. Applied
    # to a class it marks the class as skipped at decoration time if Dynamo reports it is
    # unsupported; applied to a function it returns a wrapper that raises unittest.SkipTest at call
    # time when unavailable.
    reason = "dynamo doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_dynamo_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if not torchdynamo.is_dynamo_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoInductorSupport(fn):
    # [FUNCTION DOC] Decorator that skips tests when Inductor is unsupported on the platform.
    # Applied to a class it marks the class as skipped at decoration time if Inductor reports it is
    # unsupported; applied to a function it returns a wrapper that raises unittest.SkipTest at call
    # time when unavailable.
    reason = "inductor doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_inductor_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that raises SkipTest with the stored reason when the
        # requirement is unavailable, otherwise calls the wrapped test Return value of the wrapped
        # function is discarded.
        if not torchdynamo.is_inductor_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


try:
    import torchvision  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def get_script_module(model, tracing, data):
    # [FUNCTION DOC] Returns a TorchScript version of model: traced with the given example data
    # when tracing is true, else scripted (data unused).
    return torch.jit.trace(model, data) if tracing else torch.jit.script(model)


def lengths_to_offsets(t, offset_type=np.int64, use_begin_offset=True):
    # [FUNCTION DOC] Converts a 1-D tensor of bag lengths into embedding_bag offsets. Builds a
    # zero-prefixed running sum with the given numpy offset dtype and returns it as a tensor: the
    # beginning offsets (all but the last cumulative value) when use_begin_offset is true,
    # otherwise the ending offsets (all but the first).
    """
    Convert lengths to offsets for embedding_bag
    """
    tt = np.zeros((t.shape[0] + 1,), dtype=offset_type)
    tt[1:] = t
    tt = torch.from_numpy(np.cumsum(tt, dtype=offset_type))
    if use_begin_offset:
        return tt[:-1]
    return tt[1:]


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    # [FUNCTION DOC] HELPER: Asymmetric groupwise n-bit quantization of a 2-D weight. Transposes w,
    # splits rows into groups of q_group_size, and per group computes scale (range over max
    # integer, floored at 1e-6) and zero point, then rounds and clamps to unsigned n-bit integers.
    # On non-CPU devices two 4-bit values are packed per uint8. Returns the integer weight and a
    # tensor interleaving scales and zeros per group. Raises AssertionError for wrong rank, group
    # size not above one, width not divisible by the group size, or NaNs at any stage.
    if w.dim() != 2:
        raise AssertionError(f"expected w.dim() == 2, got {w.dim()}")
    w = w.transpose(0, 1).contiguous()
    if q_group_size <= 1:
        raise AssertionError(f"expected q_group_size > 1, got {q_group_size}")
    if w.shape[-1] % q_group_size != 0:
        raise AssertionError(
            f"expected w.shape[-1] % q_group_size == 0, got w.shape[-1]={w.shape[-1]}, q_group_size={q_group_size}"
        )

    to_quant = w.reshape(-1, q_group_size)
    if torch.isnan(to_quant).sum() != 0:
        raise AssertionError("to_quant contains NaN values")

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    if torch.isnan(scales).sum() != 0:
        raise AssertionError("scales contains NaN values")

    zeros = min_val + scales * (2 ** (n_bit - 1))
    if torch.isnan(zeros).sum() != 0:
        raise AssertionError("zeros contains NaN values")

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    if torch.isnan(out).sum() != 0:
        raise AssertionError("out contains NaN values")

    out = out.to(dtype=torch.int32).reshape(w.shape)
    if out.device != torch.device("cpu"):
        out = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out, scales_and_zeros


def _group_quantize_tensor_symmetric(w, n_bit=4, groupsize=32):
    # [FUNCTION DOC] HELPER: Symmetric groupwise n-bit quantization of a K by N weight. Transposes
    # w, takes per-group absolute max as scale base over the signed max, quantizes with a rounding
    # offset, and packs two 4-bit values per uint8 byte. Returns the packed weight and the
    # flattened float32 scales. Raises AssertionError for bad rank, group size or width.
    # KNOWN ISSUE: the rank check runs after the transpose, so non-2-D input fails inside transpose
    # instead of with the assertion; the zeros tensor is computed but never returned or used.
    # W is of shape [K x N]
    # We transpose W as Quantization is applied on [N x K]
    w = w.transpose(0, 1).contiguous()
    if w.dim() != 2:
        raise AssertionError(f"Expected w.dim() == 2, got {w.dim()}")
    if groupsize <= 1:
        raise AssertionError(f"Expected groupsize > 1, got {groupsize}")
    if w.shape[-1] % groupsize != 0:
        raise AssertionError(f"Expected w.shape[-1] % groupsize == 0, got {w.shape[-1]} % {groupsize}")
    # Calculate scale and zeros
    to_quant = w.reshape(-1, groupsize)
    max_val = to_quant.abs().amax(dim=1, keepdim=True)
    eps = torch.finfo(max_val.dtype).eps
    max_int = 2 ** (n_bit - 1) - 1  # For 4-bit, this is 7
    scales = max_val.clamp(min=eps) / max_int
    zeros = torch.zeros_like(scales)

    # Quantize the weight
    scales = scales.to(torch.float32).reshape(w.shape[0], -1)
    zeros = zeros.to(torch.float32).reshape(w.shape[0], -1)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    max_int = 2**n_bit - 1
    w_int8 = to_quant.div(scales).add(8.5).to(torch.int8).clamp(max=max_int)
    # We pack 2 signed int4 values in unsigned uint8 container.
    # This reduces the weight size by half and improves load perf
    out_uint8 = (w_int8[::, 1::2] << 4 | w_int8[::, ::2]).to(torch.uint8)

    scales_and_zeros = scales.squeeze().contiguous()

    return out_uint8, scales_and_zeros


def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # [FUNCTION DOC] HELPER: Per-row (dim 1 reduction) symmetric dynamic quantization of x adapted
    # from gpt-fast. Computes scales from the larger magnitude of min and max divided by half the
    # quantized range (floored at float32 eps), uses zero points of zero, divides, rounds and
    # clamps to [quant_min, quant_max], and casts to target_dtype. Returns quantized values, scales
    # in the original dtype, and int64 zero points.
    # source: https://github.com/meta-pytorch/gpt-fast/blob/main/quantize.py
    # default setup for affine quantization of activations
    x_dtype = x.dtype
    x = x.float()
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales.to(x_dtype), zero_points


# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):
    # [CLASS DOC] Base TestCase for quantization tests. setUp builds random calibration and
    # training datasets (flat, 1D/2D/3D image lists, train variants, and a dict keyed by dimension)
    # and the lists of quant types. Provides check methods that assert module structure after
    # prepare/convert (observers, qconfig, swapped quantized modules), JIT/eager serialization
    # round trips, graph-mode (JIT and FX) quantization checks, and FX graph node inspection.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp, then creates instance attributes calib_data,
        # train_data, img_data_{1d,2d,3d} and their _train variants (lists of small random float
        # batches, targets are long zeros), img_data_dict mapping dimension to data,
        # static_quant_types (static, QAT) and all_quant_types (dynamic, static, QAT).
        super().setUp()
        self.calib_data = [[torch.rand(2, 5, dtype=torch.float)] for _ in range(2)]
        self.train_data = [
            [
                torch.rand(2, 5, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_1d = [[torch.rand(2, 3, 10, dtype=torch.float)] for _ in range(2)]
        self.img_data_2d = [
            [torch.rand(1, 3, 10, 10, dtype=torch.float)] for _ in range(2)
        ]
        self.img_data_3d = [
            [torch.rand(1, 3, 5, 5, 5, dtype=torch.float)] for _ in range(2)
        ]
        self.img_data_1d_train = [
            [
                torch.rand(2, 3, 10, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_2d_train = [
            [
                torch.rand(1, 3, 10, 10, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_3d_train = [
            [
                torch.rand(1, 3, 5, 5, 5, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]

        self.img_data_dict = {
            1: self.img_data_1d,
            2: self.img_data_2d,
            3: self.img_data_3d,
        }

        # Quant types that produce statically quantized ops
        self.static_quant_types = [QuantType.STATIC, QuantType.QAT]
        # All quant types for (fx based) graph mode quantization
        self.all_quant_types = [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT]

    def checkNoPrepModules(self, module):
        # [FUNCTION DOC] Asserts the module has no quant or dequant attribute.
        r"""Checks the module does not contain child
        modules for quantization preparation, e.g.
        quant, dequant and observer
        """
        self.assertFalse(hasattr(module, "quant"))
        self.assertFalse(hasattr(module, "dequant"))

    def checkNoQconfig(self, module):
        # [FUNCTION DOC] Recursively asserts neither the module nor any descendant has a
        # qconfig attribute.
        r"""Checks the module does not contain qconfig"""
        self.assertFalse(hasattr(module, "qconfig"))

        for child in module.children():
            self.checkNoQconfig(child)

    def checkHasPrepModules(self, module):
        # [FUNCTION DOC] Asserts the module has module, quant and dequant attributes, as produced
        # by a QuantWrapper.
        r"""Checks the module contains child
        modules for quantization preparation, e.g.
        quant, dequant and observer
        """
        self.assertTrue(hasattr(module, "module"))
        self.assertTrue(hasattr(module, "quant"))
        self.assertTrue(hasattr(module, "dequant"))

    def checkObservers(
        self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None
    ):
        # [FUNCTION DOC] Recursively asserts that qualifying leaf modules have an
        # activation_post_process observer. A module qualifies when it has a non-None qconfig, is a
        # leaf (ignoring the observer child), is not Sequential and is in the propagate list, or is
        # a registered custom float-to-observed class; DeQuantStub is exempt. Recursion skips
        # children of QAT, custom observed and fused modules, and skips Dropout children. Raises
        # AssertionError via the test assertions.
        r"""Checks the module or module's leaf descendants
        have observers in preparation for quantization
        """
        if propagate_qconfig_list is None:
            propagate_qconfig_list = get_default_qconfig_propagation_list()
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        float_to_observed_module_class_mapping = prepare_custom_config_dict.get(
            "float_to_observed_custom_module_class", {}
        )

        # check if a module is a leaf module, ignoring activation_post_process attribute
        def is_leaf_module(module):
            # [FUNCTION DOC] HELPER: Returns True when the module has no children other
            # than activation_post_process.
            submodule_name_count = 0
            for name, _ in module.named_children():
                if name != "activation_post_process":
                    submodule_name_count += 1
            return submodule_name_count == 0

        if (
            hasattr(module, "qconfig")
            and module.qconfig is not None
            and (
                (
                    is_leaf_module(module)
                    and not isinstance(module, torch.nn.Sequential)
                    and type(module) in propagate_qconfig_list
                )
                or type(module) in float_to_observed_module_class_mapping
            )
            and not isinstance(module, torch.ao.quantization.DeQuantStub)
        ):
            self.assertTrue(
                hasattr(module, "activation_post_process"),
                "module: " + str(type(module)) + " do not have observer",
            )
        # we don't need to check observers for child modules of the
        # qat modules
        if (
            type(module) not in get_default_qat_module_mappings().values()
            and type(module) not in float_to_observed_module_class_mapping.values()
            and not isinstance(module, _FusedModule)
        ):
            for child in module.children():
                if type(child) is nn.Dropout:
                    continue
                self.checkObservers(
                    child, propagate_qconfig_list, prepare_custom_config_dict
                )

    def checkQuantDequant(self, mod):
        # [FUNCTION DOC] Asserts mod.quant is exactly nnq.Quantize and mod.dequant is
        # exactly nnq.DeQuantize.
        r"""Checks that mod has nn.Quantize and
        nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    def checkWrappedQuantizedLinear(self, mod):
        # [FUNCTION DOC] Asserts the wrapped inner module is exactly nnq.Linear and that Quantize
        # and DeQuantize submodules are present.
        # KNOWN ISSUE: the docstring mentions a qint32 bias, but no bias dtype check is performed.
        r"""Checks that mod has been swapped for an nnq.Linear
        module, the bias is qint32, and that the module
        has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.checkQuantDequant(mod)

    def checkQuantizedLinear(self, mod):
        # [FUNCTION DOC] Asserts mod is exactly an nnq.Linear.
        self.assertEqual(type(mod), nnq.Linear)

    def checkDynamicQuantizedLinear(self, mod, dtype):
        # [FUNCTION DOC] Asserts mod is exactly nnqd.Linear and its packed params dtype
        # equals dtype.
        # KNOWN ISSUE: the docstring claims the bias is checked as float; it is not.
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        self.assertEqual(type(mod), nnqd.Linear)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def checkDynamicQuantizedLinearRelu(self, mod, dtype):
        # [FUNCTION DOC] Asserts mod is exactly nniqd.LinearReLU and its packed params dtype
        # equals dtype.
        # KNOWN ISSUE: the docstring says nnqd.Linear and a float bias check; code checks the fused
        # LinearReLU type only.
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        self.assertEqual(type(mod), nniqd.LinearReLU)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def check_eager_serialization(self, ref_model, loaded_model, x):
        # [FUNCTION DOC] Verifies eager serialization of a quantized module. Saves the reference
        # state dict to memory, loads it into loaded_model, and compares outputs on inputs x
        # (sequence of positional arguments); then saves and reloads the whole reference model
        # and compares again. Output comparison checks the first element and the second element
        # (which may be a tuple, compared element-wise). Uses weights_only False when loading.
        # Raises on mismatch.
        # Check state dict serialization and torch.save APIs
        model_dict = ref_model.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        # weights_only=False as we sometimes get a ScriptObject here (weird)
        loaded_dict = torch.load(b, weights_only=False)
        loaded_model.load_state_dict(loaded_dict)
        ref_out = ref_model(*x)
        load_out = loaded_model(*x)

        def check_outputs(ref_out, load_out):
            # [FUNCTION DOC] HELPER: Asserts the first elements of two outputs are equal and the
            # second elements are equal, element-wise if they are tuples.
            self.assertEqual(ref_out[0], load_out[0])
            if isinstance(ref_out[1], tuple):
                self.assertEqual(ref_out[1][0], load_out[1][0])
                self.assertEqual(ref_out[1][1], load_out[1][1])
            else:
                self.assertEqual(ref_out[1], load_out[1])

        check_outputs(ref_out, load_out)
        b = io.BytesIO()
        torch.save(ref_model, b)
        b.seek(0)
        # weights_only=False as this is legacy code that saves the model
        loaded = torch.load(b, weights_only=False)
        load_out = loaded(*x)
        check_outputs(ref_out, load_out)

    def check_weight_bias_api(self, ref_model, weight_keys, bias_keys):
        # [FUNCTION DOC] Asserts the key sets of the model get_weight and get_bias results equal
        # weight_keys and bias_keys.
        weight = ref_model.get_weight()
        bias = ref_model.get_bias()
        self.assertEqual(weight_keys ^ weight.keys(), set())
        self.assertEqual(bias_keys ^ bias.keys(), set())

    def checkDynamicQuantizedLSTM(self, mod, reference_module_type, dtype):
        # [FUNCTION DOC] Asserts mod has exactly reference_module_type and that every packed weight
        # of the module records the serialization tag matching dtype (qint8 means dynamic, float16
        # means fp16 quantization).
        r"""Checks that mod has been swapped for an nnqd.LSTM type
        module, the bias is float.
        """
        wt_dtype_map = {
            torch.qint8: "quantized_dynamic",
            torch.float16: "quantized_fp16",
        }
        self.assertEqual(type(mod), reference_module_type)
        for packed_params in mod._all_weight_values:
            self.assertEqual(
                packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype]
            )

    def checkLinear(self, mod):
        # [FUNCTION DOC] Asserts mod is exactly torch.nn.Linear (not converted).
        self.assertEqual(type(mod), torch.nn.Linear)

    def checkDynamicQuantizedModule(self, mod, reference_module_type, dtype):
        # [FUNCTION DOC] Asserts mod has exactly reference_module_type and, if it has packed
        # weight values, that each matches the serialization tag for dtype (qint8 dynamic or
        # float16 fp16).
        # KNOWN ISSUE: the docstring talks about nnqd.Linear and a float bias, but the check is
        # generic and does not inspect the bias.
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        wt_dtype_map = {
            torch.qint8: "quantized_dynamic",
            torch.float16: "quantized_fp16",
        }
        self.assertEqual(type(mod), reference_module_type)
        if hasattr(mod, "_all_weight_values"):
            for packed_params in mod._all_weight_values:
                self.assertEqual(
                    packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype]
                )

    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        # [FUNCTION DOC] Checks that orig_mod works under TorchScript: scripts it, then traces it
        # with the first calibration entry, and for each compares outputs against the original and
        # tests save/load via the helper. check_save_load controls whether the reloaded module's
        # outputs are compared.
        scripted = torch.jit.script(orig_mod)
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)

        # Use first calib_data entry as trace input
        traced = torch.jit.trace(orig_mod, calib_data[0])
        self._checkScriptable(orig_mod, traced, calib_data, check_save_load)

    # Call this twice: once for a scripted module and once for a traced module
    def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
        # [FUNCTION DOC] HELPER: Compares script_mod outputs with orig_mod over calib_data, saves
        # script_mod to an in-memory buffer and loads it back; compares the loaded module too only
        # when check_save_load is true (pending get/set state support).
        self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)

        # Test save/load
        buffer = io.BytesIO()
        torch.jit.save(script_mod, buffer)

        buffer.seek(0)
        loaded_mod = torch.jit.load(buffer)
        # Pending __get_state_ and __set_state__ support
        # See tracking task https://github.com/pytorch/pytorch/issues/23984
        if check_save_load:
            self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)

    def _checkModuleCorrectnessAgainstOrig(self, orig_mod, test_mod, calib_data):
        # [FUNCTION DOC] HELPER: For each entry of calib_data (positional args), asserts test_mod
        # output equals orig_mod output.
        for inp in calib_data:
            ref_output = orig_mod(*inp)
            scripted_output = test_mod(*inp)
            self.assertEqual(scripted_output, ref_output)

    def checkGraphModeOp(
        self,
        module,
        inputs,
        quantized_op,
        tracing=False,
        debug=False,
        check=True,
        eval_mode=True,
        dynamic=False,
        qconfig=None,
    ):
        # [FUNCTION DOC] JIT graph-mode quantization check. Scripts or traces module (eval mode
        # optionally), quantizes it with quantize_dynamic_jit (when dynamic) or quantize_jit with
        # the default engine qconfig and calibration via the eval function, each in debug and
        # non-debug mode, and runs the results. If check is true, asserts debug and non-debug
        # outputs are equal and that quantized_op appears in the non-debug graph (FileCheck).
        # Returns the non-debug quantized model.
        # KNOWN ISSUE: the loop variable debug shadows the debug parameter, so after the loop it is
        # False and the final debug graph printing never runs.
        if debug:
            print("Testing:", str(module))
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}

        if eval_mode:
            module = module.eval()
        if dynamic:
            qconfig_dict = {"": default_dynamic_qconfig if qconfig is None else qconfig}
        model = get_script_module(module, tracing, inputs[0]).eval()
        if debug:
            print("input graph:", model.graph)
        models = {}
        outputs = {}
        for debug in [True, False]:
            if dynamic:
                models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
                # make sure it runs
                outputs[debug] = models[debug](inputs)
            else:
                # module under test can contain in-place ops, and we depend on
                # input data staying constant for comparisons
                inputs_copy = copy.deepcopy(inputs)
                models[debug] = quantize_jit(
                    model,
                    qconfig_dict,
                    test_only_eval_fn,
                    [inputs_copy],
                    inplace=False,
                    debug=debug,
                )
                # make sure it runs
                outputs[debug] = models[debug](*inputs[0])

        if debug:
            print("debug graph:", models[True].graph)
            print("non debug graph:", models[False].graph)

        if check:
            # debug and non-debug option should have the same numerics
            self.assertEqual(outputs[True], outputs[False])

            # non debug graph should produce quantized op
            FileCheck().check(quantized_op).run(models[False].graph)

        return models[False]

    def checkGraphModuleNodes(
        self,
        graph_module,
        expected_node=None,
        expected_node_occurrence=None,
        expected_node_list=None,
    ):
        # [FUNCTION DOC] Asserts properties of an FX GraphModule's nodes. Builds an occurrence
        # count and ordered list of NodeSpec for call_function, call_method and call_module nodes
        # (module nodes use the module's type). Checks that expected_node is present, that each
        # entry of expected_node_occurrence has exactly the given count (zero means absent), and
        # that expected_node_list appears as an ordered subsequence. Raises AssertionError with a
        # message on failure.
        """Check if GraphModule contains the target node
        Args:
            graph_module: the GraphModule instance we want to check
            expected_node, expected_node_occurrence, expected_node_list:
               see docs for checkGraphModeFxOp
        """
        nodes_in_graph = {}
        node_list = []
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        for node in graph_module.graph.nodes:
            n = None
            if node.op == "call_function" or node.op == "call_method":
                n = NodeSpec(node.op, node.target)
            elif node.op == "call_module":
                n = NodeSpec(node.op, type(modules[node.target]))

            if n is not None:
                node_list.append(n)
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1

        if expected_node is not None:
            self.assertTrue(
                expected_node in nodes_in_graph,
                "node:" + str(expected_node) + " not found in the graph module",
            )

        if expected_node_occurrence is not None:
            for expected_node, occurrence in expected_node_occurrence.items():
                if occurrence != 0:
                    self.assertTrue(
                        expected_node in nodes_in_graph,
                        "Check failed for node:" + str(expected_node) + " not found",
                    )
                    self.assertTrue(
                        nodes_in_graph[expected_node] == occurrence,
                        "Check failed for node:"
                        + str(expected_node)
                        + " Expected occurrence:"
                        + str(occurrence)
                        + " Found occurrence:"
                        + str(nodes_in_graph[expected_node]),
                    )
                else:
                    self.assertTrue(
                        expected_node not in nodes_in_graph,
                        "Check failed for node:"
                        + str(expected_node)
                        + " expected no occurrence but found",
                    )

        if expected_node_list is not None:
            cur_index = 0
            for n in node_list:
                if cur_index == len(expected_node_list):
                    return
                if n == expected_node_list[cur_index]:
                    cur_index += 1
            self.assertTrue(
                cur_index == len(expected_node_list),
                "Check failed for graph:"
                + self.printGraphModule(graph_module, print_str=False)
                + "Expected ordered list:"
                + str(expected_node_list),
            )

    def printGraphModule(self, graph_module, print_str=True):
        # [FUNCTION DOC] Builds a string with one line per graph node (op, name, target, args,
        # kwargs, plus module type for module calls), prints it when print_str is true, and
        # returns it.
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        node_infos = []
        for n in graph_module.graph.nodes:
            node_info = " ".join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
            if n.op == "call_module":
                node_info += " module type: " + repr(type(modules[n.target]))
            node_infos.append(node_info)
        str_to_print = "\n".join(node_infos)
        if print_str:
            print(str_to_print)
        return str_to_print

    if HAS_FX:

        def assert_types_for_matched_subgraph_pairs(
            self,
            matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]],
            expected_types: dict[
                str, tuple[tuple[Callable, Callable], tuple[Callable, Callable]]
            ],
            gm_a: GraphModule,
            gm_b: GraphModule,
        ) -> None:
            # [FUNCTION DOC] Asserts the matched numeric-suite subgraph pairs have the expected
            # types. Requires the same key set (by length and lookup) as expected_types; for each
            # key compares the start and end node underlying types of both graphs (module type for
            # call_module, else the target) by identity against the expected ones. Raises
            # AssertionError on mismatch.
            """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_subgraph_pairs.

            An example successful test case:

              matched_subgraph_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

            def _get_underlying_op_type(
                node: Node, gm: GraphModule
            ) -> Callable | str:
                # [FUNCTION DOC] HELPER: Returns the module type for call_module nodes of gm,
                # otherwise the node target; raises AssertionError for ops other than call_function
                # or call_method.
                if node.op == "call_module":
                    mod = getattr(gm, node.target)
                    return type(mod)
                else:
                    if node.op not in ("call_function", "call_method"):
                        raise AssertionError(f"Expected node.op in ('call_function', 'call_method'), got {node.op!r}")
                    return node.target

            self.assertTrue(
                len(matched_subgraph_pairs) == len(expected_types),
                f"Expected length of results to match, but got {len(matched_subgraph_pairs)} and {len(expected_types)}",
            )
            for k, v in expected_types.items():
                expected_types_a, expected_types_b = v
                exp_type_start_a, exp_type_end_a = expected_types_a
                exp_type_start_b, exp_type_end_b = expected_types_b
                subgraph_a, subgraph_b = matched_subgraph_pairs[k]

                act_type_start_a = _get_underlying_op_type(subgraph_a.start_node, gm_a)
                act_type_start_b = _get_underlying_op_type(subgraph_b.start_node, gm_b)
                act_type_end_a = _get_underlying_op_type(subgraph_a.end_node, gm_a)
                act_type_end_b = _get_underlying_op_type(subgraph_b.end_node, gm_b)
                types_match = (
                    (exp_type_start_a is act_type_start_a)
                    and (exp_type_end_a is act_type_end_a)
                    and (exp_type_start_b is act_type_start_b)
                    and (exp_type_end_b is act_type_end_b)
                )
                self.assertTrue(
                    types_match,
                    f"Type mismatch at {k}: expected {(exp_type_start_a, exp_type_end_a, exp_type_start_b, exp_type_end_b)}, "
                    f"got {(act_type_start_a, act_type_end_a, act_type_start_b, act_type_end_b)}",
                )

        def assert_ns_compare_dict_valid(
            self,
            act_compare_dict: dict[str, dict[str, dict[str, Any]]],
        ) -> None:
            # [FUNCTION DOC] Validates the numeric-suite comparison result mapping. For every
            # layer and result type, requires results for exactly two models with equal
            # numbers of recorded values; compares tensor shapes (nested lists and tuples
            # handled; the weight check is skipped for functional conv1d), and checks
            # ref_node_name versus prev_node_name consistency for node output (equal) and node
            # input (different) records.
            # KNOWN ISSUE: the type equality check compares layer_data_0 type to itself, so it can
            # never fail.
            """
            Verifies that the act_compare_dict (output of Numeric Suite APIs) is valid:
            1. for each layer, results are recorded for two models
            2. number of seen tensors match
            3. shapes of each pair of seen tensors match
            """
            for layer_name, result_type_to_data in act_compare_dict.items():
                for result_type, layer_data in result_type_to_data.items():
                    self.assertTrue(
                        len(layer_data) == 2,
                        f"Layer {layer_name} does not have exactly two model results.",
                    )
                    model_name_0, model_name_1 = layer_data.keys()
                    for res_idx in range(len(layer_data[model_name_0])):
                        layer_data_0 = layer_data[model_name_0][res_idx]
                        layer_data_1 = layer_data[model_name_1][res_idx]
                        self.assertTrue(
                            layer_data_0["type"] == layer_data_0["type"],
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same type.",
                        )

                        self.assertTrue(
                            len(layer_data_0["values"]) == len(layer_data_1["values"]),
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same number of seen Tensors.",
                        )

                        # F.conv1d weight has rank 3, and toq.conv1d unpacked weight
                        # has rank 4. For now, skip the length check for conv1d only.
                        is_weight_functional_conv1d = (
                            result_type == NSSingleResultValuesType.WEIGHT.value
                            and (
                                "conv1d" in layer_data_0["prev_node_target_type"]
                                or "conv1d" in layer_data_1["prev_node_target_type"]
                            )
                        )
                        if not is_weight_functional_conv1d:
                            for idx in range(len(layer_data_0["values"])):
                                values_0 = layer_data_0["values"][idx]
                                values_1 = layer_data_1["values"][idx]
                                if isinstance(values_0, torch.Tensor):
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} "
                                        + f"have a shape mismatch at idx {idx}.",
                                    )
                                elif isinstance(values_0, list):
                                    values_0 = values_0[0]
                                    values_1 = values_1[0]
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} "
                                        + f"have a shape mismatch at idx {idx}.",
                                    )
                                else:
                                    if not isinstance(values_0, tuple):
                                        raise AssertionError(f"unhandled type {type(values_0)}")
                                    if len(values_0) != 2:
                                        raise AssertionError(f"Expected len(values_0) == 2, got {len(values_0)}")
                                    if len(values_0[1]) != 2:
                                        raise AssertionError(f"Expected len(values_0[1]) == 2, got {len(values_0[1])}")
                                    if values_0[0].shape != values_1[0].shape:
                                        raise AssertionError(
                                            f"Expected values_0[0].shape == values_1[0].shape, "
                                            f"got {values_0[0].shape} != {values_1[0].shape}"
                                        )
                                    if values_0[1][0].shape != values_1[1][0].shape:
                                        raise AssertionError(
                                            f"Expected values_0[1][0].shape == values_1[1][0].shape, "
                                            f"got {values_0[1][0].shape} != {values_1[1][0].shape}"
                                        )
                                    if values_0[1][1].shape != values_1[1][1].shape:
                                        raise AssertionError(
                                            f"Expected values_0[1][1].shape == values_1[1][1].shape, "
                                            f"got {values_0[1][1].shape} != {values_1[1][1].shape}"
                                        )

                        # verify that ref_node_name is valid
                        ref_node_name_0 = layer_data_0["ref_node_name"]
                        ref_node_name_1 = layer_data_1["ref_node_name"]
                        prev_node_name_0 = layer_data_0["prev_node_name"]
                        prev_node_name_1 = layer_data_1["prev_node_name"]
                        if (
                            layer_data_0["type"]
                            == NSSingleResultValuesType.NODE_OUTPUT.value
                        ):
                            self.assertTrue(ref_node_name_0 == prev_node_name_0)
                            self.assertTrue(ref_node_name_1 == prev_node_name_1)
                        elif (
                            layer_data_0["type"]
                            == NSSingleResultValuesType.NODE_INPUT.value
                        ):
                            self.assertTrue(ref_node_name_0 != prev_node_name_0)
                            self.assertTrue(ref_node_name_1 != prev_node_name_1)

        def checkGraphModeFxOp(
            self,
            model,
            inputs,
            quant_type,
            expected_node=None,
            expected_node_occurrence=None,
            expected_node_list=None,
            is_reference=False,
            print_debug_info=False,
            custom_qconfig_dict=None,
            prepare_expected_node=None,
            prepare_expected_node_occurrence=None,
            prepare_expected_node_list=None,
            prepare_custom_config=None,
            backend_config=None,
        ):
            # [FUNCTION DOC] Quantizes a float model with FX graph mode and checks node
            # expectations. Chooses the qconfig mapping by quant_type (QAT: train mode and
            # prepare_qat_fx; static: eval; dynamic: eval with the default dynamic qconfig),
            # optionally overridden by custom_qconfig_dict (QConfigMapping or dict). Prepares,
            # calibrates by running inputs (unless dynamic), checks the prepared graph against
            # the prepare_* expectations, converts to both the normal and the reference
            # quantized model, runs both, and checks the chosen one (reference when
            # is_reference) against the expected_* arguments. Returns a mapping of deep copies
            # and outputs under the keys prepared, quantized, quantized_reference,
            # quantized_output and quantized_reference_output.
            # KNOWN ISSUE: the docstring lists a result key and omits prepare_custom_config and
            # backend_config; the returned keys differ from those described.
            """Quantizes model with graph mode quantization on fx and check if the
            quantized model contains the quantized_node

            Args:
                model: floating point torch.nn.Module
                inputs: one positional sample input arguments for model
                expected_node: NodeSpec
                    e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                expected_node_occurrence: a dict from NodeSpec to
                    expected number of occurrences (int)
                    e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                            NodeSpec.call_method('dequantize'): 1}
                expected_node_list: a list of NodeSpec, used to check the order
                    of the occurrence of Node
                    e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                            NodeSpec.call_module(nnq.Conv2d),
                            NodeSpec.call_function(F.hardtanh_),
                            NodeSpec.call_method('dequantize')]
                is_reference: if True, enables reference mode
                print_debug_info: if True, prints debug info
                custom_qconfig_dict: overrides default qconfig_dict
                prepare_expected_node: same as expected_node, but for prepare
                prepare_expected_node_occurrence: same as
                    expected_node_occurrence, but for prepare
                prepare_expected_node_list: same as expected_node_list, but
                    for prepare

            Returns:
                A dictionary with the following structure:
               {
                   "prepared": ...,  # the prepared model
                   "quantized": ...,  # the quantized non-reference model
                   "quantized_reference": ...,  # the quantized reference model
                   "result": ...,  # the result for either quantized or
                                   # quantized_reference model depending on the
                                   # is_reference argument
               }
            """
            # TODO: make img_data a single example instead of a list
            if type(inputs) is list:
                inputs = inputs[0]

            if quant_type == QuantType.QAT:
                qconfig_mapping = get_default_qat_qconfig_mapping(
                    torch.backends.quantized.engine
                )
                model.train()
            elif quant_type == QuantType.STATIC:
                qconfig_mapping = get_default_qconfig_mapping(
                    torch.backends.quantized.engine
                )
                model.eval()
            else:
                qconfig = default_dynamic_qconfig
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                model.eval()

            if quant_type == QuantType.QAT:
                prepare = prepare_qat_fx
            else:
                prepare = prepare_fx

            # overwrite qconfig_dict with custom_qconfig_dict
            if custom_qconfig_dict is not None:
                if type(custom_qconfig_dict) not in (QConfigMapping, dict):
                    raise AssertionError("custom_qconfig_dict should be a QConfigMapping or a dict")
                if isinstance(custom_qconfig_dict, QConfigMapping):
                    qconfig_mapping = custom_qconfig_dict
                else:
                    qconfig_mapping = QConfigMapping.from_dict(custom_qconfig_dict)
            prepared = prepare(
                model,
                qconfig_mapping,
                example_inputs=inputs,
                prepare_custom_config=prepare_custom_config,
                backend_config=backend_config,
            )
            if quant_type != QuantType.DYNAMIC:
                prepared(*inputs)

            if print_debug_info:
                print()
                print("quant type:\n", quant_type)
                print("original model:\n", model)
                print()
                print("prepared model:\n", prepared)

            self.checkGraphModuleNodes(
                prepared,
                prepare_expected_node,
                prepare_expected_node_occurrence,
                prepare_expected_node_list,
            )

            prepared_copy = copy.deepcopy(prepared)
            qgraph = convert_fx(copy.deepcopy(prepared))
            qgraph_reference = convert_to_reference_fx(copy.deepcopy(prepared))
            result = qgraph(*inputs)
            result_reference = qgraph_reference(*inputs)
            qgraph_copy = copy.deepcopy(qgraph)
            qgraph_reference_copy = copy.deepcopy(qgraph_reference)

            qgraph_to_check = qgraph_reference if is_reference else qgraph
            if print_debug_info:
                print()
                print("quantized model:\n", qgraph_to_check)
                self.printGraphModule(qgraph_to_check)
                print()
            self.checkGraphModuleNodes(
                qgraph_to_check,
                expected_node,
                expected_node_occurrence,
                expected_node_list,
            )
            return {
                "prepared": prepared_copy,
                "quantized": qgraph_copy,
                "quantized_reference": qgraph_reference_copy,
                "quantized_output": result,
                "quantized_reference_output": result_reference,
            }

    def checkEmbeddingSerialization(
        self,
        qemb,
        num_embeddings,
        embedding_dim,
        indices,
        offsets,
        set_qconfig,
        is_emb_bag,
        dtype=torch.quint8,
    ):
        # [FUNCTION DOC] Tests serialization of a quantized Embedding or EmbeddingBag module
        # (is_emb_bag selects). Saves and loads its state dict and compares unpacked weights, runs
        # the eager serialization check into a freshly built module of the same shape, reloads the
        # state dict and compares packed weights, checks TorchScript save/load, then builds a float
        # embedding (optionally with a float-qparams weight observer when set_qconfig), prepares it
        # dynamically, runs it, applies from_float and asserts the quantized module's class name
        # appears in its string form. Raises on any mismatch.
        # Test serialization of dynamic EmbeddingBag module using state_dict
        if is_emb_bag:
            inputs = [indices, offsets]
        else:
            inputs = [indices]
        emb_dict = qemb.state_dict()
        b = io.BytesIO()
        torch.save(emb_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        embedding_unpack = torch.ops.quantized.embedding_bag_unpack
        # Check unpacked weight values explicitly
        for key in emb_dict:
            if isinstance(emb_dict[key], torch._C.ScriptObject):
                if not isinstance(loaded_dict[key], torch._C.ScriptObject):
                    raise AssertionError(f"Expected loaded_dict[{key!r}] to be ScriptObject")
                emb_weight = embedding_unpack(emb_dict[key])
                loaded_weight = embedding_unpack(loaded_dict[key])
                self.assertEqual(emb_weight, loaded_weight)

        # Check state dict serialization and torch.save APIs
        if is_emb_bag:
            loaded_qemb = nnq.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                include_last_offset=True,
                mode="sum",
                dtype=dtype,
            )
        else:
            loaded_qemb = nnq.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=dtype
            )
        self.check_eager_serialization(qemb, loaded_qemb, inputs)

        loaded_qemb.load_state_dict(loaded_dict)
        self.assertEqual(
            embedding_unpack(qemb._packed_params._packed_weight),
            embedding_unpack(loaded_qemb._packed_params._packed_weight),
        )

        # Test JIT serialization
        self.checkScriptable(qemb, [inputs], check_save_load=True)

        # Test from_float call
        if is_emb_bag:
            float_embedding = torch.nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                include_last_offset=True,
                scale_grad_by_freq=False,
                mode="sum",
            )
        else:
            float_embedding = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )

        if set_qconfig:
            float_qparams_observer = PerChannelMinMaxObserver.with_args(
                dtype=dtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
            )
            float_embedding.qconfig = QConfig(
                activation=default_dynamic_quant_observer, weight=float_qparams_observer
            )

        prepare_dynamic(float_embedding)

        float_embedding(*inputs)
        if is_emb_bag:
            q_embeddingbag = nnq.EmbeddingBag.from_float(float_embedding)
            expected_name = "QuantizedEmbeddingBag"
        else:
            q_embeddingbag = nnq.Embedding.from_float(float_embedding)
            expected_name = "QuantizedEmbedding"

        q_embeddingbag(*inputs)

        self.assertTrue(expected_name in str(q_embeddingbag))


class QuantizationLiteTestCase(QuantizationTestCase):
    # [CLASS DOC] QuantizationTestCase extension for mobile (lite interpreter) tests.
    # Provides helpers to build a qnnpack-quantized model and to compare scripted and lite
    # interpreter outputs.
    def _create_quantized_model(self, model_class: type[torch.nn.Module], **kwargs):
        # [FUNCTION DOC] HELPER: Under the qnnpack engine, instantiates model_class with kwargs and
        # quantizes it with the eval function on the calibration data. Returns the quantized model.
        # KNOWN ISSUE: a qconfig is fetched but unused (FIXME in code).
        # Creates quantized model for testing mobile script modules
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            # FIXME(rec): shouldn't qconfig be passed to quantize?
            qconfig = torch.ao.quantization.get_default_qconfig(qengine)  # noqa: F841
            model = model_class(**kwargs)
            model = quantize(model, test_only_eval_fn, [self.calib_data])

        return model

    def _compare_script_and_mobile(self, model: torch.nn.Module, input: torch.Tensor):
        # [FUNCTION DOC] HELPER: Under qnnpack, scripts model, saves it for the lite interpreter
        # into memory, loads it, and checks that calling it, calling forward and run_method all
        # match the script output. The whole attempt is retried up to 5 times; the last failure is
        # re-raised. Raises AssertionError if still mismatching.
        # Compares the numerical outputs for script and lite modules
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            script_module = torch.jit.script(model)
            script_module_result = script_module(input)

            max_retry = 5
            for retry in range(1, max_retry + 1):
                # retries `max_retry` times; breaks iff succeeds else throws exception
                try:
                    buffer = io.BytesIO(
                        script_module._save_to_buffer_for_lite_interpreter()
                    )
                    buffer.seek(0)
                    mobile_module = _load_for_lite_interpreter(buffer)

                    mobile_module_result = mobile_module(input)

                    torch.testing.assert_close(
                        script_module_result, mobile_module_result
                    )
                    mobile_module_forward_result = mobile_module.forward(input)
                    torch.testing.assert_close(
                        script_module_result, mobile_module_forward_result
                    )

                    mobile_module_run_method_result = mobile_module.run_method(
                        "forward", input
                    )
                    torch.testing.assert_close(
                        script_module_result, mobile_module_run_method_result
                    )
                except AssertionError as e:
                    if retry == max_retry:
                        raise e
                    else:
                        continue
                break


# Below are a series of toy models to use in testing quantization


class SingleLayerLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: a single float Linear (5 to 5) layer.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the Linear layer.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the layer.
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear wrapped in a QuantWrapper, with the default qconfig for
    # the engine.
    def __init__(self, qengine="fbgemm"):
        # [FUNCTION DOC] HELPER: Sets qconfig for the given engine and builds the wrapped Linear.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the wrapped layer.
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class SingleLayerLinearDynamicModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: single Linear with a default qconfig set, for
    # dynamic quantization.
    def __init__(self, qengine="fbgemm"):
        # [FUNCTION DOC] HELPER: Sets qconfig for the given engine and builds the Linear.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the layer.
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearAddModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 5 to 8, add of constant 5, Linear 8 to 5.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two Linear layers.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs first Linear, adds the scalar 5, then the second Linear.
        x = self.fc1(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class RNNDynamicModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model wrapping a GRU or LSTM (input and hidden size 2) with the
    # default dynamic qconfig.
    def __init__(self, mod_type):
        # [FUNCTION DOC] HELPER: Sets the dynamic qconfig and builds a GRU or LSTM according to
        # mod_type; any other value leaves no mod attribute.
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        if mod_type == "GRU":
            self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
        if mod_type == "LSTM":
            self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the recurrent module and returns its output tuple.
        x = self.mod(x)
        return x


class RNNCellDynamicModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model wrapping an RNN cell variant with the default dynamic qconfig.
    def __init__(self, mod_type):
        # [FUNCTION DOC] HELPER: Sets the dynamic qconfig and builds a GRUCell, LSTMCell or RNNCell
        # (relu or tanh) chosen by mod_type; unknown values leave no mod attribute.
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        if mod_type == "GRUCell":
            self.mod = torch.nn.GRUCell(2, 2).to(dtype=torch.float)
        if mod_type == "LSTMCell":
            self.mod = torch.nn.LSTMCell(2, 2).to(dtype=torch.float)
        if mod_type == "RNNReLU":
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity="relu").to(dtype=torch.float)
        if mod_type == "RNNTanh":
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity="tanh").to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the cell to the input.
        x = self.mod(x)
        return x


class LSTMwithHiddenDynamicModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: LSTM taking an explicit hidden state, with the
    # default qconfig.
    def __init__(self, qengine="fbgemm"):
        # [FUNCTION DOC] HELPER: Sets qconfig for the engine and builds an LSTM with sizes 2 and 2.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x, hid):
        # [FUNCTION DOC] HELPER: Runs the LSTM with the given hidden state; returns the output and
        # new hidden state.
        x, hid = self.lstm(x, hid)
        return x, hid


class ConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: one bias-free Conv2d (3 to 5 channels, kernel 3).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the convolution.
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the convolution.
        x = self.conv(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class ConvTransposeModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: one bias-free ConvTranspose2d (3 to 5 channels, kernel 3).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the transposed convolution.
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies it.
        x = self.conv(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class AnnotatedConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: QuantStub, Conv2d, DeQuantStub with the engine
    # default qconfig.
    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Sets qconfig and builds the stubs and convolution.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize stub, convolution, dequantize stub.
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class AnnotatedConvTransposeModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: QuantStub, ConvTranspose2d, DeQuantStub with the engine
    # default qconfig.
    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Sets qconfig and builds the stubs and transposed convolution.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize stub, transposed convolution, dequantize stub.
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class ConvBnModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d followed by BatchNorm2d.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates convolution and batch norm.
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Convolution then batch norm.
        x = self.conv(x)
        x = self.bn(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class AnnotatedConvBnModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: QuantStub, Conv2d, BatchNorm2d, DeQuantStub
    # with default_qconfig.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets qconfig and builds the stubs, convolution and batch norm.
        super().__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize stub, convolution, batch norm, dequantize stub.
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class ConvBnReLUModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d, BatchNorm2d and in-place ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the three layers.
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Convolution, batch norm, ReLU.
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class AnnotatedConvBnReLUModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: stubs around Conv2d, BatchNorm2d, ReLU; supports fusing the
    # three layers.
    def __init__(self, qengine="fbgemm"):
        # [FUNCTION DOC] HELPER: Sets qconfig for the engine and builds stubs and the three layers.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize stub, conv, bn, relu, dequantize stub.
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # [FUNCTION DOC] HELPER: Fuses conv, bn and relu in place, using the QAT fusion when the
        # model is in training mode and the standard fusion otherwise.
        # TODO: remove this check and define two fuse_modules function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(
                self, [["conv", "bn", "relu"]], inplace=True
            )
        else:
            torch.ao.quantization.fuse_modules(
                self, [["conv", "bn", "relu"]], inplace=True
            )

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class TwoLayerConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d (3 to 5, kernel 3) followed by Conv2d (5 to 5, kernel
    # 1), both bias-free.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two convolutions.
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both in order.
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class TwoLayerLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 5 to 8 followed by Linear 8 to 5.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two layers.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both in order.
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearModelWithSubmodule(nn.Module):
    # [CLASS DOC] HELPER: Toy model: a TwoLayerLinearModel submodule followed by a Linear 5 to 5.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the submodule and the extra Linear.
        super().__init__()
        self.subm = TwoLayerLinearModel()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the submodule then the Linear.
        x = self.subm(x)
        x = self.fc(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the submodule example inputs.
        return self.subm.get_example_inputs()


class AnnotatedTwoLayerLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: float Linear then a QuantWrapper Linear that has an
    # fbgemm qconfig.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the first Linear, the wrapped second Linear and sets
        # its qconfig.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = QuantWrapper(torch.nn.Linear(8, 5).to(dtype=torch.float))
        self.fc2.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both layers.
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class ActivationsTestModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: QuantStub, Hardswish, ELU, DeQuantStub with an fbgemm qconfig.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the qconfig and builds the stubs and activations.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        self.quant = torch.ao.quantization.QuantStub()
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)
        self.elu = torch.nn.ELU().to(dtype=torch.float)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize stub, Hardswish, ELU, dequantize stub.
        x = self.quant(x)
        x = self.hardswish(x)
        x = self.elu(x)
        x = self.dequant(x)
        return x


class LinearReluModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 5 to 5 followed by ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: ReLU of the Linear output.
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearReluLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 5 to 8, ReLU, Linear 8 to 5.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the three in order.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearReluAddModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear, ReLU, add of constant 5, Linear (5 to 5 each).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies Linear, ReLU, adds 5, applies second Linear. Side effect:
        # replaces self.relu with a new ReLU module on every call.
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        self.relu = torch.nn.ReLU()
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearBnLeakyReluModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear, optional BatchNorm1d, LeakyReLU (slope 0.01), all of
    # width 5.
    def __init__(self, with_bn=True):
        # [FUNCTION DOC] HELPER: Builds the layers and stores the with_bn flag.
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.bn1d = nn.BatchNorm1d(5)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.with_bn = with_bn

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Linear, batch norm only when with_bn is true, then LeakyReLU.
        x = self.linear(x)
        if self.with_bn:
            x = self.bn1d(x)
        x = self.leaky_relu(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class LinearTanhModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 5 to 5 then Tanh.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Linear then Tanh.
        x = self.linear(x)
        x = self.tanh(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class ConvBnAddReluModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d with optional BatchNorm, residual-style add (with a
    # second convolution or an external tensor, in either operand order, via torch.add or the plus
    # operator) and optional ReLU. Behavior is selected by constructor flags.
    def __init__(
        self,
        with_bn=True,
        with_relu=True,
        left_conv=True,
        two_conv=True,
        use_torch_add=True,
    ):
        # [FUNCTION DOC] HELPER: Builds two convolutions, batch norm, ReLU and stores the flags
        # with_bn, with_relu, left_conv, two_conv and use_torch_add.
        super().__init__()
        self.conv = nn.Conv2d(5, 5, (2, 2))
        self.conv2 = nn.Conv2d(5, 5, (2, 2))
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.two_conv = two_conv
        self.left_conv = left_conv
        self.use_torch_add = use_torch_add

    def forward(self, x1, x2):
        # [FUNCTION DOC] HELPER: Takes x1 and x2. Computes the conv branch (batch norm when
        # with_bn), adds either a second convolution of x1 (when two_conv) or x2 (conv on left or
        # right per left_conv), using torch.add or the plus operator per use_torch_add, then
        # applies ReLU when with_relu.
        if self.two_conv:
            if self.use_torch_add:
                if self.with_bn:
                    x = torch.add(self.bn(self.conv(x1)), self.conv2(x1))
                else:
                    x = torch.add(self.conv(x1), self.conv2(x1))
            else:
                if self.with_bn:
                    x = self.bn(self.conv(x1)) + self.conv2(x1)
                else:
                    x = self.conv(x1) + self.conv2(x1)
        else:
            if self.use_torch_add:
                if self.left_conv:
                    if self.with_bn:
                        x = torch.add(self.bn(self.conv(x1)), x2)
                    else:
                        x = torch.add(self.conv(x1), x2)
                else:
                    if self.with_bn:
                        x = torch.add(x2, self.bn(self.conv(x1)))
                    else:
                        x = torch.add(x2, self.conv(x1))
            else:
                if self.left_conv:
                    if self.with_bn:
                        x = self.bn(self.conv(x1)) + x2
                    else:
                        x = self.conv(x1) + x2
                else:
                    if self.with_bn:
                        x = x2 + self.bn(self.conv(x1))
                    else:
                        x = x2 + self.conv(x1)
        if self.with_relu:
            x = self.relu(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a tuple of two random tensors of shapes 1 by 5 by 3 by 3
        # and 1 by 5 by 2 by 2.
        return (torch.rand(1, 5, 3, 3), torch.rand(1, 5, 2, 2))


# TODO: self.fc should be self.conv
class ConvReluModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d (3 to 5, kernel 3) then ReLU. Attribute is named fc
    # rather than conv (TODO in code).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: ReLU of the convolution output.
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


# TODO: self.fc should be self.conv
class ConvReluConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d, ReLU, Conv2d (kernel 1). Attributes are named fc1
    # and fc2.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the three in order.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


# TODO: self.fc should be self.conv
class ConvReluAddModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Conv2d, ReLU, add of constant 5, Conv2d (kernel 1). Attributes
    # are named fc1 and fc2.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies conv, ReLU, adds 5, second conv. Side effect: replaces
        # self.relu with a new ReLU module on every call.
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        self.relu = torch.nn.ReLU()
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class NormalizationTestModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model for normalization ops: QuantStub, Linear, LayerNorm, GroupNorm
    # and 1d/2d/3d InstanceNorm.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the stub, Linear 5 to 8 and the norm layers over
        # 8 channels.
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm(8)
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantizes, applies Linear and LayerNorm, expands the tensor with
        # an extra last dimension (repeated three times for GroupNorm), then applies the instance
        # norms, adding trailing dimensions as needed.
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1).repeat(1, 1, 3))
        x = self.instance_norm1d(x)
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
        return x


class NestedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model with nested float submodules: LinearReluModel,
    # TwoLayerLinearModel and a Linear.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the submodules.
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sub1, sub2 and fc3 in order.
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x


class AnnotatedNestedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Nested model with explicit qconfigs: fc3 and sub2.fc1 are wrapped in
    # QuantWrapper with per-submodule qconfigs.
    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Builds the submodules, wraps fc3 and sub2.fc1, and gives
        # sub2.fc1 the per-channel qconfig for fbgemm or the default qconfig otherwise, fc3 the
        # default qconfig.
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        if qengine == "fbgemm":
            self.sub2.fc1.qconfig = default_per_channel_qconfig
        else:
            self.sub2.fc1.qconfig = default_qconfig

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sub1, sub2, fc3 in order.
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x


class AnnotatedSubNestedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Nested model where sub2 (whole TwoLayerLinearModel) and fc3 are wrapped
    # in QuantWrapper with default_qconfig.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds submodules, wraps them and sets their qconfigs.
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.qconfig = default_qconfig

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sub1, sub2, fc3 in order.
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x


class AnnotatedCustomConfigNestedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Nested model with a custom qconfig on sub2.fc1 and QuantWrapper around
    # fc3, sub2.fc1 and sub2.fc2.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the submodules and sets default qconfigs; assigns a custom
        # QConfig (quint8 per-tensor activation observer, default weight observer) on sub2.fc1
        # before wrapping fc1 and fc2 in QuantWrapper.
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.qconfig = default_qconfig

        custom_options = {"dtype": torch.quint8, "qscheme": torch.per_tensor_affine}
        custom_qconfig = QConfig(
            activation=default_observer.with_args(**custom_options),
            weight=default_weight_observer,
        )
        self.sub2.fc1.qconfig = custom_qconfig

        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        self.sub2.fc2 = QuantWrapper(self.sub2.fc2)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sub1, sub2, fc3 in order.
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x


class QuantSubModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: LinearReluModel, a QuantWrapper-wrapped TwoLayerLinearModel
    # and a Linear, with qconfigs on sub2 and fc3.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds submodules and sets default_qconfig on sub2 and fc3.
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        self.sub2.qconfig = default_qconfig
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.fc3.qconfig = default_qconfig

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sub1, sub2, fc3 in order.
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x


class InnerModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear-ReLU-Linear-ReLU block with a helper that fuses
    # Linear/ReLU pairs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates fc1, relu1, fc2, relu2.
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the four layers in order.
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))

    def fuse_modules(self):
        # [FUNCTION DOC] HELPER: Finds adjacent Linear then ReLU child pairs by name and fuses them
        # in place, using QAT fusion in training mode and standard fusion otherwise.
        fusable_layers = []
        named_children = list(self.named_children())
        for idx, (current_name, layer) in enumerate(named_children):
            if isinstance(layer, torch.nn.Linear):
                if idx >= len(named_children) - 1:
                    break
                if isinstance(named_children[idx + 1][1], torch.nn.ReLU):
                    fusable_layers.append([current_name, named_children[idx + 1][0]])
        # TODO: remove this check and define two fuse_modules function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, fusable_layers, inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, fusable_layers, inplace=True)


class FunctionalLinear(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: functional linear using plain (non-parameter) weight 5 by 5
    # random and zero bias tensors.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the weight and bias tensors.
        super().__init__()
        self.weight = torch.rand((5, 5))
        self.bias = torch.zeros(5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the functional linear.
        return F.linear(x, self.weight, self.bias)

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 5.
        return (torch.rand(1, 5),)


class SingleLayerFunctionalLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: one FunctionalLinear.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates it.
        super().__init__()
        self.linear1 = FunctionalLinear()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies it.
        x = self.linear1(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the inner example input.
        return self.linear1.get_example_inputs()


class TwoLayerFunctionalLinearModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: two FunctionalLinear in sequence.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates both.
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both in order.
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the first inner example input.
        return self.linear1.get_example_inputs()


class FunctionalLinearAddModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: FunctionalLinear, add of constant 5, FunctionalLinear.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates both.
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Linear, add 5, Linear.
        x = self.linear1(x)
        x = torch.add(x, 5)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the first inner example input.
        return self.linear1.get_example_inputs()


class FunctionalLinearReluModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model: FunctionalLinear followed by functional ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates it.
        super().__init__()
        self.linear = FunctionalLinear()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Linear then ReLU.
        x = self.linear(x)
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the inner example input.
        return self.linear.get_example_inputs()


class FunctionalLinearReluLinearModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model: FunctionalLinear, ReLU module, FunctionalLinear.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.relu = nn.ReLU()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the three in order.
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the first inner example input.
        return self.linear1.get_example_inputs()


class FunctionalConv2d(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: functional conv2d with plain random weight 3 by 3 by 3 by 3,
    # bias, stride 1, no padding, dilation 1, one group stored as attributes.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the weight, bias and convolution settings.
        super().__init__()
        self.weight = torch.rand(3, 3, 3, 3)
        self.bias = torch.rand(3)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls functional conv2d with the stored tensors and settings.
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random example input of
        # shape 1 by 3 by 5 by 5.
        return (torch.rand(1, 3, 5, 5),)


class SingleLayerFunctionalConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: one FunctionalConv2d.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates it.
        super().__init__()
        self.conv1 = FunctionalConv2d()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies it.
        x = self.conv1(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the inner example input.
        return self.conv1.get_example_inputs()


class TwoLayerFunctionalConvModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: two FunctionalConv2d in sequence.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates both.
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both.
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the first inner example input.
        return self.conv1.get_example_inputs()


class FunctionalConvReluModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model: FunctionalConv2d followed by functional ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates it.
        super().__init__()
        self.conv = FunctionalConv2d()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Conv then ReLU.
        x = self.conv(x)
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the inner example input.
        return self.conv.get_example_inputs()


class FunctionalConvReluConvModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model: FunctionalConv2d, ReLU module, FunctionalConv2d.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.relu = nn.ReLU()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the three in order.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Delegates to the first inner example input.
        return self.conv1.get_example_inputs()


class SkipQuantModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model showing that quantization of a submodule can be skipped: an
    # InnerModule then a float Linear.
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the submodule and Linear.
        super().__init__()
        self.sub = InnerModule()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the submodule then the Linear.
        return self.fc(self.sub(x))

    def fuse_modules(self):
        # [FUNCTION DOC] HELPER: Fuses the inner submodule.
        self.sub.fuse_modules()


class AnnotatedSkipQuantModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Annotated skip model: InnerModule wrapped in QuantWrapper under an engine
    # qconfig; the Linear fc has qconfig set to None so it is not quantized.
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """

    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Sets the engine qconfig, wraps the inner module and disables
        # quantization of fc.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.sub = QuantWrapper(InnerModule())
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # don't quantize this fc
        self.fc.qconfig = None

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the wrapped submodule then fc.
        return self.fc(self.sub(x))

    def fuse_modules(self):
        # [FUNCTION DOC] HELPER: Fuses the wrapped inner module.
        self.sub.module.fuse_modules()


class QuantStubModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model with manual QuantStub and DeQuantStub around a Linear,
    # qnnpack qconfig.
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`"""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the qconfig and builds stubs and Linear.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize, Linear, dequantize.
        x = self.quant(x)
        x = self.fc(x)
        return self.dequant(x)


class ManualLinearQATModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy QAT model: stubs around two Linear layers (5 to 1 to 10) with the
    # engine QAT qconfig.
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`"""

    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Sets the QAT qconfig and builds the layers.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize, two Linears, dequantize.
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)


class ManualDropoutQATModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy QAT model: stubs around a Linear and Dropout (0.5) with the engine
    # QAT qconfig.
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`"""

    def __init__(self, qengine):
        # [FUNCTION DOC] HELPER: Sets the QAT qconfig and builds the layers.
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize, Linear, Dropout, dequantize.
        x = self.quant(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.dequant(x)


class ManualLinearDynamicQATModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy dynamic QAT model: two Linears with a dynamic QAT qconfig (default
    # unless given).
    r"""A Module that uses a dynamic QAT by default."""

    def __init__(self, qconfig=None):
        # [FUNCTION DOC] HELPER: Sets qconfig from the argument or the default dynamic QAT qconfig
        # and builds the Linears.
        super().__init__()
        self.qconfig = qconfig or default_dynamic_qat_qconfig
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies both Linears.
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ManualConvLinearQATModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy QAT model with stubs, a Conv2d (3 to 1, kernel 3), flatten to 64
    # features and two Linears; default qnnpack QAT qconfig unless given.
    r"""A module with manually inserted `QuantStub` and `DeQuantStub`
    and contains both linear and conv modules
    """

    def __init__(self, qconfig=None):
        # [FUNCTION DOC] HELPER: Sets qconfig (argument or qnnpack QAT default) and builds the
        # stubs, convolution and Linears.
        super().__init__()
        self.qconfig = (
            qconfig
            if qconfig
            else torch.ao.quantization.get_default_qat_qconfig("qnnpack")
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3).to(dtype=torch.float)
        self.fc1 = torch.nn.Linear(64, 10).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(10, 10).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize, convolution, flatten via view to width 64, two
        # Linears, dequantize.
        x = self.quant(x)
        x = self.conv(x)
        x = x.view(-1, 64).contiguous()
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)


class ManualConvLinearSymmQATModel(ManualConvLinearQATModel):
    # [CLASS DOC] HELPER: ManualConvLinearQATModel using the symmetric qnnpack QAT qconfig
    # (qnnpack only).
    r"""Same as ManualConvLinearQATModule but with Symmetric Quantization.
    Supported only with qnnpack.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the parent constructor with the symmetric qnnpack
        # QAT qconfig.
        super().__init__(default_symmetric_qnnpack_qat_qconfig)


class ManualEmbeddingBagLinear(nn.Module):
    # [CLASS DOC] HELPER: Toy QAT model: EmbeddingBag (10 by 12, sum) with embedding QAT qconfig,
    # stubs and Linear 12 to 1, qnnpack QAT default.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the layers and sets qconfigs on the embedding and
        # the model.
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode="sum")
        self.emb.qconfig = default_embedding_qat_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig("qnnpack")

    def forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor | None = None,
        per_sample_weights: torch.Tensor | None = None,
    ):
        # [FUNCTION DOC] HELPER: Takes indices, optional offsets and optional per-sample weights;
        # embeds, quantizes, applies Linear, dequantizes.
        x = self.emb(input, offsets, per_sample_weights)
        x = self.quant(x)
        x = self.linear(x)
        return self.dequant(x)


class DeFusedEmbeddingBagLinear(nn.Module):
    # [CLASS DOC] HELPER: Toy QAT model that simulates an EmbeddingBag with a separate Embedding
    # and a sum reduction, followed by stubs and a Linear.
    r"""A module to simulate QAT embedding bag with a linear layer,
    this module uses a separate embedding and bagging op, similar
    to that which is described in the EmbeddingBag documentation.

    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the Embedding (with embedding QAT qconfig), sum operation,
        # stubs and Linear; sets the qnnpack QAT qconfig.
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=12)
        self.emb.qconfig = default_embedding_qat_qconfig
        self.bagging_op = torch.sum
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig("qnnpack")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Embeds the indices, sums over dimension 1, then quantize,
        # Linear, dequantize.
        x = self.bagging_op(self.emb(input), dim=1)
        x = self.quant(x)
        x = self.linear(x)
        return self.dequant(x)


class SubModelForFusion(nn.Module):
    # [CLASS DOC] HELPER: Fusion test submodule: bias-free 1x1 Conv2d followed by BatchNorm2d.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates both layers.
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        self.bn = nn.BatchNorm2d(2).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Convolution then batch norm.
        x = self.conv(x)
        x = self.bn(x)
        return x


class SubModelWithoutFusion(nn.Module):
    # [CLASS DOC] HELPER: Fusion test submodule that cannot fuse: Conv2d followed by
    # non-in-place ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates both layers.
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=False).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: ReLU of the convolution.
        return self.relu(self.conv(x))


class ModelForFusion(nn.Module):
    # [CLASS DOC] HELPER: Large fusion test model mixing 1d, 2d and 3d conv, bn, relu groups,
    # nested submodules, stubs and a Linear. qconfig comes from the argument; sub2 and fc have
    # qconfig None.
    def __init__(self, qconfig):
        # [FUNCTION DOC] HELPER: Creates all layers, assigns the given qconfig and disables
        # quantization for sub2 and fc.
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 1, bias=None).to(dtype=torch.float)
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.sub1 = SubModelForFusion()
        self.sub2 = SubModelWithoutFusion()
        self.fc = nn.Linear(36, 10).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.qconfig = qconfig
        self.conv2 = nn.Conv3d(3, 2, (1, 1, 1), bias=None).to(dtype=torch.float)
        self.relu2 = nn.ReLU(inplace=False).to(dtype=torch.float)
        self.bn2 = nn.BatchNorm3d(2).to(dtype=torch.float)
        self.relu3 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.conv3 = nn.Conv1d(3, 3, 2).to(dtype=torch.float)
        self.bn3 = nn.BatchNorm1d(3).to(dtype=torch.float)
        self.relu4 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # don't quantize sub2
        self.sub2.qconfig = None
        self.fc.qconfig = None

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Squeezes, quantizes, runs the 1d conv/bn/relu chain, unsqueezes,
        # runs the 2d chain and sub1, dequantizes, runs sub2, flattens to 36 features and applies
        # fc. A separate 3d branch is computed on a copy of the input and discarded.
        # KNOWN ISSUE: the y branch (conv2, relu2, bn2, relu3, dequant) never contributes to the
        # returned value.
        x = x.squeeze(2)
        x = self.quant(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu4(x)
        x = x.unsqueeze(2)
        y = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sub1(x)
        x = self.dequant(x)
        x = self.sub2(x)
        x = x.reshape(-1, 36).contiguous()
        x = self.fc(x)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.bn2(y)
        y = self.relu3(y)
        y = self.dequant(y)
        return x


class ConvBNReLU(nn.Sequential):
    # [CLASS DOC] HELPER: Sequential of bias-free 1x1 Conv2d (3 channels), BatchNorm2d and
    # non-in-place ReLU, used as a fusable block.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the Sequential with the three layers.
        super().__init__(
            nn.Conv2d(3, 3, 1, 1, bias=False), nn.BatchNorm2d(3), nn.ReLU(inplace=False)
        )


class ModelWithSequentialFusion(nn.Module):
    # [CLASS DOC] HELPER: Fusion test model: conv, relu, a Sequential of three ConvBNReLU blocks, a
    # classifier Sequential, an empty Sequential and stubs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds all components.
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.relu1 = nn.ReLU(inplace=False)
        layers = [ConvBNReLU() for _ in range(3)]
        self.features = nn.Sequential(*layers)
        head = [nn.Linear(300, 10), nn.ReLU(inplace=False)]
        self.classifier = nn.Sequential(*head)
        self.seq = nn.Sequential()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantize, conv, relu, features, reshape to 300 features,
        # classifier, empty Sequential, dequantize.
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.features(x)
        x = torch.reshape(x, (-1, 3 * 10 * 10))
        x = self.classifier(x)
        x = self.seq(x)
        x = self.dequant(x)
        return x


class ModelForFusionWithBias(nn.Module):
    # [CLASS DOC] HELPER: Fusion test model: stubs, Conv2d + BatchNorm2d + ReLU followed by Conv2d
    # + BatchNorm2d, with biases.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers.
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 5, bias=True).to(dtype=torch.float)
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=True).to(dtype=torch.float)
        self.bn2 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the chain between quantize and dequantize.
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dequant(x)
        return x


class ModelForLinearBNFusion(nn.Module):
    # [CLASS DOC] HELPER: Toy model: Linear 20 to 10 followed by BatchNorm1d with randomly
    # initialized scale and shift.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers and initializes batch norm weight and bias
        # uniformly at random.
        super().__init__()
        self.fc = nn.Linear(20, 10)
        self.bn = nn.BatchNorm1d(10)
        nn.init.uniform_(self.bn.weight)
        nn.init.uniform_(self.bn.bias)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Linear then batch norm.
        return self.bn(self.fc(x))


class DummyObserver(torch.nn.Module):
    # [CLASS DOC] HELPER: Observer stub returning fixed quantization parameters.
    def calculate_qparams(self):
        # [FUNCTION DOC] HELPER: Returns scale 1.0 and zero point 0.
        return 1.0, 0

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return x


class ModelForConvTransposeBNFusion(nn.Module):
    # [CLASS DOC] HELPER: Toy model: three ConvTranspose (1d, 2d, 3d) each followed by a
    # matching BatchNorm.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layer pairs.
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(3, 3, 1)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.ConvTranspose2d(3, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.ConvTranspose3d(3, 3, 1)
        self.bn3 = nn.BatchNorm3d(3)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each pair, adding a dimension between pairs.
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.unsqueeze(2)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ModelWithFunctionals(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model using four FloatFunctional modules (cat, add,
    # add_relu, matmul).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the functional modules.
        super().__init__()
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.mymatmul = nnq.FloatFunctional()
        # Tracing doesn't work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # self.my_scalar_add = nnq.FloatFunctional()
        # self.my_scalar_mul = nnq.FloatFunctional()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Concatenates three copies of the input, adds it to itself,
        # add_relu with itself, then matmul with the transpose. Returns the matmul result.
        y = self.mycat.cat([x, x, x])
        z = self.myadd.add(y, y)
        w = self.myadd_relu.add_relu(z, z)
        u = self.mymatmul.matmul(w, w.T)
        # Tracing doesn't work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # w = self.my_scalar_add.add_scalar(w, -0.5)
        # w = self.my_scalar_mul.mul_scalar(w, 0.5)
        return u


class ResNetBase(torch.nn.Module):
    # [CLASS DOC] HELPER: Small ResNet-like block: conv, bn, relu, identity skip with
    # FloatFunctional add, relu, global average pool, flatten and Linear (3 channels).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the layers.
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = torch.nn.Identity()
        self.myop = nn.quantized.FloatFunctional()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(inplanes, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Conv, bn, relu, skip add with the input, relu, pool,
        # flatten, Linear.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.myop.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def fuse_model(self):
        # [FUNCTION DOC] HELPER: Fuses conv1, bn1 and relu1 in place, using the QAT fusion in
        # training mode and standard fusion otherwise.
        # TODO: remove this check and define two fuse_model function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(
                self, [["conv1", "bn1", "relu1"]], inplace=True
            )
        else:
            torch.ao.quantization.fuse_modules(
                self, [["conv1", "bn1", "relu1"]], inplace=True
            )


class ModelMultipleOps(torch.nn.Module):
    # [CLASS DOC] HELPER: Model with several ops: conv, bn, relu, skip add, average pool, conv, max
    # pool, cat and Linear.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the layers and FloatFunctional modules.
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = torch.nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Conv, bn, relu, skip add with the input, relu, adaptive average
        # pool to 4 by 4, conv, 2 by 2 max pool, concatenates the result with itself, reshape to 12
        # features, Linear 12 to 6.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        out = self.cat.cat([out, out])
        out = out.reshape(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out


# Model to ensure consistency of fake quant with true quant
# Average pooling and mean operations are not modelled
# accurately with fake-quant so this model does not
# contain those operations
class ModelMultipleOpsNoAvgPool(torch.nn.Module):
    # [CLASS DOC] HELPER: Variant of ModelMultipleOps with max pooling instead of average pooling
    # and a second convolution as the skip branch. Used to check consistency between fake
    # quantization and true quantization, since average pooling and mean are not modelled exactly
    # by fake quant.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the layers and FloatFunctional modules.
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.maxpool = nn.MaxPool2d((4, 4))
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Conv, bn, relu, skip add with conv2 of the input, relu, max pool,
        # conv2, 2 by 2 max pool, concatenation with itself, reshape to 12 features, Linear.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        skip = self.conv2(x)
        out = self.skip_add.add(out, skip)
        out = self.relu2(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        out = self.cat.cat([out, out])
        out = out.reshape(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out


class EmbeddingBagModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy module wrapping an EmbeddingBag (10 by 12, sum, include last offset).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the embedding bag.
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(
            num_embeddings=10,
            embedding_dim=12,
            include_last_offset=True,
            scale_grad_by_freq=False,
            mode="sum",
        )

    def forward(self, indices, offsets, per_sample_weights):
        # [FUNCTION DOC] HELPER: Applies it to indices, offsets and per-sample weights.
        return self.emb(indices, offsets, per_sample_weights)


class EmbeddingModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy module wrapping an Embedding (10 by 12).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the embedding.
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

    def forward(self, indices):
        # [FUNCTION DOC] HELPER: Looks up the indices.
        return self.emb(indices)


class EmbeddingWithStaticLinear(torch.nn.Module):
    # [CLASS DOC] HELPER: Toy model: EmbeddingBag with a float-qparams weight-only qconfig plus a
    # statically quantized Linear branch; outputs are concatenated.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the layers, sets qconfigs and the stubs.
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12)
        self.fc = torch.nn.Linear(4, 2)
        self.emb.qconfig = float_qparams_weight_only_qconfig
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, indices, offsets, linear_in):
        # [FUNCTION DOC] HELPER: Embeds indices with offsets, quantizes linear_in, applies Linear,
        # dequantizes, and concatenates linear and embedding outputs along dimension 1.
        emb = self.emb(indices, offsets)
        q_x = self.quant(linear_in)
        fc = self.fc(q_x)
        fc = self.dequant(fc)
        features = torch.cat([fc] + [emb], dim=1)
        return features


class DenseTopMLP(nn.Module):
    # [CLASS DOC] HELPER: Toy sparse-NN top part: a dense MLP and a top MLP over the concatenated
    # dense and sparse features.
    def __init__(
        self, dense_dim, dense_out, embedding_dim, top_out_in, top_out_out
    ) -> None:
        # [FUNCTION DOC] HELPER: Builds the dense Sequential and the top Sequential from the
        # given dimensions.
        super().__init__()

        self.dense_mlp = nn.Sequential(
            nn.Linear(dense_dim, dense_out),
        )
        self.top_mlp = nn.Sequential(
            nn.Linear(dense_out + embedding_dim, top_out_in),
            nn.Linear(top_out_in, top_out_out),
        )

    def forward(
        self,
        sparse_feature: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies the dense MLP, concatenates with the sparse feature along
        # dimension 1, applies the top MLP.
        dense_feature = self.dense_mlp(dense)
        features = torch.cat([dense_feature] + [sparse_feature], dim=1)

        out = self.top_mlp(features)
        return out


# thin wrapper around embedding bag, because tracing inside nn.Embedding
# bag is not supported at the moment and this is top level
class EmbBagWrapper(nn.Module):
    # [CLASS DOC] HELPER: Thin wrapper around EmbeddingBag (sum mode), needed because tracing
    # inside the bag is unsupported at top level.
    def __init__(self, num_embeddings, embedding_dim):
        # [FUNCTION DOC] HELPER: Creates the embedding bag.
        super().__init__()
        self.emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode="sum")

    def forward(self, indices, offsets):
        # [FUNCTION DOC] HELPER: Applies it to indices and offsets.
        return self.emb_bag(indices, offsets)


class SparseNNModel(nn.Module):
    # [CLASS DOC] HELPER: Toy sparse-NN model: embedding bag wrapper plus DenseTopMLP with fixed
    # class-level dimensions.
    _NUM_EMBEDDINGS = 10
    _EMBEDDING_DIM = 5
    _DENSE_DIM = 4
    _DENSE_OUTPUT = 2
    _TOP_OUT_IN = 2
    _TOP_OUT_OUT = 2
    _TOP_MLP_DIM = 1

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the sparse and dense-top parts using the class constants.
        super().__init__()

        self.model_sparse = EmbBagWrapper(self._NUM_EMBEDDINGS, self._EMBEDDING_DIM)
        self.dense_top = DenseTopMLP(
            self._DENSE_DIM,
            self._DENSE_OUTPUT,
            self._EMBEDDING_DIM,
            self._TOP_OUT_IN,
            self._TOP_OUT_OUT,
        )

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_offsets: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Embeds the sparse inputs and passes the result and the dense
        # features to the top MLP.
        sparse_feature = self.model_sparse(sparse_indices, sparse_offsets)
        out = self.dense_top(sparse_feature, dense)

        return out


class TestHelperModules:
    # [CLASS DOC] HELPER: Namespace class grouping small nn.Module fixtures used by quantizer tests
    # (control flow, conv/linear/pooling combinations, embeddings, simple arithmetic). Holds only
    # nested classes; it has no state of its own.
    class ControlFlow(torch.nn.Module):
        # [CLASS DOC] HELPER: Module exercising the map and cond control flow operators with nested
        # branch functions.
        def forward(
            self,
            xs: torch.Tensor,
            pred1: torch.Tensor,
            pred2: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Defines nested branch functions, squares y by matrix multiply,
            # then maps map_fn over xs (with the predicates and y) via control-flow map and returns
            # the stacked result.
            def true_nested(y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns y plus y matrix-multiplied with itself.
                y = y + y
                y = torch.mm(y, y)
                return y

            def false_nested(y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns y matrix-multiplied with itself.
                return torch.mm(y, y)

            def true_fn(x: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Conditionally (on pred2) selects the nested true or false
                # branch applied to x and returns x plus the result.
                z = control_flow.cond(pred2, true_nested, false_nested, [x])
                return x + z

            def false_fn(x: torch.Tensor, _) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the cosine of x; the second argument is ignored.
                return x.cos()

            def map_fn(
                x: torch.Tensor,
                pred1: torch.Tensor,
                pred2: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Per-element map body: cosine of x plus a conditional (on
                # pred1) result computed from y, then returns the sine of the sum.
                x = x.cos()
                y = control_flow.cond(pred1, true_fn, false_fn, [y, pred2])
                x = x + y
                return x.sin()

            y = torch.mm(y, y)
            return control_flow.map(map_fn, xs, pred1, pred2, y)

        def example_inputs(self):
            # [FUNCTION DOC] HELPER: Returns example inputs: a 2 by 2 ones tensor, two false
            # boolean scalars-in-tensors and another 2 by 2 ones tensor.
            return (
                torch.ones(2, 2),
                torch.tensor([False]),
                torch.tensor([False]),
                torch.ones(2, 2),
            )

    class Conv2dPropAnnotaton(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, view to width 3, hardtanh and Linear; used for annotation
        # propagation tests.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, view to (-1, 3), hardtanh with bounds -0.5 and
            # 0.5, Linear.
            x = self.conv(x)
            x = x.view(-1, 3)
            x = torch.nn.functional.hardtanh(x, -0.5, 0.5)
            x = self.linear(x)
            return x

    class Conv2dWithObsSharingOps(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d followed by shared-observer ops: adaptive average pool,
        # hardtanh and mean.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, adaptive average pool, hardtanh, mean reduction.
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = torch.mean(x)
            return x

    class Conv2dWithTwoLinearPermute(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, permute to channels-last, then two Linears (16 to 8 no bias,
        # 8 to 8).
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, permute, Linear, Linear.
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear2(self.linear1(permute_out))

    class Conv2dWithTwoLinear(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, reshape to (2, 64), then two Linears (64 to 8 no bias, 8
        # to 8).
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(64, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, reshape to (2, 64) (assumes a matching input size),
            # Linear, Linear.
            conv_out = self.conv(x)
            reshape_out = torch.reshape(conv_out, (2, 64))
            return self.linear2(self.linear1(reshape_out))

    class ConvLinearWPermute(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, permute to channels-last, then a Linear 8 to 8.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3)
            self.linear1 = torch.nn.Linear(8, 8)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, permute, Linear.
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear1(permute_out)

    class TwoLinearModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Two Linears: 8 to 16 no bias, then 16 to 8.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.linear1 = torch.nn.Linear(8, 16, bias=False)
            self.linear2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies both.
            return self.linear2(self.linear1(x))

        def example_inputs(self):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random 2 by 8 tensor.
            return (torch.randn(2, 8),)

    class ConvMaxPool2d(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d (2 to 2, kernel 1) followed by 1 by 1 MaxPool2d.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)
            self.pool = torch.nn.MaxPool2d(1, 1)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv then pool.
            x = self.conv(x)
            x = self.pool(x)
            return x

    class ConvWithAdaptiveAvgPool2d(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d followed by AdaptiveAvgPool2d to 1 by 1.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv then pool.
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            return x

    class ConvWithBNRelu(torch.nn.Module):
        # [CLASS DOC] HELPER: Configurable conv (1d/2d/3d by dim), optional batch norm and optional
        # ReLU; disabled parts are Identity.
        def __init__(self, relu, dim=2, bn=True, bias=True, padding=0):
            # [FUNCTION DOC] HELPER: Picks Conv and BatchNorm classes by dim, applying bias and
            # padding to the convolution, and uses Identity for bn or relu when disabled.
            super().__init__()
            convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
            bns = {
                1: torch.nn.BatchNorm1d,
                2: torch.nn.BatchNorm2d,
                3: torch.nn.BatchNorm3d,
            }
            self.conv = convs[dim](3, 3, 3, bias=bias, padding=padding)

            if bn:
                self.bn = bns[dim](3)
            else:
                self.bn = torch.nn.Identity()
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, bn, relu.
            x = self.conv(x)
            x = self.bn(x)
            return self.relu(x)

    class ConvTWithBNRelu(torch.nn.Module):
        # [CLASS DOC] HELPER: Configurable transposed conv (1d or 2d by dim), optional batch norm
        # and optional ReLU; disabled parts are Identity.
        def __init__(self, relu, dim=2, bn=True, bias=True):
            # [FUNCTION DOC] HELPER: Picks ConvTranspose and BatchNorm classes by dim; Identity
            # replaces disabled bn or relu.
            super().__init__()
            convts = {1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d}
            bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
            self.convt = convts[dim](3, 3, 3, bias=bias)

            if bn:
                self.bn = bns[dim](3)
            else:
                self.bn = torch.nn.Identity()
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Transposed conv, bn, relu.
            x = self.convt(x)
            x = self.bn(x)
            return self.relu(x)

    class Conv2dThenConv1d(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, squeeze of the leading dimension, then Conv1d.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.conv1d = torch.nn.Conv1d(3, 3, 3)
            self.conv2d = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv2d, squeeze(0), Conv1d.
            x = self.conv2d(x)
            x = x.squeeze(0)
            x = self.conv1d(x)
            return x

        def example_inputs(self):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random 1 by 3 by 5 by
            # 5 tensor.
            return (torch.randn(1, 3, 5, 5),)

    class Conv2dWithCat(torch.nn.Module):
        # [CLASS DOC] HELPER: Two Conv2d branches on two inputs concatenated on dimension 1.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates both convolutions.
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Convolves x and y separately and concatenates.
            x = self.conv1(x)
            y = self.conv2(y)
            z = torch.cat([x, y], dim=1)
            return z

    class Conv2dWithTwoCat(torch.nn.Module):
        # [CLASS DOC] HELPER: Two Conv2d branches concatenated, plus an add of two raw inputs, then
        # another concatenation.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates both convolutions.
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x1, x2, x3, x4):
            # [FUNCTION DOC] HELPER: Concatenates the convolved x1, x2 on dimension 1, adds x3 and
            # x4, and concatenates that sum with the first result along dimension 0.
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
            y = torch.cat([x1, x2], dim=1)
            z = x3 + x4
            w = torch.cat([z, y])
            return w

    class Conv2dWithSplit(torch.nn.Module):
        # [CLASS DOC] HELPER: Conv2d, split into chunks along dimension 1, then concatenation back.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates two convolutions (only the first is used).
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv, split into pieces of size 2 along channels, concatenates
            # the pieces.
            x = self.conv1(x)
            # use split so we get a list of Tensors
            x1, x2 = torch.split(x, 2, dim=1)
            y = torch.cat([x1, x2], dim=1)
            return y

        def example_inputs(self):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random 1 by 3 by 16 by
            # 16 tensor.
            return (torch.randn(1, 3, 16, 16),)

    class ThreeAdd(torch.nn.Module):
        # [CLASS DOC] HELPER: Pure add module computing the sum of four inputs as pairwise adds.
        def forward(self, x1, x2, x3, x4):
            # [FUNCTION DOC] HELPER: Returns (x1 plus x2) plus (x3 plus x4).
            y = x1 + x2
            z = x3 + x4
            w = y + z
            return w

    class EmbeddingModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Embedding (10 by 12) lookup.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the embedding.
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

        def forward(self, indices):
            # [FUNCTION DOC] HELPER: Looks up the indices.
            return self.emb(indices)

    class EmbeddingConvLinearModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Embedding, unsqueeze, permute to channels-first, Conv2d (8 to 16,
        # kernel 1 by 3), permute back, squeeze, Linear 16 to 8.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.conv = torch.nn.Conv2d(8, 16, (1, 3))
            self.linear = torch.nn.Linear(16, 8)

        def forward(self, indices):
            # [FUNCTION DOC] HELPER: Runs the described pipeline on the indices.
            embeddings = self.emb(indices)
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.permute(embeddings, (0, 3, 1, 2))
            conv_out = self.conv(embeddings)
            conv_out = torch.permute(conv_out, (0, 2, 3, 1))
            conv_out = torch.squeeze(conv_out, dim=0)
            return self.linear(conv_out)

    class AddInplaceAdd(torch.nn.Module):
        # [CLASS DOC] HELPER: Module computing x plus y and then an in-place add of y.
        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Adds y out of place and then adds y in place; returns
            # the result.
            x = x + y
            x += y
            return x

    class MulInplaceMul(torch.nn.Module):
        # [CLASS DOC] HELPER: Module computing x times y and then an in-place multiply by y.
        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Multiplies by y out of place and then in place; returns
            # the result.
            x = x * y
            x *= y
            return x

    class AddMulScalar(torch.nn.Module):
        # [CLASS DOC] HELPER: Module applying scalar add 3 and multiply 3, each out of place and
        # in place.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Adds 3, multiplies by 3, adds 3 in place, multiplies by 3 in
            # place; returns x.
            x = x + 3
            x = x * 3
            x += 3
            x *= 3
            return x

    class ConvBnReLU2dAndLinearReLU(torch.nn.Module):
        # [CLASS DOC] HELPER: A ConvWithBNRelu (with ReLU), permute to channels-last, then
        # bias-free Linear 3 to 8.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers (the extra relu attribute is unused).
            super().__init__()
            self.conv_bn_relu = TestHelperModules.ConvWithBNRelu(relu=True)
            self.linear = torch.nn.Linear(3, 8, bias=False)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Conv-bn-relu, permute, Linear.
            x = self.conv_bn_relu(x)
            permute_out = torch.permute(x, (0, 2, 3, 1))
            linear_out = self.linear(permute_out)
            return linear_out

    class GroupwiseConv2d(torch.nn.Module):
        # [CLASS DOC] HELPER: Grouped Conv2d (4 to 4 channels, kernel 3, 2 groups).
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the convolution.
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 3, groups=2)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies it.
            return self.conv(x)

        def example_inputs(self):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random 2 by 4 by 10 by
            # 10 tensor.
            return (torch.randn(2, 4, 10, 10),)

    class LinearReluModel(torch.nn.Module):
        # [CLASS DOC] HELPER: Linear 5 to 5 followed by ReLU (no example input helper).
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the layers.
            super().__init__()
            self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: ReLU of the Linear output.
            x = self.relu(self.fc(x))
            return x

def _static_quantized_linear_module(
    N,
    K,
    bias,
    example_input,
    epilogue="none",
    output_dtype=torch.float32,
):
    # [FUNCTION DOC] HELPER: Builds a statically quantized Linear model (N output, K input
    # features, optional bias) that calls the oneDNN qlinear_pointwise op directly with fixed
    # quantization parameters derived from example_input (uint8 range 0 to 127) and
    # per-channel int8 weights; the returned module is in eval mode. epilogue picks the
    # post-op (none, relu or gelu) and output_dtype the op output dtype. Raises AssertionError
    # for an unsupported epilogue.
    # KNOWN ISSUE: activation scale is computed with quant_max 127 but the runtime input
    # quantization clamps at 255, unlike the binary variant which uses 127.
    """
    Generate a linear module using onednn.qlinear_pointwise directly
    with static quantization parameters (no choose_qparams at runtime).
    It is used to test fusion and lowering passes in Inductor for X86 CPU.
    Input quantization limit is 0-127 to avoid overflow on old platforms.
    Params:
        N: output feature dimension
        K: input feature dimension
        bias: boolean flag to indicate whether linear module has bias
        example_input: example input tensor to get scale/zero point
        epilogue: oneDNN qlinear pointwise post op, e.g. none/relu/gelu
        output_dtype: output dtype used by onednn.qlinear_pointwise
    Return:
        An instance of the quantized linear module
    """
    if epilogue not in ("none", "relu", "gelu"):
        raise AssertionError(f"Unsupported epilogue: {epilogue}")

    class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Inner nn.Module storing precomputed activation scale and zero point,
        # a Linear, per-channel int8 quantized weights, bias, and post-op settings, and calling the
        # oneDNN pointwise quantized linear op in forward.
        def __init__(
            self,
            N,
            K,
            bias,
            example_input,
            epilogue,
            output_dtype,
        ):
            # [FUNCTION DOC] HELPER: Computes activation qparams from example_input, builds the
            # Linear, derives per-channel weight scales and zero points and the quantized
            # weight, and records bias, output scale 1.0, zero point 0, output dtype and the
            # post-op settings.
            super().__init__()
            self.x_scale, self.x_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                example_input,
                quant_min=0,
                quant_max=127,
                eps=torch.finfo(torch.float32).eps,
                dtype=torch.uint8,
            )
            self.x_scale, self.x_zp = self.x_scale.detach().item(), self.x_zp.detach().item()
            self.linear = torch.nn.Linear(K, N, bias)
            self.w_scales, self.w_zps = torch.ops.quantized_decomposed.choose_qparams_per_token(
                self.linear.weight, dtype=torch.int8
            )
            self.w_scales = self.w_scales.detach().to(torch.float32).squeeze()
            self.w_zps = self.w_zps.detach().to(torch.int64).squeeze()
            self.qw = torch.ops.quantized_decomposed.quantize_per_channel.default(
                self.linear.weight,
                self.w_scales,
                self.w_zps,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )

            if bias:
                self.b = self.linear.bias
            else:
                self.b = None

            self.y_scale = 1.0
            self.y_zp = 0
            self.output_dtype = output_dtype
            self.post_op = epilogue
            self.unary_post_op_args = ()
            self.post_op_algo = "none"

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Prepacks the weight, quantizes x to uint8 when it is floating
            # point (otherwise assumes it is already quantized), and returns the oneDNN quantized
            # linear pointwise result.
            qw_packed = torch.ops.onednn.qlinear_prepack(self.qw, None)
            if x.is_floating_point():
                qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x,
                    self.x_scale,
                    self.x_zp,
                    quant_min=0,
                    quant_max=255,
                    dtype=torch.uint8,
                )
            else:
                qx = x
            return torch.ops.onednn.qlinear_pointwise.default(
                qx,
                self.x_scale,
                self.x_zp,
                qw_packed,
                self.w_scales,
                self.w_zps,
                self.b,
                self.y_scale,
                self.y_zp,
                self.output_dtype,
                self.post_op,
                self.unary_post_op_args,
                self.post_op_algo,
            )

    return Model(N, K, bias, example_input, epilogue, output_dtype).eval()


def _static_quantized_linear_binary_module(
    N,
    K,
    bias,
    example_input,
    example_binary_input,
    binary_post_op="add",
    epilogue="none",
    output_dtype=torch.float32,
):
    # [FUNCTION DOC] HELPER: Same idea as the unary static quantized Linear builder, but for the
    # binary-post-op variant: the model takes an extra tensor combined with the output via add or
    # sum (binary_post_op), with optional relu epilogue. For uint8 or int8 output_dtype it
    # calibrates output scale and zero point from an eager reference. Returns the eval-mode model.
    # Raises AssertionError for an unsupported epilogue or binary op.
    if epilogue not in ("none", "relu"):
        raise AssertionError(f"Unsupported epilogue: {epilogue}")
    if binary_post_op not in ("add", "sum"):
        raise AssertionError(f"Unsupported binary post op: {binary_post_op}")

    class Model(torch.nn.Module):
        # [CLASS DOC] HELPER: Inner module like the unary variant but with a second binary input,
        # output quantization parameters and the oneDNN binary pointwise op.
        def __init__(
            self,
            N,
            K,
            bias,
            example_input,
            example_binary_input,
            binary_post_op,
            epilogue,
            output_dtype,
        ):
            # [FUNCTION DOC] HELPER: Computes activation qparams, builds Linear and per-channel
            # int8 weights, derives output qparams from an eager reference when the output dtype is
            # an integer type (else scale 1.0 and zero point 0), and stores post-op settings.
            super().__init__()
            self.x_scale, self.x_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                example_input,
                quant_min=0,
                quant_max=127,
                eps=torch.finfo(torch.float32).eps,
                dtype=torch.uint8,
            )
            self.x_scale, self.x_zp = (
                self.x_scale.detach().item(),
                self.x_zp.detach().item(),
            )
            self.linear = torch.nn.Linear(K, N, bias)
            self.w_scales, self.w_zps = torch.ops.quantized_decomposed.choose_qparams_per_token(
                self.linear.weight, dtype=torch.int8
            )
            self.w_scales = self.w_scales.detach().to(torch.float32).squeeze()
            self.w_zps = self.w_zps.detach().to(torch.int64).squeeze()
            self.qw = torch.ops.quantized_decomposed.quantize_per_channel.default(
                self.linear.weight,
                self.w_scales,
                self.w_zps,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )

            if bias:
                self.b = self.linear.bias
            else:
                self.b = None

            if output_dtype in (torch.uint8, torch.int8):
                # Calibrate output quantization params from an eager reference of this binary op.
                # This mirrors the pre-pass graphs where uint8 outputs have non-trivial y_scale/y_zp.
                example_out = self.linear(example_input) + example_binary_input
                qmin, qmax, qdtype = (0, 127, torch.uint8)
                if output_dtype == torch.int8:
                    qmin, qmax, qdtype = (-128, 127, torch.int8)
                y_scale, y_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
                    example_out,
                    quant_min=qmin,
                    quant_max=qmax,
                    eps=torch.finfo(torch.float32).eps,
                    dtype=qdtype,
                )
                self.y_scale, self.y_zp = y_scale.detach().item(), y_zp.detach().item()
            else:
                self.y_scale, self.y_zp = 1.0, 0

            self.output_dtype = output_dtype
            self.binary_post_op = binary_post_op
            self.binary_alpha = 1.0
            self.unary_post_op = epilogue
            self.unary_post_op_args = ()
            self.unary_post_op_algo = ""

        def forward(self, x, other):
            # [FUNCTION DOC] HELPER: Prepacks the weight, quantizes x when floating point, and
            # returns the oneDNN binary quantized linear result combining the linear output with
            # other (unit scale and zero point).
            qw_packed = torch.ops.onednn.qlinear_prepack(self.qw, None)
            other_scale, other_zp = 1.0, 0
            if x.is_floating_point():
                qx = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x,
                    self.x_scale,
                    self.x_zp,
                    quant_min=0,
                    quant_max=127,
                    dtype=torch.uint8,
                )
            else:
                qx = x
            return torch.ops.onednn.qlinear_pointwise.binary(
                qx,
                self.x_scale,
                self.x_zp,
                qw_packed,
                self.w_scales,
                self.w_zps,
                other,
                self.b,
                self.y_scale,
                self.y_zp,
                self.output_dtype,
                other_scale,
                other_zp,
                self.binary_post_op,
                self.binary_alpha,
                self.unary_post_op,
                self.unary_post_op_args,
                self.unary_post_op_algo,
            )

    return Model(
        N,
        K,
        bias,
        example_input,
        example_binary_input,
        binary_post_op,
        epilogue,
        output_dtype,
    ).eval()
