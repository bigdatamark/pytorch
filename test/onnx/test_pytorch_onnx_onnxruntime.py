# Owner(s): ["module: onnx"]
# ruff: noqa: F841
# [MODULE DOC] ONNX export tests that run the exported model on ONNX Runtime and compare with
# eager PyTorch. The single test class TestONNXRuntime is expanded by parameterized_class over
# opset version, scripting vs tracing, and keep_initializers_as_inputs. Each test builds a small
# module, then calls run_test from the base class to export, execute in ORT and compare outputs.
# Module level helpers build torchvision detection components and deterministic quantization
# inputs. Importing the module only defines helpers and the parameterized test classes.

from __future__ import annotations

import functools
import io
import itertools
import os
import unittest
from collections import OrderedDict

import numpy as np

import onnx
import onnx_test_common
import parameterized
import torchvision
from model_defs import (
    lstm_flattening_result,
    rnn_model_with_packed_sequence,
    word_language_model,
)
from pytorch_test_common import (
    BATCH_SIZE,
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
    skipDtypeChecking,
    skipIfQuantizationBackendQNNPack,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedOpsetVersion,
    skipScriptTest,
    skipShapeChecking,
    skipTraceTest,
)

import torch
from torch import Tensor
from torch.nn.utils import rnn as rnn_utils
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import verification
from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


def _init_test_generalized_rcnn_transform():
    # [FUNCTION DOC] HELPER: Builds a torchvision GeneralizedRCNNTransform with min size 100,
    # max size 200 and the standard ImageNet mean and std lists, and returns it.
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )
    return transform


def _init_test_rpn():
    # [FUNCTION DOC] HELPER: Builds a torchvision RegionProposalNetwork for tests: five anchor
    # sizes with three aspect ratios each, an RPNHead with 256 input channels, IoU thresholds
    # 0.7 and 0.3, 256 samples per image, pre and post NMS top-n of 2000 for training and 1000
    # for testing, NMS threshold 0.7 and score threshold 0. Returns the network.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    out_channels = 256
    rpn_head = torchvision.models.detection.rpn.RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    rpn_score_thresh = 0.0

    rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )
    return rpn


def _construct_tensor_for_quantization_test(
    shape: tuple[int, ...],
    offset: int | float | None = None,
    max_val: int | float | None = None,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Creates a deterministic float tensor of the given shape holding
    # consecutive values from zero, so quantization tests are not flaky. Adds offset when it is
    # given, and if max_val is given and the largest absolute element exceeds it, rescales the
    # tensor so the largest absolute value equals max_val. Returns the tensor.
    """Helper function to generate weights and test inputs in a deterministic way.

    Due to difference in implementation details between PyTorch and ONNXRuntime, randomly generated
    test data for quantization tests can be flaky. To help stabilize the test, this helper function is
    used to generate weights and test inputs in a deterministic way.

    Args:
        shape (Tuple[int]): Shape for tensor to construct.
        offset (Optional[Union[int, float]]): Offset to be added to the generated tensor.
        max_val (Optional[Union[int, float]]): If any element within tensor has a larger absolute value than
            max_val, the tensor will be scaled by max_val / tensor.abs().max(). This step is done after
            applying offset.
    """
    tensor = torch.arange(np.prod(shape), dtype=torch.float).view(shape)
    if offset is not None:
        tensor = tensor + offset
    if max_val is not None and tensor.abs().max() > max_val:
        tensor = tensor * max_val / tensor.abs().max()
    return tensor


def _parameterized_class_attrs_and_values(
    min_opset_version: int, max_opset_version: int
):
    # [FUNCTION DOC] HELPER: Produces the keyword arguments for parameterized_class. Returns a
    # mapping with the attribute names opset_version, is_script and keep_initializers_as_inputs,
    # and a list of value tuples: opsets 7 and 8 with script on and off and initializers as
    # inputs only, then every opset from min to max inclusive crossed with both booleans. Raises
    # ValueError when min_opset_version is below 9.
    attrs = ("opset_version", "is_script", "keep_initializers_as_inputs")
    input_values = []
    input_values.extend(itertools.product((7, 8), (True, False), (True,)))
    # Valid opset versions are defined in torch/onnx/_constants.py.
    # Versions are intentionally set statically, to not be affected by changes elsewhere.
    if min_opset_version < 9:
        raise ValueError("min_opset_version must be >= 9")
    input_values.extend(
        itertools.product(
            range(min_opset_version, max_opset_version + 1),
            (True, False),
            (True, False),
        )
    )
    return {"attrs": attrs, "input_values": input_values}


def _parametrize_rnn_args(arg_name):
    # [FUNCTION DOC] HELPER: Builds the argument mapping for parameterized expansion of an RNN
    # test over one option. The name selects a table of allowed values (layers, bidirectional,
    # initial_state, packed_sequence or dropout) and a readable label per value. Returns a
    # mapping with the argument name, the allowed values, and a naming callback that maps a
    # value to its label. An unknown name raises KeyError.
    options = {
        "layers": {1: "unilayer", 3: "trilayer"},
        "bidirectional": {True: "bidirectional", False: "forward"},
        "initial_state": {True: "with_initial_state", False: "no_initial_state"},
        "packed_sequence": {
            0: "without_sequence_lengths",
            1: "with_variable_length_sequences",
            2: "with_batch_first_sequence_lengths",
        },
        "dropout": {0.2: "with_dropout", 0.0: "without_dropout"},
    }

    return {
        "arg_str": arg_name,
        "arg_values": options[arg_name].keys(),
        "name_fn": lambda val: options[arg_name][val],
    }


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        onnx_test_common.MIN_ONNX_OPSET_VERSION, onnx_test_common.MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
@common_utils.instantiate_parametrized_tests
class TestONNXRuntime(onnx_test_common._TestONNXRuntime):
    # [CLASS DOC] Main ONNX Runtime export test case. Inherits run_test and the opset,
    # script and initializer settings from the common base, and is instantiated once per
    # parameter combination produced by the module helper (class names come from the common
    # naming function). Each method is one export-and-compare scenario for a model,
    # operator or feature.
    def test_fuse_conv_bn1d(self):
        # [FUNCTION DOC] Exports a convolution followed by batch norm over 1D data and checks
        # ONNX Runtime output matches PyTorch, using a random input of shape 20 x 16 x 50.
        class Fuse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv1d (16 to 33 channels, kernel 3, stride 2)
            # followed by BatchNorm1d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution then the batch norm and
                # returns the result.
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        # [FUNCTION DOC] Exports a 2D convolution without bias followed by batch norm and checks
        # ORT matches PyTorch for a random 2 x 3 x 2 x 2 input.
        class Fuse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free Conv2d (3 to 2 channels, kernel 1,
            # stride 2, padding 3) and BatchNorm2d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution then the batch norm and
                # returns the result.
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        # [FUNCTION DOC] Exports a 3D convolution with asymmetric kernel, stride and padding
        # followed by batch norm, and compares ORT with PyTorch on a 2 x 3 x 10 x 50 x 100 input
        # using relaxed tolerances.
        class Fuse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free Conv3d (3 to 2 channels, kernel 3x5x2,
            # stride 2x1x1, padding 3x2x0) and BatchNorm3d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False
                )
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution then the batch norm and
                # returns the result.
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

    def test_fuse_conv_in_block(self):
        # [FUNCTION DOC] Checks conv plus batch norm fusion when the pair sits inside a
        # conditional block. The module is always scripted, exported with dynamic batch and
        # length axes on the input, and compared with ORT with relaxed tolerances.
        class Fuse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies Conv1d and BatchNorm1d (5
            # channels, kernel 3, padding 2) only when a flag stays true; the flag is cleared
            # when the input sum is greater than minus one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 1D convolution and batch norm submodules.
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=1,
                )
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes a flag that is false when the input sum
                # exceeds minus one; if the flag is still true applies convolution then batch
                # norm. Returns the possibly transformed input.
                results_available = True

                if x.sum() > -1:
                    results_available = False

                if results_available:
                    x = self.conv(x)
                    x = self.bn(x)

                return x

        model = Fuse()
        x = torch.randn(2, 5, 9, requires_grad=True)
        self.run_test(
            torch.jit.script(model),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            rtol=1e-3,
            atol=1e-6,
        )

    def test_conv_tbc(self):
        # [FUNCTION DOC] Exports a time-batch-channel convolution module built around the
        # conv_tbc operator with Xavier-normal weights and zero bias, and compares ORT with
        # PyTorch on a 10 x 7 x 3 input.
        from torch.nn.modules.utils import _single

        class ConvTBC(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a weight parameter of shape kernel x
            # in_channels x out_channels and a bias parameter, both initialized in
            # reset_parameters. Stores kernel size and padding as single-element tuples.
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                # [FUNCTION DOC] HELPER: Stores channel counts, converts kernel size and
                # padding to one-element tuples, allocates the weight and bias parameters
                # and initializes them.
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)

                self.weight = torch.nn.Parameter(
                    Tensor(self.kernel_size[0], in_channels, out_channels)
                )
                self.bias = torch.nn.Parameter(Tensor(out_channels))
                self.reset_parameters()

            def reset_parameters(self):
                # [FUNCTION DOC] HELPER: Re-initializes the weight with a Xavier normal
                # distribution and the bias with zeros in place.
                torch.nn.init.xavier_normal_(self.weight)
                torch.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                # [FUNCTION DOC] HELPER: Runs the conv_tbc operator on a contiguous copy of the
                # input with the stored weight, bias and first padding value.
                return torch.conv_tbc(
                    input.contiguous(), self.weight, self.bias, self.padding[0]
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Forward pass; delegates to conv_tbc and
                # returns its result.
                return self.conv_tbc(input)

        in_channels = 3
        out_channels = 5
        kernel_size = 5
        model = ConvTBC(in_channels, out_channels, kernel_size, padding=0)
        x = torch.randn(10, 7, in_channels, requires_grad=True)
        self.run_test(model, (x,), atol=1e-5)

    def test_reshape_constant_fold(self):
        # [FUNCTION DOC] Checks that reshaping a constant buffer is folded during export. The
        # module multiplies the input by a ones buffer reshaped to broadcast shape; ORT output
        # is compared with PyTorch for a 4 x 5 input.
        class Reshape(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of five ones that is reshaped to a 4D
            # broadcast shape and multiplied with the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the module and registers the ones buffer
                # of length five.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reshapes the buffer to 1 x -1 x 1 x 1 and returns the
                # input multiplied by it.
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

    def run_word_language_model(self, model_name):
        # [FUNCTION DOC] HELPER: Shared body for word language model tests. Builds a small
        # model with 50 tokens, embedding and hidden size 5, 5 layers, dropout 0.2, untied,
        # batch 5. GRU uses the tensor-hidden variant, LSTM the tuple-hidden variant, any other
        # name the plain RNN model. Runs run_test on an integer token matrix and the model's
        # initial hidden state.
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        if model_name == "GRU":
            model = word_language_model.RNNModelWithTensorHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        elif model_name == "LSTM":
            model = word_language_model.RNNModelWithTupleHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        else:
            model = word_language_model.RNNModel(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def get_image(self, rel_path: str, size: tuple[int, int]) -> Tensor:
        # [FUNCTION DOC] HELPER: Loads an image from the assets directory next to this file,
        # using a slash separated relative path, converts to RGB, resizes with bilinear
        # filtering to the given size and returns it as a tensor. Needs PIL and torchvision.
        from PIL import Image
        from torchvision import transforms

        data_dir = os.path.join(os.path.dirname(__file__), "assets")
        path = os.path.join(data_dir, *rel_path.split("/"))
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)

        return transforms.ToTensor()(image)

    def get_test_images(self) -> tuple[list[Tensor], list[Tensor]]:
        # [FUNCTION DOC] HELPER: Returns a pair of image lists: one with the Grace Hopper
        # photo resized to 100 x 320 and one with the PyTorch logo resized to 250 x 380,
        # both as tensors.
        return (
            [self.get_image("grace_hopper_517x606.jpg", (100, 320))],
            [self.get_image("rgb_pytorch.png", (250, 380))],
        )

    def test_paste_mask_in_image(self):
        # [FUNCTION DOC] Checks that the torchvision paste_masks_in_image function traces
        # correctly. Traces it with one set of random masks, boxes and image size, then compares
        # eager and traced outputs for that input and for a second differently sized set. Raises
        # AssertionError on mismatch. Does not export to ONNX.
        masks = torch.rand(10, 1, 26, 26)
        boxes = torch.rand(10, 4)
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = torch.jit.trace(
            paste_masks_in_image,
            (masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]),
        )
        out_trace = jit_trace(
            masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]
        )

        if not torch.all(out.eq(out_trace)):
            raise AssertionError("Outputs do not match")

        masks2 = torch.rand(20, 1, 26, 26)
        boxes2 = torch.rand(20, 4)
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(
            masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])]
        )

        if not torch.all(out2.eq(out_trace2)):
            raise AssertionError("Outputs do not match")

    def test_heatmaps_to_keypoints(self):
        # [FUNCTION DOC] Checks that heatmaps_to_keypoints from torchvision gives identical
        # results eagerly and after tracing, for the traced input and for a second input of
        # different shape. Compares both returned elements and raises AssertionError on
        # mismatch. Does not export to ONNX.
        maps = torch.rand(10, 1, 26, 26)
        rois = torch.rand(10, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints

        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)

        if not torch.all(out[0].eq(out_trace[0])):
            raise AssertionError("Outputs do not match at index 0")
        if not torch.all(out[1].eq(out_trace[1])):
            raise AssertionError("Outputs do not match at index 1")

        maps2 = torch.rand(20, 2, 21, 21)
        rois2 = torch.rand(20, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints

        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)

        if not torch.all(out2[0].eq(out_trace2[0])):
            raise AssertionError("Outputs do not match at index 0")
        if not torch.all(out2[1].eq(out_trace2[1])):
            raise AssertionError("Outputs do not match at index 1")

    def test_word_language_model_RNN_TANH(self):
        # [FUNCTION DOC] Runs the word language model scenario with the tanh RNN variant.
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        # [FUNCTION DOC] Runs the word language model scenario with the ReLU RNN variant.
        self.run_word_language_model("RNN_RELU")

    @skipScriptTest()  # scripting prim::unchecked_cast prim::setattr
    def test_word_language_model_LSTM(self):
        # [FUNCTION DOC] Runs the word language model scenario with the LSTM variant;
        # skipped under scripting.
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        # [FUNCTION DOC] Runs the word language model scenario with the GRU variant.
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        # [FUNCTION DOC] Exports indexing a 5D tensor with a single integer index and compares
        # ORT with PyTorch.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the first element along dim 0
            # of the input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed at position 0.
                return input[0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        # [FUNCTION DOC] Exports a slice 0:1 along the first axis of a 5D tensor and compares
        # ORT with PyTorch.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a unit-length slice of the first dimension.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the slice 0:1 on dim 0 of the input.
                return input[0:1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        # [FUNCTION DOC] Exports indexing with an integer on dim 0 and a full slice on dim 1 of
        # a 5D tensor and compares ORT with PyTorch.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input selected at index 1 on the first
            # dimension, all of the second.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed by integer 1 and
                # a full slice.
                return input[1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        # [FUNCTION DOC] Exports a slice with a negative stop (0 to -1) on the first axis of a
        # 5D tensor and compares ORT with PyTorch.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning all but the last entry of
            # the first dimension.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input sliced from 0 to minus one on dim 0
                # with a full slice on dim 1.
                return input[0:-1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        # [FUNCTION DOC] Exports advanced indexing with a mask built from a small tensor. First
        # a uint8 mask of three entries, then a boolean mask, each in its own module, compared
        # against PyTorch on a 5D input. Requires opset 9 or newer.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing its input with a constant uint8 tensor with
            # values 0, 1, 0.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed by the constant
                # uint8 mask tensor.
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing its input with a constant boolean tensor
            # with values false, true, false (redefines the class of the same name from
            # earlier in the test).
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed by the constant
                # boolean mask tensor.
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        # [FUNCTION DOC] Exports a scripted module that creates a zero tensor with the size of
        # the input's data attribute. Runs once with dynamic axes on both input dimensions and
        # once with an empty list of remaining ONNX inputs. Requires opset 9 or newer.
        class Data(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule whose script method forward returns zeros shaped
            # like the input data size.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning a new zeros tensor with the
                # size of the input data.
                return x.new_zeros(x.data.size())

        x = torch.randn(3, 4)
        self.run_test(Data(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_mask_nd(self):
        # [FUNCTION DOC] Exports boolean mask indexing where the mask is the input greater than
        # zero, on a 5D tensor, and compares ORT with PyTorch. Requires opset 11 or newer.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the elements of the input selected by
            # a positive-value mask.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed by the mask input
                # greater than zero.
                return input[input > 0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipScriptTest()
    def test_dict(self):
        # [FUNCTION DOC] Exports a module taking a dictionary input whose key is a scalar tensor
        # and whose value is a tensor; the output is a dictionary holding the sum of the value
        # and the key. Skipped under scripting.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a mapping from a tensor key to a tensor value
            # and returning a mapping with one output entry.
            def forward(self, x_in):
                # [FUNCTION DOC] HELPER: Looks up the first key of the input mapping, adds the
                # key to the value stored under it, and returns a new mapping holding the result
                # under a fixed output name.
                x_out = {}
                x_out["test_key_out"] = torch.add(
                    x_in[list(x_in.keys())[0]],  # noqa: RUF015
                    list(x_in.keys())[0],  # noqa: RUF015
                )
                return x_out

        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()
    def test_dict_str(self):
        # [FUNCTION DOC] Exports a module taking a dictionary with a string key and tensor
        # value; output is a dictionary with the value plus 2.0. Skipped under scripting.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a mapping keyed by strings and returning a
            # mapping with one entry.
            def forward(self, x_in):
                # [FUNCTION DOC] HELPER: Adds 2.0 to the entry stored under the input key and
                # returns it in a new mapping under a fixed output name.
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()  # User-defined class not supported
    def test_dict_output(self):
        # [FUNCTION DOC] Exports a module returning a user-defined ordered-dictionary output
        # class with a tensor field, an optional tuple and an optional list; compares ORT with
        # PyTorch. Skipped under scripting because user-defined classes are unsupported.
        class DictModelOutput(OrderedDict):
            # [CLASS DOC] HELPER: Ordered dictionary subclass declaring field annotations for
            # a required tensor output and optional tuple and list outputs; used as the
            # model's return type.
            tensor_out: Tensor
            tuple_out: tuple[Tensor] | None = None
            list_out: list[Tensor] | None = None

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that packs its four inputs into the output class: first
            # as the tensor field, second and third as a tuple, fourth as a one-element list.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Builds and returns the dictionary output object from
                # the four inputs.
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        # [FUNCTION DOC] Exports a model returning a tuple containing a tensor, a nested tuple
        # of two tensors and a tensor; compares against ORT with four random 2 x 3 inputs.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns its inputs regrouped as a tuple with
            # one nested pair.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Returns the first input, a tuple of the second and
                # third, and the fourth.
                return a, (b, c), d

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        # [FUNCTION DOC] Exports a model returning deeper nested tuples and compares ORT with
        # PyTorch on four 2 x 3 inputs.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning its inputs in a nested tuple structure.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Returns the first input and a tuple holding a
                # one-element tuple of the second and a pair of the third and fourth.
                return a, ((b,), (c, d))

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        # [FUNCTION DOC] Exports a model whose single argument is a tuple of two tensors, passed
        # as the only positional input, and checks ORT matches PyTorch.
        class TupleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a tuple of two tensors and
            # returning it unchanged.
            def forward(self, a: tuple[Tensor, Tensor]):
                # [FUNCTION DOC] HELPER: Returns the tuple argument as is.
                return a

        x = (torch.randn(3, 4), torch.randn(4, 3))
        self.run_test(TupleModel(), input_args=(x,))

    def test_tuple_primitive_input(self):
        # [FUNCTION DOC] Exports a model with a tuple input mixing an integer and a tensor plus
        # a separate tensor input; checks ORT matches PyTorch.
        class TupleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a tuple of an integer and a tensor,
            # and a tensor.
            def forward(self, a: tuple[int, Tensor], b):
                # [FUNCTION DOC] HELPER: Returns the integer from the tuple and the sum of the
                # tuple's tensor and the second argument.
                return a[0], a[1] + b

        x = (3, torch.randn(4, 3))
        y = torch.randn(4, 3)
        self.run_test(TupleModel(), input_args=(x, y))

    def test_nested_tuple_input(self):
        # [FUNCTION DOC] Exports a model with a tensor input and a second input holding a nested
        # tuple of tensors of broadcastable shapes, and checks ORT matches PyTorch.
        class NestedTupleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a tensor and a tuple that contains a tensor and
            # a nested pair of tensors.
            def forward(self, a, b: tuple[Tensor, tuple[Tensor, Tensor]]):
                # [FUNCTION DOC] HELPER: Returns the sum of the first argument and all three
                # tensors from the tuple argument.
                return a + b[0] + b[1][0] + b[1][1]

        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(1, 5), torch.randn(4, 1)))
        self.run_test(NestedTupleModel(), input_args=(x, y))

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_none(self):
        # [FUNCTION DOC] Exports a model with two optional tensor arguments defaulting to None,
        # called with positional arguments containing None and with keyword dictionaries
        # (including omitted keys), and compares ORT with PyTorch. Skipped under scripting;
        # requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with required x and optional y and z, all
            # defaulting to None.
            def forward(
                self,
                x,
                y: Tensor | None = None,
                z: Tensor | None = None,
            ):
                # [FUNCTION DOC] HELPER: Returns x plus y if y is present, else x plus z if z is
                # present, else x.
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        # Without kwargs dict.
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))
        # With kwargs dict.
        self.run_test(model, (x,), {"y": y, "z": None})
        self.run_test(model, (x,), {"y": None, "z": z})
        self.run_test(model, (x,), {"z": z})
        self.run_test(model, (x,), {"y": y})

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor(self):
        # [FUNCTION DOC] Exports a model whose optional arguments default to tensors of ones and
        # zeros, evaluating with y given and z None, and with y None and z given. Skipped under
        # scripting since tracing drops None inputs; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with required x and optional y and z whose defaults are
            # constant tensors of ones and zeros of shape 2 x 3.
            def forward(
                self,
                x,
                y: Tensor | None = torch.ones(2, 3),
                z: Tensor | None = torch.zeros(2, 3),
            ):
                # [FUNCTION DOC] HELPER: Returns x plus y if y is present, else x plus z if z is
                # present, else x.
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()

        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor_script(self):
        # [FUNCTION DOC] Scripted counterpart of the previous default-tensor test. Runs the
        # scripted model with all inputs, with keyword inputs, and with y only. Then for four
        # combinations where one of y and z is None it expects ValueError mentioning 1 None
        # after flattening. Skipped under tracing; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with required x and optional y and z defaulting to ones
            # and zeros tensors.
            def forward(
                self,
                x,
                y: Tensor | None = torch.ones(2, 3),
                z: Tensor | None = torch.zeros(2, 3),
            ):
                # [FUNCTION DOC] HELPER: Returns x plus y if y is present, else x plus z if z is
                # present, else x.
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = torch.jit.script(Model())

        self.run_test(model, (x, y, z), input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y, "z": z}, input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y}, input_names=("x", "y"))

        for example_inputs, example_kwargs in (
            ((x, y, None), {}),
            ((x, None, z), {}),
            ((x,), {"y": y, "z": None}),
            ((x,), {"y": None, "z": z}),
        ):
            with self.assertRaisesRegex(
                ValueError, "args contained 1 None's after flattening."
            ):
                self.run_test(
                    model, example_inputs, example_kwargs, input_names=("x", "y", "z")
                )

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_none(self):
        # [FUNCTION DOC] Exports a model whose only arguments are two optional tensors
        # defaulting to None; checks ORT for positional None and for keyword input where the
        # unused y disappears in tracing. Skipped under scripting; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with optional x and y, both defaulting to None.
            def forward(self, x: Tensor | None = None, y: Tensor | None = None):
                # [FUNCTION DOC] HELPER: Returns x if present, else y if present, else a scalar
                # tensor holding -1.0.
                if x is not None:
                    return x
                if y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(
            model,
            (),
            {"x": x, "y": None},
            # y disappears in tracing.
            input_names=("x",),
        )

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor(self):
        # [FUNCTION DOC] Exports a model with two optional tensor arguments defaulting to
        # tensors. Checks tracing with x given and with y given, then expects ValueError about
        # too many positional inputs when both are passed, because tracing removes the unused y.
        # Skipped under scripting; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with optional x and y defaulting to ones
            # and zeros tensors.
            def forward(
                self,
                x: Tensor | None = torch.ones(2, 3),
                y: Tensor | None = torch.zeros(2, 3),
            ):
                # [FUNCTION DOC] HELPER: Returns x if present, else y if present, else a scalar
                # tensor holding -1.0.
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (None, y))
        # tracing means y is never used so it's removed from the exported model inputs,
        # and we fail when trying to run ORT.
        with self.assertRaisesRegex(ValueError, "got too many positional inputs"):
            self.run_test(model, (x, y))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor_script(self):
        # [FUNCTION DOC] Scripted model with optional arguments defaulting to tensors. Checks
        # that a missing y runs, that passing only y as a keyword fails with AssertionError
        # about tensors not being close because ONNX does not support default values, and
        # that full positional and keyword calls work. Skipped under tracing; requires opset
        # 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with optional x and y defaulting to ones
            # and zeros tensors.
            def forward(
                self,
                x: Tensor | None = torch.ones(2, 3),
                y: Tensor | None = torch.zeros(2, 3),
            ):
                # [FUNCTION DOC] HELPER: Returns x if present, else y if present, else a scalar
                # tensor holding -1.0.
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = torch.jit.script(Model())

        # Optional supports None inputs
        self.run_test(model, (x,))
        # NOTE: default value is not supported on ONNX, so torch and ONNX has
        # different behavior
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            self.run_test(model, (), {"y": y}, input_names=["y"])

        self.run_test(model, (x, y))
        self.run_test(model, (), {"x": x, "y": y}, input_names=("x", "y"))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logit(self):
        # [FUNCTION DOC] Exports a module applying the logit function with a small epsilon of
        # 1e-6 to a random 1 x 3 x 640 x 640 input and compares ORT with PyTorch. Requires opset
        # 9 or newer.
        class Logit(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing an epsilon and returning the logit of the input
            # clamped by that epsilon.
            def __init__(self, eps):
                # [FUNCTION DOC] HELPER: Stores the epsilon value on the module.
                super().__init__()
                self.eps = eps

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the logit of the input using
                # the stored epsilon.
                return x.logit(self.eps)

        model = Logit(eps=1e-6)
        self.run_test(model, torch.randn(1, 3, 640, 640))

    class Atleast1d(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_1d to a list of five tensors (the
        # five arguments) and returning the resulting list.
        def forward(self, t, w, x, y, z):
            # [FUNCTION DOC] HELPER: Passes the five inputs as one sequence to the atleast_1d
            # operator and returns its result.
            return torch.atleast_1d((t, w, x, y, z))

    class Atleast2d(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_2d to a sequence of five tensors.
        def forward(self, t, w, x, y, z):
            # [FUNCTION DOC] HELPER: Passes the five inputs as one sequence to the atleast_2d
            # operator and returns its result.
            return torch.atleast_2d((t, w, x, y, z))

    class Atleast3d(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_3d to a sequence of five tensors.
        def forward(self, t, w, x, y, z):
            # [FUNCTION DOC] HELPER: Passes the five inputs as one sequence to the atleast_3d
            # operator and returns its result.
            return torch.atleast_3d((t, w, x, y, z))

    class Atleast1dTensor(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_1d to a single tensor.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the atleast_1d result of the input.
            return torch.atleast_1d(x)

    class Atleast2dTensor(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_2d to a single tensor.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the atleast_2d result of the input.
            return torch.atleast_2d(x)

    class Atleast3dTensor(torch.nn.Module):
        # [CLASS DOC] HELPER: Test module applying atleast_3d to a single tensor.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns the atleast_3d result of the input.
            return torch.atleast_3d(x)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize("module_class", (Atleast1d, Atleast2d, Atleast3d))
    def test_atleast_nd_list_input(self, module_class: torch.nn.Module):
        # [FUNCTION DOC] Parameterized over the three list-input atleast modules. Exports each
        # with five inputs of rank 0 through 4 and compares ORT with PyTorch. Skipped under
        # scripting because tracing uses list unpacking; requires opset 11 or newer.
        inputs = (
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        )
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize(
        "module_class", (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor)
    )
    @common_utils.parametrize(
        "inputs",
        [
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        ],
    )
    def test_atleast_nd_single_tensor_input(
        self, module_class: torch.nn.Module, inputs: torch.Tensor
    ):
        # [FUNCTION DOC] Parameterized over the three single-tensor atleast modules and over
        # inputs of rank 0 through 4. Exports the module with that one input and compares ORT
        # with PyTorch. Skipped under scripting; requires opset 11 or newer.
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional(self):
        # [FUNCTION DOC] Exports a model with an optional tensor argument, run once with None
        # and once with a tensor, comparing ORT with PyTorch. Skipped under scripting; requires
        # opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor x and an optional tensor y
            # without a default.
            def forward(self, x, y: Tensor | None):
                # [FUNCTION DOC] HELPER: Returns x plus y when y is present, otherwise x.
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (x, x))

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional(self):
        # [FUNCTION DOC] Exports a model whose second argument is a tuple of two optional
        # tensors, called with the first entry None and the second a tensor. Skipped under
        # scripting; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a tensor and a pair of optional tensors.
            def forward(self, x, y: tuple[Tensor | None, Tensor | None]):
                # [FUNCTION DOC] HELPER: Returns x plus the first tuple entry if present, else x
                # plus the second if present, else x.
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor(self):
        # [FUNCTION DOC] Exports a model whose tuple of optional tensors defaults to two zero
        # tensors, called with the first entry None. Skipped under scripting since tracing drops
        # None inputs; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor x and a pair of optional tensors
            # defaulting to zeros of shape 2 x 3.
            def forward(
                self,
                x,
                y: tuple[Tensor | None, Tensor | None] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                # [FUNCTION DOC] HELPER: Unpacks the pair and returns x plus the first entry if
                # present, else x plus the second if present, else x.
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor_script(self):
        # [FUNCTION DOC] Scripted variant of the tuple-of-optional default test. Expects
        # ValueError about 1 None after flattening when the first entry is None, checks that
        # full tuples run in ORT, and then only checks that exporting with a keyword dictionary
        # succeeds into an in-memory buffer (running ORT would fail because inputs flatten to
        # three). Skipped under tracing; requires opset 15 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor x and a pair of optional tensors
            # defaulting to zeros of shape 2 x 3.
            def forward(
                self,
                x,
                y: tuple[Tensor | None, Tensor | None] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                # [FUNCTION DOC] HELPER: Unpacks the pair and returns x plus the first entry if
                # present, else x plus the second if present, else x.
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = torch.randn(2, 3)
        y0 = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        model = torch.jit.script(Model())
        with self.assertRaisesRegex(
            ValueError, "args contained 1 None's after flattening."
        ):
            self.run_test(model, (x, (None, y1)))
        self.run_test(model, (x, (y0, y1)))
        # export succeeds, but running ORT through run_test would fail because the exported model
        # has the inputs flattened into 3 inputs.
        torch.onnx.export(
            model,
            (x, {"y": (y0, y1)}),
            io.BytesIO(),
            opset_version=self.opset_version,
            dynamo=False,
        )

    def test_primitive_input_integer(self):
        # [FUNCTION DOC] Exports a model taking a Python integer and a tensor and adding them,
        # comparing ORT with PyTorch.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an integer argument to a tensor.
            def forward(self, x: int, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the integer and the tensor.
                return x + y

        x = 3
        y = torch.randint(10, (2, 3, 4))
        self.run_test(Model(), (x, y))

    @skipDtypeChecking
    def test_primitive_input_floating(self):
        # [FUNCTION DOC] Exports a model taking a Python float and a tensor and adding them;
        # dtype checking is disabled for this test.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a float argument to a tensor.
            def forward(self, x: float, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the float and the tensor.
                return x + y

        x = 3.0
        y = torch.randn(2, 3, 4)
        self.run_test(Model(), (x, y))

    def test_primitive_input_bool(self):
        # [FUNCTION DOC] Exports a scripted model taking a boolean flag and two tensors and
        # returning one of them depending on the flag, with the flag true.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module choosing between two tensors using a boolean argument.
            def forward(self, flag: bool, x, y):
                # [FUNCTION DOC] HELPER: Returns x when the flag is true, otherwise y.
                if flag:
                    return x
                else:
                    return y

        flag = True
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(torch.jit.script(Model()), (flag, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cste_script(self):
        # [FUNCTION DOC] Exports a scripted module producing constants whose shape comes from
        # the input size; run with dynamic axes on both dims and again with no remaining ONNX
        # inputs. Requires opset 9 or newer.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning zeros of length equal to the first dim
            # and a ones int64 matrix of shape dim1 x dim0.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning a zeros vector sized by dim 0
                # and an int64 ones matrix sized by dim 1 and dim 0 of the input.
                return torch.zeros(x.size(0)), torch.ones(
                    (x.size(1), x.size(0)), dtype=torch.int64
                )

        x = torch.randn(3, 4)
        self.run_test(MyModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    def test_scalar_tensor(self):
        # [FUNCTION DOC] Exports a module that returns scalar tensors built from the input's
        # first two dimension sizes (second one as int64), with dynamic axes on all three input
        # dims and an additional test input of a different shape.
        class test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning scalar_tensor values of the input's size in
            # dim 0 and dim 1 (the latter int64).
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns two scalar tensors holding the sizes of the
                # first and second dimension of the input.
                return torch.scalar_tensor(input.size(0)), torch.scalar_tensor(
                    input.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_tensor(self):
        # [FUNCTION DOC] Exercises tensor construction inside scripted modules from sizes and
        # scalars: scalar from a shape entry, tensor from two shape entries, tensor from a
        # float conversion, a tensor with explicit long dtype, and a mixed shape and integer
        # tensor. Dynamic or empty remaining-input variants are run where the output depends
        # only on shape.
        class ScalarInputModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning a tensor holding the size of dim 1
            # of the input.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a tensor built from the
                # second dimension size.
                return torch.tensor(input.shape[1])

        x = torch.randn(3, 4)
        self.run_test(
            ScalarInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        class TensorInputModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning a tensor made of the first
            # two dimension sizes.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a tensor built from the sizes
                # of dim 0 and dim 1.
                return torch.tensor([input.shape[0], input.shape[1]])

        x = torch.randn(3, 4)
        self.run_test(
            TensorInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        class FloatInputModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning a one-element tensor from converting
            # the input to float.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a one-element tensor holding
                # the input converted to a float.
                return torch.tensor([float(input)])

        x = torch.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning a long tensor built from the size
            # of dim 1.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a tensor built from the second
                # dimension size with long dtype.
                return torch.tensor(input.shape[1], dtype=torch.long)

        x = torch.randn(3, 4)
        self.run_test(
            InputWithDtypeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        class MixedInputModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning a tensor made of the first dimension
            # size and the input converted to an integer.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a tensor built from the size of
                # dim 0 and the integer conversion of the input.
                return torch.tensor([input.shape[0], int(input)])

        x = torch.randn(1)
        self.run_test(MixedInputModel(), x)

    def test_hardtanh(self):
        # [FUNCTION DOC] Exports the Hardtanh module with limits -1.5 and 2.5 on integers from
        # -5 to 4 as float32 and compares ORT with PyTorch.
        model = torch.nn.Hardtanh(-1.5, 2.5)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardtanh_script_with_default_values(self):
        # [FUNCTION DOC] Exports a scripted module calling the functional hardtanh with default
        # limits on values from -5 to 4 and compares ORT with PyTorch.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule applying functional hardtanh with
            # its default arguments.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning hardtanh of the input.
                return torch.nn.functional.hardtanh(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardswish(self):
        # [FUNCTION DOC] Exports the Hardswish module for a random 3 x 3 input and for scalar
        # edge cases 3 and -3, comparing ORT with PyTorch.
        model = torch.nn.Hardswish()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardswish_script(self):
        # [FUNCTION DOC] Exports a scripted module calling functional hardswish on a random 3
        # x 3 input.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule applying functional hardswish.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning hardswish of the input.
                return torch.nn.functional.hardswish(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardsigmoid(self):
        # [FUNCTION DOC] Exports the Hardsigmoid module for a random 3 x 3 input and for scalar
        # corner cases 3 and -3, comparing ORT with PyTorch.
        model = torch.nn.Hardsigmoid()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # corner cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_tanhshrink(self):
        # [FUNCTION DOC] Exports the Tanhshrink module on a random 3 x 3 float32 input and
        # compares ORT with PyTorch.
        model = torch.nn.Tanhshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink(self):
        # [FUNCTION DOC] Exports the Hardshrink module on a random 3 x 3 input and scalar edge
        # cases 0.5 and -0.5. Requires opset 9 or newer.
        model = torch.nn.Hardshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink_dtype(self):
        # [FUNCTION DOC] Exports Hardshrink on a random float64 input and compares ORT with
        # PyTorch. Requires opset 9 or newer.
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Hardshrink(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink(self):
        # [FUNCTION DOC] Exports the Softshrink module on a random 3 x 3 input and scalar edge
        # cases 0.5 and -0.5. Requires opset 9 or newer.
        model = torch.nn.Softshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink_dtype(self):
        # [FUNCTION DOC] Exports Softshrink on a random float64 input and compares ORT with
        # PyTorch. Requires opset 9 or newer.
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Softshrink(), x)

    def test_clamp(self):
        # [FUNCTION DOC] Exports three clamp variants on random 3 x 4 inputs, each in
        # its own module: both bounds, lower bound only, and upper bound only. Compares
        # ORT with PyTorch.
        class ClampModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input to the range -0.5 to 0.5.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input clamped between -0.5 and 0.5.
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input from below at -0.5 only.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input clamped with a minimum of -0.5.
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input from above at 0.5 only.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input clamped with a maximum of 0.5.
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_clamp_dyn(self):
        # [FUNCTION DOC] Exports clamp variants whose bounds are computed or passed at runtime.
        # Scripted modules clamp using the size of dim 0 as upper bound, as lower bound, and dim
        # 0 and dim 1 sizes as both bounds on float matrices; traced modules clamp with tensor
        # min and max, tensor min only, and tensor max only. Requires opset 8 or newer.
        class ClampMaxModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule clamping with no lower bound and the first
            # dimension size as the upper bound.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the input clamped above by the
                # size of dim 0.
                return x.clamp(None, x.size(0))

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)

        class ClampMinModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule clamping with the first dimension size as the
            # lower bound and no upper bound.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the input clamped below by the
                # size of dim 0.
                return x.clamp(x.size(0), None)

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule clamping between the sizes of dim 0 and dim 1.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the input clamped between the
                # size of dim 0 and the size of dim 1.
                return x.clamp(x.size(0), x.size(1))

        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

        class ClampTensorModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input between two tensor arguments.
            def forward(self, x, min, max):
                # [FUNCTION DOC] HELPER: Returns the input clamped between the given min
                # and max tensors.
                return x.clamp(min, max)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = torch.randn(3, 4)
        self.run_test(ClampTensorModel(), (x, y, z))

        class ClampTensorMinModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input below by a tensor argument.
            def forward(self, x, min):
                # [FUNCTION DOC] HELPER: Returns the input clamped with the given
                # tensor as minimum.
                return x.clamp(min=min)

        self.run_test(ClampTensorMinModel(), (x, y))

        class ClampTensorMaxModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping its input above by a tensor argument.
            def forward(self, x, max):
                # [FUNCTION DOC] HELPER: Returns the input clamped with the given
                # tensor as maximum.
                return x.clamp(max=max)

        self.run_test(ClampTensorMaxModel(), (x, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        # [FUNCTION DOC] Exports a traced module that calls full to build a 3 x 4 long tensor
        # filled with the scalar input tensor 12. Requires opset 9 or newer.
        class FullModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing a 3 x 4 long tensor filled with
            # the input value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a 3 x 4 tensor of dtype long filled with
                # the scalar input.
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        # [FUNCTION DOC] Same as the traced full test but through a ScriptModule. Requires opset
        # 9 or newer.
        class FullModelScripting(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule producing a 3 x 4 long tensor filled with
            # the input value.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning a 3 x 4 long tensor filled with
                # the scalar input.
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_fuse_addmm(self):
        # [FUNCTION DOC] Exports a module computing the matrix product of a 3 x 3 ones
        # tensor with itself plus the tensor, a pattern fusable into addmm, and compares
        # ORT with PyTorch.
        class AddmmModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning x matmul x plus x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the input with itself
                # plus the input.
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_maxpool(self):
        # [FUNCTION DOC] Exports MaxPool1d with kernel 2 and stride 1 on a 20 x 16 x 50 input.
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_conv(self):
        # [FUNCTION DOC] Exports a module holding 1D, 2D and 3D convolutions applied to
        # three separate inputs and returning all three outputs. Compares ORT with PyTorch
        # using atol 1e-4.
        class TraceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three convolutions (1D with stride 2; 2D
            # with stride, padding and dilation; 3D with stride and padding), each mapping
            # 16 to 33 channels.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three convolution submodules.
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                # [FUNCTION DOC] HELPER: Applies each convolution to its own input and returns
                # the three results as a tuple.
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_str_padding(self):
        # [FUNCTION DOC] Same structure as the convolution test but uses string padding: valid
        # padding for the 1D and 2D convolutions and same padding for the 3D one.
        class TraceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three convolutions using string padding valid,
            # valid and same.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three convolution submodules.
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, padding="valid")
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=1, padding="valid", dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=1, padding="same"
                )

            def forward(self, input1, input2, input3):
                # [FUNCTION DOC] HELPER: Applies each convolution to its own input and returns
                # the three results as a tuple.
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_shape_inference(self):
        # [FUNCTION DOC] Checks shape inference through a 2D convolution followed by adding 2,
        # with a dynamic batch axis on the input and atol 1e-4.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 2D convolution (16 to 33 channels) whose
            # output is offset by 2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 2D convolution submodule.
                super().__init__()
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the convolution output plus 2.
                return self.conv2(input) + 2

        x = torch.randn(20, 16, 50, 100)
        self.run_test(
            Model(), x, atol=10e-5, input_names=["x"], dynamic_axes={"x": [0]}
        )

    def test_conv_transpose(self):
        # [FUNCTION DOC] Exports a module with 1D, 2D and 3D transposed convolutions applied to
        # three separate inputs and compares ORT with PyTorch with atol 1e-4.
        class TraceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three transposed convolutions (1D, 2D, 3D) mapping
            # 16 to 33 channels, with stride and, for 2D and 3D, padding and dilation settings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three transposed convolution submodules.
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                # [FUNCTION DOC] HELPER: Applies each transposed convolution to its own input
                # and returns the three results as a tuple.
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 10)
        x2 = torch.randn(20, 16, 10, 10)
        x3 = torch.randn(20, 16, 10, 10, 10)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_numpy_T(self):
        # [FUNCTION DOC] Exports a module that returns the transpose attribute of a 4 x 7 input.
        class NumpyTranspose(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the numpy-style transpose of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the transposed input.
                return x.T

        self.run_test(NumpyTranspose(), torch.randn(4, 7))

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    def test_transpose_infer_shape(self):
        # [FUNCTION DOC] Checks that transposition after a convolution needs known input rank
        # and works with ONNX shape inference. Exports a scripted module with dynamic batch and
        # height axes and tests with an additional input of different shape.
        class TransposeModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with one 2D convolution (3 to 1 channels, kernel
            # 3, stride 2) followed by swapping dim 0 and dim 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor and creates
                # the convolution submodule.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method applying the convolution then swapping
                # the first two dimensions.
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(
            TransposeModule(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=[y],
        )

    def squeeze_model_tests(self, d, x1, x2):
        # [FUNCTION DOC] HELPER: Shared body for squeeze tests. Wraps the squeeze operator in a
        # module configured with dimension d (or None to squeeze all size-one dims). If x2 is
        # given it is used as an additional test input with a dynamic input axes mapping on
        # three dims; otherwise runs a simple test with x1 only.
        class Squeeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a dimension (or None) and squeezing that
            # dimension, or all size-one dimensions when it is None.
            def __init__(self, d):
                # [FUNCTION DOC] HELPER: Stores the dimension to squeeze.
                super().__init__()
                self.d = d

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Squeezes the stored dimension when it is not None,
                # otherwise squeezes every size-one dimension.
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)

        x2 = [] if x2 is None else [x2]
        if len(x2) > 0:
            self.run_test(
                Squeeze(d),
                x1,
                input_names=["input"],
                dynamic_axes={"input": {0: "0", 1: "1", 2: "2"}},
                additional_test_inputs=x2,
            )
        else:
            self.run_test(Squeeze(d), x1)

    def test_squeeze_without_no_op(self):
        # [FUNCTION DOC] Squeeze dim 1 of a 2 x 1 x 4 tensor with no dynamic test input.
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(1, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_dynamic(self):
        # [FUNCTION DOC] Squeeze dim 1 with a dynamic input where the extra test input has size
        # 2 in dim 1 so the squeeze is a no-op. Requires opset 11 or newer.
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(1, x_squeeze, x_noop)

    def test_squeeze_neg_without_no_op(self):
        # [FUNCTION DOC] Squeeze dim -2 of a 2 x 1 x 4 tensor with no dynamic test input.
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_neg(self):
        # [FUNCTION DOC] Squeeze dim -2 with a dynamic extra input where the dim is not size
        # one. Requires opset 11 or newer.
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    def test_squeeze_all_dims(self):
        # [FUNCTION DOC] Squeeze with no dimension specified, using a dynamic extra input of
        # shape 2 x 2 x 3 that has no size-one dims.
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_no_op(self):
        # [FUNCTION DOC] Squeeze dim 2 where the first input has size 1 at dim 1 (so the squeeze
        # is a no-op) and the extra input has size 1 at dim 2. Requires opset 11 or newer.
        x_noop = torch.randn(2, 1, 4)
        x_squeeze = torch.randn(2, 2, 1)
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_runtime_dim(self):
        # [FUNCTION DOC] Exports a module that creates a zero matrix with sizes taken from
        # tensor inputs and squeezes dim 0, run with a size-one and a larger first dimension in
        # the main and additional inputs, in both orders. Requires opset 11 or newer.
        class Squeeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building a zeros matrix whose shape comes from the
            # first element of each input, then squeezing dim 0.
            def forward(self, d1, d2):
                # [FUNCTION DOC] HELPER: Returns the zeros matrix of shape given by the first
                # elements of d1 and d2, squeezed on dim 0.
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)])
        self.run_test(Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)])

    def test_squeeze(self):
        # [FUNCTION DOC] Exports a squeeze of dim -2 on a 2 x 1 x 4 tensor.
        class Squeeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squeezing dim -2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input squeezed on dim -2.
                return torch.squeeze(x, dim=-2)

        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_squeeze_dynamic_dim(self):
        # [FUNCTION DOC] Exports a squeeze with the dimension passed as a runtime integer input.
        # Requires opset 13 or newer.
        class Squeeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squeezing on a dimension given as an integer argument.
            def forward(self, x, dim: int):
                # [FUNCTION DOC] HELPER: Returns the input squeezed on the given dimension.
                return torch.squeeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = 1
        self.run_test(Squeeze(), (x, dim))

    def test_unsqueeze(self):
        # [FUNCTION DOC] Exports an unsqueeze at dim -2 of a 2 x 3 x 4 tensor.
        class Unsqueeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unsqueezing dim -2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input with a size-one dim inserted at -2.
                return torch.unsqueeze(x, dim=-2)

        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_unsqueeze_dynamic_dim(self):
        # [FUNCTION DOC] Exports unsqueeze with the dimension passed as an integer input equal
        # to -1. Requires opset 13 or newer.
        class Unsqueeze(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unsqueezing at a dimension given as
            # an integer argument.
            def forward(self, x, dim: int):
                # [FUNCTION DOC] HELPER: Returns the input with a size-one dim inserted at
                # the given dimension.
                return torch.unsqueeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = -1
        self.run_test(Unsqueeze(), (x, dim))

    def test_maxpool_default_stride(self):
        # [FUNCTION DOC] Exports functional 2D max pooling with kernel 2 and default stride on a
        # 10 x 20 x 16 x 50 input.
        class MaxPoolModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional max_pool2d with kernel 2 and
            # the default stride.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the max pool of the input.
                return torch.nn.functional.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_adaptive(self):
        # [FUNCTION DOC] Exports AdaptiveMaxPool1d with output size 5 and dynamic batch axis,
        # with an additional input of batch 32. Requires opset 8 or newer.
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        y = torch.randn(32, 16, 50, requires_grad=True)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[y],
        )

    def test_maxpool_2d(self):
        # [FUNCTION DOC] Exports MaxPool2d with kernel 5 and padding of 1 and 2 on a 1 x 20 x 16
        # x 50 input.
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        # [FUNCTION DOC] Exports MaxPool1d with kernel 3, stride 2 and ceil mode.
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        # [FUNCTION DOC] Exports MaxPool2d with kernel 3, stride 2 and ceil mode.
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        # [FUNCTION DOC] Exports MaxPool3d with kernel 3, stride 2 and ceil mode on a 5D input.
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dynamic(self):
        # [FUNCTION DOC] Exports max pooling with ceil mode, a convolution and a batch norm with
        # dynamic height and width axes on both input and output. Requires opset 10 or newer.
        class test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying MaxPool2d (kernel 2, stride 2, ceil
            # mode), a 1x1 bias-free convolution and a batch norm created through a partial
            # with eps 0.0009.
            def __init__(self, in_channels, out_channels):
                # [FUNCTION DOC] HELPER: Builds the pooling, convolution and norm submodules for
                # the given channel counts.
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the norm applied to the convolution of
                # the pooled input.
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    # TODO: Enable maxpool-ceil family after ONNX 1.15.1+ is bumped
    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_1d_ceil_corner(self):
        # [FUNCTION DOC] Exports MaxPool1d with kernel 1, stride 2 and ceil mode on a 1 x 3 x 32
        # input, a corner case where ceil mode matters. Only runs for opset 9 or older.
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_2d_ceil_corner(self):
        # [FUNCTION DOC] Exports MaxPool2d with kernel 1x1, stride 2x2 and ceil mode on a 1 x 3
        # x 32 x 32 input. Only runs for opset 9 or older.
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_3d_ceil_corner(self):
        # [FUNCTION DOC] Exports MaxPool3d with kernel 7x8x4, stride 10x11x3, padding 2 and ceil
        # mode on a 1 x 3 x 51 x 52 x 45 input. Only runs for opset 9 or older.
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_1d_ceil_corner_with_indices(self):
        # [FUNCTION DOC] Same as the 1D ceil-mode corner case but the pooling also returns
        # indices. Runs only for opset 8 to 9.
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=True
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_2d_ceil_corner_with_indices(self):
        # [FUNCTION DOC] Same as the 2D ceil-mode corner case but the pooling also returns
        # indices. Runs only for opset 8 to 9.
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_3d_ceil_corner_with_indices(self):
        # [FUNCTION DOC] Same as the 3D ceil-mode corner case but the pooling also returns
        # indices. Runs only for opset 8 to 9.
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        # [FUNCTION DOC] Exports MaxPool1d with kernel 2, stride 1 and returned indices.
        # Requires opset 8 or newer.
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        # [FUNCTION DOC] Exports MaxPool1d with kernel 2, stride 1 and dilation 2. Requires
        # opset 10 or newer.
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_default_stride(self):
        # [FUNCTION DOC] Exports functional 2D average pooling with kernel 2 and default stride
        # on a 10 x 20 x 16 x 50 input.
        class AvgPoolModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional avg_pool2d with kernel 2 and
            # the default stride.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the average pool of the input.
                return torch.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        # [FUNCTION DOC] Exports AvgPool1d with kernel 2 and stride 1 on a 20 x 16 x 50 input.
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        # [FUNCTION DOC] Exports AvgPool1d with kernel 3, stride 2 and ceil mode on a 1 x 1
        # x 7 input.
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and PyTorch has different calculation for ceil_mode (the last value).
    @common_utils.parametrize(
        "padding",
        (0, 1),
    )
    @common_utils.parametrize(
        "count_include_pad",
        (True, False),
    )
    def test_avgpool_2d(self, padding, count_include_pad):
        # [FUNCTION DOC] Parameterized over padding 0 or 1 and count_include_pad true or false.
        # Exports AvgPool2d with kernel and stride 3 on a 20 x 16 x 50 x 32 input. Ceil mode is
        # deliberately left out because ORT and PyTorch compute the last value differently.
        model = torch.nn.AvgPool2d(
            3,
            3,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and PyTorch has different calculation for ceil_mode (the last value).
    # the issue requires fix in onnx(21) (https://github.com/onnx/onnx/issues/5711)
    # a fix in ORT is planned. After the fixes in place, we can add ceil_mode to the test.
    @skipIfUnsupportedMinOpsetVersion(21)
    def test_avgpool_3d_ceil(self):
        # [FUNCTION DOC] Exports AvgPool3d with kernel 3, stride 2 and ceil mode, with dynamic
        # axes on batch and channel and an additional input of different batch and channel
        # counts. Requires opset 21 or newer.
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        y = torch.randn(32, 8, 50, 44, 31)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_avgpool_dynamic(self):
        # [FUNCTION DOC] Exports average pooling (ceil mode, no padding in the average), a 1x1
        # convolution and batch norm with dynamic height and width on input and output. Requires
        # opset 10 or newer.
        class test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying AvgPool2d (kernel 2, stride 2, ceil mode,
            # padding excluded from the average), a 1x1 bias-free convolution and a batch norm
            # with eps 0.0009.
            def __init__(self, in_channels, out_channels):
                # [FUNCTION DOC] HELPER: Builds the pooling, convolution and norm submodules for
                # the given channel counts.
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.AvgPool2d(
                    (2, 2), stride=2, ceil_mode=True, count_include_pad=False
                )
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the norm applied to the convolution of
                # the pooled input.
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floating_point(self):
        # [FUNCTION DOC] Exports scripted modules using the is-floating-point query on a tensor,
        # both branches returning same-shaped results: one with zeros shaped like the input
        # under dynamic axes (also run with no remaining ONNX inputs), another with a
        # size-dependent branch. Requires opset 9 or newer.
        class FloatingPoint(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that tests whether the input is floating point
            # and in both cases returns zeros shaped like the input.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning a zero tensor with the input's
                # shape regardless of dtype.
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that, when the first dimension exceeds one,
            # checks if input plus 2 is floating point and returns input plus 1 either way;
            # otherwise returns the input.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method with nested size and floating
                # point checks returning the input plus one, or the input unchanged for
                # batch size one.
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    # Operator rank mismatch between outputs of two branches for opsets below 11.
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_floating_point_infer_dtype(self):
        # [FUNCTION DOC] Checks dtype inference for floating-point queries inside branches.
        # First a float input with dynamic axes (and no remaining inputs) where the branches
        # differ in output rank, then an int32 input. Requires opset 11 or newer because of rank
        # mismatch between branches in older opsets.
        class FloatingPoint(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning, for batch greater than one and a
            # floating-point result of adding 2, zeros of the shape without the first dimension,
            # otherwise zeros of full shape; returns the input for batch one.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method with a size branch and a floating-point
                # branch returning zeros of two different ranks, or the input.
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning input plus one when batch exceeds one
            # and the floating-point check passes, else the input.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning input plus one in the
                # floating-point branch of the batch-greater-than-one case, otherwise the input.
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_prim_min(self):
        # [FUNCTION DOC] Exports a scripted helper that loops over a list with enumerate, which
        # creates a min operation in the graph, and a scripted module taking the minimum of an
        # element and an integer. Requires opset 12 or newer.
        @torch.jit.script
        def list_append(boxes: list[Tensor]):
            # [FUNCTION DOC] HELPER: Scripted function. For each box in the list builds a tensor
            # shaped like the box's column 1 filled with the loop index, appending to a list;
            # returns the first entry.
            temp = []
            for i, b in enumerate(
                boxes
            ):  # enumerate is creating a prim::min op in torch graph
                temp.append(torch.full_like(b[:, 1], i))
            return temp[0]

        class Min(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that builds a list of three references to the input and
            # calls the scripted list function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the result of the scripted function on a list
                # containing the input three times.
                boxes = [x for _ in range(3)]
                return list_append(boxes)

        x = torch.rand(5, 5)
        self.run_test(Min(), (x,))

        class M(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the minimum of the element at index 3
            # of the input and the integer 3.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the minimum of the input's
                # fourth element and 3.
                i = 3
                return min(x[i], i)

        x = torch.arange(6, dtype=torch.int64)
        self.run_test(M(), (x,))

    def test_arithmetic(self):
        # [FUNCTION DOC] Exports basic arithmetic (add 2, subtract 4, multiply 6, divide 8) on a
        # random 2 x 3 x 4 tensor.
        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a fixed chain of tensor and scalar arithmetic.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus 2, minus 4, times 6,
                # divided by 8.
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_long(self):
        # [FUNCTION DOC] Exports arithmetic with an integer argument used in the operations, and
        # a second module whose output is just the input's first dimension size after arithmetic
        # (no remaining ONNX inputs).
        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying arithmetic between a tensor and
            # an integer argument.
            def forward(self, x, y: int):
                # [FUNCTION DOC] HELPER: Adds, subtracts, multiplies by three times and divides
                # by four times the integer, returning the tensor.
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the first dimension size after arithmetic
            # on the tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 2 and subtracts 3 from the input and returns its
                # first dimension size.
                x = x + 2
                x = x - 3
                return x.shape[0]

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_float(self):
        # [FUNCTION DOC] Exports arithmetic with a float argument and a module returning a float
        # computed from the input shape; dtype checking is disabled.
        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying arithmetic between a tensor and
            # a float argument.
            def forward(self, x, y: float):
                # [FUNCTION DOC] HELPER: Adds, subtracts, multiplies by three times and divides
                # by four times the float, returning the tensor.
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the second dimension size halved after
            # arithmetic on the tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 2 and subtracts 3 then returns the second
                # dimension size divided by 2.
                x = x + 2
                x = x - 3
                return x.shape[1] / 2

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_bool(self):
        # [FUNCTION DOC] Exports a module with integer, boolean and float arguments (the boolean
        # selects extra multiplication), returning a tensor and the boolean; and a module
        # comparing two integers for equality. Dtype checking is disabled.
        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a tensor, an integer, a boolean flag
            # and a float.
            def forward(self, x, y: int, z: bool, t: float):
                # [FUNCTION DOC] HELPER: Adds then subtracts the integer, conditionally scales
                # by multiplication and division when the flag is true, and returns the result
                # divided by the float together with the flag.
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return x / t, z

        x = torch.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing two integer arguments.
            def forward(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: Returns whether the two integers are equal.
                return x == y

        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    @skipScriptTest(
        15,
        reason="In trace: Outputs that are always None are removed. \
                In script: Outputs that are always None are removed before opset 15. \
                After opset 15, we replace the None in output with Optional node.",
    )
    def test_tuple_with_none_outputs(self):
        # [FUNCTION DOC] Exports a module returning nested tuples that contain None values.
        # Skipped for scripting from opset 15 on, because None outputs are removed when tracing
        # and before opset 15 in scripting, and replaced by optional nodes later.
        class TupleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a nested tuple containing the input
            # and None placeholders.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of the input and a tuple with None and
                # a nested pair of the input and None.
                return (x, (x, None, (x, None)))

        x = torch.randn(3, 4)
        self.run_test(TupleModel(), (x,))

    # In scripting the first transpose node do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_arithmetic_infer_dtype(self):
        # [FUNCTION DOC] Checks dtype inference for arithmetic after a transpose in a scripted
        # module (needs ONNX shape inference since the transpose lacks shape info in scripting).
        class ArithmeticModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule transposing the input then applying a
            # fixed arithmetic chain.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the transposed input plus 2,
                # minus 4, times 6, divided by 8.
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3)
        self.run_test(ArithmeticModule(), x)

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div(self):
        # [FUNCTION DOC] Skipped test (floor division on ONNX differs from eager). Would
        # export many floor-division combinations of float, int and double tensors with
        # scalars and tensors.
        class FloorDivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of twelve floor-division results
            # mixing dtypes and scalar or tensor divisors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns floor divisions of the input and of cast
                # versions of it (float64 or int64) by scalars 3 and 2.0 and by tensor y or
                # casts of y.
                return (
                    x // 3,
                    x // 2.0,
                    x.to(dtype=torch.float64) // 3,
                    x.to(dtype=torch.float64) // 2.0,
                    x.to(dtype=torch.int64) // 3,
                    x.to(dtype=torch.int64) // 2.0,
                    x // (y + 1.0).to(dtype=torch.int64),
                    x // y,
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.float64) // y.to(dtype=torch.float64),
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.int64) // y,
                )

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div_script(self):
        # [FUNCTION DOC] Skipped test (floor division mismatch with eager). Would export
        # scripted floor division by 3, 2.0 and a tensor.
        class FloorDivModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning three floor-division results.
            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method returning the input floor-divided by 3,
                # 2.0 and y.
                return x // 3, x // 2.0, x // y

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floordiv(self):
        # [FUNCTION DOC] Skipped test (floor division mismatch with eager, see the skip reason).
        # When enabled, exports a module building zeros whose length is the integer floor
        # division of two dimension sizes, with dynamic axes and with no remaining inputs.
        # Requires opset 9 or newer.
        class FloordivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a new zero tensor whose length is dim 2 size
            # floor-divided by dim 1 size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns new zeros with a length computed by integer
                # floor division of two input dimension sizes.
                return x.new_zeros(x.size(2) // x.size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloordivModule(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    def test_div(self):
        # [FUNCTION DOC] Exports true division with both plain and function-style operators on
        # int32 inputs and then on float inputs, comparing ORT with PyTorch.
        class DivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the division operator result and the
            # true_divide result for two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair: x divided by y and the true division of
                # x by y.
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    # Note: div cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_div_promotion_trace(self):
        # [FUNCTION DOC] Checks type promotion of division for traced models only (script export
        # cannot know the scalar types). Traces the module with int32 inputs and runs it with
        # the default dtype set to float and then double.
        class DivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the division and true division
            # of two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair: x divided by y and the true division of
                # x by y.
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

    # In scripting x, y do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_div_promotion_script(self):
        # [FUNCTION DOC] Checks division type promotion in scripting with shape and type info
        # hidden by transposes: int inputs with default float dtype, int inputs with default
        # double dtype, and an int numerator with double denominator (needs known types).
        # Requires ONNX shape inference.
        class DivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that transposes dims 1 and 2 of both inputs to hide
            # shape and type info, then returns division and true division.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Transposes both inputs on dims 1 and 2 and returns a
                # pair of their division and true division.
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still available from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

    @skipDtypeChecking
    def test_div_rounding_mode(self):
        # [FUNCTION DOC] Exports division with rounding modes none, trunc and floor, each both
        # as a method and as a function. Runs eagerly, traced and scripted on int32 inputs, then
        # on random float inputs. Dtype checking disabled.
        class TrueDivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning division with no rounding mode through the
            # method and function forms.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair of the divisions of x by y with
                # rounding mode None.
                return (
                    x.div(y, rounding_mode=None),
                    torch.div(x, y, rounding_mode=None),
                )

        class TruncDivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning division with truncation rounding
            # in both forms.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair of the divisions of x by y with
                # rounding mode trunc.
                return (
                    x.div(y, rounding_mode="trunc"),
                    torch.div(x, y, rounding_mode="trunc"),
                )

        class FloorDivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning division with floor rounding in both forms.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair of the divisions of x by y with
                # rounding mode floor.
                return (
                    x.div(y, rounding_mode="floor"),
                    torch.div(x, y, rounding_mode="floor"),
                )

        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]

        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

    def test_slice_trace(self):
        # [FUNCTION DOC] Exports slicing a 1D tensor to its first element.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the first element slice of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input sliced from 0 to 1.
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_neg(self):
        # [FUNCTION DOC] Exports a slice starting at -1 of a 3 x 4 x 5 tensor.
        class NegSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module slicing the last entry along dim 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input sliced from -1 to the end.
                return x[-1:]

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        # [FUNCTION DOC] Exports a mixed slice of a 5D tensor with a negative range on dim 2 and
        # an integer -1 index on dim 4.
        class NegSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module slicing dim 2 from -3 to -1 and selecting index -1
            # on dim 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input with those slice and
                # index operations applied.
                return x[:, :, -3:-1, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large_negone(self):
        # [FUNCTION DOC] Exports selecting index -1 on the last axis of a 5D tensor.
        class NegSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting the last index on the final dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input with its last dimension
                # indexed at -1.
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        # [FUNCTION DOC] Exports an in-place slice assignment whose end index comes from the
        # size of another input. Zeros of shape 56 x 6 x 256 get a 22 x 256 tensor written.
        # Requires opset 11 or newer.
        class InputIndexSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a tensor into a slice of the input whose
            # length is the tensor's first dimension size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Writes y into the slice of x up to y's first dimension
                # size at index 0 of dim 1, and returns x (mutated in place).
                x[: y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Torchscript doesn't support 1d index.
    def test_slice_with_1d_input_index(self):
        # [FUNCTION DOC] Same as the previous test but the slice end is a one-element int64
        # tensor input. Skipped under scripting (one-dimensional index not supported); requires
        # opset 11 or newer.
        class InputIndexSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a tensor into a slice whose end is given
            # by a tensor.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Writes y into the slice of x ending at y and returns x
                # (mutated in place).
                x[:y, 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.tensor([5], dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_step_size(self):
        # [FUNCTION DOC] Exports an in-place slice assignment with end and step supplied
        # as scalar int64 tensor inputs, writing 1 into the selected region. Requires
        # opset 11 or newer.
        class InputIndexSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning the constant 1 into a strided slice with
            # tensor end and step.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Sets two strided slices of x to 1 using y as end and z
                # as step, and returns x (mutated in place).
                x[:y:z, 0::z, :] = 1
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.tensor(5, dtype=torch.int64)
        z = torch.tensor(2, dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic(self):
        # [FUNCTION DOC] Exports slices with sizes depending on the input shape inside a loop of
        # four iterations, returning a tuple of results; uses dynamic axes on all dims of input
        # and output and an additional input of different size. Skipped under scripting;
        # requires opset 10 or newer.
        class DynamicSliceExportMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of four dynamically sliced views
            # of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: For i from 0 to 3 appends a slice with a
                # shape-dependent end on dim 0 and a start of i on dims 1 and 2, then returns
                # the results as a tuple.
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        # [FUNCTION DOC] Exports a scripted module that slices dim 0 starting at 1 up to the
        # size of dim 1, on a 1 x 2 input. Requires opset 10 or newer.
        class DynamicSliceModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule slicing the first dimension from 1 up to the size
            # of the second dimension.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the dynamically sliced input.
                return x[1 : x.size(1)]

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        # [FUNCTION DOC] Exports a module building zeros from a sliced shape (dims one to the
        # size of dim 2), run with dynamic axes on all dims and with no remaining ONNX input.
        # Requires opset 10 or newer.
        class DynamicSliceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning zeros whose shape is a slice of
            # the input shape.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns new zeros of the shape sliced from index 1 to
                # the size of dim 2.
                return x.new_zeros(x.shape[1 : x.size(2)])

        x = torch.rand(1, 2, 3, 4)
        self.run_test(
            DynamicSliceModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]}
        )
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic_to_end(self):
        # [FUNCTION DOC] Exports four slices that go to the end on one axis and index by
        # input shape on another, returned as a tuple. Skipped under scripting; requires
        # opset 10 or newer.
        class DynamicSliceExportMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of four slices of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: For i from 0 to 3 appends a slice of dims 1 from
                # i and indexes dim 2 at a position derived from its size minus five;
                # returns the tuple.
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    def test_square(self):
        # [FUNCTION DOC] Exports element-wise square on a random 2 x 3 x 4 tensor.
        class Square(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squaring its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the square of the input.
                return torch.square(x)

        x = torch.randn(2, 3, 4)
        self.run_test(Square(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        # [FUNCTION DOC] Exports arange with dynamic and constant bounds, traced and scripted,
        # checking against an additional input of different size. Requires opset 9 or newer.
        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning three ranges: from the first dimension size,
            # of fixed length 12, and between the size and size plus five.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of three arange results as described.
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test(
            ArangeModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )
        self.run_test(
            torch.jit.script(ArangeModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        # [FUNCTION DOC] Exports arange with a tensor end and an output tensor argument for a
        # scalar int64 input of 8. Requires opset 9 or newer.
        class ArangeOutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange up to a tensor end with an out tensor.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Creates a one-element int64 out tensor and returns
                # arange from 0 to end written into it.
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        # [FUNCTION DOC] Exports arange with start taken from the size of one input and tensor
        # end, with an out tensor; run with dynamic axes and with only the first input
        # remaining. Requires opset 9 or newer.
        class ArangeStartOutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to a tensor
            # end with an out tensor.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Creates a one-element int64 out tensor and returns
                # arange from the size of dim 0 of start to end written into it.
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace(self):
        # [FUNCTION DOC] Exports linspace with float start 3, float end 10 and five int steps
        # given as tensors. Requires opset 9 or newer.
        class LinspaceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing linspace from tensor start, end and steps.
            def forward(self, start, end, steps):
                # [FUNCTION DOC] HELPER: Returns evenly spaced values between start and end with
                # the given number of steps.
                return torch.linspace(start, end, steps)

        x = torch.tensor(3, dtype=torch.float)
        y = torch.tensor(10, dtype=torch.float)
        z = torch.tensor(5, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace_negative_start(self):
        # [FUNCTION DOC] Exports linspace from -1 to 1 with six steps. Requires opset
        # 9 or newer.
        class LinspaceModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing linspace from tensor start, end and steps.
            def forward(self, start, end, steps):
                # [FUNCTION DOC] HELPER: Returns evenly spaced values between start and end with
                # the given number of steps.
                return torch.linspace(start, end, steps)

        x = torch.tensor(-1, dtype=torch.float)
        y = torch.tensor(1, dtype=torch.float)
        z = torch.tensor(6, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_out(self):
        # [FUNCTION DOC] Exports arange with float arguments and an out tensor: one with only an
        # end of 8.5, one with a size-based start, end 8.5 and step 1.5. The second runs with
        # dynamic axes and with limited remaining inputs. Requires opset 9 or newer.
        class ArangeModelEnd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange up to a float tensor end with a
            # float out tensor.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Creates a one-element float out tensor and returns
                # arange to end written into it.
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to an end
            # tensor with step 1.5 and a float out tensor.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Creates a float out tensor and returns arange from the
                # size of dim 0 of start to end with step 1.5 into it.
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, 1.5, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats(self):
        # [FUNCTION DOC] Exports arange with float end, with start and float step, with a
        # negative step, and with start only, each in its own module, using dynamic axes for the
        # size-based variants and a restricted remaining-input list. Requires opset 9 or newer.
        class ArangeModelEnd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange up to a float tensor end.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Returns arange from 0 to end.
                return torch.arange(end)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to an end
            # tensor with step 1.5.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Returns arange from the size of dim 0 of start to end
                # with step 1.5.
                return torch.arange(start.size(0), end, 1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStepNeg(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing a descending arange from a tensor end to a
            # size-based stop with step -1.5.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Returns arange from end down to the size of dim 0 of
                # start with step -1.5.
                return torch.arange(end, start.size(0), -1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStepNeg(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStart(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to
            # a tensor end.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Returns arange from the size of dim 0 of start to end.
                return torch.arange(start.size(0), end)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStart(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_override(self):
        # [FUNCTION DOC] Exports arange with float tensor bounds but an explicit int64
        # dtype override, for end only and for a size-based start with step 1.5. Requires
        # opset 9 or newer.
        class ArangeModelEnd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange up to a float end with dtype int64.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Returns arange from 0 to end cast to int64.
                return torch.arange(end, dtype=torch.int64)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to a tensor
            # end with step 1.5 and dtype int64.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Returns arange from the size of dim 0 of start to end
                # with step 1.5 and int64 dtype.
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_out(self):
        # [FUNCTION DOC] Exports arange with a float tensor end of 8.5 and a float out tensor.
        # Requires opset 11 or newer.
        class ArangeOutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange up to a float end with an out tensor.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Creates a one-element float out tensor and returns
                # arange from 0 to end written into it.
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_out(self):
        # [FUNCTION DOC] Exports arange with a size-based start, a float tensor end and a float
        # out tensor; run with dynamic axes and with only the second input remaining. Requires
        # opset 11 or newer.
        class ArangeStartOutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing arange from a size-based start to a float end
            # with an out tensor.
            def forward(self, start, end):
                # [FUNCTION DOC] HELPER: Creates a one-element float out tensor and returns
                # arange from the size of dim 0 of start to end written into it.
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_no_type(self):
        # [FUNCTION DOC] Exports arange with only an end and with a start of 0 and an end, for a
        # float scalar tensor 6.2 where no dtype is given. Requires opset 11 or newer.
        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two aranges over the same float end, with and
            # without an explicit start.
            def forward(self, end):
                # [FUNCTION DOC] HELPER: Returns a pair of arange results from 0 to end.
                return torch.arange(end), torch.arange(0, end)

        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        # [FUNCTION DOC] Exports size-dependent creation ops (aranges over two dimension sizes
        # and a ones tensor of the input shape), run with dynamic axes and with no remaining
        # ONNX inputs. Requires opset 9 or newer.
        class SizeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning arange over dim 0, arange over the last dim
            # and a ones tensor of the input shape.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of those three results.
                return (
                    torch.arange(input.size(0)),
                    torch.arange(input.size(-1)),
                    torch.ones(input.shape),
                )

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # x.stride() not scriptable
    def test_as_strided(self):
        # [FUNCTION DOC] Exports as_strided, once with fixed size, stride and offset and once
        # with a size and stride derived from the input. Skipped under scripting since stride is
        # not scriptable; requires opset 9 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two as_strided views of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a size list with dim 1 doubled minus one and a
                # stride list with dim 1 stride halved. Returns a fixed 3x3x3 strided view with
                # storage offset 2 and a view using the computed size and stride.
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided(
                    (3, 3, 3), (1, 4, 2), storage_offset=2
                ), x.as_strided(chunk_size, chunk_stride)

        x = torch.randn(5, 8, 7)
        self.run_test(Model(), x)

    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_tensor_index_advanced_indexing_ellipsis(self):
        # [FUNCTION DOC] Exports advanced indexing with an ellipsis followed by two index
        # tensors on a 5D input. Skipped under scripting.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing with an ellipsis and two integer tensors.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input indexed with an ellipsis and index
                # tensors on the last two dims.
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        # [FUNCTION DOC] Exports three advanced-indexing modules on a 5D tensor
        # combining slices, index tensors of differing shapes, and a new-axis entry;
        # compares ORT with PyTorch.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing with slices and three index tensors (one 2D).
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indexed input.
                return input[
                    :,
                    torch.tensor([[0, 2], [1, 1]]),
                    :,
                    torch.tensor([2, 1]),
                    torch.tensor([0, 3]),
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing with a slice, an index tensor, a new axis, a
            # range slice and a 2D index tensor.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indexed input.
                return input[
                    :, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])
                ]

        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing with a slice, two index tensors that
            # broadcast, a range slice and a 2D index tensor.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indexed input.
                return input[
                    :,
                    torch.tensor([0, 2]),
                    torch.tensor([1]),
                    2:4,
                    torch.tensor([[1], [4]]),
                ]

        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        # [FUNCTION DOC] Exports advanced indexing with adjacent index tensors followed by a new
        # axis on a 5D tensor.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a slice, two consecutive index tensors and
            # a new-axis entry.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indexed input.
                return input[
                    :, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        # [FUNCTION DOC] Exports an in-place index assignment of a vector into row 1 of a 3 x 4
        # tensor. Requires opset 11 or newer.
        class IndexPutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning an update into positions of the input given
            # by an index tensor.
            def forward(self, x, ind, update):
                # [FUNCTION DOC] HELPER: Sets the entries of x at the given index to the update
                # and returns x (mutated in place).
                x[ind] = update
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor([1], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_singular(self):
        # [FUNCTION DOC] Exports index assignment with a single index tensor, first setting
        # booleans true in a mask of 100 entries, then setting floats to 5.5 in a random vector.
        # Requires opset 11 or newer.
        class IndexPutBoolModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module setting entries of a boolean mask at given
            # indices to true.
            def forward(self, mask, indices):
                # [FUNCTION DOC] HELPER: Sets the mask at the indices to true and returns the
                # mask (mutated in place).
                mask[indices] = True
                return mask

        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module setting entries of a float vector at given indices to a
            # scalar tensor 5.5.
            def forward(self, mask, indices):
                # [FUNCTION DOC] HELPER: Sets the entries at the indices to 5.5 and returns the
                # tensor (mutated in place).
                mask[indices] = torch.tensor(5.5)
                return mask

        mask = torch.rand(100, dtype=torch.float)
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        # [FUNCTION DOC] Exports index_put with accumulate enabled, adding a vector of ones into
        # row 2 of a 3 x 4 tensor. Requires opset 11 or newer.
        class IndexPutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using index_put with accumulation.
            def forward(self, x, ind, update):
                # [FUNCTION DOC] HELPER: Returns x with the update accumulated at the index.
                return x.index_put((ind,), update, accumulate=True)

        x = torch.randn(3, 4)
        ind = torch.tensor([2], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_slice_index(self):
        # [FUNCTION DOC] Exports ten variants of in-place updates mixing slices, integer indices
        # and index tensors, with add-assign and plain assignment, including broadcast of the
        # update and a box-coordinate pattern. Each module is checked against ORT with a 3 x 4 x
        # 5 input. Requires opset 11 or newer.
        class IndexPutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update to a region selected by two slices and
            # an index tensor.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into that region of x and returns x
                # (mutated in place).
                x[1:2, 1:3, torch.tensor([1])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update at positions picked by
            # two index tensors.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update at the paired indices and returns x
                # (mutated in place).
                x[torch.tensor([0, 2]), torch.tensor([1, 2])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update at an index tensor combined
            # with a slice.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into x at those positions and returns x
                # (mutated in place).
                x[torch.tensor([0, 2]), 1:2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update at an index tensor combined with
            # an integer index.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into x at those positions and returns x
                # (mutated in place).
                x[torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update at a slice, an index tensor and
            # an integer index.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into x at those positions and returns x
                # (mutated in place).
                x[1:3, torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning an update to rows 1 and 2 at column index 0
            # using a slice and an integer.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Assigns the update into x at those positions and
                # returns x (mutated in place).
                x[1:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning an update with an open-ended slice start of 1
            # and integer 0.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Assigns the update into x at those positions and
                # returns x (mutated in place).
                x[1:, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning an update with a slice ending at 3
            # and integer 0.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Assigns the update into x at those positions and
                # returns x (mutated in place).
                x[:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(3 * 5).to(torch.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

        class IndexPutModel9(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing bounding boxes from pose coordinates, placing
            # shifted x values in column 0 of a zero boxes tensor.
            def forward(self, poses):
                # [FUNCTION DOC] HELPER: Computes the first pose column minus 15 (integer half
                # of width 32 minus one), allocates zeros of shape batch x 17 x 4, writes the
                # result into column 0 and returns the boxes.
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        self.run_test(IndexPutModel9(), (x,))

        class IndexPutModel10(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a vector broadcast to a four-dimensional
            # shape into a region selected by an index tensor and a slice.
            def forward(self, x, ind, update):
                # [FUNCTION DOC] HELPER: Reshapes and expands the update to 2 x 2 x 2 x 5,
                # assigns it into x at the index and slice and returns x (mutated in place).
                x[ind, 1:3] = update.view(1, 1, 1, 5).expand(2, 2, 2, 5)
                return x

        x = torch.randn(3, 4, 5)
        ind = torch.tensor([[0, 2], [1, 1]])
        update = torch.randn(5)
        self.run_test(IndexPutModel10(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_index_put_ellipsis(self):
        # [FUNCTION DOC] Exports in-place add-assign with an ellipsis, index tensor and slice,
        # on a 5D input, with and without a leading integer index. Skipped under scripting;
        # requires opset 11 or newer.
        class IndexPutModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update into an ellipsis region with an index
            # tensor and a slice.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into x in place and returns x.
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an update with a leading integer index, an
            # ellipsis, an index tensor and a slice.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Adds the update into x in place and returns x.
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @unittest.skip(
        "regression in 1.18: https://github.com/microsoft/onnxruntime/issues/20855"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_loop(self):
        # [FUNCTION DOC] Skipped test (regression in ONNX Runtime 1.18). When enabled, exports a
        # model calling a scripted function that fills a bias matrix with repeated in-place
        # updates inside loops, with dynamic input axes and an additional input. Requires opset
        # 11 or newer.
        @torch.jit.script
        def ngram_attention_bias(
            sequence_length: int, ngram: int, device: torch.device, dtype: torch.dtype
        ):
            # [FUNCTION DOC] HELPER: Scripted function creating a matrix of shape ngram x
            # sequence_length filled with negative infinity, then in two nested loops repeatedly
            # scales and overwrites entries by index, and returns it.
            bias = torch.ones(
                (ngram, sequence_length), device=device, dtype=dtype
            ) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5

            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class ScriptModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the scripted bias function with fixed
            # target positions 512 and ngram 2 and cropping the result to the sequence
            # length of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the ngram count 2 and the maximum
                # target positions 512.
                super().__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: Reads the sequence length from the input shape, builds
                # the bias matrix with the input's device and dtype and returns its columns up
                # to the sequence length.
                seq_length, batch_size = hidden_states.shape[:2]
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions,
                    self.ngram,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask

        x = torch.randn(6, 2)
        y = torch.randn(4, 1)
        self.run_test(
            ScriptModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": {0: "seq_length", 1: "batch_size"}},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        # [FUNCTION DOC] Exports in-place copy patterns: assigning rows, mixed slice and select
        # with updates of several shapes, an integer index tensor, and copy_ with a broadcast
        # mask. Each module is compared with ORT. Requires opset 11 or newer.
        class CopyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning data to rows 1 and 2.
            def forward(self, x, data):
                # [FUNCTION DOC] HELPER: Assigns data into x at rows 1 to 2 and returns x
                # (mutated in place).
                x[1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        class CopyModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning data into a slice of rows at column 0.
            def forward(self, x, data):
                # [FUNCTION DOC] HELPER: Assigns data into x at rows 1 to 2, column 0 and
                # returns x (mutated in place).
                x[1:3, 0] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning data into row 1 at columns 1 to 2.
            def forward(self, x, data):
                # [FUNCTION DOC] HELPER: Assigns data into x at row 1, columns 1 to 2, and
                # returns x (mutated in place).
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning data into the row given by a
            # scalar index tensor.
            def forward(self, x, ind, data):
                # [FUNCTION DOC] HELPER: Assigns data into x at the index and returns x
                # (mutated in place).
                x[ind] = data
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(torch.nn.Module):
            # [CLASS DOC] HELPER: Module copying a mask into the input in place when the
            # mask is present.
            def forward(self, x, mask):
                # [FUNCTION DOC] HELPER: If mask is not None copies it into x with broadcast and
                # returns x; otherwise returns nothing.
                if mask is not None:
                    x.copy_(mask)
                    return x

        x = torch.randn(3, 4)
        mask = torch.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Model not scriptable (output with shape doesn't match the broadcast shape)
    def test_copy_tracing(self):
        # [FUNCTION DOC] Traced-only copy test where a 1 x 2 update broadcasts into row 1
        # columns 1 to 2. Not scriptable because the output shape does not match the broadcast
        # shape; requires opset 11 or newer.
        class CopyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning data into row 1, columns 1 to 2.
            def forward(self, x, data):
                # [FUNCTION DOC] HELPER: Assigns data into x at those positions and returns x
                # (mutated in place).
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis(self):
        # [FUNCTION DOC] Exports in-place assignment of a one-element update at the last-but-one
        # position using an ellipsis, with a 3D and then a 5D input. Requires opset 11 or newer.
        class CopyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning an update at index 1 on the last axis
            # via an ellipsis.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Assigns the update into x at the ellipsis and index 1
                # and returns x (mutated in place).
                x[..., 1] = update
                return x

        x = torch.randn(2, 3, 4)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

        x = torch.randn(2, 3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis_script(self):
        # [FUNCTION DOC] Exports a copy using an ellipsis where the input is first reshaped so
        # shape and type information is unavailable in scripting without ONNX shape inference.
        # Requires opset 11 or newer.
        class CopyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes the input then assigns an update in
            # an ellipsis region.
            def forward(self, x, update):
                # [FUNCTION DOC] HELPER: Reshapes x to 4 x 3 x 5 x 6, writes the update at index
                # 2, ellipsis, columns 1 to 2 of the last axis, and returns the reshaped tensor.
                # Insert reshape node to ensure no shape/type info for
                # x in scripting, without onnx shape inference.
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x

        x = torch.randn(3, 4, 5, 6)

        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        # [FUNCTION DOC] Exports flip over a single axis. Requires opset 10 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flipping the input along dim 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input reversed along dim 0.
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint(self):
        # [FUNCTION DOC] Exports a module that adds a zero-weighted random integer tensor to the
        # input, so only shape matters for correctness. Requires opset 9 or newer.
        class RandInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing random integers in the range 1 to 9 with the
            # input shape, multiplied by zero and added to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns zero times a random integer tensor of the input
                # shape plus the input.
                randint = torch.randint(1, 10, x.shape)
                x = 0 * randint + x
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_value(self):
        # [FUNCTION DOC] Exports randint whose range of 3 to 4 only allows the value 3, so
        # results are deterministic. Requires opset 9 or newer.
        class RandInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a random integer tensor restricted to the single
            # value 3 to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns randint with low 3 and high 4 of the input
                # shape plus the input.
                # This randint call always returns 3
                return torch.randint(3, 4, x.shape) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_like(self):
        # [FUNCTION DOC] Exports randint_like with range 3 to 4, which always yields 3. Requires
        # opset 9 or newer.
        class RandInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding randint_like values restricted to 3
            # to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns randint_like of the input with range 3 to 4
                # plus the input.
                # This randint call always returns 3
                return torch.randint_like(x, 3, 4) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randn(self):
        # [FUNCTION DOC] Exports a module scaling the input by the first dimension size of a
        # random normal tensor plus the input, so the result is deterministic.
        class RandN(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the input by the first dim size of a random
            # normal tensor added to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0 of the sum of
                # a random 2 x 3 x 4 normal tensor and the input.
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand(self):
        # [FUNCTION DOC] Same as the randn test but with a uniform random tensor.
        class Rand(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the input by the first dim size of a random
            # uniform tensor added to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0 of the sum of
                # a random 2 x 3 x 4 uniform tensor and the input.
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        # [FUNCTION DOC] Checks that a random normal tensor created with double dtype keeps
        # that dtype in the exported graph. The result is multiplied by a zero float tensor
        # to stay deterministic.
        class RandN(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the float32 input by a double random normal
            # tensor and a zero scalar.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the input cast to float32, a
                # random double normal tensor and a zero float32 tensor.
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand_dtype(self):
        # [FUNCTION DOC] Same as the previous dtype test but with uniform random values.
        class Rand(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the float32 input by a double random
            # uniform tensor and a zero scalar.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the input cast to float32, a
                # random double uniform tensor and a zero float32 tensor.
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randn_dynamic_size(self):
        # [FUNCTION DOC] Exports randn with a size taken from the input size, used only for its
        # second dimension. Requires opset 9 or newer.
        class RandN(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling the input by the second dim size of a random
            # normal tensor shaped like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 1 of a random
                # normal tensor of the input size.
                return torch.mul(x, torch.randn(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_rand_dynamic_size(self):
        # [FUNCTION DOC] Exports rand with a size taken from the input size. Requires opset
        # 9 or newer.
        class Rand(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling the input by the second dim size of a random
            # uniform tensor shaped like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 1 of a random
                # uniform tensor of the input size.
                return torch.mul(x, torch.rand(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_like(self):
        # [FUNCTION DOC] Exports randn_like traced and scripted, using only the size
        # of the result.
        class RandNLike(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling the input by the first dim size of a random
            # normal tensor like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0 of a random
                # normal tensor shaped like the input.
                return torch.mul(x, torch.randn_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)
        self.run_test(torch.jit.script(RandNLike()), x)

    def test_rand_like(self):
        # [FUNCTION DOC] Exports rand_like traced and scripted, using only the size
        # of the result.
        class RandLike(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling the input by the first dim size of a random
            # uniform tensor like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0 of a random
                # uniform tensor shaped like the input.
                return torch.mul(x, torch.rand_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)
        self.run_test(torch.jit.script(RandLike()), x)

    def test_randn_like_dtype(self):
        # [FUNCTION DOC] Checks that randn_like with double dtype keeps double in
        # the exported graph.
        class RandNLike(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the float32 input by a double randn_like
            # tensor and a zero scalar.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the float32 input, a double
                # random normal tensor like it and a zero float32 tensor.
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        # [FUNCTION DOC] Checks that rand_like with double dtype keeps double in
        # the exported graph.
        class RandLike(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the float32 input by a double rand_like
            # tensor and a zero scalar.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the float32 input, a double
                # random uniform tensor like it and a zero float32 tensor.
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        # [FUNCTION DOC] Exports bernoulli sampling from probabilities given by the input, with
        # float32 and double inputs, only using the sampled tensor size.
        class Bernoulli(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling the input by the first dim size of a bernoulli
            # sample drawn with the input as probabilities.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0 of
                # the bernoulli sample.
                return torch.mul(x, torch.bernoulli(x).size(0))

        x = torch.empty(3, 3).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

        x = torch.empty(2, 3, 3, dtype=torch.double).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

    def test_bernoulli_p(self):
        # [FUNCTION DOC] Exports bernoulli sampling with a float probability 0.2 and with a
        # tensor probability, for float32 and double inputs.
        class Bernoulli_float(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using bernoulli with a fixed probability of 0.2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0
                # of the sample.
                return torch.mul(x, torch.bernoulli(x, 0.2).size(0))

        class Bernoulli_tensor(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using in-place bernoulli with a tensor of probabilities
            # on a random tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the size of dim 0
                # of the sample.
                return torch.mul(x, torch.rand_like(x).bernoulli_(x).size(0))

        x = torch.rand(3, 3)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

        x = torch.rand(2, 3, 3, dtype=torch.double)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

    @unittest.skip("Bug in ORT, skip test until rel-1.11.")
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        # [FUNCTION DOC] Skipped test (ORT bug) that would reshape an empty tensor with a zero
        # in the target shape. Requires opset 14 or newer.
        class ReshapeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping to 3 x 4 x 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input reshaped to 3 x 4 x 0.
                x = x.reshape(3, 4, 0)
                return x

        x = torch.randn(0, 3, 4)
        self.run_test(ReshapeModel(), x)

    def test_reshape_different_rank(self):
        # [FUNCTION DOC] Exports a reshape from a 4D tensor to a 6D shape with
        # one inferred dimension.
        class ReshapeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping to a rank-6 shape with an
            # inferred leading dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input reshaped to -1 x 2 x 4 x 4 x 5 x 5.
                x = x.reshape(-1, 2, 4, 4, 5, 5)
                return x

        x = torch.randn(1, 32, 5, 5)
        self.run_test(ReshapeModel(), x)

    def _interpolate(self, x, mode, use_size, is_upsample, align_corners=False):
        # [FUNCTION DOC] HELPER: Builds a module that runs interpolate on x twice, once with a
        # scalar size or scale factor and once with a per-dimension list, using the given mode
        # and use_size, is_upsample and align_corners flags, then runs run_test with atol 1e-6.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding interpolate settings as constants: mode, flags,
            # a scalar and a per-dimension size and scale (2.0 or 0.5, size 24 or 2, and lists
            # sized by the input rank). Returns two interpolations.
            __constants__ = [
                "mode",
                "use_size",
                "is_upsample",
                "size",
                "scale",
                "size_array",
                "scale_array",
                "align_corners",
            ]

            def __init__(self, mode, use_size, is_upsample, align_corners):
                # [FUNCTION DOC] HELPER: Stores mode and flags, picks scalar scale and size from
                # the upsample flag, and chooses list values by the rank of the outer tensor x
                # (3, 4 or higher).
                super().__init__()
                self.mode = mode
                self.use_size = use_size
                self.is_upsample = is_upsample
                self.align_corners = align_corners
                self.scale = 2.0 if self.is_upsample else 0.5
                self.size = 24 if self.is_upsample else 2
                if x.dim() == 3:
                    self.scale_array = [2.3]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2.3, 3.1]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2.3, 3.1, 4.6]
                    self.size_array = [16, 32, 64]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs two interpolate calls, one with the scalar and
                # one with the list value, using size when use_size is set and scale factor
                # otherwise, passing align_corners only when enabled; returns both outputs
                # as a tuple.
                if self.use_size:
                    if self.align_corners:
                        return torch.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size, align_corners=True
                        ), torch.nn.functional.interpolate(
                            x, mode=self.mode, size=self.size_array, align_corners=True
                        )
                    return torch.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size
                    ), torch.nn.functional.interpolate(
                        x, mode=self.mode, size=self.size_array
                    )
                if self.align_corners:
                    return torch.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale,
                        recompute_scale_factor=False,
                    ), torch.nn.functional.interpolate(
                        x,
                        mode=self.mode,
                        scale_factor=self.scale_array,
                        recompute_scale_factor=False,
                    )
                return torch.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale,
                    recompute_scale_factor=False,
                ), torch.nn.functional.interpolate(
                    x,
                    mode=self.mode,
                    scale_factor=self.scale_array,
                    recompute_scale_factor=False,
                )

        model = MyModel(mode, use_size, is_upsample, align_corners)
        self.run_test(model, x, atol=1e-6)

    def _interpolate_tests(self, is_upsample):
        # [FUNCTION DOC] HELPER: Runs interpolation tests for upsample or downsample. Modes are
        # nearest, linear and bicubic, but only nearest below opset 11. Inputs are 3D, 4D and 5D
        # random tensors. Skips bicubic for non-4D, skips 1D and 3D linear, maps 4D linear to
        # bilinear. Tests with sizes, with align_corners where applicable, and from opset 9 also
        # with scale factors.
        # - cubic mode is not supported for opsets below 11;
        # - linear mode does not match for opsets below 11;
        modes = ["nearest", "linear", "bicubic"]
        if self.opset_version < 11:
            modes = ["nearest"]
        x = [
            torch.randn(1, 2, 6, requires_grad=True),
            torch.randn(1, 2, 4, 6, requires_grad=True),
            torch.randn(1, 2, 4, 4, 6, requires_grad=True),
        ]

        for mode in modes:
            for xi in x:
                mode_i = mode
                # TODO: enable bicubic downsample when ORT precision loss fixed
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 3:
                        # TODO : enable when linear mode is implemented for 1d inputs in ORT
                        continue
                    elif xi.dim() == 4:
                        mode_i = "bilinear"
                    elif xi.dim() == 5:
                        # TODO : enable when linear mode is implemented for 3d inputs in ORT
                        mode_i = "trilinear"
                        continue
                self._interpolate(xi, mode_i, True, is_upsample)
                # test with align_corners if supported
                if mode != "nearest":
                    self._interpolate(xi, mode_i, True, is_upsample, True)
                # the following cases, require dynamic sizes/scales,
                # which which is not supported for opset_version < 9
                if self.opset_version >= 9:
                    self._interpolate(xi, mode_i, True, is_upsample)
                    # test with align_corners if supported
                    if mode != "nearest":
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                    self._interpolate(xi, mode_i, False, is_upsample)

    # ONNX export failed on interpolate scripting because dynamic size not supported for opsets below 9.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_upsample(self):
        # [FUNCTION DOC] Runs the upsampling interpolation tests. Requires opset 9 or newer.
        self._interpolate_tests(True)

    @skipIfUnsupportedMaxOpsetVersion(8)
    @skipScriptTest()  # Scripting supported for opsets > 8. See test_interpolate_upsample
    def test_interpolate_upsample_trace(self):
        # [FUNCTION DOC] Runs the upsampling interpolation tests in tracing mode for opsets up
        # to 8; skipped under scripting.
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_function_substitution(self):
        # [FUNCTION DOC] Checks that interpolate called inside script modules, a script
        # submodule and a scripted function used by a traced module, exports with nearest mode
        # and scale factor 2. Requires opset 9 or newer.
        class ScriptModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule upsampling its input with nearest mode and
            # scale factor 2.0.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the
                # nearest-mode upsampled input.
                return torch.nn.functional.interpolate(
                    x, mode="nearest", scale_factor=2.0
                )

        class ScriptModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that wraps a ScriptModel submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor and creates
                # the interpolation submodule.
                super().__init__()
                self.submodule = ScriptModel()

            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the submodule applied
                # to the input.
                return self.submodule(input)

        x = torch.randn(1, 2, 4, 4, 6)
        self.run_test(ScriptModule(), (x,))

        @torch.jit.script
        def script_method(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the input upsampled with
            # nearest mode and scale factor 2.0.
            return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2.0)

        class TracingModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Traced module that calls the scripted upsampling function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the scripted function applied to the input.
                return script_method(x)

        self.run_test(TracingModule(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        # [FUNCTION DOC] Runs the downsampling interpolation tests. Requires opset 10 or newer.
        self._interpolate_tests(False)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_half_pixel(self):
        # [FUNCTION DOC] Checks the choice between the half pixel and pytorch half pixel
        # coordinate modes. For linear (mapped to bilinear or trilinear by rank) and bicubic (4D
        # only) modes, interpolates each input to a size where one spatial dimension is 1.
        # Requires opset 11 or newer.
        # testing whether it uses "half_pixel" or "pytorch_half_pixel"
        # see https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module interpolating its input to a stored size with
            # a stored mode.
            def __init__(self, mode, size):
                # [FUNCTION DOC] HELPER: Stores the interpolation mode and target size.
                super().__init__()
                self.mode = mode
                self.size = size

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input interpolated to the stored
                # size and mode.
                return torch.nn.functional.interpolate(
                    x, mode=self.mode, size=self.size
                )

        modes = ["linear", "bicubic"]
        x = [
            torch.randn(1, 2, 6, requires_grad=True),
            torch.randn(1, 2, 4, 6, requires_grad=True),
            torch.randn(1, 2, 4, 4, 6, requires_grad=True),
        ]
        for mode in modes:
            for xi in x:
                mode_i = mode
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 4:
                        mode_i = "bilinear"
                    elif xi.dim() == 5:
                        mode_i = "trilinear"
                for i in range(xi.dim() - 2):
                    size = list(xi.shape[2:])
                    size[i] = 1
                    self.run_test(MyModel(mode_i, size), xi)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_no_shape(self):
        # [FUNCTION DOC] Exports a scripted module with interpolate from an input with no static
        # shape info: a bilinear resize to 16 x 16 and a nearest resize to the size of another
        # input. Run with dynamic axes and with only the first input remaining. Requires opset
        # 11 or newer.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that doubles the input then returns a bilinear
            # and a nearest-resized version.
            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method adding x to itself, resizing to a fixed
                # size and to the integer sizes of y, returning both outputs.
                x = torch.add(x, x)
                out1 = torch.nn.functional.interpolate(
                    x, mode="bilinear", size=(16, 16), align_corners=False
                )
                out2 = torch.nn.functional.interpolate(
                    x, mode="nearest", size=(int(y.size(0)), int(y.size(1)))
                )
                return out1, out2

        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        self.run_test(
            MyModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1]},
        )
        self.run_test(MyModel(), (x, y), remained_onnx_input_idx=[0])

    @skipScriptTest()  # scripting raises OnnxRuntimeError
    def test_interpolate_adaptive_pooling_error(self):
        # [FUNCTION DOC] Checks that interpolating with area mode (adaptive pooling) fails
        # during export: expects RuntimeError for both size-based and scale-based upsample.
        # Skipped under scripting, where an ORT error is raised.
        x = torch.randn(1, 2, 6, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", True, True)

        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", False, True)

    def test_groupnorm(self):
        # [FUNCTION DOC] Exports GroupNorm with 3 groups, then 1 group, then 6 groups (one per
        # channel) on 6-channel inputs.
        model = torch.nn.GroupNorm(3, 6, 0.002)
        x = torch.randn(4, 6, 36, 36, 18)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_groupnorm_noaffine(self):
        # [FUNCTION DOC] Exports GroupNorm without affine parameters for several group counts
        # and input shapes.
        model = torch.nn.GroupNorm(4, 8, 0.002, affine=False)
        x = torch.randn(3, 8, 224, 224)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted(self):
        # [FUNCTION DOC] Exports a scripted module that unpacks the input shape into two
        # integers and builds zeros from them; run with dynamic axes and with no remaining ONNX
        # inputs. Requires opset 9 or newer.
        class ListUnpack(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that unpacks a 2D shape into two sizes and returns
            # zeros of that shape.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns new zeros with the two dimension sizes
                # of the input.
                a, b = x.shape
                return x.new_zeros((a, b))

        x = torch.randn(2, 3)
        self.run_test(
            torch.jit.script(ListUnpack()),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        self.run_test(torch.jit.script(ListUnpack()), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted_runs_without_error_with_constructed_list_as_input(
        self,
    ):
        # [FUNCTION DOC] Checks that a scripted module building a list from two inputs and
        # unpacking it exports and runs without error when the constructed list is used; only
        # the first input remains in the ONNX graph. Requires opset 9 or newer.
        class PackUnpack(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that creates a two-element list from its inputs and
            # unpacks it; the docstring shows the expected scripted graph.
            """Create and unpack a list of tensors.

            When scripted, it should produce a graph similar to

            ```
            graph(%self : __torch__.PackUnpack,
                %a.1 : Tensor,
                %b.1 : Tensor):
            %packed.1 : Tensor[] = prim::ListConstruct(%a.1, %b.1)
            %c.1 : Tensor, %8 : Tensor = prim::ListUnpack(%packed.1)
            return (%c.1)
            ```
            """

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Builds a list of the two inputs, unpacks it and returns
                # the first element.
                packed = [a, b]
                c, _ = packed
                return c

        self.run_test(
            torch.jit.script(PackUnpack()),
            (torch.tensor(0), torch.tensor([42])),
            remained_onnx_input_idx=[0],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_slice_scripted(self):
        # [FUNCTION DOC] Exports a scripted module that unpacks a slice of the input shape (the
        # last two dims) into zeros; run with dynamic axes on all four dims and with no
        # remaining inputs. Requires opset 9 or newer.
        class ListUnpackSlice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unpacking the trailing two dimensions of the shape into
            # a zeros matrix.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns new zeros using the sizes from the shape slice
                # starting at dimension 2.
                a, b = x.shape[2:]
                return x.new_zeros((a, b))

        x = torch.randn(2, 3, 4, 5)
        self.run_test(
            torch.jit.script(ListUnpackSlice()),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        self.run_test(
            torch.jit.script(ListUnpackSlice()), x, remained_onnx_input_idx=[]
        )

    @skipDtypeChecking
    def test_pow(self):
        # [FUNCTION DOC] Exports power with tensor exponents across dtype mixes (float, int64
        # with int32, int64, double with int64), a scalar base with tensor exponent for float,
        # int and double, and a power result used as an index. Dtype checking disabled.
        class PowModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising the first tensor to the power of the second.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first input raised to the second.
                return x.pow(y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        x = torch.randn(2, 3, 4).to(dtype=torch.float64)
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        class PowModule2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising the constant 2 to a tensor power.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns 2 raised to the input.
                return torch.pow(2, x)

        x = torch.randn(1, 10)
        self.run_test(PowModule2(), (x,))

        x = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule2(), (x,))

        x = torch.randn(1, 10).to(dtype=torch.float64)
        self.run_test(PowModule2(), (x,))

        class PowModule3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing a vector with the result of 2 raised to
            # a tensor power.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns y indexed by two raised to x.
                return y[torch.pow(2, x)]

        x = torch.randint(5, (2, 3, 4))
        y = torch.rand(100)
        self.run_test(PowModule3(), (x, y))

    # the arithmeticOps(Add\Sub\Mul\Div\Gemm\Pow\Mod) with low precision include unit8 will be failed in ORT
    # add to(dtype=torch.long) to avoid ORT output type does not match expected type.
    # will be fixed in ONNX version 14.
    @skipIfUnsupportedMaxOpsetVersion(13)
    @skipDtypeChecking
    def test_arithmeticOps_with_low_precision(self):
        # [FUNCTION DOC] Exports add, subtract, multiply, divide and power on low-precision
        # types (uint8, int8, int16 and mixes with float and int types) to check ORT output
        # types. Only for opsets up to 13; dtype checking disabled.
        class AddModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module subtracting two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x minus y.
                return x - y

        class MulModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x times y.
                return x * y

        class DivModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x divided by y.
                return x / y

        class PowModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising one tensor to the power of another.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x raised to y.
                return x.pow(y)

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        z = torch.tensor([1], dtype=torch.uint8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        z = torch.tensor([1], dtype=torch.int8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        z = torch.tensor([1], dtype=torch.int16)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float32)
        z = torch.tensor([1], dtype=torch.float64)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int64)
        z = torch.tensor([1], dtype=torch.int32)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

    def test_mul_bool(self):
        # [FUNCTION DOC] Exports multiplication with boolean tensors alone and mixed with
        # floats, in both argument orders.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the product of the two inputs.
                return torch.mul(x, y)

        x_t = torch.tensor([True, False, True, False])
        y_t = torch.tensor([True, True, False, False])
        z_t = torch.tensor([1.0, 2.0, 3.0, 0.0])
        self.run_test(MyModel(), (x_t, y_t))
        self.run_test(MyModel(), (x_t, z_t))
        self.run_test(MyModel(), (z_t, y_t))

    # fmod was added in version 10
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfUnsupportedMaxOpsetVersion(13)
    def test_mod_with_low_precision(self):
        # [FUNCTION DOC] Exports the fmod operator cast to long for low precision integer types
        # (uint8, int8, int16, mixes with int32 and double). Requires opset 10 to 13.
        class ModModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the remainder of two tensors
            # converted to long.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns fmod of the two inputs cast to int64.
                return torch.fmod(x, y).to(dtype=torch.long)

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int32)
        self.run_test(ModModule(), (x, y))

        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float64)
        self.run_test(ModModule(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_empty_constant_shape(self):
        # [FUNCTION DOC] Exports modules that create a zero-rank tensor with zeros, ones, full
        # or empty-then-fill and add a scalar input in place. Requires opset 9 or newer.
        class Zeros(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the input to a zero-rank zeros tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a scalar zero tensor, adds the input in place
                # and returns it.
                y = torch.zeros(())
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Zeros(), x)

        class Ones(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the input to a zero-rank ones tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a scalar ones tensor, adds the input in place
                # and returns it.
                y = torch.ones(())
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Ones(), x)

        class Full(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the input to a zero-rank tensor filled with 1.0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a scalar tensor via full with 1.0, adds the
                # input in place and returns it.
                y = torch.full((), 1.0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Full(), x)

        class Empty(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the input to a zero-rank tensor created
            # uninitialized and filled with 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a scalar tensor, fills it with 0, adds the
                # input in place and returns it.
                y = torch.empty(()).fill_(0)
                y += x
                return y

        x = torch.tensor(42.0)
        self.run_test(Empty(), x)

    def test_std(self):
        # [FUNCTION DOC] Exports the standard deviation over all elements with biased
        # and unbiased estimators.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased standard deviation of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the standard deviation with the
                # unbiased flag false.
                return torch.std(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased standard deviation of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the standard deviation with the
                # unbiased flag true.
                return torch.std(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_along_dims(self):
        # [FUNCTION DOC] Exports standard deviation reduced over dims 0 and 1,
        # biased and unbiased.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased standard deviation over dims
            # 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the biased standard deviation reduced over dims
                # 0 and 1.
                return torch.std(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased standard deviation over dims
            # 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the unbiased standard deviation reduced over
                # dims 0 and 1.
                return torch.std(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_keepdim(self):
        # [FUNCTION DOC] Exports standard deviation over dims 0 and 1 keeping reduced
        # dimensions, biased and unbiased.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased standard deviation over dims 0 and
            # 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the biased standard deviation over dims 0 and
                # 1 keeping dims.
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased standard deviation over dims 0
            # and 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the unbiased standard deviation over dims 0 and
                # 1 keeping dims.
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_correction(self):
        # [FUNCTION DOC] Exports standard deviation over dims 0 and 1 with a correction value of
        # 3 and keepdim.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the standard deviation with correction 3 over
            # dims 0 and 1, keeping dims.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the standard deviation with correction 3 over
                # dims 0 and 1 keeping dims.
                return torch.std(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_var(self):
        # [FUNCTION DOC] Exports variance over all elements biased and unbiased, plus a square
        # root of the variance over dim 1 plus a small epsilon on a large 5D input.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased variance of the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance with the unbiased flag false.
                return torch.var(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased variance of the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance with the unbiased flag true.
                return torch.var(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

        class VarianceSqrt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the square root of the variance over dim
            # 1 plus 1e-8.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the square root of the dim 1 variance plus
                # a small epsilon.
                y = torch.var(input, 1)
                return torch.sqrt(y + 1e-8)

        x = torch.randn(1, 2, 3, 300, 300)
        model = VarianceSqrt()
        self.run_test(model, x)

    def test_var_along_dims(self):
        # [FUNCTION DOC] Exports variance over dims 0 and 1, biased and unbiased.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased variance over dims 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the biased variance over dims 0 and 1.
                return torch.var(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased variance over dims 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the unbiased variance over dims 0 and 1.
                return torch.var(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_keepdim(self):
        # [FUNCTION DOC] Exports variance over dims 0 and 1 with keepdim, biased and unbiased.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased variance over dims 0 and
            # 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the biased variance with kept dimensions.
                return torch.var(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased variance over dims 0 and
            # 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the unbiased variance with kept dimensions.
                return torch.var(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_correction(self):
        # [FUNCTION DOC] Exports variance with correction 3 over dims 0 and 1 and keepdim.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the variance with correction 3 over dims 0
            # and 1, keeping dims.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance with correction 3 keeping dims.
                return torch.var(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_var_mean(self):
        # [FUNCTION DOC] Exports var_mean over all elements, biased and unbiased.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased variance and mean of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the pair of variance and mean with the
                # unbiased flag false.
                return torch.var_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased variance and mean of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the pair of variance and mean with the
                # unbiased flag true.
                return torch.var_mean(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        # [FUNCTION DOC] Exports var_mean over dims 0 and 1, biased and unbiased.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the biased variance and mean over dims
            # 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns variance and mean reduced over dims 0
                # and 1, biased.
                return torch.var_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the unbiased variance and mean over dims
            # 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns variance and mean reduced over dims 0
                # and 1, unbiased.
                return torch.var_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        # [FUNCTION DOC] Exports var_mean with unusual dimension lists: reversed dims, skipped
        # dims and dims not starting at zero, all biased.
        class ReverseDims(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning variance and mean over dims 2 and
            # 1 (reversed order).
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns variance and mean reduced over dims 2 and 1.
                return torch.var_mean(input, dim=(2, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = ReverseDims()
        self.run_test(model, x)

        class SkipDims(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning variance and mean over dims 0 and 2, skipping
            # the middle one.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns variance and mean reduced over dims 0 and 2.
                return torch.var_mean(input, dim=(0, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = SkipDims()
        self.run_test(model, x)

        class NonZeroDims(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning variance and mean over dims 1 and 2.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns variance and mean reduced over dims 1 and 2.
                return torch.var_mean(input, dim=(1, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = NonZeroDims()
        self.run_test(model, x)

    def test_var_mean_keepdim(self):
        # [FUNCTION DOC] Exports var_mean over dims 0 and 1 with keepdim, biased and unbiased.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning biased variance and mean over dims 0 and
            # 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance and mean pair with
                # kept dimensions, biased.
                return torch.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning unbiased variance and mean over dims 0 and
            # 1 with keepdim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance and mean pair with
                # kept dimensions, unbiased.
                return torch.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_correction(self):
        # [FUNCTION DOC] Exports var_mean with correction 3 over dims 0 and 1 with keepdim.
        class Variance(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning variance and mean with correction 3 over dims
            # 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance and mean pair with correction 3
                # and kept dimensions.
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_std_mean(self):
        # [FUNCTION DOC] Exports std_mean over all elements, biased and unbiased.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning biased standard deviation and mean of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair with the
                # unbiased flag false.
                return torch.std_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning unbiased standard deviation and mean of
            # the whole input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair with the
                # unbiased flag true.
                return torch.std_mean(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        # [FUNCTION DOC] Exports std_mean over dims 0 and 1, biased and unbiased.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning biased std and mean over dims 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair reduced over dims 0
                # and 1, biased.
                return torch.std_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning unbiased std and mean over dims 0 and 1
            # (named like the variance module although it computes standard deviation).
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair reduced over dims 0
                # and 1, unbiased.
                return torch.std_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        # [FUNCTION DOC] Exports std_mean over dims 0 and 1 with keepdim, biased and unbiased.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning biased std and mean over dims 0 and
            # 1 keeping dims.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair with
                # kept dimensions, biased.
                return torch.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning unbiased std and mean over dims 0 and
            # 1 keeping dims.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the std and mean pair with
                # kept dimensions, unbiased.
                return torch.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_correction(self):
        # [FUNCTION DOC] Exports a module with correction 3 over dims 0 and 1 with keepdim.
        # KNOWN ISSUE: despite the test name, the module calls var_mean rather than std_mean, so
        # the std_mean correction path is not covered.
        class StandardDeviation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module named for standard deviation but computing variance and
            # mean with correction 3 over dims 0 and 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the variance and mean pair with correction 3
                # and kept dimensions.
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_bitshift(self):
        # [FUNCTION DOC] Exports bit shifts on an int64 tensor: right shift by 1, left shift by
        # 3, right shift by a two-element tensor and left shift by 4.
        class BitshiftModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning four shifted versions of its input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of the input shifted right by 1, left
                # by 3, right by a tensor of shifts and left by 4.
                return (
                    input >> 1,
                    input << 3,
                    input >> torch.tensor([1, 2]),
                    input << 4,
                )

        input = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), input)

    @common_utils.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64]
    )
    def test_bitshift_signed(self, dtype):
        # [FUNCTION DOC] Parameterized over int8, int16, int32 and int64. Exports arithmetic
        # right shifts of boundary values (-128 to 127) by 1 and 3.
        class RightShiftModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning right shifts by 1 and by 3.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a pair of the input shifted right by 1
                # and by 3.
                return input >> 1, input >> 3

        input = torch.tensor([-128, -7, -4, -1, 0, 1, 3, 127], dtype=dtype)
        self.run_test(RightShiftModel(), input)

    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64]
    )
    def test_bitwise_right_shift_signed(self, dtype):
        # [FUNCTION DOC] Parameterized over four signed integer dtypes. Exports
        # bitwise_right_shift by 2 of boundary values. Requires opset 11 or newer.
        class BitwiseRightShiftModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying bitwise_right_shift by 2.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input shifted right by 2 with
                # the function form.
                return torch.bitwise_right_shift(input, 2)

        input = torch.tensor([-128, -7, -4, -1, 0, 1, 3, 127], dtype=dtype)
        self.run_test(BitwiseRightShiftModel(), input)

    @skipIfUnsupportedMinOpsetVersion(18)
    def test_bitwise_and(self):
        # [FUNCTION DOC] Exports bitwise and with a constant, with a tensor, and with an int32
        # constant tensor, for uint8 and int8 inputs. Requires opset 18 or newer.
        class BitwiseAndModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning three bitwise-and results.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns a tuple: input and 20, bitwise_and of input and
                # other, and other and a two-element int32 tensor.
                return (
                    input & 20,
                    torch.bitwise_and(input, other),
                    other & torch.tensor([1, 2], dtype=torch.int32),
                )

        input = torch.randint(0, 255, (3, 4, 2), dtype=torch.uint8)
        other = torch.randint(-128, 127, (3, 4, 2), dtype=torch.int8)
        self.run_test(BitwiseAndModel(), (input, other))

    # uint8 not implemented in ORT for Mul used in
    # exporting bitshift for opset_version < 10
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_bitshift_uint8(self):
        # [FUNCTION DOC] Exports bit shifts on uint8 tensors, avoiding the multiplication path
        # unsupported in ORT for uint8 under opset 10. Requires opset 11 or newer.
        class BitshiftModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning four shifted uint8 results.
            def forward(self, input, input2):
                # [FUNCTION DOC] HELPER: Returns a tuple: input right 1, input left 3, input2
                # right by a uint8 tensor and input2 left 4.
                return (
                    input >> 1,
                    input << 3,
                    input2 >> torch.tensor([1, 2], dtype=torch.uint8),
                    input2 << 4,
                )

        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow(self):
        # [FUNCTION DOC] Exports narrow on dim 0 from start 0 with length 2 on a 3 x 3 tensor.
        class NarrowModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module narrowing dim 0 to its first two entries.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the narrowed input.
                return torch.narrow(input, 0, 0, 2)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_narrow_dynamic(self):
        # [FUNCTION DOC] Exports narrow with a length computed from the input size minus one.
        # Requires opset 11 or newer.
        class NarrowModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module narrowing dim 0 to all but its last entry.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input narrowed on dim 0 from 0 with length
                # equal to the dim size minus one.
                return torch.narrow(input, 0, 0, input.shape[0] - 1)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        # [FUNCTION DOC] Exports index_fill on dim 2 with indices 2 and 0, fill value -1.
        # Requires opset 9 or newer.
        class IndexFillModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling selected positions on dim 2 with -1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Builds the index tensor and returns the input filled
                # at those indices.
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexFillModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        # [FUNCTION DOC] Exports index_copy for dims 1 and -2 (the same axis) of a 3 x 4 x 5
        # tensor with source of ones. Requires opset 9 or newer.
        class IndexCopyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module copying a ones source into the input along a stored
            # dimension at indices 2 and 0.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Stores the dimension used for the copy.
                super().__init__()
                self.dim = dim

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Creates the index tensor and a ones source of shape 3 x
                # 2 x 5 and returns the input after index_copy along the stored dimension.
                index = torch.tensor([2, 0])
                source = torch.ones(3, 2, 5)
                return input.index_copy(self.dim, index, source)

        x = torch.randn(3, 4, 5, requires_grad=True)
        for dim in (1, -2):
            self.run_test(IndexCopyModel(dim), x)

    def test_select(self):
        # [FUNCTION DOC] Exports selecting column 1 from a 3 x 4 tensor.
        class Select(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting column 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input indexed by a full slice and 1.
                return x[:, 1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        # [FUNCTION DOC] Exports selecting the last column using index -1.
        class Select(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting column -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input indexed by a full slice and -1.
                return x[:, -1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_index_select_constant_scaler_index(self):
        # [FUNCTION DOC] Exports index_select with a scalar constant index tensor on dim 1.
        class IndexSelectScalerIndexModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying index_select on dim 1 with a scalar index
            # tensor of value 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns index_select of the input with scalar index 2
                # on dim 1.
                index = 2
                return torch.index_select(x, 1, torch.tensor(index))

        x = torch.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_index_select_scaler_index(self):
        # [FUNCTION DOC] Exports index_select where the scalar index is computed as a stored
        # base plus an input offset, base 1 and offset 2.
        class IndexSelectScalerIndexModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a base index tensor and selecting on dim 1 at
            # the base plus an offset input.
            def __init__(self, index_base):
                # [FUNCTION DOC] HELPER: Stores the base as a scalar tensor.
                super().__init__()
                self.index_base = torch.tensor(index_base)

            def forward(self, x, index_offset):
                # [FUNCTION DOC] HELPER: Returns the index_select of x on dim 1 using the base
                # plus the offset.
                index = self.index_base + index_offset
                return torch.index_select(x, 1, index)

        x = torch.randn(3, 4)
        offset = 2
        index_offset = torch.tensor(offset)
        base = 1
        self.run_test(IndexSelectScalerIndexModel(base), (x, index_offset))

    def test_take(self):
        # [FUNCTION DOC] Exports take of flat indices from a 4D tensor.
        class TakeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking elements of the input at flat indices.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the elements of x at flat positions y.
                return torch.take(x, y)

        x = torch.randn(6, 4, 3, 3)
        y = torch.tensor([4, 1, 7, 15, 63])
        self.run_test(TakeModel(), (x, y))

    def test_topk(self):
        # [FUNCTION DOC] Exports topk with k of 3 on a five-element vector.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the top three values and indices.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the top 3 result of the input.
                return torch.topk(x, 3)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_int32_k(self):
        # [FUNCTION DOC] Exports topk with k passed as an int32 tensor. Requires opset
        # 10 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the top-k with a tensor k.
            def forward(self, x, k):
                # [FUNCTION DOC] HELPER: Returns the top-k values and indices of x.
                return torch.topk(x, k)

        x = torch.arange(1.0, 6.0)
        k = torch.tensor(3, dtype=torch.int32)
        self.run_test(Model(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_topk_smallest_unsorted(self):
        # [FUNCTION DOC] Exports topk for the smallest values in sorted and unsorted modes,
        # sorting the unsorted values so the comparison is stable. Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sorted bottom-k and sorted values of
            # the unsorted bottom-k.
            def forward(self, x, k):
                # [FUNCTION DOC] HELPER: Computes the k smallest unsorted and sorted; returns
                # the sorted result and the sorted values of the unsorted result, since unsorted
                # order is not guaranteed to match.
                # When sorted=False, order of elements in the output tensors
                # are not expected to match between PyTorch and ORT
                topk_unsorted = torch.topk(x, k, largest=False, sorted=False)
                topk_sorted = torch.topk(x, k, largest=False, sorted=True)
                return topk_sorted, torch.sort(topk_unsorted.values).values

        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModule(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        # [FUNCTION DOC] Exports a ScriptModule calling topk with a tensor k. Requires opset
        # 10 or newer.
        class MyModuleDynamic(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the top-k of its input with a tensor k.
            @torch.jit.script_method
            def forward(self, x, k):
                # [FUNCTION DOC] HELPER: Script method returning the top-k values and indices.
                return torch.topk(x, k)

        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModuleDynamic(), (x, k))

    @skipScriptTest()  # Python builtin apply of FunctionMeta object is currently not supported in Torchscript.
    @skipIfUnsupportedMinOpsetVersion(11)  # Clip op min is an input since opset 11.
    def test_auto_grad(self):
        # [FUNCTION DOC] Exports custom autograd functions via a registered symbolic for the
        # Python op node, checking a clip function with a tensor minimum and a relu function.
        # The symbolic is registered at opset 1 and unregistered through a test cleanup. Skipped
        # under scripting; requires opset 11 or newer.
        class MyClip(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function clamping its input from below at a scalar
            # and saving the input for backward.
            @staticmethod
            def forward(ctx, input, scalar):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns the input
                # clamped with the given minimum.
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function clamping its input at zero and saving the
            # input for backward.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns the input
                # clamped at minimum zero.
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        def symbolic_python_op(g, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Symbolic function for Python operator nodes. Reads the
            # node name from keyword arguments and emits a Clip op for the clip function and a
            # Relu op for the relu function; for any other name returns the internal
            # unimplemented error result.
            name = kwargs["name"]
            if name == "MyClip":
                return g.op("Clip", args[0], args[1])
            elif name == "MyRelu":
                return g.op("Relu", args[0])
            else:
                # TODO(justinchuby): Remove reference to internal names in symbolic_helper
                return torch.onnx.symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        torch.onnx.register_custom_op_symbolic("prim::PythonOp", symbolic_python_op, 1)
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "prim::PythonOp", 1)

        class MyClipModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom clip function.
            def forward(self, x, min):
                # [FUNCTION DOC] HELPER: Returns the clip autograd function applied to x
                # with the minimum.
                return MyClip.apply(x, min)

        x = torch.randn(3, 3)
        min = torch.tensor([0.0])
        self.run_test(MyClipModule(), (x, min))

        class MyReluModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom relu function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the relu autograd function applied to x.
                return MyRelu.apply(x)

        x = torch.randn(3, 3)
        self.run_test(MyReluModule(), x)

    def test_clip_int(self):
        # [FUNCTION DOC] Exports clamp between 0 and 1 on an int64 tensor.
        class MyClipInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clamping integers to the range 0 to 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input clamped to 0 and 1.
                return torch.clamp(x, 0, 1)

        self.run_test(MyClipInt(), torch.randn(3, 3).to(torch.int64))

    def test_relu_int(self):
        # [FUNCTION DOC] Exports ReLU on an int32 tensor.
        self.run_test(torch.nn.ReLU(), torch.randn(3, 3).to(torch.int32))

    def test_pad_int(self):
        # [FUNCTION DOC] Exports functional padding by 1 on both sides of the last dim on
        # an int32 tensor.
        class MyPadInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module padding the last dimension by one on each side.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the padded input.
                return torch.nn.functional.pad(x, (1, 1))

        self.run_test(MyPadInt(), torch.randn(3, 3).to(torch.int32))

    def test_min_int(self):
        # [FUNCTION DOC] Exports elementwise minimum of an int32 tensor and itself plus 1.
        class MyMinInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the elementwise minimum of the input and the
            # input plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the elementwise minimum.
                return torch.min(x, x + 1)

        self.run_test(MyMinInt(), torch.randn(3, 3).to(torch.int32))

    def test_max_int(self):
        # [FUNCTION DOC] Exports elementwise maximum of an int32 tensor and itself plus 1.
        class MyMaxnInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the elementwise maximum of the input and the
            # input plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the elementwise maximum.
                return torch.max(x, x + 1)

        self.run_test(MyMaxnInt(), torch.randn(3, 3).to(torch.int32))

    @skipIfUnsupportedOpsetVersion([7])
    def test_normalize(self):
        # [FUNCTION DOC] Exports functional normalize with defaults on a 3 x 3 tensor; not run
        # for opset 7.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional normalize.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the normalized input.
                return torch.nn.functional.normalize(x)

        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_norm_with_dtype(self):
        # [FUNCTION DOC] Exports the aten norm overload that takes a dtype argument (order 2
        # over dim 1, keepdim, float64) on a 3 x 3 tensor, since the public norm function does
        # not decompose to it.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the aten norm operator with a float64 dtype.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the order-2 norm along dim 1 with kept dim
                # and float64 dtype.
                # TODO(bowbao): There is a slight gap in today's test infrastructure
                # to directly test aten ops. OpInfo `torch.norm`` in `common_methods_invocations.py`
                # will not decompose to below aten op.
                return torch.ops.aten.norm(
                    x, p=2, dim=[1], keepdim=True, dtype=torch.float64
                )

        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_layer_norm(self):
        # [FUNCTION DOC] Exports LayerNorm over the last three dims for all four combinations of
        # elementwise affine and bias on a 5D input; the input has at least three dimensions so
        # axis 2 and axis -2 differ.
        # As layer_norm works on the last D dimension, please keep
        # this test case at least three dimension to prevent the
        # situation of axis=2 mapping to the same axis as axis=-2
        for elementwise_affine in (True, False):
            for bias in (True, False):
                model = torch.nn.LayerNorm(
                    [10, 10, 10], elementwise_affine=elementwise_affine, bias=bias
                )
                x = torch.randn(20, 5, 10, 10, 10)
                self.run_test(model, x)

    def test_batchnorm1d(self):
        # [FUNCTION DOC] Exports affine BatchNorm1d on a 10 x 10 input and on a 10 x 10
        # x 128 input.
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_noaffine(self):
        # [FUNCTION DOC] Exports BatchNorm1d without affine parameters on 2D and 3D inputs.
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_norunningstats(self):
        # [FUNCTION DOC] Exports BatchNorm1d with running statistics disabled on 2D
        # and 3D inputs.
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, track_running_stats=False)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm2d(self):
        # [FUNCTION DOC] Exports affine BatchNorm2d on a 10 x 3 x 128 x 128 input.
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm2d_noaffine(self):
        # [FUNCTION DOC] Exports BatchNorm2d without affine parameters.
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=False)
        self.run_test(model, x)

    def test_batchnorm2d_norunningstats(self):
        # [FUNCTION DOC] Exports BatchNorm2d with running statistics disabled.
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, track_running_stats=False)
        self.run_test(model, x)

    def test_batchnorm3d(self):
        # [FUNCTION DOC] Exports affine BatchNorm3d on a 10 x 3 x 64 x 64 x 64 input.
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm3d_noaffine(self):
        # [FUNCTION DOC] Exports BatchNorm3d without affine parameters on a 5D input.
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm1d_runningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm1d with running statistics tracking, affine
        # and non-affine, on a 10 x 5 x 128 input. Requires opset 9 or newer because of
        # the constant-of-shape operator.
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm1d_norunningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm1d without running statistics,
        # affine and non-affine.
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm2d_runningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm2d with running statistics tracking, affine and
        # non-affine, on a 4D input. Requires opset 9 or newer.
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm2d_norunningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm2d without running statistics,
        # affine and non-affine.
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because ConstantOfShape op is not supported for opset < 9
    def test_instancenorm3d_runningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm3d with running statistics tracking, affine and
        # non-affine, on a 5D input. Requires opset 9 or newer.
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm3d_norunningstats(self):
        # [FUNCTION DOC] Exports InstanceNorm3d without running statistics,
        # affine and non-affine.
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)

        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar(self):
        # [FUNCTION DOC] Exports scatter along dim 1 of a 3 x 3 double zeros tensor with the
        # scalar 1.0 as the update. Requires opset 9 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scattering a scalar value into the input along dim 1 at
            # the given indices.
            def forward(self, input, indices):
                # [FUNCTION DOC] HELPER: Returns the input with 1.0 scattered at the indices
                # on dim 1.
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar_different_types(self):
        # [FUNCTION DOC] Same scalar scatter but with a float32 input, so the scalar update has
        # a different type than the input. Requires opset 9 or newer.
        # Tests the case when scalar src (updates values) type is different
        # from self type. Happens only with scalar src - PyTorch does not
        # allow this when src is a tensor.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scattering a Python float scalar into a float32 input
            # along dim 1.
            def forward(self, input, indices):
                # [FUNCTION DOC] HELPER: Returns the input with 1.0 scattered at the indices
                # on dim 1.
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter(self):
        # [FUNCTION DOC] Exports scatter with tensor values for four cases: a 3 x 3 zeros tensor
        # with two index patterns, a 4D tensor with expanded indices and a 3D tensor with 3D
        # indices. Requires opset 9 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scattering a value tensor into the input along dim 1.
            def forward(self, input, indices, values):
                # [FUNCTION DOC] HELPER: Returns the input with values scattered at the indices
                # on dim 1.
                return input.scatter(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 5, 6)
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 2)
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        self.run_test(ScatterModel(), (input, indices, values))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_add(self):
        # [FUNCTION DOC] Exports scatter_add on a small tensor, then through a scripted helper
        # that sums source rows into a zeros tensor with in-place scatter_add. Requires opset
        # 9 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding values into the input along dim 1
            # at given indices.
            def forward(self, input, indices, values):
                # [FUNCTION DOC] HELPER: Returns the result of scatter_add on dim 1.
                return input.scatter_add(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @torch.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            # [FUNCTION DOC] HELPER: Scripted function creating zeros with the shape and
            # dtype of the source and adding the source into it along dim 1 at the index,
            # returning the result.
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module delegating to the scripted scatter-sum function.
            def forward(self, src, index):
                # [FUNCTION DOC] HELPER: Returns the scripted function applied to src and index.
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_index_not_unique(self):
        # [FUNCTION DOC] Same as the scatter_add test but with indices that repeat, so
        # several sources add into the same position (both direct and scripted). Requires
        # opset 16 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding values into the input along dim 1
            # at given indices.
            def forward(self, input, indices, values):
                # [FUNCTION DOC] HELPER: Returns the result of scatter_add on dim 1.
                return input.scatter_add(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @torch.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            # [FUNCTION DOC] HELPER: Scripted function creating zeros with the shape and
            # dtype of the source and adding the source into it along dim 1 at the index,
            # returning the result.
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module delegating to the scripted scatter-sum function.
            def forward(self, src, index):
                # [FUNCTION DOC] HELPER: Returns the scripted function applied to src and index.
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_different_size_index_src(self):
        # [FUNCTION DOC] Exports scatter_add where the index and source are smaller than the
        # input (a 1 x 5 index and 2 x 5 source into 3 x 5 zeros) on dim 0. Requires opset
        # 16 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding src into the input on dim 0 at the indices.
            def forward(self, input, indices, src):
                # [FUNCTION DOC] HELPER: Returns the result of scatter_add on dim 0.
                return input.scatter_add(0, indices, src)

        src = torch.ones((2, 5))
        input = torch.zeros(3, 5, dtype=src.dtype)
        indices = torch.tensor([[0, 1, 2, 0, 0]])
        self.run_test(ScatterModel(), input_args=(input, indices, src))

    @common_utils.parametrize(
        "src, indices",
        [
            common_utils.subtest(
                [torch.ones((1, 5)), torch.tensor([[0, 1, 2, 0, 0]])],
                name="src_indices_dynamic_combination1",
            ),
            common_utils.subtest(
                [torch.ones((2, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination2",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])],
                name="src_indices_dynamic_combination3",
            ),
            common_utils.subtest(
                [torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1]])],
                name="src_indices_dynamic_combination4",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_dynamic_index(self, src, indices):
        # [FUNCTION DOC] Parameterized with four named combinations of source and index shapes.
        # Exports scatter_add on dim 0 with dynamic axes for the index and source sizes.
        # Requires opset 16 or newer.
        class ScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding src into the input on dim 0 at the indices.
            def forward(self, input, indices, src):
                # [FUNCTION DOC] HELPER: Returns the result of scatter_add on dim 0.
                return input.scatter_add(0, indices, src)

        input = torch.zeros(3, 5, dtype=src.dtype)
        self.run_test(
            ScatterModel(),
            input_args=(input, indices, src),
            input_names=["input", "indices", "src"],
            dynamic_axes={"indices": {0: "a", 1: "b"}, "src": {0: "c", 1: "d"}},
        )

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce(self):
        # [FUNCTION DOC] Exports scatter_reduce with amax, sum, amin and prod reductions on a
        # six-element source into a four-element tensor, with the model in evaluation mode.
        # Requires opset 16 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying four scatter_reduce reductions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, x, index, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of scatter_reduce results on dim 0 with
                # max, sum, min and product.
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1.0, 2.0, 3.0, 8.0])

        self.run_test(model, (src, index, input))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce_self_rank_zero(self):
        # [FUNCTION DOC] Exports the same four scatter_reduce reductions with empty source,
        # index and input tensors. Requires opset 16 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying four scatter_reduce reductions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor only.
                super().__init__()

            def forward(self, x, index, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of scatter_reduce results on dim 0 with
                # max, sum, min and product.
                y_max = input.scatter_reduce(0, index, x, reduce="amax")
                y_sum = input.scatter_reduce(0, index, x, reduce="sum")
                y_min = input.scatter_reduce(0, index, x, reduce="amin")
                y_mul = input.scatter_reduce(0, index, x, reduce="prod")
                return y_max, y_sum, y_min, y_mul

        model = Model()
        model.eval()

        empty_tensor = torch.tensor([])
        empty_idx = torch.tensor([], dtype=torch.int64)

        self.run_test(model, (empty_tensor, empty_idx, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_bucketize(self):
        # [FUNCTION DOC] Exports bucketize of a 2 x 3 integer tensor against sorted boundaries,
        # left-closed and right-closed variants. Requires opset 9 or newer.
        class BucketModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning bucketize results for default
            # and right-closed mode.
            def forward(self, input, boundaries):
                # [FUNCTION DOC] HELPER: Returns a tuple of the bucket indices of the input for
                # the boundaries, without and with the right flag.
                return torch.bucketize(input, boundaries), torch.bucketize(
                    input, boundaries, right=True
                )

        input = torch.tensor([[2, 5, 10], [6, 8, 3]])
        boundaries = torch.tensor([1, 5, 7, 8, 10])
        self.run_test(BucketModel(), (input, boundaries))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_one_hot(self):
        # [FUNCTION DOC] Exports one_hot for a constant class count 15, and for a class
        # count given as a float tensor converted to int32 inside the model. Requires opset
        # 9 or newer.
        class OneHot(torch.nn.Module):
            # [CLASS DOC] HELPER: Module one-hot encoding its input with a stored
            # number of classes.
            def __init__(self, num_classes):
                # [FUNCTION DOC] HELPER: Stores the number of classes.
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns one_hot of x with the stored class count.
                return torch.nn.functional.one_hot(x, self.num_classes)

        x = torch.arange(10)
        self.run_test(OneHot(15), (x))

        class OneHot(torch.nn.Module):
            # [CLASS DOC] HELPER: Module one-hot encoding with the class count taken from
            # a tensor input.
            def forward(self, x, num_classes):
                # [FUNCTION DOC] HELPER: Converts the class count tensor to int32 and returns
                # one_hot of x with its first element.
                num_classes = num_classes.to(torch.int32)
                return torch.nn.functional.one_hot(x, num_classes[0])

        x = torch.arange(10)
        num_classes = 15 * torch.ones(1)
        self.run_test(OneHot(), (x, num_classes))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gather(self):
        # [FUNCTION DOC] Exports gather on dim 1 of a 3 x 3 float tensor with a 3 x 2 index.
        # Requires opset 9 or newer.
        class GatherModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module gathering along dim 1.
            def forward(self, input, indices):
                # [FUNCTION DOC] HELPER: Returns the gather of the input at the indices
                # on dim 1.
                return input.gather(1, indices)

        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(GatherModel(), input_args=(input, indices))

    @skipScriptTest()  # Scripting error: Cannot instantiate nn module
    def test_gather_constant_fold(self):
        # [FUNCTION DOC] Exports constant folding of gather over constant inputs in three
        # modules: shape-derived clamp bound with an embedding lookup, a zero padding whose
        # widths come from a buffer shape, and an add of an indexed buffer into the input with
        # dynamic axes. Skipped under scripting since nn module instantiation fails.
        class GatherModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of five ones and an embedding of 8
            # entries by 3, producing a clamped input and an embedding lookup of constant ids.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ones buffer and embedding.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))
                # torch.nn.Embedding is converted to ONNX::Gather.
                # Constant folding will be triggered for constant inputs.
                # This pattern is common for constant mask inputs in transformer models.
                self.embed = torch.nn.Embedding(8, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Derives a clamp minimum from five minus the
                # buffer length, builds constant ids and returns the clamped input and
                # the embedding lookup.
                # shape is of rank 0
                shape = self.weight.shape[0]
                m = 5 - shape
                y = torch.ones(1, 4, dtype=torch.long)
                return x.clamp(min=m), self.embed(y)

        x = torch.randn(1)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer of two ones used to compute padding
            # widths for a 2D zero pad.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ones buffer.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds padding values from the buffer length and
                # returns the zero-padded input.
                # shape is of rank 0
                shape = self.weight.shape[0]
                pad = [1, shape, shape, shape]
                zero_pad = torch.nn.ZeroPad2d(pad)
                return zero_pad(x)

        x = torch.randn(1, 3, 2)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random buffer of shape 1 x 1 x 3 x 1 x 1 that is
            # added to the input in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the random buffer.
                super().__init__()
                self.rb = torch.nn.Buffer(torch.randn(1, 1, 3, 1, 1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the buffer's first slice into x in place
                # and returns x.
                x += self.rb[0]
                return x

        x = torch.randn(1, 3, 224, 224)
        self.run_test(
            GatherModule(),
            (x,),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "class", 2: "height", 3: "width"},
            },
            input_names=["input"],
            output_names=["output"],
        )

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_expand(self):
        # [FUNCTION DOC] Exports expand with fixed sizes and an inferred dimension, with
        # inferred size from the input shape, and with a size given as a tensor. Not run for
        # opset 13; requires opset 9 or newer.
        class ExpandModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding to sizes 2, 3 and an inferred last dim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input expanded to 2 x 3 x -1.
                return input.expand(2, 3, -1)

        input = torch.randn(2, 1, 4)
        self.run_test(ExpandModel(), input_args=(input))

        class ExpandInferDimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding to an inferred first dim and the first dim
            # size as second dim.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input expanded to -1 and the input's
                # first dimension size.
                return input.expand(-1, input.size(0))

        input = torch.randn(3, 1)
        self.run_test(ExpandInferDimModel(), input_args=(input))

        class ExpandTensorSizeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding to a size given as a tensor.
            def forward(self, input, size):
                # [FUNCTION DOC] HELPER: Returns the input expanded to the given size tensor.
                return input.expand(size)

        input = torch.randn(
            3,
        )
        size = torch.tensor(-1)
        self.run_test(ExpandTensorSizeModel(), input_args=(input, size))

    @skipIfUnsupportedMinOpsetVersion(11)  # index_put is supported in opsets >= 11
    def test_dynamic_expand_as(self):
        # [FUNCTION DOC] Exports dynamic in-place slice assignments with a start that depends
        # on the input size (a scalar zero, a constant vector) and expand_as of a constant
        # column against the input, with dynamic axes and additional inputs. Requires opset
        # 11 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module zeroing the columns of the input from the index equal
            # to the first dimension size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets columns from the dim 0 size onward to zero and
                # returns x (mutated in place).
                x[:, x.size(0) :] = 0
                return x

        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a constant three-element vector to columns
            # from the dim 0 size onward.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assigns the vector 1, 2, 3 broadcast into those columns
                # and returns x (mutated in place).
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[x2],
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding a constant 3 x 1 column to the shape
            # of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the constant column expanded to the shape of x.
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test(
            Model(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[x2],
        )

    def test_multinomial(self):
        # [FUNCTION DOC] Exports multinomial sampling on weights where each row has a single
        # nonzero entry, so the sampled indices are deterministic: three draws with replacement
        # and one draw without replacement.
        class Multinomial(torch.nn.Module):
            # [CLASS DOC] HELPER: Module drawing three samples with replacement from
            # a weight tensor.
            def forward(self, weight):
                # [FUNCTION DOC] HELPER: Returns the multinomial samples drawn from the weights.
                return torch.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Module drawing a single sample without replacement from
            # a weight tensor.
            def forward(self, weight):
                # [FUNCTION DOC] HELPER: Returns one multinomial sample per row.
                return torch.multinomial(weight, 1)

        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        self.run_test(Multinomial(), (weight,))
        self.run_test(MultinomialNoReplacement(), (weight,))

    def _test_reduced_ops(self, op):
        # [FUNCTION DOC] HELPER: Shared body for reduction tests of the given reduction function
        # applied over the last dim. Runs uint8, int8, int16, int32 and int64 inputs unless the
        # operation is mean; double unless prod or mean; half unless prod; and always float.
        # Reasons are limits in dtype support of mean, ORT prod and half prod.
        class ReducedOpModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the chosen reduction along the last dimension.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the reduction applied to the input over dim -1.
                return op(input, dim=-1)

        if op != torch.mean:  # torch.mean only supports float types
            x = torch.randint(10, (4, 4), dtype=torch.uint8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int16)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int32)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int64)
            self.run_test(ReducedOpModule(), x)

        # torch.mean only supports float types
        # ORT does not support double ReduceProd for double
        if op != torch.prod and op != torch.mean:
            x = torch.randn(4, 5, dtype=torch.double)
            self.run_test(ReducedOpModule(), x)

        if op != torch.prod:  # torch.prod not implemented for Half
            x = torch.randn(4, 4, dtype=torch.half)
            self.run_test(ReducedOpModule(), x)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedOpModule(), x)

    def test_reduced_sum(self):
        # [FUNCTION DOC] Runs the reduction dtype tests for the sum operation and returns
        # the helper's result.
        return self._test_reduced_ops(op=torch.sum)

    def test_reduced_mean(self):
        # [FUNCTION DOC] Runs the reduction dtype tests for the mean operation and returns
        # the helper's result.
        return self._test_reduced_ops(op=torch.mean)

    def test_reduced_prod(self):
        # [FUNCTION DOC] Runs the reduction dtype tests for the product operation and returns
        # the helper's result.
        return self._test_reduced_ops(op=torch.prod)

    def test_reduced_sum_dtypes(self):
        # [FUNCTION DOC] Exports sum with a float output dtype over all elements and over the
        # last dim, on a half precision input.
        class NoDimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing all elements into a float.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the sum of the input as float.
                return input.sum(dtype=torch.float)

        class DimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing over the last dimension into a float.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the sum over dim -1 as float.
                return input.sum(dim=-1, dtype=torch.float)

        input = torch.randn((4, 4), dtype=torch.half)
        self.run_test(NoDimModel(), input)
        self.run_test(DimModel(), input)

    def test_reduced_min_max(self):
        # [FUNCTION DOC] Exports min over the last dim and max over dim 0, taking only the
        # values, for int32, int64 and float inputs.
        class ReducedMinMaxModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the minimum values over the last dim and
            # maximum values over dim 0.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of the min along dim -1 and the max
                # along dim 0, values only.
                return torch.min(input, dim=-1)[0], torch.max(input, dim=0)[0]

        x = torch.randint(10, (4, 4), dtype=torch.int32)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randint(10, (4, 4), dtype=torch.int64)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedMinMaxModule(), x)

    def test_reduce_log_sum_exp(self):
        # [FUNCTION DOC] Exports logsumexp over dim 0 added to logsumexp over dims 0 and 1.
        class ReduceLogSumExpModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing two logsumexp reductions.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns logsumexp over dim 0 plus logsumexp over dims
                # 0 and 1.
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_softmax(self):
        # [FUNCTION DOC] For softmax dims from -4 to 2 on a 4D input, exports Softmax directly
        # and a scripted wrapper that reshapes the input first so the rank is unknown at export.
        for i in range(-4, 3):
            model = torch.nn.Softmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying softmax on a given dim after reshaping to
                # 3 x 4 x 5 x 6.
                def __init__(self, i):
                    # [FUNCTION DOC] HELPER: Creates a Softmax submodule for
                    # the given dimension.
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the softmax of the input reshaped to 3 x 4
                    # x 5 x 6.
                    return self.softmax(x.reshape(3, 4, 5, 6))

            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_softmax_large_values(self):
        # [FUNCTION DOC] Exports softmax over very large and very small values (plus and minus
        # 1e12) for dims -2 to 0, directly and via a scripted wrapper with unknown rank, to
        # check numerical stability.
        input = torch.tensor(
            [[-1e12, -1e12, -1e12], [1e12, 0.0, -5.0], [3.0, 4.0, 5.0]]
        )
        for i in range(-2, 1):
            model = torch.nn.Softmax(dim=i)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying softmax on a given dim after reshaping to
                # 3 x 3.
                def __init__(self, i):
                    # [FUNCTION DOC] HELPER: Creates a Softmax submodule for
                    # the given dimension.
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the softmax of the input reshaped to 3 x 3.
                    return self.softmax(x.reshape(3, 3))

            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_logsoftmax(self):
        # [FUNCTION DOC] Exports LogSoftmax on ones tensors of rank 2 through 6, with the
        # softmax dim one less than the rank.
        for i in range(7)[2:]:
            model = torch.nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_test(model, input)

    def test_logsoftmax_dim(self):
        # [FUNCTION DOC] Exports LogSoftmax for dims -4 to 2 on a 4D random input.
        for i in range(-4, 3):
            model = torch.nn.LogSoftmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

    def test_logsoftmax_dtype(self):
        # [FUNCTION DOC] Exports functional log_softmax over dim 1 with float64 output dtype on
        # a float input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying log_softmax with a double output dtype.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the log softmax over dim 1 as float64.
                return torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float64)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(Model(), x)

    def test_softplus(self):
        # [FUNCTION DOC] Exports functional softplus with default beta, beta 2 and beta 1.7 on a
        # random 3D input.
        class BetaOneModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying softplus with default beta.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the softplus of the input.
                return torch.nn.functional.softplus(x)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaOneModel(), x)

        class BetaModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying softplus with beta 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the softplus of the input with beta 2.
                return torch.nn.functional.softplus(x, beta=2)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaModel(), x)

        class BetaFloatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying softplus with beta 1.7.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the softplus of the input with beta 1.7.
                return torch.nn.functional.softplus(x, beta=1.7)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaFloatModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_hidden(self):
        # [FUNCTION DOC] Exports a single-layer LSTM with no initial state given on a 10 x 16 x
        # 16 input. Requires opset 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM of input and hidden size 16.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule.
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the LSTM output and states for the input.
                return self.rnn(x)

        input = torch.randn((10, 16, 16))
        self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_proj_no_hidden(self):
        # [FUNCTION DOC] Checks that exporting an LSTM with a projection size of 8 raises
        # RuntimeError. Requires opset 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM of size 16 and projection size 8.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the projected LSTM submodule.
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the LSTM output and states for the input.
                return self.rnn(x)

        input = torch.randn((10, 16, 16))
        with self.assertRaises(RuntimeError):
            self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        # [FUNCTION DOC] Exports a unidirectional single-layer LSTM with explicit initial hidden
        # and cell states, using the shared RNN size constants. Requires opset 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a unidirectional one-layer LSTM taking
            # explicit initial states.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule.
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                # [FUNCTION DOC] HELPER: Returns the LSTM result for the input with the initial
                # hidden and cell states.
                return self.rnn(x, (h0, c0))

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(LSTMModel(), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_cell(self):
        # [FUNCTION DOC] Exports an LSTM cell with and without bias, given inputs and initial
        # hidden and cell states. Requires opset 9 or newer.
        class LSTMCellModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM cell, optionally without bias.
            def __init__(self, bias):
                # [FUNCTION DOC] HELPER: Creates the LSTM cell with the given bias flag.
                super().__init__()
                self.lstm_cell = torch.nn.LSTMCell(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, bias=bias
                )

            def forward(self, x, h0, c0):
                # [FUNCTION DOC] HELPER: Returns the new hidden and cell states computed from x
                # and the initial states.
                return self.lstm_cell(x, (h0, c0))

        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        for bias in [True, False]:
            self.run_test(LSTMCellModel(bias), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        # [FUNCTION DOC] Exports an LSTM with the default initial state. Requires opset
        # 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a unidirectional one-layer LSTM using
            # the default state.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule.
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the LSTM result for the input.
                return self.rnn(x)

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        # [FUNCTION DOC] Exports an LSTM whose initial states are created from the runtime batch
        # size with a fixed batch size export, also testing with another input of the same batch
        # size. Requires opset 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM that builds ones initial states from the
            # input batch size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule and stores the hidden size.
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Reads the batch size from the input, creates ones
                # states, and returns the LSTM result.
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # verify with different input of same batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(
            LSTMModel(), input, fixed_batch_size=True, additional_test_inputs=[input2]
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        # [FUNCTION DOC] Exports an LSTM with initial states created from the batch size, using
        # dynamic sequence and batch axes and checking with an input of a different batch size.
        # Requires opset 9 or newer.
        class LSTMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM that builds ones initial states from the
            # input batch size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule and stores the hidden size.
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Reads the batch size from the input, creates ones
                # states and returns the LSTM result.
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # verify with different input of different batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(
            model,
            input,
            input_names=["input.1"],
            dynamic_axes={"input.1": {0: "seq", 1: "batch"}},
            additional_test_inputs=[input2],
        )

    def test_lstm_constant_folding(self):
        # [FUNCTION DOC] Exports multi-layer LSTMs, one bidirectional with 2 layers and one
        # unidirectional with 3 layers and different sizes, with constant folding enabled.
        class LstmNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an LSTM taking an input and a tuple
            # of initial states.
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule with the given sizes,
                # layers and directionality.
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state: tuple[Tensor, Tensor]):
                # [FUNCTION DOC] HELPER: Returns the LSTM result for the input
                # and initial states.
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # [FUNCTION DOC] HELPER: Builds an LSTM net and matching random input and initial
            # hidden and cell states, where the state first dimension is layers times
            # directions. Returns the model and the input tuple.
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, (h0, c0))

        batch_size1 = 3
        model1, input1 = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_bias(self):
        # [FUNCTION DOC] Exports LSTMs without bias for four combinations of layer count and
        # directionality. Requires opset 9 or newer.
        class LstmNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free LSTM of given
            # layers and directionality.
            def __init__(self, num_layers, bidirectional):
                # [FUNCTION DOC] HELPER: Creates the bias-free LSTM submodule.
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers,
                    bias=False,
                    bidirectional=bidirectional,
                )

            def forward(self, input, initial_state: tuple[Tensor, Tensor]):
                # [FUNCTION DOC] HELPER: Returns the LSTM result for the input
                # and initial states.
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(num_layers, bidirectional):
            # [FUNCTION DOC] HELPER: Builds a model and random input with initial states sized
            # by layers and directions, and returns the model with its input tuple.
            input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            num_directions = 2 if bidirectional else 1
            model = LstmNet(num_layers, bidirectional)
            h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            return model, (input, (h0, c0))

        num_layers = [1, 1, 2, 3]
        bidirectional = [True, False, True, False]
        models_and_inputs = [
            get_LstmNet_model_and_inputs(n, b)
            for n, b in zip(num_layers, bidirectional)
        ]
        for model, input in models_and_inputs:
            self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_sequence(self):
        # [FUNCTION DOC] Exports two stacked bidirectional batch-first LSTMs with linear layers
        # between them on a zero input, with dynamic axes on all input and output dims. Requires
        # opset 9 or newer.
        class LstmNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of two bidirectional LSTMs of size 8 each followed by
            # a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two LSTMs and the two linear layers.
                super().__init__()
                self.rnn1 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear1 = torch.nn.Linear(8 * 2, 8)
                self.rnn2 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear2 = torch.nn.Linear(8 * 2, 8)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Runs the LSTM and linear layer twice in sequence and
                # returns the final linear output.
                rnn_output1, _ = self.rnn1(input)
                linear_output1 = self.linear1(rnn_output1)
                rnn_output2, _ = self.rnn2(linear_output1)
                linear_output2 = self.linear2(rnn_output2)
                return linear_output2

        input = torch.zeros((1, 100, 8), dtype=torch.float32)
        self.run_test(
            LstmNet(),
            input,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "w", 2: "h"},
                "output": {0: "batch_size", 1: "w", 2: "h"},
            },
        )

    @skipScriptTest()
    def test_rnn_no_bias(self):
        # [FUNCTION DOC] Exports a bias-free RNN for several layer counts and the three packed
        # sequence variants (none, variable lengths, batch first). Skipped under scripting.
        def make_model(layers, packed_sequence):
            # [FUNCTION DOC] HELPER: Creates a bias-free unidirectional RNN with the given layer
            # count, batch first when the packed sequence mode is 2, and wraps it in the packed
            # sequence helper for modes 1 and 2.
            batch_first = packed_sequence == 2
            model = torch.nn.RNN(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional=False,
                batch_first=batch_first,
                bias=False,
            )

            if packed_sequence == 1:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, False
                )
            if packed_sequence == 2:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(
                    model, True
                )
            return model

        def make_input(batch_size, layers, packed_sequence):
            # [FUNCTION DOC] HELPER: Builds a padded batch of random sequences with random
            # decreasing lengths, an initial hidden state, and for packed modes the length
            # tensor. Returns the single input tensor or a tuple of the tensors.
            batch_first = packed_sequence == 2
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            h0 = torch.randn(layers, batch_size, RNN_HIDDEN_SIZE)
            inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        models = [make_model(l, p) for l, p in zip(layers, packed_sequence)]
        inputs = [
            make_input(RNN_BATCH_SIZE, l, p) for l, p in zip(layers, packed_sequence)
        ]

        for model, input in zip(models, inputs):
            self.run_test(model, input)

    def test_gru_no_bias(self):
        # [FUNCTION DOC] Exports bias-free GRUs with two configurations (bidirectional 2-layer
        # and unidirectional 3-layer, different sizes), with constant folding enabled.
        class GruNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a bias-free GRU taking an input and
            # an initial state.
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                # [FUNCTION DOC] HELPER: Creates the bias-free GRU submodule with the given
                # sizes, layers and directionality.
                super().__init__()
                self.mygru = torch.nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers,
                    bidirectional=bidirectional,
                    bias=False,
                )

            def forward(self, input, initial_state):
                # [FUNCTION DOC] HELPER: Returns the GRU result for the input and initial state.
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # [FUNCTION DOC] HELPER: Builds a GRU net with a random input and initial state
            # sized by layers times directions; returns the model and the input tuple.
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        input_size = [7, 5]
        hidden_size = [3, 4]
        num_layers = [2, 3]
        batch_size = [3, 4]
        seq_len = [5, 7]
        bidirectional = [True, False]
        models_and_inputs = [
            get_GruNet_model_and_inputs(i, h, n, b, s, bi)
            for i, h, n, b, s, bi in zip(
                input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
            )
        ]
        for model, input in models_and_inputs:
            self.run_test(model, input, do_constant_folding=True)

    def test_gru_constant_folding(self):
        # [FUNCTION DOC] Exports GRUs with bias, a bidirectional 2-layer and a unidirectional
        # 3-layer, with constant folding.
        class GruNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a GRU taking an input and an initial state.
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                # [FUNCTION DOC] HELPER: Creates the GRU submodule with the given sizes,
                # layers and directionality.
                super().__init__()
                self.mygru = torch.nn.GRU(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state):
                # [FUNCTION DOC] HELPER: Returns the GRU result for the input and initial state.
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            # [FUNCTION DOC] HELPER: Builds a GRU net with a random input and initial state
            # sized by layers times directions; returns the model and the input tuple.
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        batch_size1 = 3
        model1, input1 = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_max_tensors(self):
        # [FUNCTION DOC] Exports elementwise max of two tensors with broadcasting between 4 x 4
        # and 4 x 1. Requires opset 8 or newer.
        class MaxModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the elementwise maximum of two tensors.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the elementwise maximum.
                return torch.max(input, other)

        model = MaxModel()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    def test_amax_amin(self):
        # [FUNCTION DOC] Exports amax over dim 0 with keepdim and amin over dims 0 and
        # 1 without keepdim.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning an amax and an amin reduction.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the amax over dim 0 keeping the dim and the
                # amin over dims 0 and 1.
                return torch.amax(x, dim=0, keepdim=True), torch.amin(
                    x, dim=[0, 1], keepdim=False
                )

        model = Model()
        x = torch.randn(4, 4)
        self.run_test(model, x)

    def test_aminmax(self):
        # [FUNCTION DOC] Exports aminmax along dim 1 with keepdim and over all elements.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two aminmax results.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the min and max along dim 1 keeping the dim,
                # and the global min and max.
                return torch.aminmax(x, dim=1, keepdim=True), torch.aminmax(
                    x, keepdim=False
                )

        model = Model()
        x = torch.randn(3, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        # [FUNCTION DOC] Exports arange of the first dimension size as float, viewed as a column
        # and added to the input, both scripted and traced. Requires opset 9 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding a float arange column to the input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning a float arange over dim 0
                # viewed as a column plus the input.
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a float arange column to the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns a float arange over dim 0 viewed as a column
                # plus the input.
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_end_notype(self):
        # [FUNCTION DOC] Exports arange of the first dimension size without dtype, scripted and
        # traced, with dynamic axes and with no remaining inputs. Requires opset 11 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning arange over the first dimension size.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning the arange.
                return torch.arange(a.size(0))

        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(ArangeScript(), x, remained_onnx_input_idx=[])

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning arange over the first dimension size.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the arange.
                return torch.arange(a.size(0))

        self.run_test(ArangeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(ArangeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end(self):
        # [FUNCTION DOC] Exports arange with start 2 and end from the size plus 2 as
        # float, viewed as a column and added to the input, scripted and traced. Requires
        # opset 9 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding a float arange from 2 to size plus 2 as a
            # column to the input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning that arange column
                # plus the input.
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a float arange from 2 to size plus 2 as a column
            # to the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns that arange column plus the input.
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_notype(self):
        # [FUNCTION DOC] Same as the start-end test but with a float start 2.7 and no dtype.
        # Requires opset 11 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding an arange from 2.7 to size plus 2 as a
            # column to the input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning that arange column
                # plus the input.
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an arange from 2.7 to size plus 2 as a column
            # to the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns that arange column plus the input.
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end_step(self):
        # [FUNCTION DOC] Exports arange with start 2, an end and a step derived from input sizes
        # as float, viewed as a column and added to the input, scripted and traced. Requires
        # opset 9 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding a float arange with size-derived end and
            # step to the input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning that arange column
                # plus the input.
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a float arange with size-derived end and step
            # to the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns that arange column plus the input.
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_step_notype(self):
        # [FUNCTION DOC] Same as the start-end-step test but with start 2.7 and no dtype.
        # Requires opset 11 or newer.
        class ArangeScript(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding an arange from 2.7 with size-derived end
            # and step to the input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Script method returning that arange column
                # plus the input.
                return (
                    torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an arange from 2.7 with size-derived end and
            # step to the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns that arange column plus the input.
                return (
                    torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1)
                    + a
                )

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test__dim_arange(self):
        # [FUNCTION DOC] Exports the internal _dim_arange operator (arange over the second
        # dimension) with dynamic axes, then with a remaining-input list that is empty for opset
        # 11 or newer and unset otherwise. Requires opset 9 or newer.
        class DimArange(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the range over dim 1 of the input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the internal dim arange of the input
                # along dim 1.
                return torch._dim_arange(input, 1)

        x = torch.ones(5, 6)
        self.run_test(DimArange(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        remained_onnx_input_idx = None if self.opset_version < 11 else []
        self.run_test(DimArange(), x, remained_onnx_input_idx=remained_onnx_input_idx)

    def _test_compare_ops(self, model, num_inputs):
        # [FUNCTION DOC] HELPER: Shared body for comparison tests. With two inputs runs the
        # model on all combinations of a float 4D and an int32 2D tensor as first and second
        # argument; with one input runs on a float tensor and an int tensor.
        x_float = torch.randn(1, 2, 3, 4, requires_grad=True)
        x_int = torch.randint(10, (3, 4), dtype=torch.int32)
        if num_inputs > 1:
            y_float = torch.randn(1, 2, 3, 4, requires_grad=True)
            y_int = torch.randint(10, (3, 4), dtype=torch.int32)
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_and_or_xor(self):
        # [FUNCTION DOC] Exports xor, or, and and invert operators on boolean tensors. Requires
        # opset 9 or newer.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning bitwise xor, or, and of two boolean tensors
            # and the inversion of the first.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a tuple of x xor y, x or y, x and y and not x.
                return x ^ y, x | y, x & y, ~x

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(MyModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_and(self):
        # [FUNCTION DOC] Exports logical_and for boolean, int32, double, and float with long
        # mixed inputs. Requires opset 9 or newer.
        class AndModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying logical and.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the logical and of the two inputs.
                return torch.logical_and(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(AndModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(AndModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_or(self):
        # [FUNCTION DOC] Exports logical_or for boolean, int32, double and mixed float and long
        # inputs. Requires opset 9 or newer.
        class OrModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying logical or.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the logical or of the two inputs.
                return torch.logical_or(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(OrModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(OrModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_xor(self):
        # [FUNCTION DOC] Exports logical_xor for boolean, int32, double and mixed float and long
        # inputs. Requires opset 9 or newer.
        class XorModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying logical xor.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the logical xor of the two inputs.
                return torch.logical_xor(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(XorModel(), input_args=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(XorModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_not(self):
        # [FUNCTION DOC] Exports logical_not for boolean, int32, double and float inputs.
        # Requires opset 9 or newer.
        class NotModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying logical not.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the logical not of the input.
                return torch.logical_not(x)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(NotModel(), input_args=(x,))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        self.run_test(NotModel(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(11)  # float equal added after opset 11
    def test_eq(self):
        # [FUNCTION DOC] Exports the equality comparison using the shared comparison helper with
        # two inputs. Requires opset 11 or newer because float equality was added then.
        class EqualModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing two tensors for equality.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the elementwise equality of input and other.
                return input == other

        self._test_compare_ops(EqualModel(), 2)

    def test_gt(self):
        # [FUNCTION DOC] Exports the greater-than comparison with two inputs using the
        # shared comparison helper.
        class GreaterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing two tensors with greater-than.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns whether input is greater than other.
                return input > other

        self._test_compare_ops(GreaterModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge(self):
        # [FUNCTION DOC] Exports greater-or-equal with two inputs using the shared helper.
        # Requires opset 9 or newer.
        class GreaterOrEqualModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing two tensors with greater-or-equal.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns whether input is greater than or
                # equal to other.
                return input >= other

        self._test_compare_ops(GreaterOrEqualModel(), 2)

    def test_gt_scalar(self):
        # [FUNCTION DOC] Exports greater-than against the scalar 1 with one input using
        # the shared helper.
        class GreaterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing the input with the scalar
            # 1 using greater-than.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns whether input is greater than 1.
                return input > 1

        self._test_compare_ops(GreaterModel(), 1)

    def test_gt_primitive(self):
        # [FUNCTION DOC] Exports a comparison between a stored integer attribute and an integer
        # input, run with the value 3.
        class GreaterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute 2 compared with
            # an integer argument.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the integer attribute 2.
                super().__init__()
                self.y: int = 2

            def forward(self, x: int):
                # [FUNCTION DOC] HELPER: Returns whether the stored integer is greater
                # than the argument.
                return self.y > x

        x = 3
        self.run_test(GreaterModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge_scalar(self):
        # [FUNCTION DOC] Exports greater-or-equal against scalar 1 with one input. Requires
        # opset 9 or newer.
        class GreaterOrEqualModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing the input with the scalar
            # 1 using greater-or-equal.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns whether input is greater than or equal to 1.
                return input >= 1

        self._test_compare_ops(GreaterOrEqualModel(), 1)

    def test_lt(self):
        # [FUNCTION DOC] Intended to export the less-than comparison on two inputs with
        # the shared helper.
        # KNOWN ISSUE: the module actually returns a greater-than comparison, so the less-than
        # path is not exercised by this test.
        class LessModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module named for less-than but comparing with greater-than.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns whether input is greater than other
                # (not less than).
                return input > other

        self._test_compare_ops(LessModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le(self):
        # [FUNCTION DOC] Exports less-or-equal on two inputs using the shared helper. Requires
        # opset 9 or newer.
        class LessOrEqualModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing two tensors with less-or-equal.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns whether input is less than or equal to other.
                return input <= other

        self._test_compare_ops(LessOrEqualModel(), 2)

    def test_lt_scalar(self):
        # [FUNCTION DOC] Exports less-than against the scalar 1 with one input using
        # the shared helper.
        class LessModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing the input with the scalar 1 using less-than.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns whether input is less than 1.
                return input < 1

        self._test_compare_ops(LessModel(), 1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le_scalar(self):
        # [FUNCTION DOC] Exports less-or-equal against scalar 1 with one input. Requires opset
        # 9 or newer.
        class LessOrEqualModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module comparing the input with the scalar
            # 1 using less-or-equal.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns whether input is less than or equal to 1.
                return input <= 1

        self._test_compare_ops(LessOrEqualModel(), 1)

    def test_matmul(self):
        # [FUNCTION DOC] Exports matmul for float 3 x 4 by 4 x 5 and integer matrices of
        # the same shapes.
        class MatmulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matrix product of two tensors.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the matmul of input and other.
                return torch.matmul(input, other)

        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        # [FUNCTION DOC] Exports batched matmul for float and integer 3D tensors of shapes 2 x 3
        # x 4 and 2 x 4 x 5.
        class MatmulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the batched matrix product of two tensors.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the matmul of input and other.
                return torch.matmul(input, other)

        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def _argmin_argmax_model(self, input):
        # [FUNCTION DOC] HELPER: Shared body for argmin and argmax tests. Builds a module
        # returning six results (global argmin and argmax, with keepdim, and dim-specific ones)
        # and runs it on the given input.
        class ArgminArgmaxModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning six argmin and argmax variants.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of argmin and argmax over all elements,
                # with keepdim, and argmin on dim 0 and argmax on dim 1 with keepdim.
                return (
                    torch.argmin(input),
                    torch.argmax(input),
                    torch.argmin(input, keepdim=True),
                    torch.argmax(input, keepdim=True),
                    torch.argmin(input, dim=0, keepdim=True),
                    torch.argmax(input, dim=1, keepdim=True),
                )

        self.run_test(ArgminArgmaxModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_argmin_argmax(self):
        # [FUNCTION DOC] Runs the argmin and argmax checks on a random 7 x 3 x 5 tensor.
        # Requires opset 9 or newer.
        input = torch.randn(7, 3, 5)
        self._argmin_argmax_model(input)

    # Argmin and Argmax with "select_last_index" is not supported before opset 12
    # "select_last_index" was added in opset 12 to deal with corner case where the
    # same value appears multiple times in the tensor
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_argmin_argmax_select_last_index(self):
        # [FUNCTION DOC] Runs the argmin and argmax checks on tensors with repeated values (a
        # small matrix and a ones tensor) to exercise the last-index selection added in opset
        # 12. Requires opset 12 or newer.
        input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
        self._argmin_argmax_model(input)

        input = torch.ones(7, 3, 5)
        self._argmin_argmax_model(input)

    def test_repeat(self):
        # [FUNCTION DOC] Exports repeat of a vector a number of times taken from another
        # tensor's size, added to a column view of that tensor.
        class RepeatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating x rows by the first dimension size of y and
            # adding a column view of y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Repeats x along a new leading dim as many times as y
                # has elements, reshapes y to a column and returns the sum.
                x2 = x.repeat(y.shape[0], 1)
                y1 = y.view(-1, 1)
                return x2 + y1

        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 8, 9])
        self.run_test(RepeatModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave(self):
        # [FUNCTION DOC] Exports repeat_interleave in several forms: flattened with repeat 2
        # for 1D, 2D and 3D inputs, scalar repeats along dim 1, tensor repeats of one entry
        # along dim 1, and tensor repeats with two entries along dim 0 and dim 1. Requires
        # opset 9 or newer.
        class FlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating each element twice on the flattened input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the flattened repeat_interleave of the input
                # with repeat 2.
                return x.repeat_interleave(2)

        for shape in ([3], [3, 4], [2, 3, 4]):
            x = torch.randn(shape)
            self.run_test(FlattenModel(), (x,))

        class DimsModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating each element four times along dim 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input repeat-interleaved 4 times
                # along dim 1.
                return x.repeat_interleave(4, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel(), (x,))

        class DimsModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 1 using a one-element repeats
            # tensor with value 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave with the tensor repeats
                # along dim 1.
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel2(), (x,))

        class RepeatsDimsModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating rows along dim 0 with per-row repeats
            # 1 and 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns repeat_interleave along dim 0
                # with per-entry repeats.
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel(), (x,))

        class RepeatsDimsModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating columns along dim 1 with per-column repeats
            # 1 and 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns repeat_interleave along dim 1
                # with per-entry repeats.
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel2(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave_noop(self):
        # [FUNCTION DOC] Exports repeat_interleave with repeat 1, which does nothing, on dim 1
        # of a 4 x 1 x 8 tensor. Requires opset 9 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating each element once along dim 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input repeat-interleaved once along dim 1.
                return x.repeat_interleave(1, dim=1)

        x = torch.randn(4, 1, 8)
        self.run_test(Model(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_dynamic_repeat_interleave(self):
        # [FUNCTION DOC] Exports repeat_interleave with dynamic input sizes or dynamic repeats:
        # scalar repeat 4 on dim 1 and dim -1, one-element repeats on dim 0 for float, repeats
        # supplied as input, repeats as the only dynamic input, and flattened repeat with a
        # dynamic length. Requires opset 13 or newer.
        class SingleDynamicModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 1 with a scalar tensor repeat of 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave along dim 1
                # with repeats 4.
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(
            SingleDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class NegDynamicModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along the last dimension with a scalar tensor
            # repeat of 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave along dim -1
                # with repeats 4.
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=-1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(
            NegDynamicModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "w"}},
        )

        class SingleDynamicModelFloat(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating a float input along dim 0 with a one-element
            # repeats tensor of 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave along dim 0.
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        another_x = torch.tensor([[7.1, 8.1], [5.1, 6.1]])
        self.run_test(
            SingleDynamicModelFloat(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "h"}},
        )

        class DynamicRepeatsModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 1 with repeats given as an input.
            def forward(self, x, repeats):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave of x with the repeats
                # input along dim 1.
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(another_x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"input_1": {1: "w"}, "repeats_1": {0: "r"}},
        )

        class DynamicRepeatsModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 1 with repeats given as an input;
            # used with only the repeats axis dynamic.
            def forward(self, x, repeats):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave of x with the repeats
                # input along dim 1.
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

        class DynamicFlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating each element twice on the flattened input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the flattened repeat_interleave of the input
                # with repeat 2.
                return x.repeat_interleave(2)

        x = torch.tensor([1, 2, 3])
        self.run_test(
            DynamicFlattenModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": {0: "w"}},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_multiple_dynamic_repeat_interleave(self):
        # [FUNCTION DOC] Exports repeat_interleave with multi-entry repeats given as input,
        # along dim 1 (three repeats) and along dim 0 (two repeats), with a dynamic repeats
        # axis. Requires opset 13 or newer.
        class DynamicRepeatsModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 1 with a repeats input.
            def forward(self, x, repeats):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave of x with the repeats
                # input along dim 1.
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3, 4])
        another_repeats = torch.tensor([4, 3, 2])
        self.run_test(
            DynamicRepeatsModel(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

        class DynamicRepeatsModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeating along dim 0 with a repeats input.
            def forward(self, x, repeats):
                # [FUNCTION DOC] HELPER: Returns the repeat_interleave of x with the repeats
                # input along dim 0.
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3])
        another_repeats = torch.tensor([4, 3])
        self.run_test(
            DynamicRepeatsModel2(),
            (x, repeats),
            additional_test_inputs=[(x, another_repeats)],
            input_names=["input_1", "repeats_1"],
            dynamic_axes={"repeats_1": {0: "r"}},
        )

    def test_view(self):
        # [FUNCTION DOC] Exports view to a fixed 4 x 24 shape on an int32 4D tensor.
        class ViewModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping the input into 4 x 24.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input viewed as 4 x 24.
                return input.view(4, 24)

        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_view_dynamic(self):
        # [FUNCTION DOC] Exports view where the target shape comes from another tensor's shape,
        # with dynamic axes on both inputs and then with only the first input remaining.
        class ViewModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module viewing the input with the shape of another tensor.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the input viewed with the shape of other.
                return input.view(other.shape)

        x = torch.randn(2, 3, 4)
        shape = torch.randn(6, 4)
        self.run_test(
            ViewModel(),
            (x, shape),
            input_names=["x", "shape"],
            dynamic_axes={"x": [0, 1, 2], "shape": [0, 1]},
        )
        self.run_test(ViewModel(), (x, shape), remained_onnx_input_idx=[0])

    def test_view_dynamic_zero_dim(self):
        # [FUNCTION DOC] Exports two consecutive views with inferred sizes, using a dynamic axis
        # and an additional empty input of length zero.
        class ViewModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module viewing the input as rows of two and then as
            # a single row.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input viewed as N x 2 and then as 1 x N.
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        another_x = torch.empty((0,))
        self.run_test(
            ViewModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={
                "input_1": [
                    0,
                ]
            },
        )

    def test_view_as(self):
        # [FUNCTION DOC] Exports view_as with a target tensor of shape 6 x 4.
        class ViewModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module viewing the input like another tensor.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the input viewed as other.
                return input.view_as(other)

        x = torch.randn(2, 3, 4)
        y = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, y))

    def test_linear(self):
        # [FUNCTION DOC] Exports a module applying the same linear layer twice, then functional
        # linear with inputs of rank 2 and rank 3.
        class LinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 16 to 16 linear layer applied twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.fc = torch.nn.Linear(16, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer twice and returns the result.
                out = self.fc(x)
                out = self.fc(out)
                return out

        x = torch.randn(3, 16)
        self.run_test(LinearModel(), (x,))

        class LinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling functional linear with explicit
            # weight and bias.
            def forward(self, input, weight, bias):
                # [FUNCTION DOC] HELPER: Returns the functional linear of input,
                # weight and bias.
                return torch.nn.functional.linear(input, weight, bias)

        # input of rank 2
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))

        # input of rank 3
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))

    @skipScriptTest()
    def test_weight_norm(self):
        # [FUNCTION DOC] Exports weight-normalized layers: a linear layer on 3D and 2D
        # input (dim 1), convolutions with default, negative dim and named weight.
        # Skipped under scripting.
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3))
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3), dim=-2)
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 6, 3), name="weight")
        x = torch.randn(3, 3, 5, requires_grad=True)
        self.run_test(model, x)

    @skipScriptTest()
    def test_weight_norm_nodim(self):
        # [FUNCTION DOC] Exports a weight-normalized linear layer with dim None on 3D and 2D
        # inputs. Skipped under scripting.
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

    def test_flatten(self):
        # [FUNCTION DOC] Exports flatten of all dims for 4D integer, 0D and 1D inputs.
        class FlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening its input completely.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the flattened input.
                return torch.flatten(input)

        model = FlattenModel()

        # flatten with 4d input
        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(model, x)

        # flatten with 0d input
        x = torch.randn([])
        self.run_test(model, x)

        # flatten with 1d input
        x = torch.randn(4)
        self.run_test(model, x)

    def test_flatten2d(self):
        # [FUNCTION DOC] Exports flatten from dim 1 onward on an integer 4D tensor.
        class FlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening all dims from the second.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input flattened from dim 1.
                return torch.flatten(input, 1)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d_neg(self):
        # [FUNCTION DOC] Exports flatten with negative end dims and varying start dims on an
        # integer 4D tensor.
        class FlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning three flatten results using
            # negative end dims.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of flatten from 1 to -1, from 0 to -2
                # and from 1 to -2.
                return (
                    torch.flatten(x, 1, -1),
                    torch.flatten(x, 0, -2),
                    torch.flatten(x, 1, -2),
                )

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_flatten_dynamic_axes(self):
        # [FUNCTION DOC] Exports flatten of dims 2 and 3 with a dynamic batch axis on input and
        # output and an additional input of different batch size. Requires opset 9 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening dims 2 and 3.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input flattened between dims 2 and 3.
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_getitem(self):
        # [FUNCTION DOC] Exports a scripted module that builds a list of three tensors with
        # different first dimensions and selects one by an index tensor, with positive and
        # negative indices. Requires opset 11 or newer.
        class GetItemModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule choosing one of three tensors by index.
            @torch.jit.script_method
            def forward(self, x, y, z, ind):
                # [FUNCTION DOC] HELPER: Script method placing x, y and z in a list and
                # returning the entry at the index.
                # this will create prim::ListConstruct(x, y, z) + aten::__getitem__
                arr = [x, y, z]
                return arr[ind]

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        ind = torch.tensor(1, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

        ind = torch.tensor(-2, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

    @skipDtypeChecking
    def test_item(self):
        # [FUNCTION DOC] Exports a scripted function that extracts a scalar via item from an
        # indexed element and returns it as an integer. Dtype checking disabled.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module indexing x by an entry of y selected by integer i and
            # returning its scalar value as an integer.
            def forward(self, x, y, i: int):
                # [FUNCTION DOC] HELPER: Returns the integer value of x indexed with y
                # at position i.
                return int(x[y[i]].item())

        x = torch.arange(6, dtype=torch.float)
        y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        i = 3
        self.run_test(torch.jit.script(M()), (x, y, i))

    @skipScriptTest()  # torch.nonzero(x, as_tuple=True) is not scriptable.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nonzero(self):
        # [FUNCTION DOC] Exports nonzero both as a matrix of indices and as a tuple per
        # dimension on a sparse tensor. Skipped under scripting because the tuple form is not
        # scriptable; requires opset 9 or newer.
        class NonzeroModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning nonzero indices in both forms.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair: the nonzero index matrix and the tuple
                # of index vectors.
                return x.nonzero(), x.nonzero(as_tuple=True)

        x = torch.randn(60).index_fill_(0, torch.randint(0, 60, (20,)), 0).view(3, 4, 5)
        self.run_test(NonzeroModel(), (x,))

    def test_unbind(self):
        # [FUNCTION DOC] Exports unbind on dim 0 of a 3 x 4 x 5 tensor, selecting the middle
        # slice, and also variants unbinding on dims 1 and -2.
        class UnbindModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unbinding dim 0 into three slices and returning
            # the middle one.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the second slice from unbinding the input
                # along dim 0.
                _, out, _ = input.unbind()
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unbinding dim 1 into four slices and
            # returning the second.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the second slice from unbinding the input
                # along dim 1.
                _, out, _, _ = input.unbind(1)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

        class UnbindModel3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unbinding dim -2 into four slices and
            # returning the second.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the second slice from unbinding the input
                # along dim -2.
                _, out, _, _ = input.unbind(-2)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_len(self):
        # [FUNCTION DOC] Exports a scripted module adding the number of unbound slices of the
        # input to the input, with a dynamic sequence axis and an additional input of a
        # different length. Requires opset 11 or newer.
        class LenModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding the length of the unbound list of the
            # input to the input.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the number of slices from
                # unbinding dim 0 plus the input.
                return len(input.unbind()) + input

        x = torch.randn(4, 5)
        self.run_test(
            LenModel(),
            x,
            input_names=["input"],
            dynamic_axes={"input": {0: "seq"}},
            additional_test_inputs=(torch.randn(5, 5),),
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_len_list(self):
        # [FUNCTION DOC] Exports a scripted module creating a ones tensor whose length is the
        # rank of the input; no input remains in the ONNX graph. Requires opset 9 or newer.
        class LenListModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning ones with length equal to the number of
            # dimensions of the input.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning a ones vector sized by the rank
                # of the input.
                return torch.ones(len(input.shape))

        x = torch.randn(4, 5)
        self.run_test(LenListModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unbind_dynamic(self):
        # [FUNCTION DOC] Exports scripted modules that unbind and then index the result (second
        # slice on dim 0 and on the last dim). Requires opset 11 or newer.
        class UnbindModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the second slice from unbinding dim 0.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the slice at position 1 of
                # the unbound input.
                return input.unbind()[1]

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the second slice from unbinding
            # the last dimension.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the slice at position 1 of the
                # input unbound on dim -1.
                return input.unbind(-1)[1]

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_split_script
    def test_split(self):
        # [FUNCTION DOC] Exports split with a list of sizes in tracing: sizes 2,1,2 and the
        # first of a 3,2 split; sizes on dim -2; and a plain list split. Skipped under
        # scripting, which is covered by the script variant for opsets above 11.
        class SplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a three-way split of dim 0 and the first part
            # of a two-way split.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 2 and the first
                # element of the split with sizes 3 and 2.
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting along dim -2.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 1 and the last part
                # of a split with sizes 2, 2, both along dim -2.
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a three-way split of dim 0.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 2.
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_script(self):
        # [FUNCTION DOC] Same split cases as the tracing test, run with the script export mode
        # for opsets from 11. Requires opset 11 or newer.
        class SplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a three-way split of dim 0 and the first part
            # of a two-way split.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 2 and the first
                # element of the split with sizes 3 and 2.
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting along dim -2.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 1 and the last part
                # of a split with sizes 2, 2, both along dim -2.
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a three-way split of dim 0.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the split with sizes 2, 1, 2.
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_split_size_as_list(self):
        # [FUNCTION DOC] Exports split with split sizes passed as a list of scalar tensors,
        # concatenating all parts. Skipped under scripting; requires opset 11 or newer.
        class SplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting the input by a list of sizes and
            # concatenating the pieces along dim 0.
            def forward(self, input, split_sizes: list[int]):
                # [FUNCTION DOC] HELPER: Splits the input by the given sizes, appends each part
                # to a list and returns their concatenation along dim 0.
                out = []
                split_list: list[Tensor] = input.split(split_sizes)

                for ob in split_list:
                    out.append(ob)  # noqa: PERF402
                return torch.cat(out, dim=0)

        x = torch.randn(6, 4, 3)
        split_sizes = [torch.tensor(2), torch.tensor(4)]
        self.run_test(SplitModel(), (x, split_sizes))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_size_with_slice(self):
        # [FUNCTION DOC] Exports split where sizes come from the dim 1 sizes of two inputs,
        # with dynamic axes on all inputs and with only the third input remaining. Requires
        # opset 11 or newer.
        class SplitModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting a tensor on dim 1 into parts sized like
            # two other inputs.
            def forward(self, x, y, t):
                # [FUNCTION DOC] HELPER: Splits t on dim 1 by the second-dimension sizes of x
                # and y and returns both parts.
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        self.run_test(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        self.run_test(SplitModule(), (x, y, t), remained_onnx_input_idx=[2])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic(self):
        # [FUNCTION DOC] Exports scripted modules that split with a fixed chunk size and index
        # the second chunk, on dim 0 and dim -3. Requires opset 11 or newer.
        class SplitModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the second chunk of the input split
            # into pieces of size two.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the chunk at position 1.
                return input.split(2)[1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the second chunk of a split
            # along dim -3.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the chunk at position 1 for
                # split size 2 on dim -3.
                return input.split(2, -3)[1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic_axes(self):
        # [FUNCTION DOC] Exports splitting into size-1 pieces along the last dim with a dynamic
        # batch axis. Requires opset 11 or newer.
        class Split(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting into pieces of size one along
            # the last dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the split of the input with size 1
                # along dim -1.
                return x.split(1, dim=-1)

        x = torch.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            Split(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_chunk(self):
        # [FUNCTION DOC] Exports chunk into 3 parts along dim 1 and dim -1 with a dynamic axes
        # on batch and dim, checking inputs whose dim sizes are 13 to 15 as additional inputs.
        # Requires opset 11 or newer.
        class ChunkModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module chunking its input into three pieces along
            # a stored dimension.
            def __init__(self, dim=1):
                # [FUNCTION DOC] HELPER: Stores the chunk dimension (default 1).
                super().__init__()
                self.dim = dim

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input chunked into 3 parts along
                # the stored dimension.
                return torch.chunk(x, 3, dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(1, 18)

        for dim_size_ in range(13, 16):
            y = torch.randn(1, dim_size_)
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_chunk(self):
        # [FUNCTION DOC] Exports chunk where the number of chunks equals the first dimension
        # size, for dim 1 and dim -1, checking additional inputs with dim sizes 13 to 15.
        # Requires opset 11 or newer.
        class ChunkModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module chunking its input into as many pieces as its first
            # dimension size along a stored dimension.
            def __init__(self, dim=1):
                # [FUNCTION DOC] HELPER: Stores the chunk dimension (default 1).
                super().__init__()
                self.dim = dim

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input chunked into dim-0-size parts along
                # the stored dimension.
                return torch.chunk(x, x.size(0), dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(3, 18)

        for dim_size_ in range(13, 16):
            y = torch.randn(3, dim_size_)
            self.run_test(
                model,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

            self.run_test(
                model_neg_dim,
                x,
                additional_test_inputs=[y],
                input_names=["x"],
                dynamic_axes={"x": {0: "batch_size", 1: "dims"}},
            )

    def test_concat(self):
        # [FUNCTION DOC] Exports concatenation of three tensors with different first dimensions.
        class ConcatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating three tensors along dim 0.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the concatenation of x, y and z.
                return torch.cat((x, y, z))

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        self.run_test(ConcatModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_concat_dynamic(self):
        # [FUNCTION DOC] Exports a scripted module concatenating the slices from unbinding the
        # input. Requires opset 11 or newer.
        class ConcatDynamicModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule concatenating all unbound slices of the input.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the concatenation of
                # the unbound input.
                return torch.cat(x.unbind())

        x = torch.randn(4, 5, 6)
        self.run_test(ConcatDynamicModel(), x)

    def test_stack(self):
        # [FUNCTION DOC] Exports stacking three equal-shaped tensors along dim 1.
        class StackModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking three tensors along dim 1.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the stack of x, y and z along dim 1.
                return torch.stack((x, y, z), 1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        z = torch.randn(3, 4, 5)
        self.run_test(StackModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_stack_dynamic(self):
        # [FUNCTION DOC] Exports a scripted module stacking all unbound slices of the input
        # along dim 1. Requires opset 11 or newer.
        class StackDynamicModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule stacking the unbound slices of the input
            # along dim 1.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the stack of the unbound input.
                return torch.stack(x.unbind(), 1)

        x = torch.randn(4, 5, 6)
        self.run_test(StackDynamicModel(), x)

    def test_loop_dynamic(self):
        # [FUNCTION DOC] Exports a scripted loop whose trip count is the size of dim 2, adding
        # the loop index to the input each iteration, on a long zeros tensor.
        class LoopModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule looping over the size of dim 2 adding
            # the loop counter.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method adding each loop index to x
                # and returning x.
                for i in range(x.size(2)):
                    x = x + i
                return x

        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_loop_nested(self):
        # [FUNCTION DOC] Exports a scripted loop nested with a while loop that counts to four,
        # adding it to the input in each of five outer iterations. Requires opset 9 or newer.
        class NestedLoopsModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a for loop of five iterations containing
            # a while loop.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method that, five times, counts a variable up to
                # 4 in a while loop, adds it to x, and returns x.
                for _ in range(5):
                    a = 0
                    while a < 4:
                        a += 1
                    x = x + a
                return x

        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_with_list(self):
        # [FUNCTION DOC] Exports a scripted loop that builds several lists and a tensor
        # accumulator while iterating over a split of the input, returning lists, a tensor and
        # two stacked results. Requires opset 11 or newer.
        class ListLoopModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule looping over a seven-way split and accumulating
            # results into lists and a tensor.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method splitting the input into seven parts,
                # then for each part appends sums from the front and from the back to two lists,
                # increments a long accumulator and extends two more lists; returns the lists,
                # the accumulator and the two stacks.
                res = []
                res1 = []
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                res2 = torch.zeros(3, 4, dtype=torch.long)
                res3 = []
                res4 = []
                for i in range(len(arr)):
                    res.append(arr[i].sum(0, False))
                    res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                    res3 = res3 + [arr[i].sum(0, False)]
                    res4 += [arr[-1 - i].sum(0, False)]
                return res, res1, res2, torch.stack(res3), torch.stack(res4)

        model = ListLoopModel()
        inputs = torch.randn(16)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_transpose(self):
        # [FUNCTION DOC] Exports a scripted loop that adds the transposed first slice to an
        # accumulator once per first-dimension entry. Requires opset 11 or newer.
        class LoopModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing the transpose of the first slice repeatedly.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Starts from zeros like the first slice and adds that
                # slice transposed once per entry of dim 0; returns the sum.
                res = torch.zeros_like(x[0])
                for _ in range(x.size(0)):
                    res += x[0].transpose(0, 1)
                return res

        model = torch.jit.script(LoopModel())
        x = torch.randn(5, 3, 3)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_multi_dim(self):
        # [FUNCTION DOC] Exports a scripted loop iterating over a flipped narrow of the
        # input, using an indexed element as a gather into y each iteration. Requires opset
        # 11 or newer.
        class LoopMultiDimModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule iterating over a reversed seven-row narrow of the
            # input and indexing with the running index tensor.
            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method that, for each row of the flipped
                # narrowed x, replaces y by the row's first entry indexed with y; returns y.
                for x_ in torch.flip(x.narrow(0, 0, 7), [0]):
                    y = x_[0][y]
                return y

        model = LoopMultiDimModel()
        x = torch.randint(0, 5, (8, 1, 17), dtype=torch.long)
        y = torch.ones(1, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list(self):
        # [FUNCTION DOC] Exports a scripted module that manipulates a Python-style list with
        # append, pop, insert, in-place add and list concatenation and returns ones of the final
        # length. Requires opset 11 or newer.
        class ListModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that edits a list of unbound tensors and returns
            # ones sized by the list length.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method building and editing a list from the
                # unbound slices through several list operations, returning ones of that length.
                tensors = x.unbind()
                res = []
                res.append(tensors[0])
                res.append(tensors[1])
                res.pop(1)

                res.insert(0, tensors[1])
                res.append(tensors[2])
                res += [tensors[3], tensors[4]]
                res = res + [tensors[5]]
                return torch.ones(len(res))

        model = ListModel()
        inputs = torch.randn(16, 1)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append(self):
        # [FUNCTION DOC] Exports a scripted module that appends matmul results into a list in a
        # loop and returns the list. Requires opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of per-row matrix products.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: For each row of x appends its matmul with y to a list
                # and returns the list.
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested(self):
        # [FUNCTION DOC] Exports a scripted module appending matmul results in nested loops over
        # the first two dimensions. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of matrix products from
            # two nested loops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: For each entry over dims 0 and 1 appends its matmul
                # with y to a list and returns it.
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)  # Need onnx::Identity of sequence in opset 14
    def test_list_append_nested_2(self):
        # [FUNCTION DOC] Exports a scripted module with list appends in a conditional inside a
        # loop, appending from nested loops and referencing earlier list items. Requires opset
        # 14 because an identity of sequences is needed.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling two lists with slices from nested loops once
            # the first list is long enough.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Loops over dim 0; while the first list has more
                # than two entries, appends all dim 1 slices to it, appends its last item to
                # the second list and appends the second list's last item to the first.
                # Returns both lists.
                res = []
                res_replicate = []
                for i in range(x.size(0)):
                    if len(res) > 2:
                        for j in range(x.size(1)):
                            res.append(x[i][j])
                        res_replicate.append(res[-1])
                        res.append(res_replicate[-1])
                return res, res_replicate

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested_mixed_dtype(self):
        # [FUNCTION DOC] Exports a scripted module appending boolean comparison results of
        # different shapes in nested loops. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of equality or inequality results
            # chosen by comparing loop indices.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Loops over dims 0 and 1; appends the equality of x and
                # y when the indices match, otherwise the inequality; returns the list.
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        if i == j:
                            res.append(x == y)
                        else:
                            res.append(x != y)
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(3, 4)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_pop(self):
        # [FUNCTION DOC] Exports a scripted module that builds a list of matmul results then
        # pops the last one. Requires opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of matrix products with the
            # last entry removed.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends the matmul of each row with y to a list, pops
                # the last element and returns the list.
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                res.pop()
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_nested(self):
        # [FUNCTION DOC] Exports a scripted module that pops list entries inside nested loops
        # and appends more entries in the outer loop. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list built with nested-loop
            # appends and pops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: For each outer index, for each inner index appends a
                # matmul result and pops it, then appends one more result for the first inner
                # entry; returns the list.
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        res.pop()
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_del(self):
        # [FUNCTION DOC] Exports a scripted module that appends matmul results then deletes the
        # entry at position 2. Requires opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of matrix products with
            # one entry deleted.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends the matmul of each row with y, deletes item 2
                # and returns the list.
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                del res[2]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_nested(self):
        # [FUNCTION DOC] Exports a scripted module with deletion of list entries by outer index
        # inside nested loops. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list built with nested-loop
            # appends and deletions.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: For each outer and inner index appends a matmul result
                # and deletes the item at the outer index, then appends one more result per
                # outer iteration; returns the list.
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        del res[i]
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_set(self):
        # [FUNCTION DOC] Exports a scripted module that builds a list of rows and
        # overwrites one entry selected by a tensor index with the row at that index.
        # Requires opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of the rows of x with
            # one entry reassigned.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends each row of x to a list, sets the item at index
                # y to the row at index y and returns the list.
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                res[y] = x[y]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(12, 4)
        y = torch.tensor(2, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_idx_sum(self):
        # [FUNCTION DOC] Exports a scripted module that indexes a list of rows by the sum of a
        # prefix of an index range. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the row chosen by summing the first y entries
            # of an arange.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds an arange over the first dimension, collects
                # rows in a list and returns the entry at the sum of the first y indices.
                indices = torch.arange(x.size(0))
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                return res[torch.sum(indices[:y])]

        model = torch.jit.script(ListModel())
        x = torch.randn(12, 4)
        y = torch.tensor(2, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        # [FUNCTION DOC] Exports zeros plus ones created from the input size, run with dynamic
        # axes and with no remaining ONNX inputs. Requires opset 9 or newer.
        class TensorFactory(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding zeros and ones tensors built from
            # the input size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns zeros of the input size plus ones of
                # the input size.
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        # [FUNCTION DOC] Same as the factories test but through a ScriptModule using the shape
        # and an explicit float dtype. Requires opset 9 or newer.
        class TensorFactory(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding float zeros and ones tensors built from
            # the input shape.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning float zeros plus float ones of
                # the input shape.
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(
                    x.shape, dtype=torch.float
                )

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        # [FUNCTION DOC] Exports a scripted module using zeros_like and ones_like with explicit
        # dtype, strided layout and cpu device, with dynamic axes and no remaining inputs.
        # Requires opset 9 or newer.
        class TensorFactory(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding zeros_like and ones_like results
            # with explicit options.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the sum of float zeros_like and
                # ones_like of the input on the cpu.
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return zeros + ones

        x = torch.randn(2, 3, 4)
        self.run_test(
            TensorFactory(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split(self):
        # [FUNCTION DOC] Exports tensor_split with index lists in several forms: default dim,
        # output indexing, along dim -2, along dim -2 with output indexing, and with an end
        # index beyond the dimension size. Requires opset 13 or newer.
        class TensorSplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning five tensor_split variants.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of splits at indices 1 and 3, the first
                # part of a split at 2 and 4, a split along dim -2, the last part of another
                # split along dim -2, and a split with an out-of-range index.
                return (
                    input.tensor_split([1, 3]),
                    # test with output indexing.
                    input.tensor_split([2, 4])[0],
                    # test split on specific dim.
                    input.tensor_split([1, 3, 4], dim=-2),
                    # test split on specific dim and output indexing.
                    input.tensor_split([0, 2], dim=-2)[-1],
                    # test with out of bound end index (5).
                    input.tensor_split([2, 3, 5]),
                )

        self.run_test(TensorSplitModel(), torch.randn(5, 4, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_scalar(self):
        # [FUNCTION DOC] Exports tensor_split with a section count taken from the input's second
        # dimension size. Requires opset 13 or newer.
        class TensorSplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting the input into as many sections as its
            # dim 1 size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tensor_split of the input
                # into dim-1-size sections.
                return torch.tensor_split(x, x.size(1))

        self.run_test(TensorSplitModel(), torch.randn(1, 2, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_dynamic_axes(self):
        # [FUNCTION DOC] Exports tensor_split into one section along the last dimension with a
        # dynamic batch axis. Requires opset 13 or newer.
        class TensorSplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting the input into one section along
            # the last dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tensor_split of the input with one section
                # along dim -1.
                return x.tensor_split(1, dim=-1)

        x = torch.randn(4, 384, 2)
        input_names = ["logits"]
        self.run_test(
            TensorSplitModel(),
            x,
            input_names=input_names,
            dynamic_axes={input_names[0]: {0: "batch"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_eye(self):
        # [FUNCTION DOC] Exports eye with sizes from the input shape and several dtypes, using
        # dynamic axes and an additional input of different shape. Requires opset 9 or newer.
        class TensorFactory(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning five identity-like matrices with sizes
            # derived from the input or fixed.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of eye results: size from dim 1 by 3, a
                # fixed 4 x 4 long, size from dim 1 by 2 long, a square from dim 0, and a double
                # square from dim 0.
                return (
                    torch.eye(x.size()[1], 3),
                    torch.eye(4, 4, dtype=torch.long),
                    torch.eye(x.size()[1], 2, dtype=torch.long),
                    torch.eye(x.shape[0]),
                    torch.eye(x.shape[0], dtype=torch.float64),
                )

        x = torch.randn(2, 3, 4)
        another_x = torch.randn(5, 6, 7)
        self.run_test(
            TensorFactory(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_diagonal(self):
        # [FUNCTION DOC] Exports diagonal in six variants: default, negative offset, positive
        # offset, explicit dims, negative dims and offsets beyond the extent. Each uses dynamic
        # axes and an additional input of different shape. Requires opset 13 or newer.
        class DiagonalModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the default diagonal.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the diagonal of the input.
                return torch.diagonal(x)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModel(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelNegOffset(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the diagonal with offset -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the diagonal with offset -1.
                return torch.diagonal(x, offset=-1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelNegOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelPosOffset(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the diagonal with offset 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the diagonal with offset 1.
                return torch.diagonal(x, offset=1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelPosOffset(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithDims(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the diagonal with offset -1 taken over dims
            # 1 and 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the diagonal with offset -1 over dims 1 and 2.
                return torch.diagonal(x, offset=-1, dim1=1, dim2=2)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelWithNegativeDims(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the diagonal over the last two dims given
            # as negative values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the diagonal with offset 0 over dims -2 and -1.
                return torch.diagonal(x, offset=0, dim1=-2, dim2=-1)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelWithNegativeDims(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

        class DiagonalModelOffsetOverrun(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning diagonals with offsets whose range overruns
            # the matrix size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of diagonals with offsets -2 and 5.
                return torch.diagonal(x, offset=-2), torch.diagonal(x, offset=5)

        x = torch.randn(2, 4, 5, 2)
        # Other test inputs to test dynamic behavior
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(
            DiagonalModelOffsetOverrun(),
            x,
            additional_test_inputs=[another_x],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        # [FUNCTION DOC] Exports the in-place zero_ operator returning the modified tensor
        # and the original, with dynamic axes and with no remaining ONNX inputs. Requires
        # opset 9 or newer.
        class Zero_(torch.nn.Module):
            # [CLASS DOC] HELPER: Module zeroing its input in place and returning the result
            # plus the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of the zeroed input and the input.
                return x.zero_(), x

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_zero_qkv(self):
        # [FUNCTION DOC] Exports zeroing a slice of the first dimension in place and returning
        # the slice, with dynamic axes. Requires opset 11 or newer.
        class Zero_(torch.nn.Module):
            # [CLASS DOC] HELPER: Module zeroing rows 2 and 3 of the input in place.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns rows 2 to 3 of the input after zeroing
                # them in place.
                return x[2:4].zero_()

        x = torch.randn(24, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros(self):
        # [FUNCTION DOC] Exports new_zeros with shapes taken from slices of the input shape,
        # one with long dtype, using dynamic axes and with no remaining inputs. Requires opset
        # 9 or newer.
        class Zero_(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two new_zeros tensors with shapes from
            # the input shape.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns zeros with shape slice 1:2 and long zeros with
                # shape slice 2:.
                return x.new_zeros(x.shape[1:2]), x.new_zeros(
                    x.shape[2:], dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros_with_dtype(self):
        # [FUNCTION DOC] Checks new_zeros keeps the integer dtype of its input when used as
        # embedding indices, with dynamic axes. Requires opset 9 or newer.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that builds zeros like the integer input and looks them
            # up in an embedding of 50 by 64.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the embedding.
                super().__init__()
                self.emb = torch.nn.Embedding(50, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the embedding lookup of new zeros shaped and
                # typed like the input.
                inp = x.new_zeros(x.shape)
                return self.emb(inp)

        model = MyModel()
        x = torch.Tensor([[2, 5, 6], [3, 2, 5]]).to(torch.int64)
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_ones(self):
        # [FUNCTION DOC] Exports new_ones with shapes taken from slices of the input shape,
        # one with long dtype, using dynamic axes and with no remaining inputs. Requires opset
        # 9 or newer.
        class OnesModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two new_ones tensors with shapes from
            # the input shape.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ones with shape slice 1:2 and long ones with
                # shape slice 2:.
                return x.new_ones(x.shape[1:2]), x.new_ones(
                    x.shape[2:], dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(OnesModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(OnesModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # torch.zeros/torch.ones with size tensor of dim != 0 not scriptable.
    def test_zeros_ones_with_tensor_input(self):
        # [FUNCTION DOC] Exports zeros and ones created with a size given by a tensor input.
        # Skipped under scripting; requires opset 9 or newer.
        class ZeroAndOnes(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning zeros and ones whose first size comes
            # from a tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns zeros and ones of shape x by 1.
                return torch.zeros(x, 1), torch.ones(x, 1)

        x = torch.tensor([2])
        self.run_test(ZeroAndOnes(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipShapeChecking
    def test_tolist(self):
        # [FUNCTION DOC] Exports a scripted module converting an integer tensor to a Python list
        # of integers. Shape checking disabled; requires opset 9 or newer.
        class List(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule returning the input values as a list of integers.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning the tolist conversion as a
                # list of integers.
                res: list[int] = input.tolist()
                return res

        self.run_test(List(), (torch.randint(100, (1,)),))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_pass(self):
        # [FUNCTION DOC] Exports new_zeros with shapes built by combining shape pieces:
        # concatenated shape slices, concatenated shapes, explicit integer arrays, and
        # converted lists. Each runs with dynamic axes and with no remaining inputs. Requires
        # opset 9 or newer.
        class Slice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building zeros with a shape that joins slices of
            # two input shapes.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns new zeros with the shape slice of x from dim 2
                # joined with that of y from dim 1.
                return x.new_zeros(x.shape[2:] + y.shape[1:])

        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(1, 2, 3, 4)
        self.run_test(
            Slice(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(Slice(), (x, y), remained_onnx_input_idx=[])

        class Size(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building zeros with the joined shapes of both inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns new zeros with shape equal to the shape of x
                # followed by the shape of y.
                return x.new_zeros(x.shape + y.shape)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            Size(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Size(), (x, y), remained_onnx_input_idx=[])

        class Array(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building zeros with a shape from lists of integers.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds a list from two sizes of x and the constant
                # 2, another from two sizes of y, and returns new zeros with the joined
                # list as shape.
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            Array(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(Array(), (x, y), remained_onnx_input_idx=[])

        class List(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building zeros from shapes converted to lists.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Converts both shapes to lists, joins them and returns
                # new zeros with that shape.
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(
            List(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        self.run_test(List(), (x, y), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_empty(self):
        # [FUNCTION DOC] Exports new_empty filled with zero and multiplied by zero for long
        # dtype, with dynamic axes and with no remaining inputs. Requires opset 9 or newer.
        class Empty(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning an uninitialized tensor filled with 0 and an
            # uninitialized long tensor times 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of new_empty of size dim 0 filled with
                # 0 and a long new_empty times 0.
                return (
                    x.new_empty(x.shape[0]).fill_(0),
                    x.new_empty(x.shape[0], dtype=torch.long) * 0,
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Empty(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Empty(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_full(self):
        # [FUNCTION DOC] Exports new_full with shape from the input shape: fill 5 and fill
        # 1.3 with long dtype, using dynamic axes and with no remaining inputs. Requires
        # opset 9 or newer.
        class Full(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two new_full tensors.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns new_full with shape slice 1:2 filled with 5 and
                # long new_full with shape slice 0:1 filled with 1.3.
                return x.new_full(x.shape[1:2], 5), x.new_full(
                    x.shape[0:1], 1.3, dtype=torch.long
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Full(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Full(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_list(self):
        # [FUNCTION DOC] Exports a scripted module that concatenates an in-place added x and an
        # in-place zero-filled y, with dynamic axes on both inputs and with only the first input
        # remaining in the ONNX graph. Requires opset 9 or newer.
        class Arithmetic(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule concatenating the in-place add of 3 on x and the
            # in-place zero fill of y.
            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method that adds 3 to x in place, fills y with 0
                # in place and returns their concatenation.
                return torch.cat([x.add_(3), y.fill_(0)])

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(
            Arithmetic(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
        )
        self.run_test(Arithmetic(), (x, y), remained_onnx_input_idx=[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        # [FUNCTION DOC] Exports in-place fill with 3 returning the filled tensor and the input,
        # with dynamic axes and with no remaining inputs. Requires opset 9 or newer.
        class Fill_(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling its input with 3 in place and returning the
            # result and the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of the filled input and the input.
                return x.fill_(3), x

        x = torch.randn(2, 3, 4)
        self.run_test(Fill_(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(Fill_(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic(self):
        # [FUNCTION DOC] Exports a scripted module with in-place add and multiply on its inputs
        # and returns both.
        class Arithmetic(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding 3 to x in place then multiplying y by
            # x in place.
            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method that mutates x and y in place
                # and returns both.
                x.add_(3)
                y.mul_(x)
                return x, y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(Arithmetic(), (x, y))

    def test_inplace_arithmetic_half(self):
        # [FUNCTION DOC] Exports in-place add and multiply with a half precision tensor and a
        # float tensor, with relaxed tolerances of 1e-2.
        class InplaceAddModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding y into x in place.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the result of adding y into x in place.
                return x.add_(y)

        class InplaceMulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying x by y in place.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the result of multiplying x by y in place.
                return x.mul_(y)

        x = torch.randn(2, 2, dtype=torch.half)
        y = torch.randn(2, 2, dtype=torch.float)
        self.run_test(InplaceAddModel(), (x, y), rtol=1e-2, atol=1e-2)
        self.run_test(InplaceMulModel(), (x, y), rtol=1e-2, atol=1e-2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop(self):
        # [FUNCTION DOC] Exports a scripted module that adds ones to a ones vector in place
        # inside a ten-iteration loop and then adds the input. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that accumulates in place in a loop.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a ones vector of length 12, adds ones to it ten
                # times in place and returns it plus the input.
                a = torch.ones(
                    12,
                )
                for _ in range(10):
                    a.add_(
                        torch.ones(
                            12,
                        )
                    )
                return a + x

        m = M()
        x = torch.randn(
            12,
        )
        self.run_test(torch.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop_2(self):
        # [FUNCTION DOC] Exports a scripted module with several aliases mutated in and outside
        # loops: an accumulator, an alias of it, a copy of the input and an alias of that copy,
        # with a nested loop on the iteration where the counter is 3. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module testing in-place updates with aliasing in loops.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Loops ten times updating an accumulator and a bias in
                # place, with an inner five-iteration loop at counter 3; adds ones to an alias
                # of a copy; returns the bias plus input, the accumulator, the copy and its
                # alias. The in-place update of the first alias is commented out because its
                # value is incorrect.
                _bias = torch.ones(
                    12,
                )
                a = torch.ones(
                    12,
                )  # used in loop, altered.
                a_ref = a  # not used in loop, should be altered.
                b = x.clone()  # used in loop, not be altered.
                b_ref = b  # not used in loop, should not be altered.
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            a += _bias
                            _bias.add_(
                                torch.ones(
                                    12,
                                )
                            )
                            b = b + torch.ones(
                                12,
                            )

                    _bias.add_(
                        torch.ones(
                            12,
                        )
                    )
                    a += _bias
                # TODO: value for a_ref is incorrect.
                # a_ref += torch.ones(12,)
                b_ref += torch.ones(
                    12,
                )
                return _bias + x, a, b, b_ref

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(torch.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_with_loop(self):
        # [FUNCTION DOC] Exports a scripted module that updates a stored attribute tensor in
        # place in nested loops (in-place add of an arange) and adds it to the input. Requires
        # opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an arange attribute that is reassigned and
            # updated in loops.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the attribute tensor with values 0 to 11.
                super().__init__()
                self._bias = torch.arange(
                    12,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Resets the attribute to an arange, adds an arange to it
                # five times in place when the loop counter is 3 of ten iterations, and returns
                # the attribute plus the input.
                self._bias = torch.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias += torch.arange(
                                12,
                            )
                return self._bias + x

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(torch.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_copy_with_loop(self):
        # [FUNCTION DOC] Exports a scripted module that updates a stored attribute tensor with
        # in-place copy inside loops. Requires opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an arange attribute updated by copy_ in loops.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the attribute tensor with values 0 to 11.
                super().__init__()
                self._bias = torch.arange(
                    12,
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Resets the attribute and in ten iterations copies into
                # it an arange, itself plus an arange, with extra copies at counter 3; returns
                # the attribute plus the input.
                self._bias = torch.arange(
                    12,
                )
                for i in range(10):
                    if i == 3:
                        for _ in range(5):
                            self._bias.copy_(
                                torch.arange(
                                    12,
                                )
                            )
                        self._bias.copy_(
                            self._bias
                            + torch.arange(
                                12,
                            )
                        )

                    self._bias.copy_(
                        self._bias
                        + torch.arange(
                            12,
                        )
                    )
                return self._bias + x

        m = M()
        x = torch.zeros(
            12,
        )
        self.run_test(torch.jit.script(M()), (x))

    @skipIfUnsupportedMinOpsetVersion(14)  # Need onnx::Identity of sequence in opset 14
    def test_inplace_sequence_with_loop(self):
        # [FUNCTION DOC] Exports a scripted module that appends tokens to a list in loops
        # through a helper method, with a while loop over the sequence length. Requires opset 14
        # because of sequence identity.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module collecting tokens of the input into a list through a
            # helper method, called repeatedly in a while loop.
            def process(self, beam_hyps: list[Tensor], done: Tensor, x):
                # [FUNCTION DOC] HELPER: For each batch entry not marked done, appends up to six
                # tokens to the hypotheses list, then marks the entry done when more than four
                # tokens are collected. Returns the list and the done tensor.
                batch_size = x.shape[0]
                for i in range(batch_size):
                    if done[i]:
                        continue

                    beam_idx = 0
                    for _, token in enumerate(x[i]):
                        beam_hyps.append(token)
                        beam_idx += 1  # noqa: SIM113

                        if beam_idx == 6:
                            break

                    done[i] = len(beam_hyps) > 4

                return beam_hyps, done

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Starts with an empty list and an all-false done tensor,
                # and calls the helper once per time step of the input, each time with the first
                # slice along dim 1; returns the list.
                beam_hyps: list[Tensor] = []
                batch_size = x.shape[0]
                cur_len = 0
                max_len = x.shape[1]
                done = torch.zeros(batch_size, dtype=torch.bool)
                while cur_len < max_len:
                    beam_hyps, done = self.process(beam_hyps, done, x[:, 0, :])
                    cur_len = cur_len + 1

                return beam_hyps

        m = torch.jit.script(M())
        x = torch.randn(8, 4, 3)
        self.run_test(torch.jit.script(M()), (x))

    @skipScriptTest()  # Sort with dynamic dim not supported in ONNX
    def test_sort(self):
        # [FUNCTION DOC] Exports descending sort over dims -2 to 1, collecting each result in a
        # list. Skipped under scripting (sort with a dynamic dim is unsupported).
        class SortModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of descending sorts
            # along several dims.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a list holding the descending sort of the input
                # for each dim from -2 to 1.
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=True))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Sort with dynamic dim not supported in ONNX
    def test_sort_ascending(self):
        # [FUNCTION DOC] Same as the descending sort test but ascending. Skipped under
        # scripting; requires opset 11 or newer.
        class SortModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of ascending sorts along several dims.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a list holding the ascending sort of the input
                # for each dim from -2 to 1.
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=False))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_argsort(self):
        # [FUNCTION DOC] Exports argsort along dim 1 ascending. Requires opset 11 or newer.
        class ArgSortModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning argsort indices.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the ascending argsort of the input along dim 1.
                return torch.argsort(x, dim=1, descending=False)

        x = torch.randn(3, 4)
        self.run_test(ArgSortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        # [FUNCTION DOC] Exports masked_fill with a constant boolean mask that broadcasts and
        # with a mask from a comparison. Requires opset 9 or newer.
        class MaskedFillModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling masked positions with 2 using a constant 2
            # x 3 mask.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds the constant boolean mask and returns x filled
                # with 2 where the mask is true.
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                return x.masked_fill(mask, 2)

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling positions greater than 3 with -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x filled with -1 where x exceeds 3.
                return x.masked_fill(x > 3, -1)

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill_inplace(self):
        # [FUNCTION DOC] Exports scripted in-place masked_fill with a constant mask and a
        # comparison mask. Requires opset 9 or newer.
        class MaskedFillModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule filling masked positions with 2 in place.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method that fills x with 2 where a constant mask
                # is true and returns x.
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                x.masked_fill_(mask, 2)
                return x

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule filling positions greater than 3 with
            # -1 in place.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method that fills x with -1 where x exceeds 3
                # and returns x.
                x.masked_fill_(x > 3, -1)
                return x

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_scatter(self):
        # [FUNCTION DOC] Exports masked_scatter where positions with values of at least 0.5
        # receive entries of a 100 x 100 tensor of fives. Requires opset 11 or newer.
        class MaskedScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scattering a source of fives into masked positions.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x with masked positions replaced by
                # the source values.
                return torch.masked_scatter(x, x.ge(0.5), torch.ones(100, 100) * 5)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedScatterModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_select(self):
        # [FUNCTION DOC] Exports masked_select picking the entries at least 0.5. Requires opset
        # 11 or newer.
        class MaskedSelectModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting entries not below 0.5.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the one-dimensional tensor of selected entries.
                return torch.masked_select(x, x.ge(0.5))

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedSelectModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_fill(self):
        # [FUNCTION DOC] Exports a pattern of index assignment with boolean masks that should
        # convert to masked fill: set entries not equal to a constant to 1, entries equal to it
        # to 0. Requires opset 11 or newer.
        class MaskedFillModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module replacing entries by comparison with a constant
            # using index assignment.
            def forward(self, input_mask, some_const):
                # [FUNCTION DOC] HELPER: Clones the input, sets entries different from the
                # constant to 1 and entries equal to it to 0, and returns the clone.
                mask = input_mask.clone()
                mask[mask != some_const] = 1
                mask[mask == some_const] = 0
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedFillModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_scatter(self):
        # [FUNCTION DOC] Exports index assignment of a ones vector into masked positions that
        # should convert to masked scatter. Requires opset 11 or newer.
        class MaskedScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a vector of eight ones into masked positions.
            def forward(self, input_mask, some_const):
                # [FUNCTION DOC] HELPER: Clones the input, writes eight ones into entries
                # different from the constant and returns the clone.
                mask = input_mask.clone()
                mask[mask != some_const] = torch.ones(8)
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_with_1d_mask_to_masked_scatter(self):
        # [FUNCTION DOC] Exports index assignment with a one-dimensional boolean mask over the
        # first dimension. Requires opset 11 or newer.
        class MaskedScatterModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module assigning a tensor into the rows selected by a 1D mask.
            def forward(self, tensor, mask, some_const):
                # [FUNCTION DOC] HELPER: Writes the constant tensor into the masked rows of the
                # input and returns the input (mutated in place).
                tensor[mask] = some_const
                return tensor

        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        tensor = torch.randn(8, 4, 5, requires_grad=True)
        some_const = torch.randn(4, 4, 5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (tensor, mask, some_const))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_shuffle(self):
        # [FUNCTION DOC] Exports pixel_shuffle with upscale factor 2 on a 4D tensor, also with
        # dynamic axes and an additional input. Requires opset 9 or newer.
        class PixelShuffle(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying pixel shuffle with upscale factor 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pixel-shuffled input.
                return torch.pixel_shuffle(x, upscale_factor=2)

        x = torch.randn(2, 16, 4, 3, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelShuffle(), x)
        self.run_test(
            PixelShuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_unshuffle(self):
        # [FUNCTION DOC] Exports pixel_unshuffle with downscale factor 2, also with dynamic axes
        # and an additional input. Requires opset 9 or newer.
        class PixelUnshuffle(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying pixel unshuffle with downscale factor 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pixel-unshuffled input.
                return torch.pixel_unshuffle(x, downscale_factor=2)

        x = torch.randn(2, 16, 4, 6, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelUnshuffle(), x)
        self.run_test(
            PixelUnshuffle(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_reciprocal(self):
        # [FUNCTION DOC] Exports reciprocal on the values 2 and 4 as long, float and double.
        # Requires opset 9 or newer.
        class ReciprocalModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the reciprocal of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns one over the input.
                return torch.reciprocal(x)

        model = ReciprocalModel()
        x = torch.tensor([2, 4])
        self.run_test(model, x.to(torch.long))
        self.run_test(model, x.to(torch.float))
        self.run_test(model, x.to(torch.double))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type(self):
        # [FUNCTION DOC] Checks scalar type handling in exported arithmetic: scalar
        # multiplication and subtraction, mixed integer and float comparisons, repeated matrix
        # products, mm plus add, full with a tensor fill value, and concatenation of half and
        # float tensors. Requires opset 9 or newer.
        class ArithmeticModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the input by twice its first dimension size
            # and returning it with 2 minus the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of size times 2 times x and 2 minus x.
                return x.size(0) * 2 * x, 2 - x

        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ArithmeticModel(), x)

        class ComparisonModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning several comparisons mixing int, float
            # and scalar operands.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the combined and of comparisons of x and y,
                # plus x greater than y, x less than y, and a constant compared with the
                # first dim size.
                a = torch.tensor([12.0])
                return x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0))

        x = torch.ones(2, 3, dtype=torch.int32)
        y = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ComparisonModel(), (x, y))

        class MatMulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two matrix products added to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns mm of x with itself plus x plus another mm of x
                # with itself plus x.
                return torch.mm(x, x) + x + torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(MatMulModel(), x)

        class AddMMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a matrix product added to the input, a
            # pattern for addmm.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns mm of x with itself plus x.
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddMMModel(), x)

        class FullModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating a 3 x 4 tensor filled with the tensor input.
            # add is used for exporting full
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a 3 x 4 tensor filled with the scalar input.
                return torch.full((3, 4), x)

        x = torch.tensor(12.0)
        self.run_test(FullModel(), x)

        class CatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating a half precision and a float tensor.
            def forward(self, fp16, fp32):
                # [FUNCTION DOC] HELPER: Returns the concatenation of fp16 and fp32 tensors.
                return torch.cat([fp16, fp32])

        fp16 = Tensor([0.5])
        fp16 = fp16.half()
        fp32 = Tensor([1.5])
        self.run_test(CatModel(), (fp16, fp32))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_does_not_trigger_upcast_type_promotion(self):
        # [FUNCTION DOC] Exports a float16 input multiplied by a scalar scale (last dim size to
        # the power -0.5) and checks the result stays float16: the scalar becomes a float32
        # rank-0 tensor, which must not promote the multiplication. Requires opset 9 or newer.
        class DoNotUpcastModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by the inverse square root of its
            # last dimension size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes the scale as last dim size to the power -0.5
                # and returns the input times that scale.
                scale = x.size()[-1] ** -0.5
                # 'scale' is exported as onnx float32 rank 0 tensor.
                # The following 'Mul' should NOT be promoted to float32.
                return x * scale

        x = torch.ones(2, 3, dtype=torch.float16)
        self.run_test(DoNotUpcastModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_two_prim_const(self):
        # [FUNCTION DOC] Exports where with a condition tensor and two Python scalar
        # branches (an integer and a float), exercising constant type promotion. Requires
        # opset 9 or newer.
        class TwoPrimConstCastWhereModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between the scalars 0 and 1.0 by
            # a boolean tensor.
            def forward(self, c):
                # [FUNCTION DOC] HELPER: Returns the where result with 0 where the condition is
                # true and 1.0 elsewhere.
                return torch.where(c, 0, 1.0)

        c = torch.ones(8, dtype=torch.bool)
        self.run_test(TwoPrimConstCastWhereModel(), (c))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_one_prim_const(self):
        # [FUNCTION DOC] Exports where with a half precision tensor branch and a float scalar
        # branch. Requires opset 9 or newer.
        class OnePrimConstCastWhereModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between a tensor and the scalar 1.0.
            def forward(self, c, x):
                # [FUNCTION DOC] HELPER: Returns x where c is true and 1.0 elsewhere.
                return torch.where(c, x, 1.0)

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        self.run_test(OnePrimConstCastWhereModel(), (c, x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_promotion_onnx_where_one_tensor_const(self):
        # [FUNCTION DOC] Exports where with a half precision tensor and a float64 rank-0 tensor
        # constant. Requires opset 9 or newer.
        class OneTensorConstCastWhereModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between a tensor and a double scalar tensor.
            def forward(self, c, x):
                # [FUNCTION DOC] HELPER: Returns x where c is true and a double
                # scalar one elsewhere.
                return torch.where(c, x, torch.ones(size=(), dtype=torch.float64))

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        self.run_test(OneTensorConstCastWhereModel(), (c, x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_upcast_type_promotion_onnx_where_no_const(self):
        # [FUNCTION DOC] Exports where with half and float tensors, which upcasts to float.
        # Requires opset 9 or newer.
        class OnnxWhereUpcastModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between two tensors of
            # different float precisions.
            def forward(self, c, x, y):
                # [FUNCTION DOC] HELPER: Returns x where c is true and y elsewhere.
                return torch.where(c, x, y)

        c = torch.ones(8, dtype=torch.bool)
        x = torch.ones(8, dtype=torch.float16)
        y = torch.ones(8, dtype=torch.float32)

        self.run_test(OnnxWhereUpcastModel(), (c, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_like(self):
        # [FUNCTION DOC] Exports full_like with fill value 1.3 and int dtype on a scalar long
        # tensor. Requires opset 9 or newer.
        class FullLikeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating an int tensor shaped like the input
            # filled with 1.3.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns full_like of the input with value 1.3
                # and int dtype.
                return torch.full_like(x, 1.3, dtype=torch.int)

        x = torch.tensor(12)
        self.run_test(FullLikeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipDtypeChecking
    def test_full_like_value(self):
        # [FUNCTION DOC] Exports full_like whose fill value is a tensor computed from an input.
        # Dtype checking disabled; requires opset 9 or newer.
        class FullLikeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating a tensor like x filled with the value
            # y plus 2.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns full_like of x filled with y plus two.
                out = y + 2
                return torch.full_like(x, out)

        x = torch.tensor(12)
        y = torch.tensor(2)
        self.run_test(FullLikeModel(), (x, y))

    def test_l1_norm(self):
        # [FUNCTION DOC] Exports the L1 norm over the last dim without keepdim on a 4 x 2
        # x 3 input.
        class NormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the L1 norm over dim -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the order-1 norm over the last dim.
                return torch.norm(x, p=1, dim=-1, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_l2_norm(self):
        # [FUNCTION DOC] Exports the L2 norm over dim -2 without keepdim.
        class NormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the L2 norm over dim -2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the order-2 norm over dim -2.
                return torch.norm(x, p=2, dim=-2, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm(self):
        # [FUNCTION DOC] Exports the Frobenius norm over dim 0 without keepdim.
        class NormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the Frobenius norm over dim 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the Frobenius norm over dim 0.
                return torch.norm(x, p="fro", dim=0, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm_keepdim(self):
        # [FUNCTION DOC] Exports the Frobenius norm over dims 0 and 1 with keepdim.
        class NormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the Frobenius norm over dims 0 and
            # 1 keeping dims.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the Frobenius norm over dims 0 and 1
                # with kept dims.
                return torch.norm(x, p="fro", dim=(0, 1), keepdim=True)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_unfold(self):
        # [FUNCTION DOC] Exports unfold along dim 2 with size 2 and step 2, with dynamic axes on
        # dims 0 and 1 and an additional input of a different shape.
        class UnfoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unfolding dim 2 into windows of size two with step two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the unfolded input.
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(4, 2, 3, requires_grad=True)
        y = torch.randn(2, 1, 3, requires_grad=True)
        self.run_test(
            UnfoldModel(),
            x,
            dynamic_axes={"x": [0, 1]},
            input_names=["x"],
            additional_test_inputs=[y],
        )

    def test_unfold_infer_shape(self):
        # [FUNCTION DOC] Exports a scripted module with a 1D convolution followed by unfold,
        # relying on shape inference through the convolution.
        class UnfoldModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a Conv1d (3 to 1 channels, kernel 3, stride
            # 2) followed by unfold.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor and
                # creates the convolution.
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method applying the convolution then unfolding
                # dim 2 with size 2 and step 2.
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(32, 3, 64)
        self.run_test(UnfoldModule(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_unfold_dynamic_inputs(self):
        # [FUNCTION DOC] Exports unfold whose window size comes from the input shape, with the
        # step equal to that size minus one and with step one. Requires opset 12 or newer.
        class UnfoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unfolding dim 2 with window size from dim 1 and
            # step one less.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unfolded with a shape-derived
                # window and step.
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

        class UnfoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module unfolding dim 2 with window size from dim 1
            # and step one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unfolded with a shape-derived window
                # and unit step.
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_mv(self):
        # [FUNCTION DOC] Exports matrix-vector product for float and integer inputs. Requires
        # opset 9 or newer because matmul on long inputs arrived then.
        class MatmulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a matrix-vector product.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns mv of input and other.
                return torch.mv(input, other)

        x = torch.randn(4, 5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (4, 5))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)  # MatMul long inputs is added in ONNX opset 9.
    def test_dot(self):
        # [FUNCTION DOC] Exports dot product of two vectors for float and integer inputs.
        # Requires opset 9 or newer.
        class MatmulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a vector dot product.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns dot of input and other.
                return torch.dot(input, other)

        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (5,))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipScriptTest()  # SpectralNorm not TorchScript compatible.
    def test_spectral_norm(self):
        # [FUNCTION DOC] Exports a Linear layer wrapped with spectral norm on a 6 x 2 input.
        # Skipped under scripting (not compatible with TorchScript).
        m = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 4))

        x = torch.randn(6, 2)
        self.run_test(m, (x,))

    def test_prelu(self):
        # [FUNCTION DOC] Exports a PReLU layer with dynamic axes 1 and 2 and an additional input
        # of different shape.
        class PReluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a PReLU activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PReLU submodule.
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the PReLU of the input.
                return self.prelu(x)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        self.run_test(
            PReluModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_prelu_scalar(self):
        # [FUNCTION DOC] Exports PReLU on a scalar (rank 0) tensor.
        x = torch.scalar_tensor(1.0)
        self.run_test(torch.nn.PReLU(), x, input_names=["x"])

    def test_relu6(self):
        # [FUNCTION DOC] Exports ReLU6 with large-magnitude inputs, dynamic axes 1 and 2, and
        # an additional input.
        class Relu6Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a ReLU6 activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU6 submodule.
                super().__init__()
                self.relu6 = torch.nn.ReLU6()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the ReLU6 of the input.
                return self.relu6(x)

        x = torch.randn(2, 3, 4) * 100.0
        y = torch.randn(2, 4, 5) * 100.0
        self.run_test(
            Relu6Model(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [1, 2]},
            additional_test_inputs=[y],
        )

    def test_silu(self):
        # [FUNCTION DOC] Exports a SiLU activation on a random 2 x 3 x 4 tensor.
        class SiLUModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a SiLU activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the SiLU submodule.
                super().__init__()
                self.silu = torch.nn.SiLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the SiLU of the input.
                return self.silu(x)

        x = torch.randn(2, 3, 4)
        self.run_test(SiLUModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_tril(self):
        # [FUNCTION DOC] Exports tril in four forms: default, positive diagonal, negative
        # diagonal and a diagonal passed as an integer input of 5. Requires opset 14 or newer.
        class trilModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the lower triangle.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the lower triangular part of the input.
                return torch.tril(x)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModel(), (x))

        class trilModelwithDiagonal(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the lower triangle with diagonal 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the lower triangle with diagonal offset 1.
                return torch.tril(x, diagonal=1)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithDiagonal(), (x))

        class trilModelwithNegDiagonal(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the lower triangle with diagonal -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the lower triangle with diagonal offset -1.
                return torch.tril(x, diagonal=-1)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithNegDiagonal(), (x))

        class trilModelWithDiagonalInput(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the lower triangle with the diagonal taken
            # from an integer argument.
            def forward(self, x, diagnonal: int):
                # [FUNCTION DOC] HELPER: Returns the lower triangle with the
                # given diagonal offset.
                return torch.tril(x, diagonal=diagnonal)

        x = torch.randn(2, 3, 4)
        self.run_test(trilModelWithDiagonalInput(), (x, 5))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_triu(self):
        # [FUNCTION DOC] Exports triu in four forms: default, positive diagonal, negative
        # diagonal and a diagonal passed as an integer input of 5. Requires opset 14 or newer.
        class triuModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the upper triangle.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the upper triangular part of the input.
                return torch.triu(x)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModel(), (x))

        class triuModelwithDiagonal(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the upper triangle with diagonal 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the upper triangle with diagonal offset 1.
                return torch.triu(x, diagonal=1)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithDiagonal(), (x))

        class triuModelwithNegDiagonal(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the upper triangle with diagonal -1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the upper triangle with diagonal offset -1.
                return torch.triu(x, diagonal=-1)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithNegDiagonal(), (x))

        class triuModelWithDiagonalInput(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the upper triangle with the diagonal taken
            # from an integer argument.
            def forward(self, x, diagnonal: int):
                # [FUNCTION DOC] HELPER: Returns the upper triangle with the
                # given diagonal offset.
                return torch.triu(x, diagonal=diagnonal)

        x = torch.randn(2, 3, 4)
        self.run_test(triuModelWithDiagonalInput(), (x, 5))

    def test_mish(self):
        # [FUNCTION DOC] Exports a Mish activation on a random 2 x 3 x 4 tensor.
        class MishModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Mish activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Mish submodule.
                super().__init__()
                self.mish = torch.nn.Mish()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the Mish of the input.
                return self.mish(x)

        x = torch.randn(2, 3, 4)
        self.run_test(MishModel(), (x))

    def test_remainder(self):
        # [FUNCTION DOC] Exports the remainder operator for broadcast floats, long, then a float
        # numerator with long denominator, two floats, and int32 with float.
        class RemainderModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the remainder of two tensors.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the remainder of input divided by other.
                return torch.remainder(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(RemainderModel(), (x, y))

        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        y = torch.tensor([2], dtype=torch.long)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        y = y.to(torch.float)
        self.run_test(RemainderModel(), (x, y))

        x = x.to(torch.int32)
        self.run_test(RemainderModel(), (x, y))

    def test_remainder_scalar(self):
        # [FUNCTION DOC] Exports remainder with a scalar divisor: default float 2.55 on an
        # integer input, and integer 2 on a long vector.
        class RemainderModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the remainder against a stored scalar.
            def __init__(self, scalar=2.55):
                # [FUNCTION DOC] HELPER: Stores the scalar divisor (default 2.55).
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the remainder of the input divided by
                # the stored scalar.
                return torch.remainder(input, self.scalar)

        x = torch.randint(10, (2, 3))
        self.run_test(RemainderModel(), x)

        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        self.run_test(RemainderModel(2), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod(self):
        # [FUNCTION DOC] Exports fmod on broadcasted float tensors. Requires opset 10 or newer.
        class FModModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the fmod of two tensors.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns fmod of input and other.
                return torch.fmod(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(FModModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod_scalar(self):
        # [FUNCTION DOC] Exports fmod with the scalar 2.55 on an integer tensor. Requires opset
        # 10 or newer.
        class FModModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning fmod against the scalar 2.55.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns fmod of the input and 2.55.
                return torch.fmod(input, 2.55)

        x = torch.randint(10, (2, 3))
        self.run_test(FModModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_glu(self):
        # [FUNCTION DOC] Exports functional glu with default dim on a 4D tensor. Requires opset
        # 9 or newer.
        class GluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying glu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the glu of the input.
                return torch.nn.functional.glu(x)

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        # [FUNCTION DOC] Exports functional gelu with the exact (none) approximation. Requires
        # opset 9 or newer.
        class GeluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying gelu with exact approximation.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the exact gelu of the input.
                return torch.nn.functional.gelu(x, approximate="none")

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tanh_gelu(self):
        # [FUNCTION DOC] Exports functional gelu with the tanh approximation. Requires opset
        # 9 or newer.
        class GeluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying gelu with the tanh approximation.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tanh-approximated gelu of the input.
                return torch.nn.functional.gelu(x, approximate="tanh")

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    def test_add_inplace(self):
        # [FUNCTION DOC] Exports an in-place add of 12 on a random tensor.
        class InplaceAddModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 12 to its input in place.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 12 in place and returns the input.
                x += 12
                return x

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(InplaceAddModel(), x)

    def test_addcmul(self):
        # [FUNCTION DOC] Exports addcmul with default and explicit value 2.2
        # on broadcastable inputs.
        class AddcmulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two addcmul results.
            def forward(self, x, t1, t2):
                # [FUNCTION DOC] HELPER: Returns addcmul of x with the product of t1 and t2,
                # with default and with value 2.2.
                return torch.addcmul(x, t1, t2), torch.addcmul(x, t1, t2, value=2.2)

        x = torch.randn(1, 3)
        t1 = torch.randn(3, 1)
        t2 = torch.randn(1, 3)
        self.run_test(AddcmulModel(), (x, t1, t2))

    def test_rsqrt(self):
        # [FUNCTION DOC] Exports reciprocal square root on double precision random input
        # (with requires_grad set).
        class RsqrtModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing rsqrt.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the rsqrt of the input.
                return x.rsqrt()

        x = torch.randn(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        # [FUNCTION DOC] Exports reciprocal square root on an all-zero double tensor, covering
        # the infinity result.
        class RsqrtModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing rsqrt.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the rsqrt of the input.
                return x.rsqrt()

        x = torch.zeros(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique(self):
        # [FUNCTION DOC] Exports unique with sorted output, no inverse and counts on a small
        # long vector. Requires opset 11 or newer.
        class UniqueModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sorted unique values and their counts.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns unique of the input, sorted, with counts and
                # without the inverse.
                return torch.unique(
                    x, sorted=True, return_inverse=False, return_counts=True
                )

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique_along_dim(self):
        # [FUNCTION DOC] Exports unique along dim 0 with sorted output, returning the inverse
        # indices and not the counts, on a small long vector. Requires opset 11 or newer.
        class UniqueModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning sorted unique values along dim 0 with
            # the inverse indices.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns unique of the input along dim 0, sorted, with
                # the inverse and without counts.
                return torch.unique(
                    x, dim=0, sorted=True, return_inverse=True, return_counts=False
                )

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum(self):
        # [FUNCTION DOC] Exports cumulative sum along dim 0 on a random 3D tensor. Requires
        # opset 11 or newer.
        class CumSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a cumulative sum over dim 0.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the cumsum of the input along dim 0.
                return torch.cumsum(input, dim=0)

        x = torch.randn(2, 3, 4)
        model = CumSum()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum_with_cast(self):
        # [FUNCTION DOC] Exports cumsum with a float32 output dtype for int32 and boolean
        # inputs. Requires opset 11 or newer.
        class CumSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a cumulative sum cast to float32.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the cumsum along dim 0 as float32.
                return torch.cumsum(input, dim=0, dtype=torch.float32)

        model = CumSum()
        x = torch.tensor([2, 3, 4], dtype=torch.int32)
        self.run_test(model, x)
        x = torch.tensor([False, True, True])
        self.run_test(model, x)

    @skipScriptTest()  # error in propagate as assign input shape
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_embedding_bag(self):
        # [FUNCTION DOC] Exports EmbeddingBag layers: sum mode with gradient scaling and
        # offsets, sum mode including the last offset, and max mode with a 2D input. Skipped
        # under scripting; requires opset 10 or newer.
        model = torch.nn.EmbeddingBag(10, 5, mode="sum", scale_grad_by_freq=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode="sum", include_last_offset=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode="max")
        input = torch.randint(10, (7, 5))
        self.run_test(model, (input))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_1d_per_sample_weights(self):
        # [FUNCTION DOC] Exports functional embedding_bag with sum mode, offsets and per-sample
        # weights for a 1D input. Requires opset 11 or newer.
        class EmbeddingModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling functional embedding_bag with sum mode, offsets
            # and per sample weights.
            def forward(self, embedding_matrix, input, offset, weights):
                # [FUNCTION DOC] HELPER: Returns the embedding bag of the input indices over the
                # embedding matrix with the offsets and weights.
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offset,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel()
        x = torch.randint(7, (6,))
        w = torch.randn(
            6,
        )
        offset = torch.tensor([0, 2, 5])
        embedding_matrix = torch.rand(10, 15)
        self.run_test(model, (embedding_matrix, x, offset, w))

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip(
        "This test is broken with ONNXRuntime(17): "
        "when running with onnxruntime 1.17.0 this test fails with the following error:"
        "FAIL : Non-zero status code returned while running If node. "
        "Name:'/If' Status Message: if.cc:253 Compute "
        "If nodes condition input must have exactly one element"
        "https://github.com/pytorch/pytorch/issues/119442"
    )
    def test_embedding_bag_2d_per_sample_weights(self):
        # [FUNCTION DOC] Skipped test (ORT 1.17 failure, see the linked issue). Would export
        # embedding_bag with 2D input and per-sample weights, with dynamic axes on indices and
        # weights and an additional input. Requires opset 11 or newer.
        class EmbeddingModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling functional embedding_bag with sum mode and per
            # sample weights on a 2D input.
            def forward(self, embedding_matrix, input, weights):
                # [FUNCTION DOC] HELPER: Returns the embedding bag of the input over the matrix
                # with the weights.
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        embedding_matrix = torch.rand(10, 15)
        model = EmbeddingModel()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)

        x2 = torch.randint(7, (4, 3))
        w2 = torch.randn(4, 3)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            input_names=["embed", "x", "w"],
            dynamic_axes={"x": [0], "w": [0]},
            additional_test_inputs=[(embedding_matrix, x2, w2)],
        )

    @skipScriptTest()  # scripting prim::Uninitialized, prim::dtype, prim::unchecked_cast
    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip(
        "Due to ONNX Loop shape inference issue. "
        "https://msdata.visualstudio.com/Vienna/_workitems/edit/1352001"
    )
    def test_embedding_bag_dynamic_input(self):
        # [FUNCTION DOC] Skipped test (ONNX Loop shape inference issue). Would export
        # embedding_bag with 1D and 2D inputs and dynamic axes for the matrix, indices, offsets
        # and weights. Skipped under scripting; requires opset 11 or newer.
        class EmbeddingModel1D(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling embedding_bag for 1D input with
            # offsets and weights.
            def forward(self, embedding_matrix, input, weights, offsets):
                # [FUNCTION DOC] HELPER: Returns the embedding bag of the input over the matrix
                # with the weights and offsets.
                return torch.nn.functional.embedding_bag(
                    input,
                    embedding_matrix,
                    offsets=offsets,
                    mode="sum",
                    per_sample_weights=weights,
                )

        model = EmbeddingModel1D()
        x = torch.randint(7, (6,))
        w = torch.randn(
            6,
        )
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (2,))
        w2 = torch.randn(
            2,
        )
        embedding_matrix2 = torch.rand(12, 25)
        offsets2 = torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
        )
        self.run_test(
            model,
            (embedding_matrix, x, w, offsets),
            additional_test_inputs=[(embedding_matrix2, x2, w2, offsets2)],
            input_names=["embedding_matrix", "x", "offsets", "w"],
            dynamic_axes={
                "embedding_matrix": [0, 1],
                "x": [0],
                "offsets": [0],
                "w": [0],
            },
        )

        class EmbeddingModel2D(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling embedding_bag for 2D input with weights.
            def forward(self, embedding_matrix, input, weights):
                # [FUNCTION DOC] HELPER: Returns the embedding bag of the input over the matrix
                # with the weights.
                return torch.nn.functional.embedding_bag(
                    input, embedding_matrix, mode="sum", per_sample_weights=weights
                )

        model = EmbeddingModel2D()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (3, 5))
        w2 = torch.randn(3, 5)
        embedding_matrix2 = torch.rand(12, 25)
        self.run_test(
            model,
            (embedding_matrix, x, w),
            additional_test_inputs=[(embedding_matrix2, x2, w2)],
            input_names=["embedding_matrix", "x", "w"],
            dynamic_axes={"embedding_matrix": [0, 1], "x": [0, 1], "w": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        # [FUNCTION DOC] Exports meshgrid of three vectors with different lengths. Requires
        # opset 8 or newer.
        class Meshgrid(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the three outputs of meshgrid.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the three grids from meshgrid of x, y and z.
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        x = torch.randn(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_indexing(self):
        # [FUNCTION DOC] Exports meshgrid with indexing xy and ij. Requires opset 8 or newer.
        class Meshgrid(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning meshgrid outputs with a stored indexing mode.
            def __init__(self, indexing):
                # [FUNCTION DOC] HELPER: Stores the indexing mode.
                super().__init__()
                self.indexing = indexing

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the three grids from meshgrid with
                # the stored indexing.
                output1, output2, output3 = torch.meshgrid(
                    x, y, z, indexing=self.indexing
                )
                return output1, output2, output3

        x = torch.randn(5, requires_grad=True)
        y = torch.zeros(6, requires_grad=True)
        z = torch.randn(7, requires_grad=True)
        for indexing in ("xy", "ij"):
            self.run_test(Meshgrid(indexing), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        # [FUNCTION DOC] Exports meshgrid with two vectors and a scalar. Requires opset
        # 8 or newer.
        class Meshgrid(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the three outputs of meshgrid.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the three grids from meshgrid of x, y and z.
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.tensor(2.0)
        self.run_test(Meshgrid(), (x, y, z))

    def test_baddbmm(self):
        # [FUNCTION DOC] Exports baddbmm with a tensor alpha of 5 and beta 3.5
        # on batched matrices.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing batched matrix multiply plus input.
            def forward(self, input, batch1, batch2):
                # [FUNCTION DOC] HELPER: Returns baddbmm of the input and the two batches with
                # alpha 5 as tensor and beta 3.5.
                return torch.baddbmm(
                    input, batch1, batch2, alpha=torch.tensor(5), beta=3.5
                )

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2))

    def test_baddbmm_dynamic(self):
        # [FUNCTION DOC] Exports baddbmm with alpha and beta passed as tensor inputs.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing baddbmm with runtime alpha and beta.
            def forward(self, input, batch1, batch2, alpha, beta):
                # [FUNCTION DOC] HELPER: Returns baddbmm of the input and batches with the given
                # alpha and beta.
                return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = torch.tensor(5)
        beta = torch.tensor(3.5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    def test_numel(self):
        # [FUNCTION DOC] Exports multiplication of the input by its element count, with dynamic
        # axes and an additional larger input.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the input by its number of elements.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the element count times the input.
                return input.numel() * input

        x = torch.randn(2, 3, 5)
        x2 = torch.randn(4, 5, 6)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
            additional_test_inputs=[(x2,)],
        )

    def test_numel_empty(self):
        # [FUNCTION DOC] Exports multiplication by the element count with an empty input and
        # dynamic axis, plus an additional vector of length 4.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the input by its number of elements.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the element count times the input.
                return input.numel() * input

        x = torch.randn(0)
        x2 = torch.randn(4)
        model = MyModule()
        self.run_test(
            model,
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[(x2,)],
        )

    def test_dtype(self):
        # [FUNCTION DOC] Exports a scripted module casting the input to the dtype of
        # another and adding.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule casting input to the dtype of other
            # and adding other.
            @torch.jit.script_method
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Script method returning the cast input plus other.
                return input.to(dtype=other.dtype) + other

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_dtype_eq(self):
        # [FUNCTION DOC] Exports a scripted module comparing the dtypes of two inputs in a
        # conditional; with equal dtypes it adds them.
        class MyModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding the inputs if their dtypes are equal and
            # otherwise returning the first.
            @torch.jit.script_method
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Script method returning the sum if dtypes match, else
                # the first input.
                if input.dtype == other.dtype:
                    return input + other
                return input

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_cast_to(self):
        # [FUNCTION DOC] Exports a scripted module casting the input to the type of a long
        # tensor and adding it.
        class MyModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule casting the input to the type and device of other
            # and adding other.
            @torch.jit.script_method
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Script method returning the cast input plus other.
                return input.to(other) + other

        x = torch.randn(2, 3, 4)
        y = torch.tensor([1], dtype=torch.int64)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_cast_to_bool(self):
        # [FUNCTION DOC] Exports casting a float tensor to the type of a boolean tensor
        # and concatenating them.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module casting the input to the type of
            # other and concatenating.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the concatenation along dim 0 of the cast
                # input and other.
                return torch.cat((input.to(other), other), 0)

        x = torch.randn(2, 3, 4)
        y = torch.zeros([2, 3, 4], dtype=torch.bool)
        model = MyModule()
        self.run_test(model, (x, y))

    # ONNX supports bfloat16 for opsets >= 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_cast_type_as_with_bfloat16(self):
        # [FUNCTION DOC] Exports type_as to a bfloat16 tensor followed by conversion to float16.
        # Requires opset 13 or newer because bfloat16 support starts there.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module converting through bfloat16 back to half precision.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Converts x to the type of a bfloat16 ones tensor and
                # then to float16.
                y = torch.ones((3, 4), dtype=torch.bfloat16)
                x = x.type_as(y)
                return x.to(dtype=torch.float16)

        x = torch.ones(3, 4, dtype=torch.float16)
        model = MyModule()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_type_as(self):
        # [FUNCTION DOC] Exports type_as to a float tensor for boolean, double and int64 inputs.
        # Requires opset 9 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module converting the input to the type of a float tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input cast to float32 using type_as of a
                # one-element float tensor.
                y = torch.tensor([1.0])
                return x.type_as(y)

        a = torch.tensor([True, False], dtype=torch.bool)
        b = torch.randn(3, 4, dtype=torch.double)
        c = torch.ones((2, 2), dtype=torch.int64)
        model = MyModule()
        self.run_test(model, a)
        self.run_test(model, b)
        self.run_test(model, c)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ones_bool(self):
        # [FUNCTION DOC] Exports casting a float input to the type of a ones boolean tensor and
        # applying logical and. Requires opset 9 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module converting the input to boolean and combining with a
            # ones boolean tensor.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input cast to boolean and-ed with a boolean
                # ones tensor of the same shape.
                true = torch.ones(input.shape, dtype=torch.bool)
                return input.to(true) & true

        x = torch.randn(2, 3, 4)
        model = MyModule()
        self.run_test(model, x)

    def test_log(self):
        # [FUNCTION DOC] Exports natural log on a random positive tensor.
        class Log(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing log.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the natural log of the input.
                return torch.log(input)

        x = torch.rand(2, 3, 4)
        model = Log()
        self.run_test(model, x)

    def test_log1p(self):
        # [FUNCTION DOC] Exports log1p on a random positive tensor.
        class Log1p(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing log1p.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns log1p of the input.
                return torch.log1p(input)

        x = torch.rand(2, 3, 4)
        model = Log1p()
        self.run_test(model, x)

    def test_log10(self):
        # [FUNCTION DOC] Exports log base 10 on a random positive tensor.
        class Log10(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing log10.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the base-10 log of the input.
                return torch.log10(input)

        x = torch.rand(2, 3, 4)
        model = Log10()
        self.run_test(model, x)

    def test_log2(self):
        # [FUNCTION DOC] Exports log base 2 on the scalar 1.0.
        class Log2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing log2.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the base-2 log of the input.
                return torch.log2(input)

        x = torch.tensor(1.0)
        model = Log2()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round(self):
        # [FUNCTION DOC] Exports round on floats with ties (-1.5, 3.5) and on int32 values.
        # Requires opset 11 or newer.
        class Round(torch.nn.Module):
            # [CLASS DOC] HELPER: Module rounding its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the rounded input.
                return torch.round(x)

        x = torch.tensor([0.9920, -1.0362, -1.5000, 3.5000], requires_grad=True)
        self.run_test(Round(), x)

        int_x = torch.tensor([9920, 1036, -1500, 35], dtype=torch.int32)
        self.run_test(Round(), int_x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round_with_decimals(self):
        # [FUNCTION DOC] Exports round with decimals 0, -2 and 3. Requires opset 11 or newer.
        class Round(torch.nn.Module):
            # [CLASS DOC] HELPER: Module rounding to a stored number of decimals.
            def __init__(self, decimals):
                # [FUNCTION DOC] HELPER: Stores the number of decimals.
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input rounded to the stored decimals.
                return torch.round(x, decimals=self.decimals)

        x = torch.tensor([0.9920, -1234.0362, -1.58960, 3.5000])
        for decimals in (0, -2, 3):
            self.run_test(Round(decimals), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_default(self):
        # [FUNCTION DOC] Exports a short-time Fourier transform with n_fft 16, no centering and
        # real-valued output, on a 1 x 32 signal with atol 1e-6. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with n_fft 16
            # and no centering.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft of the input with default
                # hop and window.
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_hop_length(self):
        # [FUNCTION DOC] Exports the STFT with n_fft 16 and hop length 4, no centering,
        # real-valued output. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with n_fft 16 and
            # hop length 4.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft of the input with hop length 4.
                n_fft = 16
                hop_length = 4
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_non_divisible_hop_length(self):
        # [FUNCTION DOC] Exports an STFT with n_fft 16 and hop length 5 that does not evenly
        # divide the signal. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with n_fft 16 and
            # hop length 5.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft of the input with hop length 5.
                n_fft = 16
                hop_length = 5
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    hop_length=hop_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_same_size(self):
        # [FUNCTION DOC] Exports an STFT where the integer window length equals n_fft (16).
        # Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with window length 16.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with n_fft 16 and win_length 16.
                n_fft = 16
                win_length = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_different_size(self):
        # [FUNCTION DOC] Exports an STFT where the integer window length 9 is smaller than
        # n_fft. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with window length 9.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with n_fft 16 and win_length 9.
                n_fft = 16
                win_length = 9
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    win_length=win_length,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_custom(self):
        # [FUNCTION DOC] Exports an STFT with a Hann window of length 16 passed explicitly.
        # Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT with a Hann window.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a Hann window of size 16 and returns the
                # stft with it.
                n_fft = 16
                window = torch.hann_window(16)
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    window=window,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_custom_window_size(self):
        # [FUNCTION DOC] Checks that a custom window of length 10 with n_fft 16 and no window
        # length fails: expects AssertionError or RuntimeError from run_test. Requires opset
        # 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT with a Hann window of
            # mismatched length 10.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates a Hann window of size 10 and returns the stft
                # with n_fft 16 and that window.
                n_fft = 16
                window = torch.hann_window(10)
                return torch.stft(
                    x, n_fft=n_fft, window=window, center=False, return_complex=False
                )

        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_window_length(self):
        # [FUNCTION DOC] Checks that a window length of 17 larger than n_fft 16 fails with
        # RuntimeError. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT with window length 17 and n_fft 16.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with win_length 17.
                n_fft = 16
                win_len = 17
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_size_with_win_len(self):
        # [FUNCTION DOC] Exports an STFT with a Hann window of length 10 and a matching explicit
        # window length of 10. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT with a Hann window and matching
            # window length 10.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with n_fft 16, a Hann window of 10
                # and win_length 10.
                n_fft = 16
                window = torch.hann_window(10)
                win_len = 10
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    window=window,
                    win_length=win_len,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_one_dimension(self):
        # [FUNCTION DOC] Exports an STFT on a one-dimensional signal of 32 samples. Requires
        # opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a real-valued STFT on a 1D signal.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with n_fft 16 and no centering.
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_input_size(self):
        # [FUNCTION DOC] Checks that a 3D input to the STFT fails with RuntimeError. Requires
        # opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft of the input with n_fft 16.
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)

        x = torch.randn((1, 1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_return_complex(self):
        # [FUNCTION DOC] Checks that requesting complex output from the STFT export raises
        # SymbolicValueError. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT with complex output requested.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft of the input with return_complex true.
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=True)

        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises(errors.SymbolicValueError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_normalize(self):
        # [FUNCTION DOC] Exports a normalized STFT on a 1D signal. Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a normalized STFT.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with normalization enabled
                # and real output.
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    normalized=True,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_not_onesided(self):
        # [FUNCTION DOC] Exports an STFT with the full two-sided spectrum on a 1D signal.
        # Requires opset 17 or newer.
        class STFT(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an STFT with onesided disabled.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stft with onesided false and real output.
                n_fft = 16
                return torch.stft(
                    x,
                    n_fft=n_fft,
                    center=False,
                    onesided=False,
                    return_complex=False,
                )

        x = torch.randn((32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-6)

    def test_constant_pad(self):
        # [FUNCTION DOC] Exports ConstantPad1d (pad 2, value 3.5) and ConstantPad2d (pad
        # 3,0,2,1, value 3.5) layers on random inputs.
        model = torch.nn.ConstantPad1d(2, 3.5)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @common_utils.parametrize(
        "pad",
        [
            common_utils.subtest([2, 4], name="scalar_list"),
            common_utils.subtest(
                [
                    torch.tensor(2, dtype=torch.int64),
                    torch.tensor(4, dtype=torch.int64),
                ],
                name="scalar_tensor_list",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(11)  # Dynamic padding is added in opset 11
    def test_pad_types(self, pad):
        # [FUNCTION DOC] Parameterized over padding given as a list of integers and as a list of
        # int64 scalar tensors. Exports functional pad of a 4D tensor. Requires opset 11 or
        # newer (dynamic padding).
        # Test for different pad integer types
        class Pad(torch.nn.Module):
            # [CLASS DOC] HELPER: Module padding the input by a list of integers.
            def forward(self, x, pad: list[int]):
                # [FUNCTION DOC] HELPER: Returns the input padded by the list.
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        self.run_test(Pad(), (x, pad))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular(self):
        # [FUNCTION DOC] Exports circular padding of 1 and 2 on the last two dims of a 4D
        # tensor. Requires opset 11 or newer.
        class PadModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying circular padding of 1,2,1,2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the circularly padded input.
                out = torch.nn.functional.pad(x, (1, 2, 1, 2), mode="circular")
                return out

        x = torch.randn(2, 3, 3, 4)
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_negative(self):
        # [FUNCTION DOC] Exports circular padding with negative amounts (-1, -2) on the last dim
        # of a 3D tensor. Requires opset 11 or newer.
        # Test for different pad integer types
        class PadModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying circular padding with negative amounts.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input padded circularly by -1 and -2.
                out = torch.nn.functional.pad(x, (-1, -2), mode="circular")
                return out

        x = torch.randn(2, 3, 6)
        self.run_test(PadModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_dynamic_axes(self):
        # [FUNCTION DOC] Exports circular padding with dynamic axes on all four input dims.
        # Requires opset 11 or newer.
        class PadModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying circular padding of 2,1,2,1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the circularly padded input.
                out = torch.nn.functional.pad(x, (2, 1, 2, 1), mode="circular")
                return out

        x = torch.randn(4, 3, 5, 6)
        self.run_test(
            PadModel(),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMaxOpsetVersion(10)
    @skipScriptTest()  # TODO: the logic in symbolic_opset9 doesn't handle script
    def test_unsupported_pad(self):
        # [FUNCTION DOC] Checks that a pad list passed as a runtime argument fails to export
        # for opset 10 or older with RuntimeError saying the padding sizes must be constant.
        # Skipped under scripting.
        class Pad(torch.nn.Module):
            # [CLASS DOC] HELPER: Module padding by a list given as an argument.
            def forward(self, x, pad: list[int]):
                # [FUNCTION DOC] HELPER: Returns the input padded by the list.
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        y = [2, 4]

        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Unsupported: ONNX export of Pad.*"
                + "The sizes of the padding must be constant"
            ),
        ):
            self.run_test(Pad(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_if_fold(self):
        # [FUNCTION DOC] Exports ten variants of module with conditionals on dimension count,
        # element count and dtype that can be folded at export time: equality, element count,
        # inequality, greater-or-equal, less-or-equal, combined conditions, and conditions
        # comparing the element counts of two inputs. Requires opset 9 or newer.
        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 and 2 when the input has 2 dimensions,
            # otherwise subtracting 1.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 6 for a rank-2 input, else y minus 1.
                if y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    y = y - 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 when the input has more than one element,
            # otherwise adding 2.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 4 if the element count exceeds 1, else
                # y plus 2.
                if y.numel() > 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 6 when the rank is not 3, otherwise
            # returning the input.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 6 if the rank differs from 3, else y.
                if y.dim() != 3:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 when the rank is at least 1,
            # otherwise subtracting 1.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 4 if the rank is at least 1, else
                # y minus 1.
                if y.dim() >= 1:
                    y = y + 4
                else:
                    y = y - 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 when the rank is at most 1,
            # otherwise adding 2.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 4 if the rank is at most 1, else
                # y plus 2.
                if y.dim() <= 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 6 when the rank is below 3 and the dtype is int,
            # otherwise returning the input.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 6 for rank below 3 and int
                # dtype, else y.
                if y.dim() < 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 6 when the rank is 3 and the dtype is int,
            # otherwise adding 1.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 6 for rank 3 with int dtype, else
                # y plus 1.
                if y.dim() == 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    y = y + 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 6 when the input is non-empty and has rank 2,
            # otherwise returning the input.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns y plus 6 if the element count is nonzero and
                # rank is 2, else y.
                if y.numel() != 0 and y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding two inputs when their element counts are
            # equal, otherwise subtracting.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y if both have the same number of
                # elements, else y minus x.
                if x.numel() == y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

        class IfFoldModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding two inputs when their element counts
            # differ, otherwise subtracting.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y if the element counts differ, else
                # y minus x.
                if x.numel() != y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized(self):
        # [FUNCTION DOC] Exports a model with a variable that is defined only in some branches
        # of nested conditionals (uninitialized value), on a 3 x 4 int tensor. Requires opset
        # 11 or newer.
        class UninitializedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 in a nested conditional that may not assign
            # in all branches.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: If dim 1 is below 5 and dim 0 equals 1 adds 4; if dim 0
                # differs returns the input early; returns the input.
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(UninitializedModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized_dynamic(self):
        # [FUNCTION DOC] Same as the uninitialized test but with dynamic axes on both input dims
        # and an additional larger input. Requires opset 11 or newer.
        class UninitializedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding 4 in a nested conditional that may not assign
            # in all branches.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: If dim 1 is below 5 and dim 0 equals 1 adds 4; if dim 0
                # differs returns the input early; returns the input.
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((6, 7), dtype=torch.int)
        self.run_test(
            UninitializedModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList(self):
        # [FUNCTION DOC] Exports a scripted model that returns a one-element list of the input
        # from several branches of nested conditionals, on an int tensor. Requires opset 14
        # because sequence identity is needed.
        class UninitializedTensorListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input wrapped in a list
            # from nested branches.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: If the first dimension of the first slice is below 5:
                # when dim 0 is 1 adds 4 to x, otherwise returns x in a list. Finally returns x
                # in a list.
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        return [x]
                return [x]

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(torch.jit.script(UninitializedTensorListModel()), x)

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_dynamic(self):
        # [FUNCTION DOC] Exports a scripted model with list results from nested conditionals,
        # where one branch doubles the input in place, with dynamic axes on a double tensor.
        # Requires opset 14 or newer.
        class UninitializedTensorListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the rows of x as a list from nested branches.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: If the size of the first slice is below 5: when dim 0
                # is 1 doubles x in place, otherwise returns the list of rows. Finally returns
                # the list of rows.
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x += x
                    else:
                        return list(x)
                return list(x)

        x = torch.ones((3, 4), dtype=torch.double)
        self.run_test(
            torch.jit.script(UninitializedTensorListModel()),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_intList(self):
        # [FUNCTION DOC] Exports a scripted model producing a list of integers from the
        # first dimension size, appended to in a conditional, with dynamic axes. Requires
        # opset 14 or newer.
        class UninitializedListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building a list of integers and appending to it when
            # the first dim is 3.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a range list over dim 0; if its first entry is
                # below 5, appends 10 when dim 0 equals 3, otherwise returns the list early;
                # returns the list.
                y = list(range(x.size(0)))
                if y[0] < 5:
                    # if x.size(0) != 3, ORT will throw type error.
                    if x.size(0) == 3:
                        y.append(10)
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(
            torch.jit.script(UninitializedListModel()),
            x,
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # onnx::Identity of sequence supported for ONNX opset >= 14
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_shape(self):
        # [FUNCTION DOC] Exports a scripted model returning lists of tensors from branches: one
        # list extended with the input, another with the input twice. Uses dynamic axes and an
        # additional input. Requires opset 14 or newer.
        class UninitializedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning lists of tensors from nested branches.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: If dim 1 is below 5: when dim 0 is 1 adds 4 to x,
                # otherwise returns a list of the rows with x appended. Finally returns a list
                # of x twice.
                if x.shape[1] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        x_list = list(x)
                        x_list.append(x)
                        return x_list
                return [x, x]

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((4, 6), dtype=torch.int)
        self.run_test(
            torch.jit.script(UninitializedModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    # Sequence type as loop-carried dependencies only supported for ONNX opset >= 13
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequance_loopcarried(self):
        # [FUNCTION DOC] Exports a scripted loop with a list as loop-carried value, stacking
        # three copies of the input and transposing. Requires opset 13 or newer.
        class SequanceLoopModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module accumulating a list of copies of the input in a loop.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Appends the input to a list three times, stacks the
                # list and swaps dims 0 and 1.
                outputs = []
                for _ in range(3):
                    outputs += [x]
                return torch.stack(outputs).transpose(0, 1)

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(torch.jit.script(SequanceLoopModel()), x)

    def test_reflection_pad(self):
        # [FUNCTION DOC] Exports ReflectionPad1d (pad 2) and ReflectionPad2d (pad 3,0,2,1)
        # on random inputs.
        model = torch.nn.ReflectionPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReflectionPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_replication_pad(self):
        # [FUNCTION DOC] Exports ReplicationPad1d (pad 2) and ReplicationPad2d (pad 3,0,2,1)
        # on random inputs.
        model = torch.nn.ReplicationPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReplicationPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_im2col(self):
        # [FUNCTION DOC] Exports functional unfold (im2col) three times with different
        # kernel size, dilation, padding and stride on a 1 x 1 x 200 x 100 input. Requires
        # opset 11 or newer.
        class Unfold(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning three unfold results.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns a tuple of three unfolds with kernels 10x15,
                # 2x2 and 1x1 and varying dilation, padding and stride.
                return (
                    torch.nn.functional.unfold(
                        input, kernel_size=(10, 15), dilation=2, padding=5, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(2, 2), dilation=1, padding=0, stride=3
                    ),
                    torch.nn.functional.unfold(
                        input, kernel_size=(1, 1), dilation=5, padding=2, stride=3
                    ),
                )

        x = torch.rand(1, 1, 200, 100)
        self.run_test(Unfold(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_det(self):
        # [FUNCTION DOC] Exports linalg det on batched 5 x 5 matrices. Skipped when LAPACK is
        # missing; requires opset 11 or newer.
        class Det(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the determinant.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the determinant of the input.
                return torch.linalg.det(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(Det(), x)

    def test_linalg_norm(self):
        # [FUNCTION DOC] Exports linalg norm for several cases: a single dim with norm orders
        # None, 2, plus and minus infinity, -4 and 1.5; two dims with Frobenius, infinity and 1
        # orders; no dim or order on inputs of ranks 4, 2 and 1; no dim on 1D; and no dim on 2D
        # with matrix orders.
        class LinalgSingleDimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg norm over dim 1 with a stored order.
            def __init__(self, ord_val):
                # [FUNCTION DOC] HELPER: Stores the norm order.
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linalg norm of the input over dim 1 with
                # the stored order.
                return torch.linalg.norm(x, ord=self.ord, dim=1)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgSingleDimModel(None), x)
        self.run_test(LinalgSingleDimModel(2), x)
        self.run_test(LinalgSingleDimModel(float("inf")), x)
        self.run_test(LinalgSingleDimModel(-float("inf")), x)
        self.run_test(LinalgSingleDimModel(-4), x)
        self.run_test(LinalgSingleDimModel(1.5), x)

        class LinalgMultiDimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg norm over dims 0 and 2 with
            # a stored order.
            def __init__(self, ord_val):
                # [FUNCTION DOC] HELPER: Stores the norm order.
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linalg norm over dims 0 and 2 with
                # the stored order.
                return torch.linalg.norm(x, ord=self.ord, dim=(0, 2))

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgMultiDimModel("fro"), x)
        self.run_test(LinalgMultiDimModel(float("inf")), x)
        self.run_test(LinalgMultiDimModel(-float("inf")), x)
        self.run_test(LinalgMultiDimModel(1), x)
        self.run_test(LinalgMultiDimModel(-1), x)

        class LinalgNoDimNoOrdModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg norm with no dim and no order.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linalg norm of the input.
                return torch.linalg.norm(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgNoDimNoOrdModel(), x)
        y = torch.randn(2, 3)
        self.run_test(LinalgNoDimNoOrdModel(), y)
        z = torch.randn(2)
        self.run_test(LinalgNoDimNoOrdModel(), z)

        class LinalgNoDim1DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg norm of a 1D input with
            # a stored order.
            def __init__(self, ord_val):
                # [FUNCTION DOC] HELPER: Stores the norm order.
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linalg norm with the stored order.
                return torch.linalg.norm(x, ord=self.ord)

        x = torch.randn(2)
        self.run_test(LinalgNoDim1DModel(None), x)
        self.run_test(LinalgNoDim1DModel(2), x)
        self.run_test(LinalgNoDim1DModel(float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-float("inf")), x)
        self.run_test(LinalgNoDim1DModel(-4), x)
        self.run_test(LinalgNoDim1DModel(1.5), x)

        class LinalgNoDim2DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg norm of a 2D input with a
            # stored matrix order.
            def __init__(self, ord_val):
                # [FUNCTION DOC] HELPER: Stores the norm order.
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linalg norm with the stored order.
                return torch.linalg.norm(x, ord=self.ord)

        x = torch.randn(2, 3)
        self.run_test(LinalgNoDim2DModel("fro"), x)
        self.run_test(LinalgNoDim2DModel(float("inf")), x)
        self.run_test(LinalgNoDim2DModel(-float("inf")), x)
        self.run_test(LinalgNoDim2DModel(1), x)
        self.run_test(LinalgNoDim2DModel(-1), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_linalg_vector_norm_zero(self):
        # [FUNCTION DOC] Exports linalg vector_norm with order 0 on a 4D tensor. Requires opset
        # 11 or newer.
        class LinalgVectorNormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing linalg vector_norm with a stored order.
            def __init__(self, ord_val):
                # [FUNCTION DOC] HELPER: Stores the norm order.
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the vector_norm of the input with
                # the stored order.
                return torch.linalg.vector_norm(x, ord=self.ord)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgVectorNormModel(0), x)

    def test_linalg_vector_norm(self):
        # [FUNCTION DOC] Exports linalg vector_norm for orders 2, plus and minus infinity, -4
        # and 1.5, each with four dim and keepdim combinations.
        class LinalgVectorNormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing vector_norm with a stored order,
            # dim and keepdim.
            def __init__(self, ord_val, dim_info):
                # [FUNCTION DOC] HELPER: Stores the order and unpacks the pair of
                # dim and keepdim.
                super().__init__()
                self.ord = ord_val
                self.dim, self.keepdim = dim_info

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the vector_norm of the input with the stored
                # order, dim and keepdim.
                return torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        x = torch.randn(2, 3, 5, 5)
        ord_options = [2, float("inf"), -float("inf"), -4, 1.5]
        dim_options = [(None, False), (1, False), ((1, 2), False), ((1, 2), True)]
        for ord_val in ord_options:
            for dim_info in dim_options:
                self.run_test(LinalgVectorNormModel(ord_val, dim_info), x)

    def test_linalg_matrix_norm(self):
        # [FUNCTION DOC] Exports linalg matrix_norm for orders Frobenius, plus and minus
        # infinity, 1 and -1, with default dims, dims 0 and 2, and dims 0 and 2 with keepdim.
        class LinalgMatrixNormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing matrix_norm with a stored order, dim pair
            # and keepdim flag.
            def __init__(self, ord_val, dim_val=(-2, -1), keepdim_val=False):
                # [FUNCTION DOC] HELPER: Stores the order, the dim pair (default last two) and
                # the keepdim flag.
                super().__init__()
                self.ord = ord_val
                self.dim = dim_val
                self.keepdim = keepdim_val

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix_norm of the input with
                # the stored settings.
                return torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

        x = torch.randn(2, 3, 5, 5)
        ord_options = ["fro", float("inf"), -float("inf"), 1, -1]
        for ord_val in ord_options:
            self.run_test(LinalgMatrixNormModel(ord_val), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2)), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2), True), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linalg_cross(self):
        # [FUNCTION DOC] Exports linalg cross on broadcast inputs, with dim 1 and with the
        # default dim. Requires opset 9 or newer.
        class Cross(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two cross products.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the cross product along dim 1 and with
                # the default dim.
                return torch.linalg.cross(x, y, dim=1), torch.linalg.cross(x, y)

        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(1, 3, 1, 3)
        self.run_test(Cross(), input_args=(x, y))

    # This test checks output scalar type in the ONNX graph should not be null
    # https://github.com/pytorch/pytorch/issues/28607
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_trace_script(self):
        # [FUNCTION DOC] Checks that the output scalar type in the exported graph is not
        # null when a scripted slice helper is called from a traced module. Requires opset
        # 10 or newer.
        @torch.jit.script
        def center_slice_helper(input, h_offset):
            # [FUNCTION DOC] HELPER: Scripted function returning the input sliced from the given
            # offset on dim 1.
            return input[:, h_offset:]

        class CenterCrop(torch.nn.Module):
            # [CLASS DOC] HELPER: Module slicing the input from its second dimension size minus
            # one, through the scripted helper.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the scripted helper applied with an offset
                # tensor equal to dim 1 size minus one.
                return center_slice_helper(input, torch.tensor(input.shape[1] - 1))

        x = torch.randn(3, 4)
        self.run_test(CenterCrop(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_logdet(self):
        # [FUNCTION DOC] Exports logdet on batched matrices. Skipped when LAPACK is missing;
        # requires opset 11 or newer.
        class LogDet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the log determinant.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the logdet of the input.
                return torch.logdet(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LogDet(), x)

    def test_dim(self):
        # [FUNCTION DOC] Exports a scripted module multiplying the output by the input's rank,
        # for an empty 1D input and a 3D input.
        class DimModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule doubling the input and scaling it by its
            # number of dimensions.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method returning twice the input times
                # the input rank.
                out = input * 2
                out *= out.dim()
                return out

        empty_input = torch.randn(0, requires_grad=True)
        multi_dim_input = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dim_1(self):
        # [FUNCTION DOC] Exports a scripted loop over boxes clipped to a 2 x 3 image using
        # torchvision, with a dynamic batch axis. Requires opset 11 or newer.
        class M(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule clipping zero-initialized boxes to the image
            # size, one set per input pose.
            @torch.jit.script_method
            def forward(self, poses):
                # [FUNCTION DOC] HELPER: Script method creating zero boxes of shape poses by 2
                # by 4, clipping each set to the image and returning the list of results.
                boxes = torch.zeros([poses.shape[0], 2, 4])
                batch_boxes = []
                for kp_boxes in boxes:
                    kp_boxes = torchvision.ops.clip_boxes_to_image(kp_boxes, (2, 3))
                    batch_boxes.append(kp_boxes)
                return batch_boxes

        dummy_inputs = torch.rand(2, 2, 3)
        self.run_test(M(), (dummy_inputs,), input_names=["x"], dynamic_axes={"x": [0]})

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipDtypeChecking
    def test_outer(self):
        # [FUNCTION DOC] Exports outer product for four dtype pairs: int64 with int64, float32
        # with long, float32 with float64 and int32 with long. Dtype checking disabled; requires
        # opset 12 or newer.
        class Outer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the outer product of two vectors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the outer product of x and y.
                return torch.outer(x, y)

        x = torch.arange(1, 5)
        y = torch.arange(1, 4)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(1, 6).to(dtype=torch.float32)
        y = torch.arange(1, 4).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(2, 5).to(dtype=torch.float32)
        y = torch.arange(2, 4).to(dtype=torch.float64)
        self.run_test(Outer(), input_args=(x, y))

        x = torch.arange(3, 6).to(dtype=torch.int32)
        y = torch.arange(4, 7).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_movedim(self):
        # [FUNCTION DOC] Exports movedim with single and tuple dimension arguments, including
        # identity moves, on a 4D tensor. Requires opset 9 or newer.
        class MovedimModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning six movedim variants.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of six movedim results with integer and
                # tuple source and destination dimensions.
                return (
                    x.movedim(1, 3),
                    x.movedim(2, 0),
                    x.movedim(1, 1),
                    x.movedim((1, 2, 3), (3, 0, 1)),
                    x.movedim((0, 1, 2), (1, 2, 3)),
                    x.movedim((1, 3, 2), (1, 3, 2)),
                )

        x = torch.randn(5, 3, 4, 2)

        self.run_test(MovedimModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_moveaxis(self):
        # [FUNCTION DOC] Exports the moveaxis alias with the same six cases as the movedim test.
        # Requires opset 9 or newer.
        # moveaxis is an alias of movedim; thus, mostly copied from `test_movedim`.
        class MoveaxisModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning six moveaxis variants.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of six moveaxis results with integer
                # and tuple source and destination dimensions.
                return (
                    x.moveaxis(1, 3),
                    x.moveaxis(2, 0),
                    x.moveaxis(1, 1),
                    x.moveaxis((1, 2, 3), (3, 0, 1)),
                    x.moveaxis((0, 1, 2), (1, 2, 3)),
                    x.moveaxis((1, 3, 2), (1, 3, 2)),
                )

        x = torch.randn(5, 3, 4, 2)

        self.run_test(MoveaxisModel(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        # [FUNCTION DOC] Exports einsum for a batch diagonal (float and bool), batched matrix
        # multiply, inner product and a transpose (float and bool). Requires opset 12 or newer.
        class EinsumModelBatchDiagonal(torch.nn.Module):
            # [CLASS DOC] HELPER: Module extracting the diagonal over the last two dims
            # with an einsum.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the einsum batch diagonal of the input.
                eqn = "...ii ->...i"
                return torch.einsum(eqn, x)

        for x in [torch.randn(3, 5, 5), torch.randn(3, 5, 5).to(dtype=torch.bool)]:
            self.run_test(EinsumModelBatchDiagonal(), input_args=(x,))

        class EinsumModelBatchMatmul(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a batched matrix product with an einsum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the einsum batched product of x and y.
                eqn = "bij, bjk -> bik"
                return torch.einsum(eqn, x, y)

        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input_args=(x, y))

        class EinsumModelInnerProd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an inner product with an einsum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the einsum inner product of x and y.
                eqn = "i,i"
                return torch.einsum(eqn, x, y)

        x = torch.randn(5)
        y = torch.randn(5)
        self.run_test(EinsumModelInnerProd(), input_args=(x, y))

        class EinsumModelTranspose(torch.nn.Module):
            # [CLASS DOC] HELPER: Module transposing a matrix with an einsum.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the einsum transpose of the input.
                eqn = "ij->ji"
                return torch.einsum(eqn, x)

        for x in [torch.randn(3, 4), torch.randn(3, 4).to(dtype=torch.bool)]:
            self.run_test(EinsumModelTranspose(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cosine_similarity(self):
        # [FUNCTION DOC] Exports a CosineSimilarity layer over dim 2 on two 5 x 3 x 2 inputs.
        # Requires opset 9 or newer.
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.CosineSimilarity(dim=2), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pairwise_distance(self):
        # [FUNCTION DOC] Exports a PairwiseDistance layer with p of 2.0 on two 5 x 3 x 2 inputs.
        # Requires opset 9 or newer.
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.PairwiseDistance(p=2.0), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cross(self):
        # [FUNCTION DOC] Exports the cross product along dim 3 and with the default dim on two
        # 4D inputs. Requires opset 9 or newer.
        class Cross(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two cross products.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the cross product along dim 3 and with
                # the default dim.
                return torch.cross(x, y, dim=3), torch.cross(x, y)

        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(5, 3, 2, 3)
        self.run_test(Cross(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist(self):
        # [FUNCTION DOC] Exports cdist (pairwise distance between two batches of point sets) on
        # 5 x 3 x 3 and 5 x 2 x 3 inputs. Requires opset 9 or newer.
        class Cdist(torch.nn.Module):
            # [CLASS DOC] HELPER: Module for pairwise distance between sets of points.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the pairwise distances between the points of
                # x and y.
                return torch.cdist(x, y)

        x = torch.randn(5, 3, 3)
        y = torch.randn(5, 2, 3)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_euclid_dist(self):
        # [FUNCTION DOC] Exports cdist with p 2 and the matrix-multiply based Euclidean mode on
        # broadcast batches. Requires opset 9 or newer.
        class Cdist(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing cdist with the matrix
            # multiply Euclidean mode.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns cdist of x and y with p 2.0 and
                # compute mode use_mm_for_euclid_dist.
                return torch.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist")

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_euclid_dist_if_necessary(self):
        # [FUNCTION DOC] Exports cdist with the compute mode that uses matrix multiplication
        # only if necessary. Requires opset 9 or newer.
        class Cdist(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing cdist with
            # the matrix-multiply-if-necessary mode.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns cdist of x and y with p 2.0 and the
                # if-necessary compute mode.
                return torch.cdist(
                    x, y, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"
                )

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_no_euclid_dist(self):
        # [FUNCTION DOC] Exports cdist with the compute mode that never uses matrix
        # multiplication. Requires opset 9 or newer.
        class Cdist(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing cdist without the matrix multiply mode.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns cdist of x and y with p 2.0 and the
                # no-matrix-multiply compute mode.
                return torch.cdist(
                    x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
                )

        x = torch.randn(2, 64, 4)
        y = torch.randn(1, 32, 4)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist_dynamic_axes_script(self):
        # [FUNCTION DOC] Exports a scripted cdist model with dynamic first axes on both inputs.
        # Requires opset 9 or newer.
        class CdistModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing cdist of two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns cdist of a and b.
                return torch.cdist(a, b)

        model = torch.jit.script(CdistModule())
        a = torch.randn(10, 3)
        b = torch.randn(10, 3)
        self.run_test(
            model,
            input_args=(a, b),
            input_names=["a", "b"],
            output_names=["out"],
            dynamic_axes={"a": {0: "size"}, "b": {0: "size"}},
        )

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_crossentropyloss(self):
        # [FUNCTION DOC] Runs the shared cross entropy checks for ignore index -100 (the
        # default) and 1 on inputs of rank 2, 3 and 4 with random labels where label 1 is
        # replaced by the ignore index. Requires opset 12 or newer.
        for ignore_index in [-100, 1]:
            x = torch.randn(3, 5)
            y = torch.empty(3, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index

            self._crossentropyloss(x, y, ignore_index)

            x = torch.randn(3, 5, 2)
            y = torch.empty(3, 2, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

            x = torch.randn(3, 5, 2, 7)
            y = torch.empty(3, 2, 7, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

    def _crossentropyloss(self, x, y, ignore_index):
        # [FUNCTION DOC] HELPER: Runs run_test for cross entropy loss in six variants (reduction
        # none, sum, mean, each with and without class weights) using the given ignore index;
        # the default index is not passed explicitly.
        class CrossEntropyLossNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using reduction none.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss, passing ignore_index only if it
                # differs from the default -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction="none")
                else:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="none", ignore_index=ignore_index
                    )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNone(ignore_index), input_args=(x, y))

        class CrossEntropyLossNoneWeight(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using reduction none and
            # random class weights.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss with five random weights, passing
                # ignore_index only if it differs from -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="none", weight=torch.randn(5)
                    )
                else:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="none",
                        weight=torch.randn(5),
                        ignore_index=ignore_index,
                    )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNoneWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using reduction sum.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss, passing ignore_index only if it
                # differs from -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
                else:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="sum", ignore_index=ignore_index
                    )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSum(ignore_index), input_args=(x, y))

        class CrossEntropyLossSumWeight(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using reduction sum and
            # random class weights.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss with five random weights, passing
                # ignore_index only if it differs from -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="sum", weight=torch.randn(5)
                    )
                else:
                    self.loss = torch.nn.CrossEntropyLoss(
                        reduction="sum",
                        weight=torch.randn(5),
                        ignore_index=ignore_index,
                    )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSumWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using the default mean reduction.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss, passing ignore_index only if it
                # differs from -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss()
                else:
                    self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMean(ignore_index), input_args=(x, y))

        class CrossEntropyLossMeanWeight(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with CrossEntropyLoss using mean reduction and
            # random class weights.
            def __init__(self, ignore_index):
                # [FUNCTION DOC] HELPER: Creates the loss with five random weights, passing
                # ignore_index only if it differs from -100.
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(
                        weight=torch.randn(5), ignore_index=ignore_index
                    )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMeanWeight(ignore_index), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_MSELoss(self):
        # [FUNCTION DOC] Exports three MSE losses with reductions none, sum and mean on the same
        # 2 x 3 x 5 inputs. Requires opset 9 or newer.
        class MSELoss(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding three MSELoss instances
            # with different reductions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three loss modules.
                super().__init__()
                self.loss1 = torch.nn.MSELoss(reduction="none")
                self.loss2 = torch.nn.MSELoss(reduction="sum")
                self.loss3 = torch.nn.MSELoss(reduction="mean")

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns a tuple of the three loss values for the
                # input and target.
                return (
                    self.loss1(input, target),
                    self.loss2(input, target),
                    self.loss3(input, target),
                )

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        self.run_test(MSELoss(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_kldiv_loss(self):
        # [FUNCTION DOC] Runs the shared KL divergence checks on log-probability inputs of rank
        # 1, 3 and 4 with random probability targets. Requires opset 9 or newer.
        x = torch.rand(5).log()
        y = torch.rand(5)
        self._kldiv_loss(x, y)

        x = torch.rand(2, 3, 5).log()
        y = torch.rand(2, 3, 5)
        self._kldiv_loss(x, y)

        x = torch.rand(2, 3, 5, 7).log()
        y = torch.rand(2, 3, 5, 7)
        self._kldiv_loss(x, y)

    def _kldiv_loss(self, x, y):
        # [FUNCTION DOC] HELPER: Runs run_test for KLDivLoss in five variants: none with log
        # target, mean, sum with log target, batchmean, and batchmean through the legacy
        # size_average flag with log target.
        class KLDivLossNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with KLDivLoss reduction none and log targets.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss.
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction="none", log_target=True)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the log
                # of the target.
                return self.loss(input, target.log())

        self.run_test(KLDivLossNone(), input_args=(x, y))

        class KLDivLossMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with KLDivLoss reduction mean and probability targets.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss.
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction="mean", log_target=False)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(KLDivLossMean(), input_args=(x, y))

        class KLDivLossSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with KLDivLoss reduction sum and log targets.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss.
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction="sum", log_target=True)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the log
                # of the target.
                return self.loss(input, target.log())

        self.run_test(KLDivLossSum(), input_args=(x, y))

        class KLDivLossBatchMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with KLDivLoss reduction batchmean
            # and probability targets.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss.
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the target.
                return self.loss(input, target)

        self.run_test(KLDivLossBatchMean(), input_args=(x, y))

        class KLDivLossMiniBatchMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with KLDivLoss reduction batchmean via the legacy
            # size_average setting, with log targets.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss.
                super().__init__()
                self.loss = torch.nn.KLDivLoss(
                    reduction="batchmean", size_average=False, log_target=True
                )

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss of the input against the log
                # of the target.
                return self.loss(input, target.log())

        self.run_test(KLDivLossMiniBatchMean(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss(self):
        # [FUNCTION DOC] Exports NLLLoss with reduction none after a log softmax of doubled
        # inputs, with some labels set to the default ignore index. Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying LogSoftmax to twice the input and an
            # unreduced NLL loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss and log softmax submodules.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of twice the
                # input against the target.
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_none(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with reduction none after a convolution and log
        # softmax, with some labels set to the default ignore index. Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution to C classes, a log softmax and an
            # unreduced NLL loss; C is a variable of the enclosing test.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss, convolution (16 to C channels, kernel
                # 3) and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with mean reduction, same setup as the none variant.
        # Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, log softmax and mean NLL loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss, convolution and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="mean")
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_sum(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with sum reduction, same setup as the none variant.
        # Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, log softmax and sum NLL loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss, convolution and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="sum")
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_weights(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with mean reduction and random class weights.
        # Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, log softmax and weighted
            # mean NLL loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weighted loss, convolution and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="mean", weight=torch.randn(C))
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with mean reduction and ignore index 1 on random
        # labels. Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, log softmax and mean NLL loss
            # ignoring class 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the loss, convolution and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="mean", ignore_index=1)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_dynamic_ignore_index(self):
        # [FUNCTION DOC] Exports a label-smoothing style loss module where the ignore index is
        # computed from an input's size at run time. Requires opset 12 or newer.
        import torch.nn.functional as F

        def linear_combination(x, y, epsilon):
            # [FUNCTION DOC] HELPER: Returns a linear blend of x and y weighted by epsilon.
            # Defined but not used by this test.
            return epsilon * x + (1 - epsilon) * y

        def reduce_loss(loss, reduction="mean"):
            # [FUNCTION DOC] HELPER: Reduces a loss tensor by mean or sum, or returns it
            # unchanged for another reduction name. Defined but not used by this test.
            return (
                loss.mean()
                if reduction == "mean"
                else loss.sum()
                if reduction == "sum"
                else loss
            )

        class LabelSmoothingCrossEntropy(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an NLL loss over log softmax predictions
            # whose ignore index depends on an input's second dimension size, plus the
            # float start positions.
            def __init__(self, epsilon: float = 0.1, reduction="mean"):
                # [FUNCTION DOC] HELPER: Stores epsilon and the reduction mode.
                super().__init__()
                self.epsilon = epsilon
                self.reduction = reduction

            def forward(self, preds, target, start_position):
                # [FUNCTION DOC] HELPER: Computes the log softmax of the predictions along the
                # last dim, takes the ignore index from the size of dim 1 of the start
                # positions, and returns the NLL loss plus the start positions as float.
                n = preds.size()[-1]
                log_preds = F.log_softmax(preds, dim=-1)
                ignore_index = start_position.size(1)
                nll = F.nll_loss(
                    log_preds,
                    target,
                    reduction=self.reduction,
                    ignore_index=ignore_index,
                )
                return nll + start_position.float()

        N = 5
        preds = torch.randn(N, 16)
        target = torch.randint(5, (N,))
        start_position = torch.randint(10, (N, N))
        self.run_test(LabelSmoothingCrossEntropy(), (preds, target, start_position))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index_weights(self):
        # [FUNCTION DOC] Exports 2D NLLLoss with mean reduction, random class weights and ignore
        # index 1. Requires opset 12 or newer.
        class NLLModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, log softmax and weighted mean NLL
            # loss ignoring class 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weighted loss, convolution and log softmax.
                super().__init__()
                self.loss = torch.nn.NLLLoss(
                    reduction="mean", weight=torch.randn(C), ignore_index=1
                )
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the NLL loss of the log softmax of the
                # convolved input against the target.
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_binary_cross_entropy_with_logits(self):
        # [FUNCTION DOC] Exports binary cross entropy with logits in four setups through shared
        # helpers: plain on a vector, with a weight tensor, with a positive-class weight, and
        # with both. Requires opset 12 or newer.
        x = torch.randn(5)
        y = torch.empty(5).random_(2)
        self._bce_logits(x, y)

        x = torch.randn(3, 4)
        y = torch.empty(3, 4).random_(2)
        weight = torch.tensor([3])
        self._bce_logits_wegiht(x, y, weight)

        x = torch.randn(3, 2, 4)
        y = torch.empty(3, 2, 4).random_(2)
        pos_weight = torch.empty([2, 4]).random_(2)
        self._bce_logits_posweight(x, y, pos_weight)

        x = torch.randn(3, 3, 4)
        y = torch.empty(3, 3, 4).random_(2)
        weight = torch.tensor([3])
        pos_weight = torch.empty([3, 4]).random_(2)
        self._bce_logits_loss_weight_posweight(x, y, weight, pos_weight)

    def _bce_logits(self, x, y):
        # [FUNCTION DOC] HELPER: Runs run_test for functional binary cross entropy with logits
        # using reductions none, mean and sum, without weights.
        class BCEWithLogitsLossNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the unreduced loss.
            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss with reduction none.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="none"
                )

        self.run_test(BCEWithLogitsLossNone(), input_args=(x, y))

        class BCEWithLogitsLossMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the mean loss.
            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss with reduction mean.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossMean(), input_args=(x, y))

        class BCEWithLogitsLossSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the summed loss.
            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the loss with reduction sum.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossSum(), input_args=(x, y))

    def _bce_logits_wegiht(self, x, y, weight):
        # [FUNCTION DOC] HELPER: Runs run_test for the logits loss with a weight tensor input
        # for reductions none, mean and sum. The name keeps a historical misspelling of weight.
        class BCEWithLogitsLossWegihtNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the unreduced loss with a weight input.
            def forward(self, input, target, weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the weight and reduction none.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="none"
                )

        self.run_test(BCEWithLogitsLossWegihtNone(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the mean loss with a weight input.
            def forward(self, input, target, weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the weight and reduction mean.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossWegihtMean(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the summed loss with a weight input.
            def forward(self, input, target, weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the weight and reduction sum.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossWegihtSum(), input_args=(x, y, weight))

    def _bce_logits_posweight(self, x, y, pos_weight):
        # [FUNCTION DOC] HELPER: Runs run_test for the logits loss with a positive-class weight
        # input for reductions none, mean and sum.
        class BCEWithLogitsLossPosWegihtNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the unreduced loss with a
            # positive weight input.
            def forward(self, input, target, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the positive weight
                # and reduction none.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="none"
                )

        self.run_test(BCEWithLogitsLossPosWegihtNone(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the mean loss with a positive weight input.
            def forward(self, input, target, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the positive weight
                # and reduction mean.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="mean"
                )

        self.run_test(BCEWithLogitsLossPosWegihtMean(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the summed loss with a positive weight input.
            def forward(self, input, target, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with the positive weight
                # and reduction sum.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, pos_weight=pos_weight, reduction="sum"
                )

        self.run_test(BCEWithLogitsLossPosWegihtSum(), input_args=(x, y, pos_weight))

    def _bce_logits_loss_weight_posweight(self, x, y, weight, pos_weight):
        # [FUNCTION DOC] HELPER: Runs run_test for the logits loss with both a weight and a
        # positive-class weight input for reductions none, mean and sum.
        class BCEWithLogitsLossWeightPosweightNone(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the unreduced loss with weight and
            # positive weight inputs.
            def forward(self, input, target, weight, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with both weights and reduction none.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="none",
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightNone(),
            input_args=(x, y, weight, pos_weight),
        )

        class BCEWithLogitsLossWeightPosweightMean(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the mean loss with weight and
            # positive weight inputs.
            def forward(self, input, target, weight, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with both weights and reduction mean.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input,
                    target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="mean",
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightMean(),
            input_args=(x, y, weight, pos_weight),
        )

        class BCEWithLogitsLossWeightPosweightSum(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the summed loss with weight and
            # positive weight inputs.
            def forward(self, input, target, weight, pos_weight):
                # [FUNCTION DOC] HELPER: Returns the loss with both weights and reduction sum.
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    input, target, weight=weight, pos_weight=pos_weight, reduction="sum"
                )

        self.run_test(
            BCEWithLogitsLossWeightPosweightSum(), input_args=(x, y, weight, pos_weight)
        )

    def test_torch_mm(self):
        # [FUNCTION DOC] Exports a matrix product of a 2 x 3 and a 3 x 3 matrix.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a matrix product of two matrices.
            def forward(self, mat1, mat2):
                # [FUNCTION DOC] HELPER: Returns mm of the two matrices.
                mm = torch.mm(mat1, mat2)
                return mm

        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because where op is not supported for opset < 9.
    def test_where_with_bool_tensor(self):
        # [FUNCTION DOC] Exports where with a condition from comparing a tensor to zero.
        # Requires opset 9 or newer because where is not supported earlier.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting mat1 where it is positive, else mat2.
            def forward(self, mat1, mat2):
                # [FUNCTION DOC] HELPER: Returns where mat1 is greater than zero choose
                # mat1 else mat2.
                out = torch.where(mat1 > 0, mat1, mat2)
                return out

        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(
        9
    )  # Because where op is not supported for opset < 9.
    def test_where_with_byte_tensor(self):
        # [FUNCTION DOC] Exports where with a uint8 condition tensor, one entry set to zero.
        # Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between two tensors by a uint8 condition.
            def forward(self, cond, mat1, mat2):
                # [FUNCTION DOC] HELPER: Returns the where result of the condition,
                # mat1 and mat2.
                out = torch.where(cond, mat1, mat2)
                return out

        cond = torch.ones(2, 3, dtype=torch.uint8)
        cond[1, 2] = 0
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(cond, mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(10)  # ONNX IsInf op is added in opset 10.
    def test_isinf(self):
        # [FUNCTION DOC] Exports isinf on a tensor with infinities and NaN. Requires opset
        # 10 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module checking for infinite values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a boolean tensor marking infinite entries.
                return x.isinf()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_isfinite(self):
        # [FUNCTION DOC] Exports isfinite on a tensor with infinities and NaN. Requires opset
        # 10 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module checking for finite values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a boolean tensor marking finite entries.
                return x.isfinite()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)  # ONNX IsNaN op is added in opset 9.
    def test_isnan(self):
        # [FUNCTION DOC] Exports isnan on a tensor with NaN and infinities. Requires opset
        # 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module checking for NaN values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a boolean tensor marking NaN entries.
                return x.isnan()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), float("inf")]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(
        10
    )  # ONNX IsNaN, IsInf op is added in opset 9, 10 respectively.
    def test_nan_to_num(self):
        # [FUNCTION DOC] Exports nan_to_num with defaults on float, int and half inputs, and
        # with explicit replacements for NaN, positive and negative infinity. Requires opset
        # 10 or newer.
        class NoParams(torch.nn.Module):
            # [CLASS DOC] HELPER: Module replacing NaN and infinities with default values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns nan_to_num of the input.
                return x.nan_to_num()

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        xint = torch.ones((2, 4), dtype=torch.int)
        xhalf = torch.ones((2, 4), dtype=torch.half)
        self.run_test(NoParams(), (x,))
        self.run_test(NoParams(), (xint,))
        self.run_test(NoParams(), (xhalf,))

        class WithParams(torch.nn.Module):
            # [CLASS DOC] HELPER: Module replacing NaN with 2.3, positive infinity with 4.5 and
            # negative infinity with 6.7.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns nan_to_num with those replacement values.
                return x.nan_to_num(nan=2.3, posinf=4.5, neginf=6.7)

        x = torch.tensor([[1, 2, float("inf")], [2, float("nan"), -float("inf")]])
        self.run_test(WithParams(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_maximum_minimum(self):
        # [FUNCTION DOC] Exports maximum and minimum on a vector containing NaN broadcast
        # against a 1 x 3 tensor. Requires opset 9 or newer.
        class ModelWithNan(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning elementwise maximum and minimum
            # of two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair: maximum and minimum of x and y.
                return torch.maximum(x, y), torch.minimum(x, y)

        x = torch.tensor([-2, -2, float("nan")])
        y = torch.rand(1, 3)
        self.run_test(ModelWithNan(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_minimum_dtypes(self):
        # [FUNCTION DOC] Exports minimum over mixed dtype pairs: half with float, half
        # with int16, int16 with int32, and int with a boolean-filled tensor. Requires
        # opset 12 or newer.
        class MinimumModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing elementwise minimum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the elementwise minimum of x and y.
                return torch.minimum(x, y)

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MinimumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MinimumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_maximum_dtypes(self):
        # [FUNCTION DOC] Exports maximum over the same mixed dtype pairs as the minimum test.
        # Requires opset 12 or newer.
        class MaximumModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing elementwise maximum.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the elementwise maximum of x and y.
                return torch.maximum(x, y)

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MaximumModel(), (x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MaximumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_any(self):
        # [FUNCTION DOC] Exports any over all elements, along dim 1, and along dim 1 with
        # keepdim. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with any over all elements.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns whether any element is true.
                return x.any()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with any along dim 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns any along dim 1.
                return x.any(dim=1)

        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with any along dim 1 keeping the dim.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns any along dim 1 with keepdim.
                return x.any(dim=1, keepdim=True)

        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_all(self):
        # [FUNCTION DOC] Exports all over all elements, along dim 1, and along dim 1 with
        # keepdim. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with all over all elements.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns whether all elements are true.
                return x.all()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with all along dim 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns all along dim 1.
                return x.all(dim=1)

        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reducing with all along dim 1 keeping the dim.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns all along dim 1 with keepdim.
                return x.all(dim=1, keepdim=True)

        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    def test_dropout(self):
        # [FUNCTION DOC] Exports a dropout layer with probability 0.3 on a 10 x 3 x 53 input;
        # the base run_test exports in evaluation mode so dropout is the identity.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Dropout layer with probability 0.3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dropout submodule.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dropout of the input.
                dropout = self.dropout(x)
                return dropout

        x = torch.randn(10, 3, 53)
        self.run_test(M(), (x))

    def test_rrelu_eval(self):
        # [FUNCTION DOC] Exports RReLU with bounds 0.1 and 0.3 in evaluation mode on
        # a two-element input.
        x = torch.tensor([0.5, -0.5])
        self.run_test(torch.nn.RReLU(0.1, 0.3).eval(), x)

    def test_shape_constant_fold(self):
        # [FUNCTION DOC] Exports a module that adds the length of a ones buffer, taken from its
        # shape, to the input; the shape is constant folded.
        class ShapeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer of length five whose shape is added
            # to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ones buffer.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the first dimension size
                # of the buffer.
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        self.run_test(ShapeModule(), (x,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu(self):
        # [FUNCTION DOC] Exports CELU with alpha 1.0 on a two-element tensor. Requires opset
        # 12 or newer.
        class Celu(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a CELU activation with alpha 1.0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CELU submodule.
                super().__init__()
                self.celu = torch.nn.CELU(alpha=1.0)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the CELU of the input.
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_default(self):
        # [FUNCTION DOC] Exports CELU with default alpha. Requires opset 12 or newer.
        class Celu(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a CELU activation with default alpha.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CELU submodule.
                super().__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the CELU of the input.
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_alpha(self):
        # [FUNCTION DOC] Exports CELU with alpha 2.0. Requires opset 12 or newer.
        class Celu(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a CELU activation with alpha 2.0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CELU submodule.
                super().__init__()
                self.celu = torch.nn.CELU(alpha=2.0)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the CELU of the input.
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_cast(self):
        # [FUNCTION DOC] Exports CELU with default alpha on a double precision tensor. Requires
        # opset 12 or newer.
        class Celu(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a CELU activation with default alpha.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CELU submodule.
                super().__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the CELU of the input.
                return self.celu(input)

        input = torch.randn(2, 5, 7, dtype=torch.float64)
        self.run_test(Celu(), (input,))

    def test_lower_tuple(self):
        # [FUNCTION DOC] Exports a model that builds, rebuilds and unpacks tuples inside nested
        # loops and conditionals, to exercise lowering of tuples in the graph.
        class TupleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module juggling several tuples of its three inputs through
            # loops and a conditional.
            def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Starts with tuples of the inputs, and in five outer and
                # two inner iterations rebuilds them based on whether the third input length
                # differs from the last dim of the first; finally returns the sum of the three
                # entries of the last tuple.
                a = (input1, input2)
                b = a
                c = (input1, input2, input3)
                for _ in range(5):
                    d = a[0]
                    for _ in range(2):
                        e, f = a
                        a = (d, f)
                        f = c[2]
                        if f.size(0) != input1.size(-1):
                            g = b[1]
                            b = (g, f)
                        else:
                            k = c[1:]
                            b = (f, k[0])
                    m, n = b
                    c = (input1, n, m)
                p, q, r = c
                return p + q + r

        input1 = torch.randn(2)
        input2 = torch.randn(2)
        input3 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2, input3))

    def test_lower_tuple_2(self):
        # [FUNCTION DOC] Exports a model that repeatedly unpacks and repacks a tuple in a loop
        # and returns the tuple.
        class TupleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repacking a pair of inputs five times and returning it.
            def forward(self, input1: Tensor, input2: Tensor) -> tuple[Tensor, Tensor]:
                # [FUNCTION DOC] HELPER: Returns the pair of tensors after five unpack
                # and repack cycles.
                a = (input1, input2)
                for _ in range(5):
                    c, d = a
                    a = (c, d)
                return a

        input1 = torch.randn(2)
        input2 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2))

    def test_lower_tuple_3(self):
        # [FUNCTION DOC] Exports a model taking two tuples of tensors, swapping and updating
        # them across five loop iterations depending on the equality of first dimension sizes,
        # and returning a pair of tuples.
        class TupleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repacking two tuple inputs in a loop and returning them
            # as a pair of tuples.
            def forward(
                self,
                input1: tuple[Tensor, Tensor],
                input2: tuple[Tensor, Tensor],
            ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
                # [FUNCTION DOC] HELPER: Five times unpacks both tuples; if the first dim of the
                # first element of each matches adds one to the other tensor, else adds to the
                # second; then swaps the roles; returns both tuples.
                a = input1
                b = input2
                for _ in range(5):
                    c, d = a
                    e, f = b
                    if c.shape[0] == e.shape[0]:
                        e = e + c
                    else:
                        f = f + d
                    a = (e, f)
                    b = (c, d)
                return a, b

        input1 = (torch.randn(2), torch.randn(2))
        input2 = (torch.randn(2), torch.randn(2))
        self.run_test(TupleModule(), (input1, input2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where(self):
        # [FUNCTION DOC] Exports where with a boolean condition of shape 2 x 3 x 4 broadcast
        # with inputs of shapes 2 x 1 x 4 and 2 x 3 x 1. Requires opset 9 or newer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting between two tensors by a condition.
            def forward(self, cond, input, other):
                # [FUNCTION DOC] HELPER: Returns where the condition picks input else other.
                return torch.where(cond, input, other)

        x = torch.randint(0, 1, (2, 3, 4), dtype=torch.bool)
        y = torch.randn(2, 1, 4)
        z = torch.ones(2, 3, 1)
        self.run_test(Model(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # scripting tests run for opsets > 11. See: test_where_condition_script
    def test_where_condition(self):
        # [FUNCTION DOC] Exports single-argument where (returns index tensors) stacked along dim
        # 1, for a comparison with a constant and between two tensors. Skipped under scripting;
        # requires opset 9 or newer.
        class Model1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking the indices where the input exceeds 0.5.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indices of true entries of the comparison
                # stacked along dim 1.
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking the indices where input exceeds other.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the indices of true entries of the comparison
                # stacked along dim 1.
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_where_condition_script(self):
        # [FUNCTION DOC] Same as the where-condition test in script mode. Not run for opset 13;
        # requires opset 11 or newer.
        class Model1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking the indices where the input exceeds 0.5.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the indices of true entries of the comparison
                # stacked along dim 1.
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module stacking the indices where input exceeds other.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the indices of true entries of the comparison
                # stacked along dim 1.
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    def test_empty_branch(self):
        # [FUNCTION DOC] Exports a scripted module with nested conditionals whose else
        # branches do nothing.
        class EmptyBranchModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule adding 3 only when the rank of input plus one
            # exceeds 3, with empty other branches.
            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method adding 1, then adding 3 if the rank
                # exceeds 3; other branches do nothing; returns the result.
                out = input + 1
                if out.dim() > 2:
                    if out.dim() > 3:
                        out += 3
                    else:
                        pass
                else:
                    pass
                return out

        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(EmptyBranchModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_derive_index_scripting(self):
        # [FUNCTION DOC] Exports four modules looping over reversed or forward index ranges with
        # steps 2 and 3 derived from the length of the input, appending products of the input
        # and a row to a list. Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from the last index downwards by 2.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the descending range appends the
                # input times that row to a list and returns the list.
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from minus length upwards by 2.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the ascending range appends the input
                # times that row to a list and returns the list.
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from the last index downwards by 3.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the descending range appends the
                # input times that row to a list and returns the list.
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from minus length upwards by 3.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the ascending range appends the input
                # times that row to a list and returns the list.
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    @skipScriptTest()  # Scripting fails for add lists for opsets < 11. Check test_derive_index_scripting
    def test_derive_index(self):
        # [FUNCTION DOC] Tracing variant of the derived-index loops; same four range patterns as
        # the scripted test. Skipped under scripting.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from the last index downwards by 2.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the descending range appends the
                # input times that row to a list and returns the list.
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from minus length upwards by 2.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the ascending range appends the input
                # times that row to a list and returns the list.
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from the last index downwards by 3.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the descending range appends the
                # input times that row to a list and returns the list.
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module iterating from minus length upwards by 3.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: For each index in the ascending range appends the input
                # times that row to a list and returns the list.
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_if_transpose(self):
        # [FUNCTION DOC] Exports a scripted model that transposes and conditionally transposes
        # back based on a dimension size, with dynamic output axes. Requires opset 11 or newer.
        class IfModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module transposing the input and, if its first dim is
            # 2, transposing again.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Transposes x; returns it transposed back if dim 0 is
                # then 2, else the transposed tensor.
                x = x.transpose(0, 1)
                if x.size(0) == 2:
                    return x.transpose(0, 1)
                else:
                    return x

        x = torch.randn(2, 3)
        self.run_test(
            torch.jit.script(IfModel()),
            x,
            output_names=["output_1"],
            dynamic_axes={"output_1": [0, 1]},
        )

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_list(self):
        # [FUNCTION DOC] Exports a scripted model that builds a list containing x or y depending
        # on a boolean tensor condition. Requires opset 13 or newer.
        class IfModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a one-element list holding x or y depending
            # on the condition.
            def forward(self, x, y, cond):
                # [FUNCTION DOC] HELPER: Returns a list with x if cond is true, else with y.
                res = []
                if cond:
                    res = res + [x]
                else:
                    res = res + [y]
                return res

        x = torch.randn(2, 3)
        y = torch.randn(3, 3)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(IfModel()), (x, y, cond))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_view(self):
        # [FUNCTION DOC] Exports a scripted model that reshapes x to the batch and sequence dims
        # of y when the condition holds, then transposes dims 1 and 2, with a dynamic output
        # axis. Requires opset 13 or newer.
        class IfModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module viewing x like y's leading dims or using
            # y, then transposing.
            def forward(self, x, y, cond):
                # [FUNCTION DOC] HELPER: Reads batch and sequence sizes from y; if cond is true
                # views x as batch by sequence by inferred, else uses y; returns the result
                # transposed on dims 1 and 2.
                bs, seq = y.shape[:2]
                if cond:
                    res = x.view(bs, seq, -1)
                else:
                    res = y
                return res.transpose(1, 2)

        x = torch.randn(2, 16, 2, 2)
        y = torch.randn(2, 16, 8)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(
            torch.jit.script(IfModel()),
            (x, y, cond),
            output_names=["output_1"],
            dynamic_axes={"output_1": [1]},
        )

    @skipScriptTest(
        skip_before_opset_version=11, reason="dynamic split support added in 11"
    )
    def test_split_tensor_scalar(self):
        # [FUNCTION DOC] Exports split using the second dimension size as split size. Skipped
        # for scripting before opset 11 because dynamic split was added then.
        class SplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting with the size of dim 1 as the chunk size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the split of x with chunk size equal to its
                # dim 1 size.
                return torch.split(x, x.size(1))

        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(SplitModel(), x)

    def test_split_tensor_multi(self):
        # [FUNCTION DOC] Checks that building the model with a mistaken argument raises
        # TypeError: the nested runner constructs the module class with a tensor, which the
        # class does not accept. The export itself is not exercised.
        class SplitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting with a tensor of ones as split sizes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the split of x with a tensor of three ones
                # as split sizes.
                return torch.split(x, torch.ones(3))

        x = torch.randn(1, 2, 3, requires_grad=True)

        def run_model():
            # [FUNCTION DOC] HELPER: Calls the model class with the tensor as constructor
            # argument, which fails with TypeError because the class takes none.
            SplitModel(x)

        self.assertRaises(TypeError, run_model)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding(self):
        # [FUNCTION DOC] Exports functional embedding with padding index 1 for 1D and 3D
        # index tensors, also in training mode, and then without a padding index. Requires
        # opset 9 or newer.
        class EmbedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module looking up indices in a weight matrix with
            # padding index 1.
            def forward(self, input, emb):
                # [FUNCTION DOC] HELPER: Returns the embedding of the input from emb with
                # padding index 1.
                return torch.nn.functional.embedding(input, emb, padding_idx=1)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = torch.rand(10, 3)
        self.run_test(model, (x, embedding_matrix))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x, embedding_matrix))
        self.run_test(
            model, (x, embedding_matrix), training=torch.onnx.TrainingMode.TRAINING
        )

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            # [CLASS DOC] HELPER: Module looking up indices in a weight matrix without
            # a padding index.
            def forward(self, input, emb):
                # [FUNCTION DOC] HELPER: Returns the embedding of the input from emb.
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x, embedding_matrix))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding_module(self):
        # [FUNCTION DOC] Exports Embedding modules with padding index 1 (one with the padding
        # row set to ones) for 1D and 3D indices, then an Embedding without padding index.
        # Requires opset 9 or newer.
        class EmbedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two embeddings of 4 by 3 with padding index 1, the
            # second having its padding row set to ones.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates both embeddings and overwrites the padding row
                # of the second without tracking gradients.
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3, padding_idx=1)
                self.emb2 = torch.nn.Embedding(4, 3, padding_idx=1)
                with torch.no_grad():
                    self.emb2.weight[1] = torch.ones(3)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the lookups of the input in both embeddings.
                return self.emb(input), self.emb2(input)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        self.run_test(model, (x,))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x,))

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one embedding of 4 by 3 without padding index.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the embedding.
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the lookup of the input in the embedding.
                return self.emb(input)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_renorm(self):
        # [FUNCTION DOC] Exports Embedding layers with max norm renormalization: 0.2 with
        # default norm type and 0.5 with norm type 1. Requires opset 11 or newer.
        n, d = 7, 5
        embedding = torch.nn.Embedding(n, d, max_norm=0.2)
        idx = torch.tensor([2, 1])
        self.run_test(embedding, idx)

        embedding = torch.nn.Embedding(n, d, max_norm=0.5, norm_type=1.0)
        idx = torch.tensor([4, 3, 4, 2])
        self.run_test(embedding, idx)

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Dispatches to the Elman, LSTM or GRU test helper by name,
        # forwarding all arguments. An unknown name does nothing.
        if name == "elman":
            self._elman_rnn_test(*args, **kwargs)
        if name == "lstm":
            self._lstm_test(*args, **kwargs)
        if name == "gru":
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(
        self,
        layers,
        nonlinearity,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Builds and exports an Elman RNN with the given layers,
        # nonlinearity, directionality, initial state option, packed sequence mode (0 none, 1
        # variable lengths, 2 batch first) and dropout, then checks it with a random padded
        # batch and also with a different batch size.
        class ElmanWithStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an RNN that takes a packed sequence and an
            # optional initial state.
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the RNN.
                # KNOWN ISSUE: the bidirect argument is ignored; the RNN uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()

                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    layers,
                    nonlinearity=nonlinearity,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence, hx=None):
                # [FUNCTION DOC] HELPER: Returns the inner RNN output for the input
                # and initial state.
                return self.inner_model(input, hx)

        class ElmanWithoutStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an RNN that takes a packed sequence
            # without initial state.
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the RNN.
                # KNOWN ISSUE: the bidirect argument is ignored; the RNN uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    layers,
                    nonlinearity=nonlinearity,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence):
                # [FUNCTION DOC] HELPER: Returns the inner RNN output for the input.
                return self.inner_model(input)

        batch_first = packed_sequence == 2

        if initial_state:
            model = ElmanWithStateModel(
                layers=layers,
                bidirect=bidirectional,
                nonlinearity=nonlinearity,
                dropout=dropout,
                batch_first=batch_first,
            )
            if packed_sequence:
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
        else:
            model = ElmanWithoutStateModel(
                layers=layers,
                bidirect=bidirectional,
                nonlinearity=nonlinearity,
                dropout=dropout,
                batch_first=batch_first,
            )
            if packed_sequence:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )

        def make_input(batch_size):
            # [FUNCTION DOC] HELPER: Builds a random padded batch with random decreasing
            # lengths, an optional initial state, and for packed modes a lengths tensor. Returns
            # the input (single tensor or tuple) and a matching list of input names.
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append("h0")
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    def _lstm_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Builds and exports an LSTM test model with the given layers,
        # directionality, initial state option (hidden and cell states), packed sequence mode
        # and dropout. Uses a flattening helper model, with the packed sequence wrappers for
        # packed modes. Runs run_test on a random padded batch and on a different batch size
        # with dynamic axes.
        batch_first = packed_sequence == 2

        if packed_sequence:
            model = lstm_flattening_result.LstmFlatteningResultWithSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )
            if initial_state:
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
            else:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )
        else:
            model = lstm_flattening_result.LstmFlatteningResultWithoutSeqLength(
                RNN_INPUT_SIZE,
                RNN_HIDDEN_SIZE,
                layers,
                bidirectional,
                dropout,
                batch_first,
            )

        def make_input(batch_size):
            # [FUNCTION DOC] HELPER: Builds a random padded batch with random decreasing
            # lengths, optional initial hidden and cell states as a tuple, and for packed modes
            # the lengths tensor. Returns the input and the list of input names.
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]
            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                c0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append((h0, c0))
                input_names.append("h0")
                input_names.append("c0")
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1], "c0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    def _gru_test(
        self,
        layers,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
        **extra_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Builds and exports a GRU test model, with or without
        # initial state and packed sequence wrappers, using the given layers, directionality
        # and dropout. Runs run_test on a random padded batch and on a different batch size,
        # with dynamic axes.
        class GRUWithStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a GRU that takes a packed sequence and
            # an initial state.
            def __init__(self, layers, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the GRU.
                # KNOWN ISSUE: the bidirect argument is ignored; the GRU uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()

                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence, hx):
                # [FUNCTION DOC] HELPER: Returns the GRU output for the packed input
                # and initial state.
                return self.inner_model(input, hx)

        class GRUWithoutStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a GRU that takes a packed sequence
            # without initial state.
            def __init__(self, layers, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the GRU.
                # KNOWN ISSUE: the bidirect argument is ignored; the GRU uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input: rnn_utils.PackedSequence):
                # [FUNCTION DOC] HELPER: Returns the GRU output for the packed input.
                return self.inner_model(input)

        class GRUNoSeqLengthWithoutStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a GRU that takes a padded tensor without
            # lengths or initial state.
            def __init__(self, layers, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the GRU.
                # KNOWN ISSUE: the bidirect argument is ignored; the GRU uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the GRU output for the input.
                return self.inner_model(input)

        class GRUNoSeqLengthWithStateModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a GRU that takes a padded tensor and
            # an initial state.
            def __init__(self, layers, bidirect, dropout, batch_first):
                # [FUNCTION DOC] HELPER: Stores the batch-first flag and creates the GRU.
                # KNOWN ISSUE: the bidirect argument is ignored; the GRU uses the directionality
                # variable of the enclosing helper instead.
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(
                    RNN_INPUT_SIZE,
                    RNN_HIDDEN_SIZE,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, input, hx):
                # [FUNCTION DOC] HELPER: Returns the GRU output for the input and initial state.
                return self.inner_model(input, hx)

        batch_first = packed_sequence == 2

        if packed_sequence:
            if initial_state:
                model = GRUWithStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                model = (
                    rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(
                        model, batch_first
                    )
                )
            else:
                model = GRUWithoutStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(
                    model, batch_first
                )
        else:
            if initial_state:
                model = GRUNoSeqLengthWithStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )
            else:
                model = GRUNoSeqLengthWithoutStateModel(
                    layers=layers,
                    bidirect=bidirectional,
                    dropout=dropout,
                    batch_first=batch_first,
                )

        def make_input(batch_size):
            # [FUNCTION DOC] HELPER: Builds a random padded batch with random decreasing
            # lengths, an optional initial state, and for packed modes the lengths tensor.
            # Returns the input and the list of input names.
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ["input"]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append("h0")
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append("seq_lengths")
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input, input_names

        input, input_names = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {"input": [0, 1], "seq_lengths": [0]}
        if initial_state:
            dynamic_axes.update({"h0": [1]})
        export_options = {"input_names": input_names, "dynamic_axes": dynamic_axes}

        # test that the model still runs with a different batch size
        other_input, _ = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(
            model, input, additional_test_inputs=[other_input], **export_options
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quantize_per_tensor(self):
        # [FUNCTION DOC] Exports fake quantization per tensor with scale 1/127, zero point 0 and
        # range -128 to 127. Requires opset 10 or newer.
        class FakeQuantizePerTensorModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying per-tensor fake quantization
            # with constant parameters.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the fake-quantized input.
                scale = 1.0 / 127
                zero_point = 0
                quant_min = -128
                quant_max = 127
                return torch.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerTensorModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_tensor_dynamic_scale_zeropoint(self):
        # [FUNCTION DOC] Exports per-tensor fake quantization with scale and zero point passed
        # as tensor inputs. Requires opset 13 or newer.
        class FakeQuantizePerTensorModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying per-tensor fake quantization with tensor scale
            # and zero point.
            def forward(self, input, scale, zero_point):
                # [FUNCTION DOC] HELPER: Returns the fake-quantized input with range
                # -128 to 127.
                quant_min = -128
                quant_max = 127
                return torch.fake_quantize_per_tensor_affine(
                    input, scale, zero_point, quant_min, quant_max
                )

        x = torch.randn(6, 4, 3, 3)
        scale = torch.tensor(1.0 / 127)
        zero_point = torch.tensor(0)
        self.run_test(FakeQuantizePerTensorModel(), (x, scale, zero_point))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_channel(self):
        # [FUNCTION DOC] Exports per-channel fake quantization on axis 1 applied twice with
        # different ranges (0 to 255, then -128 to 127). Requires opset 13 or newer.
        class FakeQuantizePerChannelModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying per-channel fake quantization twice on axis 1.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Builds a scale of ones divided by 127 and zero int zero
                # points for four channels, quantizes with range 0 to 255, then quantizes the
                # result with range -128 to 127; returns the result.
                amax = torch.ones(4)
                scale = amax / 127.0
                zero_point = torch.zeros_like(amax, dtype=torch.int)
                # Quantize twice to test different branches
                y = torch.fake_quantize_per_channel_affine(
                    input, scale, zero_point, 1, 0, 255
                )
                return torch.fake_quantize_per_channel_affine(
                    y, scale, zero_point, 1, -128, 127
                )

        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerChannelModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(13)
    # RuntimeError: Can't redefine method:
    # forward on class: __torch__.torch.nn.modules.linear.Linear
    @skipScriptTest()
    def test_fake_quantize_activation(self):
        # [FUNCTION DOC] Exports a Linear layer prepared for quantization-aware training with
        # the default fake quantizers, observers disabled after calibration and evaluation mode,
        # using fixed weights, bias and inputs to check overflow handling of the restricted
        # activation range. Skipped under scripting; requires opset 13 or newer.
        from torch.ao import quantization

        m = torch.nn.Linear(1, 1)
        m.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        quantization.prepare_qat(m.train(), inplace=True)
        m.apply(quantization.enable_observer)
        m.apply(quantization.enable_fake_quant)
        for module in m.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()

        m.apply(quantization.disable_observer)
        m.eval()

        # Fake quantize activation is a special case, as it restricts quantized range to be (0, 127),
        # while standard 8bit quantization range is (-128, 127) or (0, 255).
        # Set fixed weight, bias and inputs to test if ONNX handles the overflow correctly.
        m.weight = torch.nn.Parameter(torch.tensor([[1.0], [1.0], [1.0]]))
        m.bias = torch.nn.Parameter(torch.tensor([0.0]))
        x = torch.tensor([[150.0], [127.0], [-5.0]])
        self.run_test(m, x)

    def test_batchnorm_training(self):
        # [FUNCTION DOC] Exports a convolutional model with batch norm layers in training mode:
        # once with the training export mode and once with the preserve mode after calling
        # train. Uses relaxed tolerances.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of batch norm and convolution layers alternating, with
            # affine and non-affine norms.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three batch norm layers and two convolutions.
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies bn1, cv1, bn2, cv2, bn3 in sequence.
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 20, 20) * 2
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_training_mode_fix_layer(self):
        # [FUNCTION DOC] Exports the same style of model but with the last batch norm layer set
        # to evaluation mode, in training export mode and then in preserve mode after calling
        # train on the model.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of alternating batch norm and convolution layers with
            # the last norm in evaluation mode.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three batch norm layers and two convolutions
                # and sets the last norm to evaluation mode.
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.eval()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in sequence.
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_batchnorm_eval_mode_train_layer(self):
        # [FUNCTION DOC] Exports a model with the last batch norm layer left in training mode:
        # in evaluation export mode, then in preserve mode after calling eval on the model.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of alternating batch norm and convolution layers with
            # the last norm in training mode.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three batch norm layers and two convolutions
                # and sets the last norm to training mode.
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.train()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in sequence.
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training(self):
        # [FUNCTION DOC] Exports instance norm layers alternating with convolutions:
        # once in the training export mode and once in preserve mode after calling
        # train. Uses relaxed tolerances.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of alternating instance norm and convolution layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three instance norm layers
                # and two convolutions.
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in sequence.
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_training_mode_fix_layer(self):
        # [FUNCTION DOC] Exports instance norm and convolution layers with the last
        # instance norm in evaluation mode, in training export mode and then preserve mode
        # after calling train.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of alternating instance norm and convolution layers
            # with the last norm in evaluation mode.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three instance norm layers and two convolutions
                # and sets the last norm to evaluation mode.
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)
                self.in3.eval()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in sequence.
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.train()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_instancenorm_eval_mode_train_layer(self):
        # [FUNCTION DOC] Exports instance norm and convolution layers with eight channels
        # and the last norm in training mode: in evaluation export mode, then preserve mode
        # after calling eval.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of alternating instance norm and convolution layers
            # with the last norm in training mode.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three instance norm layers and two convolutions
                # and sets the last norm to training mode.
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(8, affine=True)
                self.cv1 = torch.nn.Conv2d(8, 8, 10)
                self.in2 = torch.nn.InstanceNorm2d(8, affine=False)
                self.cv2 = torch.nn.Conv2d(8, 8, 10)
                self.in3 = torch.nn.InstanceNorm2d(8, affine=True)
                self.in3.train()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in sequence.
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x

        x = torch.randn(10, 8, 128, 128)
        model_export = MyModule()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.EVAL,
            rtol=1e-3,
            atol=1e-5,
        )
        model_export.eval()
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.PRESERVE,
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training(self):
        # [FUNCTION DOC] Exports dropout in training mode twice with an in-memory buffer,
        # runs each with ONNX Runtime, and checks the result differs from the input (some
        # elements dropped). Raises AssertionError if the output equals the input. Requires
        # opset 12 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping Dropout with probability 0.4.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dropout submodule.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dropout of the input.
                dropout = self.dropout(x)
                return dropout

        model = MyModule()
        x = torch.randn(10)
        model.train()

        model_onnx = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        if torch.all(torch.eq(x, torch.from_numpy(ort_outs[0]))):
            raise AssertionError("Outputs should not match input after dropout")

        output = model(x)
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_outs = verification._run_onnx(ort_sess, (x,))
        if torch.all(torch.eq(x, torch.from_numpy(ort_outs[0]))):
            raise AssertionError("Outputs should not match input after dropout")

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training_zero(self):
        # [FUNCTION DOC] Exports dropout with probability 0.5 in training mode and compares the
        # fraction of non-zero outputs between PyTorch and ORT within 1 percent tolerance,
        # twice. Requires opset 12 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping Dropout with probability 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dropout submodule.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dropout of the input.
                dropout = self.dropout(x)
                return dropout

        model = MyModule()

        # ensure there are no zeros in the input
        x = torch.randn(10, 3, 128, 128)
        y = x.numpy()
        y_mask = np.where(y == 0, 1, y)
        input = torch.from_numpy(y_mask)
        nb_elements = torch.numel(input)

        model.train()
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))

        y = model(input)
        output = y.cpu().numpy()
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)

        y = model(input)
        output = y.cpu().numpy()
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            model_onnx,
            opset_version=self.opset_version,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)

    def test_conv_bn(self):
        # [FUNCTION DOC] Exports a convolution followed by batch norm in evaluation and
        # training export modes.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 to 16 channel convolution with bias and an
            # affine batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and batch norm.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the batch norm of the convolved input.
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model_export = MyModule()
        x = torch.randn(10, 3, 128, 128)
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )

    def test_multiple_conv_bn(self):
        # [FUNCTION DOC] Exports a small ResNet-like stack of convolutions, batch norms,
        # ReLU and max pool, with a shared second batch norm reused, in training and
        # evaluation export modes.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three bias-free convolutions, two batch norms,
            # in-place ReLU and max pooling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                self.conv2 = torch.nn.Conv2d(
                    64, 2, kernel_size=1, stride=1, padding=0, bias=False
                )
                self.conv3 = torch.nn.Conv2d(
                    2, 2, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(64)
                self.bn2 = torch.nn.BatchNorm2d(2)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1, bn, relu, maxpool, conv2, bn2, relu,
                # conv3, bn2 again and relu; returns the result.
                x = self.conv1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        model_export = MyModule()
        x = torch.randn(2, 3, 224, 224)
        self.run_test(
            model_export,
            (x,),
            training=torch.onnx.TrainingMode.TRAINING,
            rtol=1e-3,
            atol=1e-5,
        )
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nms(self):
        # [FUNCTION DOC] Exports torchvision nms with IoU threshold 0.5 on 100 random boxes.
        # Requires opset 11 or newer.
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling torchvision nms with threshold 0.5.
            def forward(self, boxes, scores):
                # [FUNCTION DOC] HELPER: Returns the indices kept by non-maximum suppression.
                return torchvision.ops.nms(boxes, scores, 0.5)

        self.run_test(Module(), (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_batched_nms(self):
        # [FUNCTION DOC] Exports torchvision batched_nms with threshold 0.5 on 100 boxes with
        # five class indices. Requires opset 11 or newer.
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)
        idxs = torch.randint(0, 5, size=(num_boxes,))

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling torchvision batched_nms with threshold 0.5.
            def forward(self, boxes, scores, idxs):
                # [FUNCTION DOC] HELPER: Returns the indices kept by
                # per-class non-maximum suppression.
                return torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)

        self.run_test(Module(), (boxes, scores, idxs))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_clip_boxes_to_image(self):
        # [FUNCTION DOC] Exports clip_boxes_to_image with the image size derived from a tensor's
        # shape, dynamic axes for the size input and additional inputs with two sizes. Skipped
        # under scripting; requires opset 11 or newer.
        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)

        size_2 = torch.randn(300, 400)

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module clipping boxes to the height and width given by the
            # shape of a tensor.
            def forward(self, boxes, size):
                # [FUNCTION DOC] HELPER: Builds a size pair from the tensor shape and returns
                # the clipped boxes.
                shape = (size.shape[0], size.shape[1])
                return torchvision.ops.boxes.clip_boxes_to_image(boxes, shape)

        self.run_test(
            Module(),
            (boxes, size),
            input_names=["boxes", "size"],
            dynamic_axes={"size": [0, 1]},
            additional_test_inputs=[(boxes, size), (boxes, size_2)],
        )

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_align(self):
        # [FUNCTION DOC] Exports torchvision RoIAlign with 5 x 5 output, scale 1.0 and sampling
        # ratio 2 on one box. Skipped under scripting; requires opset 11 or newer.
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = torchvision.ops.RoIAlign((5, 5), 1.0, 2)
        self.run_test(model, (x, single_roi))

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_roi_align_aligned(self):
        # [FUNCTION DOC] Exports RoIAlign with the aligned option for four configurations of
        # output size, spatial scale and sampling ratio. Skipped under scripting; requires opset
        # 16 or newer.
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        model1 = torchvision.ops.RoIAlign((5, 5), 1.0, 2, aligned=True)
        self.run_test(model1, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model2 = torchvision.ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model3 = torchvision.ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model4 = torchvision.ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    @skipScriptTest(
        reason="Conditioning on input type via prim::isinstance unsupported in ONNX"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_pool(self):
        # [FUNCTION DOC] Exports torchvision RoIPool with a 5 x 5 output and scale 2.0 on one
        # box. Skipped under scripting; requires opset 11 or newer.
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = torchvision.ops.RoIPool((pool_h, pool_w), 2.0)
        self.run_test(model, (x, rois))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_resize_images(self):
        # [FUNCTION DOC] Exports the resize step of a GeneralizedRCNN transform with dynamic
        # axes and additional inputs of different sizes, returning the resized images. Requires
        # opset 11 or newer.
        class TransformModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running only the resize of a GeneralizedRCNN transform.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the transform with the test helper.
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                # [FUNCTION DOC] HELPER: Returns the first element of the transform resize
                # with no targets.
                return self.transform.resize(images, None)[0]

        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_test(
            TransformModule(),
            (input,),
            input_names=["input1"],
            dynamic_axes={"input1": [0, 1, 2]},
            additional_test_inputs=[(input,), (input_test,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_transform_images(self):
        # [FUNCTION DOC] Exports the full GeneralizedRCNN transform on a list of two images and
        # checks with additional inputs. Skipped under scripting; requires opset 11 or newer.
        class TransformModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running the whole GeneralizedRCNN transform and
            # returning the batched image tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the transform with the test helper.
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: list[Tensor]):
                # [FUNCTION DOC] HELPER: Returns the batched tensors of the transformed images.
                return self.transform(images)[0].tensors

        input = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        input_test = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        self.run_test(
            TransformModule(),
            (input,),
            additional_test_inputs=[(input,), (input_test,)],
        )

    def get_features(self, images):
        # [FUNCTION DOC] HELPER: Builds an ordered mapping of five random feature maps (names 0
        # to 4) with 256 channels and spatial sizes the image size divided by 4, 8, 16, 32 and
        # 64. Returns the mapping.
        s0, s1 = images.shape[-2:]
        features = [
            ("0", torch.rand(2, 256, s0 // 4, s1 // 4)),
            ("1", torch.rand(2, 256, s0 // 8, s1 // 8)),
            ("2", torch.rand(2, 256, s0 // 16, s1 // 16)),
            ("3", torch.rand(2, 256, s0 // 32, s1 // 32)),
            ("4", torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_rpn(self):
        # [FUNCTION DOC] Exports a torchvision region proposal network on random images and
        # feature maps. Skipped under scripting; requires opset 11 or newer.
        class RPNModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping the test region proposal network.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the network with the test helper.
                super().__init__()
                self.rpn = _init_test_rpn()

            def forward(self, images, features: dict[str, Tensor]):
                # [FUNCTION DOC] HELPER: Builds an image list from the images with their width
                # and height, runs the network on it and the features and returns its outputs.
                images_m = torchvision.models.detection.image_list.ImageList(
                    images, [(i.shape[-1], i.shape[-2]) for i in images]
                )
                return self.rpn(images_m, features)

        images = torch.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)

        model = RPNModule()
        model.eval()
        model(images, features)
        self.run_test(
            model,
            (images, features),
            input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
            dynamic_axes={
                "input1": [0, 1, 2, 3],
                "input2": [0, 1, 2, 3],
                "input3": [0, 1, 2, 3],
                "input4": [0, 1, 2, 3],
                "input5": [0, 1, 2, 3],
                "input6": [0, 1, 2, 3],
            },
            additional_test_inputs=[(images, features), (images2, test_features)],
            # dict_check=False,
        )

    @skipIfUnsupportedMaxOpsetVersion(15)  # TODO: Opset 16 RoiAlign result mismatch
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_multi_scale_roi_align(self):
        # [FUNCTION DOC] Exports torchvision MultiScaleRoIAlign over two feature maps with
        # a fixed image size, with an additional input set. Only runs for opset 15 or older
        # because of an opset 16 result mismatch; skipped under scripting; requires opset
        # 11 or newer.
        class TransformModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping MultiScaleRoIAlign with two feature names,
            # output size 3 and sampling ratio 2, and a fixed image size 512 x 512.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the pooler and stores the image sizes list.
                super().__init__()
                self.model = torchvision.ops.MultiScaleRoIAlign(
                    ["feat1", "feat2"], 3, 2
                )
                self.image_sizes = [(512, 512)]

            def forward(self, input: dict[str, Tensor], boxes: list[Tensor]) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the pooled features for the feature mapping and
                # the list of boxes.
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i["feat1"] = torch.rand(1, 5, 64, 64)
        i["feat2"] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1["feat1"] = torch.rand(1, 5, 64, 64)
        i1["feat2"] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_test(
            TransformModule(),
            (
                i,
                [boxes],
            ),
            additional_test_inputs=[
                (
                    i,
                    [boxes],
                ),
                (
                    i1,
                    [boxes1],
                ),
            ],
        )

    def test_set_(self):
        # [FUNCTION DOC] Exports a module that replaces the storage of x with y through
        # set_, with only the second input remaining; then also with dynamic axes and an
        # additional input pair.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling set_ on x with y and returning x.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Sets x to y and returns x (mutated in place).
                x.set_(y)
                return x

        x = torch.ones(2, 3)
        y = torch.randn(4, 6)
        self.run_test(M(), (x, y), remained_onnx_input_idx=[1])

        y2 = torch.randn(5, 2)
        self.run_test(
            M(),
            (x, y),
            remained_onnx_input_idx=[1],
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
            additional_test_inputs=[(y, y2)],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules(self):
        # [FUNCTION DOC] Exports nested modules that reassign tensor attributes and constants in
        # forward: embedding-like weights computed by static helpers, moved to the dtype and
        # device of a buffer and scaled by a constant. Run with dynamic axes and with no
        # remaining inputs. Requires opset 9 or newer.
        class InnerModule2(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module holding sinusoid-like weights, a buffer and a
            # constant that forward reassigns.
            def __init__(self, embedding_dim):
                # [FUNCTION DOC] HELPER: Builds the weights with the static helper, creates a
                # one-element float buffer and sets the constant to 2.
                super().__init__()
                self.weights = InnerModule2.get_embedding(embedding_dim)
                self._float_tensor = torch.nn.Buffer(torch.FloatTensor(1))
                self.const = 2

            @staticmethod
            def get_embedding(embedding_dim: int):
                # [FUNCTION DOC] HELPER: Static helper computing exponentials of a half-range
                # arange scaled by a negative factor derived from the embedding dimension;
                # returns the vector.
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            def forward(self, input, incremental_state: Tensor | None = None):
                # [FUNCTION DOC] HELPER: Sets the constant to 3, converts and scales the
                # weights, and then either returns one row expanded when an incremental state is
                # given or returns the weights selected by repeated index 1, reshaped to batch
                # by sequence by features.
                # KNOWN ISSUE: the guarded branch for missing weights uses an embedding_dim
                # attribute that this class never defines; it is not reached because the weights
                # are always set.
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 3
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.weights = self.weights.to(self._float_tensor)
                self.weights = self.weights * self.const
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                return self.weights.index_select(
                    0, torch.ones((bsz * seq_len), dtype=torch.int64)
                ).view(bsz, seq_len, -1)

        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module holding its own weights and the inner module.
            def __init__(self, embedding_dim):
                # [FUNCTION DOC] HELPER: Builds weights with the static helper and creates the
                # inner module with embedding dimension 8.
                super().__init__()
                self.weights = InnerModule.get_embedding(embedding_dim)
                self.module = InnerModule2(embedding_dim=8)

            @staticmethod
            def get_embedding(embedding_dim: int):
                # [FUNCTION DOC] HELPER: Static helper computing exponentials of a half-range
                # arange scaled by a negative factor derived from the embedding dimension;
                # returns the vector.
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the inner module output plus
                # the stored weights.
                return self.module(x) + self.weights

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Top-level module wrapping the outer module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the wrapped module with embedding dimension 8.
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the wrapped module output.
                return self.module(x)

        x = torch.randn(3, 256)
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules_2(self):
        # [FUNCTION DOC] Same style as the previous test with a single inner module that
        # recomputes its weights and sets a float constant in forward. Run with dynamic axes and
        # with no remaining inputs. Requires opset 9 or newer.
        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that recomputes weights in forward and scales selected
            # rows by a constant it reassigns.
            def __init__(self, embedding_dim):
                # [FUNCTION DOC] HELPER: Stores the embedding dimension, the constant 2.5, the
                # weights and a one-element float buffer.
                super().__init__()
                self.embedding_dim = embedding_dim
                self.const = 2.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                self._float_tensor = torch.nn.Buffer(torch.FloatTensor(1))

            @staticmethod
            def get_embedding(embedding_dim: int):
                # [FUNCTION DOC] HELPER: Static helper computing exponentials of a half-range
                # arange scaled by a negative factor derived from the embedding dimension;
                # returns the vector.
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(
                    torch.arange((embedding_dim // 2), dtype=torch.float) * -emb
                )
                return emb

            def forward(self, input, incremental_state: Tensor | None = None):
                # [FUNCTION DOC] HELPER: Sets the constant to 1.5, recomputes the weights,
                # selects rows by repeated index 1, reshapes and multiplies by the constant.
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 1.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                return (
                    self.weights.index_select(
                        0, torch.ones((bsz * seq_len), dtype=torch.int64)
                    ).view(bsz, seq_len, -1)
                ) * self.const

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Top-level module wrapping the inner module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the inner module with embedding dimension 8.
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the inner module output.
                return self.module(x)

        x = torch.randn(3, 256)
        self.run_test(Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    def test_set_attr(self):
        # [FUNCTION DOC] Exports a scripted module that assigns tensor attributes on a
        # convolution in forward (replacing the weight with the input weight, then softmax, then
        # doubling it) and branches on a boolean attribute. Checked against ORT.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution and a boolean attribute
            # that forward mutates.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Conv1d (3 to 10 channels, kernel 2) and sets
                # the flag false.
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 10, 2)
                self.b = False

            def forward(self, box_regression, weight):
                # [FUNCTION DOC] HELPER: Sets the flag, replaces the convolution weight with the
                # given tensor, replaces it with its softmax along dim 0 and then with twice
                # that; adds or subtracts it from box_regression depending on the flag.
                self.b = True
                self.conv.weight = weight
                w = torch.softmax(self.conv.weight, dim=0)
                self.conv.weight = w + w
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight

        model = torch.jit.script(MyModule())
        weight = torch.ones(3, 2)
        box_regression = torch.randn(3, 2)
        self.run_test(model, (box_regression, weight))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_2(self):
        # [FUNCTION DOC] Exports a scripted module where a helper method sets a convolution bias
        # from the input plus the existing bias. Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution whose bias is replaced by a zeros
            # parameter and then updated.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Conv1d (10 to 3 channels, kernel 3) and
                # replaces its bias with a zeros parameter of shape 3 x 10 x 3.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                # [FUNCTION DOC] HELPER: If the bias exists, sets it to anchors plus the old
                # bias; otherwise creates a random weight and sets the bias from it.
                if self.conv.bias is not None:
                    b = self.conv.bias
                    assert b is not None  # noqa: S101
                    self.conv.bias = anchors + b
                elif self.conv.weight is not None:
                    self.conv.weight = torch.randn(3, 10)
                    self.conv.bias = self.conv.weight[:]

            def forward(self, anchors) -> Tensor | None:
                # [FUNCTION DOC] HELPER: Calls the helper with the anchors and returns
                # the convolution bias.
                self.set_cell_anchors(anchors)
                return self.conv.bias

        model = torch.jit.script(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_3(self):
        # [FUNCTION DOC] Exports a scripted module whose helper replaces convolution weight and
        # bias and also writes into an input box tensor in place; returns the weight and boxes.
        # Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution whose weight and bias
            # are zeros parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and replaces its weight and
                # bias by zeros parameters.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                # [FUNCTION DOC] HELPER: Sets the weight to ones; if the bias exists replaces
                # it with random values, sets the weight to anchors plus ones and zeroes the
                # boxes in place.
                self.conv.weight = torch.ones(3, 10)
                if self.conv.bias is not None:
                    self.conv.bias = torch.randn(3, 10, 3)
                    self.conv.weight = anchors + self.conv.weight
                    boxes[:] = torch.zeros(2, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                # [FUNCTION DOC] HELPER: Creates ones boxes, calls the helper and returns the
                # weight and boxes if a bias exists, otherwise the anchors and boxes.
                boxes = torch.ones(2, 2, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = torch.jit.script(MyModule())
        anchors = torch.rand(3, 10)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_4(self):
        # [FUNCTION DOC] Exports a scripted module with a helper that sets a convolution weight
        # to zeros and updates the bias, returning the bias and the feature maps through a list.
        # Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution whose bias is a zeros parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and replaces its bias by
                # a zeros parameter.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                # [FUNCTION DOC] HELPER: Sets the weight to zeros; sets the bias to anchors plus
                # the old bias when a bias exists, otherwise to ones.
                self.conv.weight = torch.zeros(10, 3)
                if self.conv.bias is not None:
                    w = self.conv.bias
                    assert w is not None  # noqa: S101
                    self.conv.bias = anchors + w
                else:
                    self.conv.bias = torch.ones(3, 10, 3)

            def forward(self, feature_maps, anchors) -> tuple[Tensor, Tensor]:
                # [FUNCTION DOC] HELPER: Calls the helper, collects the bias (if present) and
                # the feature maps in a list and returns the first two entries.
                self.set_cell_anchors(anchors)
                result = []
                if self.conv.bias is not None:
                    a = self.conv.bias
                    assert a is not None  # noqa: S101
                    result += [a]
                result += [feature_maps]
                return result[0], result[1]

        model = torch.jit.script(MyModule())
        x = torch.rand(5, 11, 30)
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (x, anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_5(self):
        # [FUNCTION DOC] Exports a scripted module where a helper updates a convolution
        # weight in nested loops and the bias in a loop; returns weight and bias. Requires
        # opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution whose bias is a zeros parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and replaces its bias by
                # a zeros parameter.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                # [FUNCTION DOC] HELPER: Sets the weight to an arange of ten; for ten
                # iterations, with an inner ten-iteration loop at index 3, adds aranges to the
                # weight and adds anchors to the bias when a bias exists.
                self.conv.weight = torch.arange(10)
                for i in range(10):
                    if i == 3:
                        for _ in range(10):
                            w = self.conv.weight
                            self.conv.weight = torch.arange(10) + w

                    self.conv.weight = self.conv.weight + torch.arange(10)
                    # NOTE: `is not None` and `assert` is for passing torchscript.
                    if self.conv.bias is not None:
                        a = self.conv.bias
                        assert a is not None  # noqa: S101
                        self.conv.bias = anchors + a

            def forward(self, anchors):
                # [FUNCTION DOC] HELPER: Calls the helper and returns the weight and bias.
                self.set_cell_anchors(anchors)
                return self.conv.weight, self.conv.bias

        model = torch.jit.script(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_in_loop(self):
        # [FUNCTION DOC] Exports a scripted module that sets attributes inside nested loops and
        # mutates an input boxes tensor in place. Requires opset 11 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution whose weight and bias
            # are zeros parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and replaces weight and bias
                # by zeros parameters.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                # [FUNCTION DOC] HELPER: Sets a random weight; in nested loops sets a
                # random bias, sets the weight to anchors times the outer index and adds
                # ones to a box row.
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        boxes[j] += torch.ones(3, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                # [FUNCTION DOC] HELPER: Creates ones boxes, calls the helper and returns the
                # weight and boxes if a bias exists, else the anchors and boxes.
                boxes = torch.ones(10, 3, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = torch.jit.script(MyModule())
        anchors = torch.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_set_attr_in_loop_with_list(self):
        # [FUNCTION DOC] Exports a scripted module that sets attributes and appends to a list
        # attribute inside nested loops. Requires opset 13 or newer.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution and a list attribute of tensors that
            # forward resets and fills.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution with zero parameters and a
                # one-element placeholder list required by TorchScript.
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))
                self.boxes: list[Tensor] = [
                    torch.ones(1)
                ]  # Workaround placeholder for TorchScript

            def set_cell_anchors(self, anchors):
                # [FUNCTION DOC] HELPER: Sets a random weight; in nested loops sets a random
                # bias, sets the weight to anchors times the outer index and appends a ones
                # matrix to the list.
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for _ in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        self.boxes.append(torch.ones(3, 3))

            def forward(self, anchors) -> tuple[Tensor, list[Tensor]]:
                # [FUNCTION DOC] HELPER: Clears the list, calls the helper and returns the
                # weight and list if a bias exists, otherwise the anchors and list.
                self.boxes = []
                self.set_cell_anchors(anchors)
                if self.conv.bias is not None:
                    return self.conv.weight, self.boxes
                return anchors, self.boxes

        model = torch.jit.script(MyModule())
        anchors = torch.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if(self):
        # [FUNCTION DOC] Exports a scripted function that conditionally writes a slice of state
        # tensors depending on whether a previous state is empty, with dynamic axes and with no
        # remaining inputs. Requires opset 11 or newer.
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function creating two zeros state tensors of shape
            # 2 x batch x hidden x height x width. If the previous state is empty, fills the
            # first from zeros plus itself and the second with scaled ones then scaled zeros;
            # otherwise sets the first to four times ones. Returns both.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    + state[:]
                )
                state_copy[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state_copy[:] = (
                    torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
            else:
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            return state, state_copy

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                # [FUNCTION DOC] HELPER: Returns the two state tensors from the scripted
                # function for the input and previous state.
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_2(self):
        # [FUNCTION DOC] Exports a scripted function with a conditional chain on the size of the
        # previous state (empty, one, two) writing slices of state tensors, then run with
        # dynamic axes plus an additional input, with only the second input remaining, and with
        # no remaining inputs. Requires opset 11 or newer.
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function creating two zeros state tensors. If the
            # previous state is empty, loops twice assigning ones times the loop index into
            # slices of both; if it has one entry, adds the previous state into the first; if
            # two, sets the first to four times ones. Returns both states.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                for i in range(2):
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
                    state_copy[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * i
                    )
            elif prev_state.size(0) == 1:
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            return state, state_copy

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                # [FUNCTION DOC] HELPER: Returns the two state tensors from
                # the scripted function.
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        random_state = torch.rand((1, 1, 10, 30, 30))
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],
            dynamic_axes={"data": [0, 1, 2], "state": [0, 1, 2, 3, 4]},
            additional_test_inputs=[(random_data, random_state)],
        )
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["data", "state"],
            dynamic_axes={"state": [0, 1, 2, 3, 4]},
            additional_test_inputs=[(random_data, random_state)],
            remained_onnx_input_idx=[1],
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_3(self):
        # [FUNCTION DOC] Exports a scripted function that multiplies a zero state by 3 and
        # overwrites slices in an inner conditional on an empty previous state, else adds 2;
        # with dynamic axes and with no remaining inputs. Requires opset 11 or newer.
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted function creating a zeros state; if the previous
            # state has fewer than two entries multiplies the state by 3, then if the previous
            # state is empty sets all of it to 3 else adds 2. Returns the state.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                state = state * 3
                if prev_state.size(0) == 0:
                    state[:] = (
                        torch.ones(
                            batch_size, hidden_size, spatial_size_0, spatial_size_1
                        )
                        * 3
                    )
                else:
                    state = state + 2

            return state

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                # [FUNCTION DOC] HELPER: Returns the state from the scripted function.
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_4(self):
        # [FUNCTION DOC] Exports a scripted function that, on an empty previous state,
        # alternates additions and slice overwrites; otherwise adds 2. With dynamic axes and
        # with no remaining inputs. Requires opset 11 or newer.
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted function creating a zeros state; when the previous
            # state is empty adds 3, overwrites with 3, adds 3 and overwrites with 4; otherwise
            # adds 2. Returns the state.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                # [FUNCTION DOC] HELPER: Returns the state from the scripted function.
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_5(self):
        # [FUNCTION DOC] Exports a scripted function that overwrites slices of a state and keeps
        # an alias to the original, returning both. With dynamic axes and with no remaining
        # inputs. Requires opset 11 or newer.
        @torch.jit.script
        def check_init(
            input_data: Tensor, hidden_size: int, prev_state: Tensor
        ) -> tuple[Tensor, Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function creating a zeros state and an alias; when
            # the previous state is empty overwrites the state with 3, rebinds it to state plus
            # 3 and overwrites the new binding with 4; otherwise rebinds to state plus 2.
            # Returns the state and the alias.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_ref = state
            if prev_state.size(0) == 0:
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
                state = state + 3
                state[:] = (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 4
                )
            else:
                state = state + 2
            return state, state_ref

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                # [FUNCTION DOC] HELPER: Returns the state and the alias from
                # the scripted function.
                prev_state, state_ref = check_init(
                    input_data, self.hidden_size, prev_state
                )
                return prev_state, state_ref

        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(
            model,
            (random_data, empty_tensor),
            input_names=["random_data", "empty_tensor"],
            dynamic_axes={"random_data": [0, 1, 2, 3], "empty_tensor": [0, 1, 2, 3, 4]},
        )
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append_in_block(self):
        # [FUNCTION DOC] Exports a scripted loop appending matmul results to a list. Requires
        # opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of per-row matrix products built
            # in a loop.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends the matmul of each row of x with y and
                # returns the list.
                res = []
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_in_nested_block(self):
        # [FUNCTION DOC] Exports scripted nested loops appending matmul results to a list.
        # Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a list built in nested loops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends the matmul of each entry over dims 0 and 1 with
                # y and returns the list.
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res.append(torch.matmul(x[i][j], y))
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_in_block(self):
        # [FUNCTION DOC] Exports a scripted module that builds a list, pops all items in a loop,
        # then appends and pops again. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module exercising list append and pop in loops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Fills a list with matmul results, pops it empty in a
                # loop, then appends and pops each result, finally returning the outcome of
                # appending the last popped element.
                # KNOWN ISSUE: the returned value is the result of the append call, which is
                # None, so the model actually returns nothing useful.
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for _ in range(x.size(0)):
                    elem = res.pop()
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    elem = res.pop()
                return res.append(elem)

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_in_block(self):
        # [FUNCTION DOC] Exports a scripted module that builds a list, deletes the first item
        # repeatedly, then appends and deletes again. Requires opset 13 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module exercising list append and delete in loops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Fills a list with matmul results, deletes the first
                # item in a loop, then appends and deletes each time, finally returning the
                # outcome of appending the first element's product.
                # KNOWN ISSUE: the returned value is the result of the append call, which is
                # None, so the model actually returns nothing useful.
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for _ in range(x.size(0)):
                    del res[0]
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    del res[0]
                return res.append(elem)

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_unpack(self):
        # [FUNCTION DOC] Exports a scripted loop that builds a list of three products and
        # unpacks it into three values, returning the first two. Requires opset 11 or newer.
        class ListModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building a three-element list and unpacking it.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Appends the matmul of each of three rows with y,
                # unpacks the list into three names and returns the first two.
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                a, b, c = res
                return a, b

        model = torch.jit.script(ListModel())
        x = torch.randn(3, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_inplace_ops(self):
        # [FUNCTION DOC] Exports a scripted function applying in-place add and divide on a state
        # slice, conditionally and in a loop, with dynamic axes and with no remaining inputs.
        # Requires opset 11 or newer.
        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted function creating a zeros state; if the batch size
            # is 1 adds twos and divides by threes in a slice; then for each batch index adds
            # ones and divides by ones times the index into the slice. Returns the state.
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if input_data.size(0) == 1:
                state[1] += (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 2
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * 3
                )
            for i in range(input_data.size(0)):
                state[1] += torch.ones(
                    batch_size, hidden_size, spatial_size_0, spatial_size_1
                )
                state[1] /= (
                    torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                    * i
                )
            return state

        class Example(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a hidden size and calling the
            # scripted state function.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Stores the hidden size.
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data):
                # [FUNCTION DOC] HELPER: Returns the state from the scripted function
                # for the input.
                state = check_init(input_data, self.hidden_size)
                return state

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        self.run_test(
            model,
            (random_data),
            input_names=["random_data"],
            dynamic_axes={"random_data": [0, 1, 2, 3]},
        )
        self.run_test(model, (random_data), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_input_mask_model(self):
        # [FUNCTION DOC] Exports in-place assignment of zeros into rows chosen by boolean masks
        # computed from integer inputs: one mask over rows, then two masks combined as row and
        # column selectors. Requires opset 11 or newer.
        class InputMaskModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a zero-initialized bias parameter that zeroes
            # masked rows of the input and adds the bias.
            def __init__(self, output_size):
                # [FUNCTION DOC] HELPER: Creates the bias parameter of the given size and zeroes
                # it without tracking gradients.
                super().__init__()
                self.bias = torch.nn.Parameter(
                    torch.empty(output_size, dtype=torch.float)
                )
                with torch.no_grad():
                    self.bias.zero_()

            def forward(self, model_input, y):
                # [FUNCTION DOC] HELPER: Builds a mask for values at most 0 or above 25, zeroes
                # those rows of y in place and returns y plus the bias.
                input_mask = (model_input <= 0) | (model_input > 25)
                y[input_mask, :] = 0.0
                output = y + self.bias
                return output

        output_size = 4
        m = InputMaskModel(output_size)
        x = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        y = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float,
        )
        self.run_test(m, (x, y))

        class InputMaskModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module zeroing y entries selected by two boolean masks used as
            # row and column indices.
            def __init__(self, output_size):
                # [FUNCTION DOC] HELPER: Accepts the output size without using it.
                super().__init__()

            def forward(self, model_input_1, model_input_2, y):
                # [FUNCTION DOC] HELPER: Builds two masks from the two inputs, zeroes the
                # selected entries of y in place and returns y.
                input_mask_1 = (model_input_1 <= 0) | (model_input_1 > 25)
                input_mask_2 = (model_input_2 < 1) | (model_input_2 >= 12)
                y[input_mask_1, input_mask_2] = 0.0
                return y

        output_size = 4
        m = InputMaskModel(output_size)
        x1 = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        x2 = torch.tensor([0, 3, 12, 15], dtype=torch.int64)
        y = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float,
        )
        self.run_test(m, (x1, x2, y))

    @skipScriptTest()
    def test_unsafe_chunk(self):
        # [FUNCTION DOC] Exports unsafe_chunk into 3 parts along dim 1. Skipped under scripting.
        class ChunkModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting its input into three chunks along dim 1 with
            # the unsafe variant.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the unsafe chunks of the input.
                return torch.unsafe_chunk(x, 3, dim=1)

        model = ChunkModel()
        model.eval()
        x = torch.randn(1, 18)
        self.run_test(model, x, input_names=["x"])

    def test_symbolic_shape_inference(self):
        # [FUNCTION DOC] Exercises symbolic shape inference through shape, reshape, transpose
        # and gather operators (constant-of-shape and tile are covered by other tests), using
        # dynamic axes and several modules including a reshape built from list construction.
        # ConstantOfShape is tested in test_embedding_bag
        # Tile is tested in test_repeat
        # test Shape, Reshape, Transpose, Gather
        class ShapeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping y to the first three dims of x plus an
            # inferred dim, then transposing.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds a shape from x's first three dimension sizes and
                # an inferred one, reshapes y and returns it transposed on dims 1 and 2.
                shape = x.size()[:3] + (-1,)  # shape [4], ("batch", 3, 4, -1)
                y = y.reshape(shape)  # batch, 3, 4, 10/batch
                return y.transpose(1, 2)

        model = ShapeModel()
        model.eval()
        x = torch.ones(2, 3, 4, 5)
        y = torch.ones(3, 4, 5, 2)
        self.run_test(
            model,
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        )
        self.run_test(model, (x, y), remained_onnx_input_idx=[1])

        class ViewModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening its input with view of minus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input viewed as a one-dimensional tensor.
                return x.view(-1)

        model = ViewModel()
        model.eval()
        x = torch.tensor(2.0)
        self.run_test(model, (x,))

        # test prim::ListConstruct for Reshape input 1
        class ViewModel_2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reshaping with a shape made of several shape entries
            # via list construction.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the batch, channel, height and width sizes (dims
                # 0, 2, 3, 4) of a 5D input, views it as N by inferred by C by H by W, permutes
                # dims to 0,3,4,1,2 and reshapes to N by inferred by C.
                N, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)

        model = ViewModel_2()
        model.eval()
        x = torch.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_arange(self):
        # [FUNCTION DOC] Exercises shape inference for the range operator: an arange whose end
        # is computed from the input shape through integer arithmetic. Run with dynamic axes,
        # with no remaining inputs, and with additional inputs of different sizes. Requires
        # opset 9 or newer.
        # test Range
        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing an arange over a frame length derived from
            # the input size.
            def forward(self, signal):
                # [FUNCTION DOC] HELPER: Computes several integer quantities from the input
                # shape (frame count and length, subframe length and sizes) and returns an
                # arange up to the computed output subframe count; most derived values besides
                # that count are unused.
                frame_step = 2
                outer_dimensions = signal.size()[:-2]
                frames, frame_length = signal.size()[-2:]

                subframe_length = signal.size()[0]
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                frame = torch.arange(0, output_subframes)
                return frame

        model = ArangeModel()
        model.eval()
        M, C, K, N = 1, 2, 3, 4
        x = torch.randint(5, (M, C, K, N))
        y = torch.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]})
        self.run_test(model, x, remained_onnx_input_idx=[])
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box(self):
        # [FUNCTION DOC] Exercises shape inference through the non-zero operator for box
        # filtering, with a static input and with dynamic axes and additional inputs. Requires
        # opset 11 or newer.
        # test NonZero
        class BoxModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the indices of boxes whose width and height
            # are at least a minimum size.
            def forward(self, boxes):
                # [FUNCTION DOC] HELPER: Computes widths and heights from the box columns and
                # returns the first index tensor of where the minimum-size condition holds.
                min_size = 1e-2
                ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                keep = (ws >= min_size) & (hs >= min_size)
                keep = torch.where(keep)[0]
                return keep

        model = BoxModel()
        model.eval()
        x = torch.ones(2, 4)
        y = torch.ones(3, 5)
        self.run_test(model, x)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[(x,), (y,)],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box_if(self):
        # [FUNCTION DOC] Exercises shape inference through a conditional that depends on the
        # number of selected boxes. Requires opset 11 or newer.
        # test If
        class BoxIfModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module selecting boxes with positive scores and doubling the
            # result if few were selected.
            def forward(self, boxes, scores):
                # [FUNCTION DOC] HELPER: Selects the boxes at indices where the scores exceed
                # zero; returns them if more than three elements, otherwise doubled.
                score_thresh = 0.0
                inds = torch.where(scores > score_thresh)[0]
                boxes_1 = boxes[inds]
                if boxes_1.numel() > 3:
                    return boxes_1
                else:
                    return boxes_1 * 2

        model = BoxIfModel()
        model.eval()
        boxes = torch.ones(2, 4)
        scores = torch.ones(1, 4)
        self.run_test(model, (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipDtypeChecking
    def test_symbolic_shape_inference_arange_2(self):
        # [FUNCTION DOC] Exercises shape inference for arange with size-derived start, end 8.5
        # and step 1.5, for int64 and double dtypes, with dynamic axes and no remaining inputs.
        # Dtype checking disabled; requires opset 11 or newer.
        # test Range
        class ArangeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing an int64 arange from a size-based start to
            # 8.5 with step 1.5.
            def forward(self, start):
                # [FUNCTION DOC] HELPER: Returns the int64 arange.
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        self.run_test(
            ArangeModel(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel(), (x,), remained_onnx_input_idx=[])

        class ArangeModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing a double arange from a size-based start to
            # 8.5 with step 1.5.
            def forward(self, start):
                # [FUNCTION DOC] HELPER: Returns the double arange.
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.double)

        x = torch.randn(2, 3, 4)
        self.run_test(
            ArangeModel2(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ArangeModel2(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_nonzero(self):
        # [FUNCTION DOC] Exercises shape inference for nonzero of ones-like and zeros-like
        # tensors on 1D and 3D inputs, with dynamic axes and no remaining inputs. Requires opset
        # 9 or newer.
        class OneLikeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the nonzero positions of a ones tensor shaped
            # like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds float ones with strided layout on cpu like the
                # input and returns their nonzero indices.
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return torch.nonzero(ones)

        x = torch.randn(2)
        self.run_test(OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(
            OneLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])

        class ZeroLikeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the nonzero positions of a zeros tensor
            # shaped like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds float zeros with strided layout on cpu like the
                # input and returns their nonzero indices.
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return torch.nonzero(zeros)

        x = torch.randn(2)
        self.run_test(ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0]})
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(
            ZeroLikeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_1(self):
        # [FUNCTION DOC] Exercises shape inference for expand of a 6 x 1 tensor to 4 x 6 x 2.
        # Requires opset 9 or newer.
        class ExpandModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module expanding the input to 4 x 6 x 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the expanded input.
                return x.expand(4, 6, 2)

        x = torch.randn(6, 1, requires_grad=True)
        self.run_test(ExpandModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_2(self):
        # [FUNCTION DOC] Exercises shape inference for a causal mask built from an arange,
        # repeat and comparison, with dynamic axes and no remaining inputs. Requires opset
        # 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module building a causal attention mask from the input's batch
            # and sequence sizes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds an arange over the sequence length, compares a
                # repeated version with a transposed-axis version and returns the mask
                # transposed on dims 0 and 1.
                input_shape = x.size()
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                return causal_mask.transpose(0, 1)

        x = torch.randn(3, 16)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(M(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_symbolic_shape_inference_slice(self):
        # [FUNCTION DOC] Exercises shape inference for slicing a position bias to its trailing
        # sequence length, with dynamic axes on both inputs and then with only the second input
        # remaining. Requires opset 10 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module cropping a position bias to the last sequence length
            # rows and transposing.
            def forward(self, x, position_bias):
                # [FUNCTION DOC] HELPER: Takes the sequence length from x, slices the position
                # bias to its last that many entries on dim 2 and returns it transposed on dims
                # 0 and 1.
                input_shape = x.size()
                batch_size, seq_length = input_shape
                position_bias = position_bias[:, :, -seq_length:, :]
                return position_bias.transpose(0, 1)

        x = torch.randn(3, 16)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(
            M(),
            (x, position_bias),
            input_names=["x", "position_bias"],
            dynamic_axes={"x": [0, 1], "position_bias": [0, 1, 2, 3]},
        )
        self.run_test(M(), (x, position_bias), remained_onnx_input_idx=[1])

    def test_symbolic_shape_inference_slice_2(self):
        # [FUNCTION DOC] Exercises shape inference for a slice with a fixed negative start on
        # dim 2 followed by a transpose.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module slicing the last two entries of dim 2 and transposing.
            def forward(self, position_bias):
                # [FUNCTION DOC] HELPER: Returns the position bias sliced to its last two
                # entries on dim 2 and transposed on dims 0 and 1.
                position_bias = position_bias[:, :, -2:, :]
                return position_bias.transpose(0, 1)

        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (position_bias,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_symbolic_shape_inference_time(self):
        # [FUNCTION DOC] Exercises shape inference through LSTM, GRU and RNN layers with
        # dynamic sequence and batch axes on the input. Skipped under scripting; requires opset
        # 9 or newer.
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        model_lstm = torch.nn.LSTM(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
        )
        self.run_test(
            model_lstm,
            (input, (h0, c0)),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1]},
        )
        model_gru = torch.nn.GRU(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        self.run_test(
            model_gru, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )
        model_rnn = torch.nn.RNN(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False
        )
        self.run_test(
            model_rnn, (input, h0), input_names=["x", "y"], dynamic_axes={"x": [0, 1]}
        )

    def test_symbolic_shape_inference_dynamic_axes(self):
        # [FUNCTION DOC] Exercises shape inference for a view with an inferred size followed by
        # a transpose, with named dynamic batch and sequence axes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module viewing the input as rows of its last
            # dimension and transposing.
            def forward(self, input_ids):
                # [FUNCTION DOC] HELPER: Reshapes to minus one by the last dimension size and
                # returns the transpose of dims 0 and 1.
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                return input_ids.transpose(0, 1)

        x = torch.randn(3, 16)
        self.run_test(
            M(),
            (x,),
            input_names=["input_ids"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_periodic(self):
        # [FUNCTION DOC] Exports hann_window with periodic enabled and a window length passed as
        # an integer input of 100, added to a vector. Requires opset 9 or newer.
        class HannWindowModule_Periodic(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a periodic Hann window of runtime length
            # to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the stored window length to zero.
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                # [FUNCTION DOC] HELPER: Stores the window length and returns the input plus a
                # periodic float Hann window of that length.
                self.window_length = window_length
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=True, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        module = HannWindowModule_Periodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_not_periodic(self):
        # [FUNCTION DOC] Exports hann_window with periodic disabled and length 100. Requires
        # opset 9 or newer.
        class HannWindowModule_NotPeriodic(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a non-periodic Hann window of runtime length
            # to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the stored window length to zero.
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                # [FUNCTION DOC] HELPER: Stores the window length and returns the input plus a
                # non-periodic float Hann window of that length.
                self.window_length = window_length
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=False, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        module = HannWindowModule_NotPeriodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_hann_window_default_values(self):
        # [FUNCTION DOC] Exports hann_window with default options, followed by relu, added to
        # the input. Also evaluates the module once eagerly. Skipped under scripting; requires
        # opset 9 or newer.
        class HannWindowModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the relu of a default Hann window to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the stored window length to zero.
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                # [FUNCTION DOC] HELPER: Stores the window length and returns the input plus the
                # relu of a default Hann window of that length.
                import torch.nn.functional as F

                self.window_length = window_length
                return torch.add(x, F.relu(torch.hann_window(self.window_length)))

        win_length = 100
        x = torch.randn(win_length, dtype=torch.float)
        module = HannWindowModule()

        output = module(x, win_length)
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_count(self):
        # [FUNCTION DOC] Exports tensordot over the last two dims of the first and first two of
        # the second (integer dim count 2) on integer tensors. Requires opset 12 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module contracting two tensors over two dimensions.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the tensordot of x and y with
                # two contracted dimensions.
                output = torch.tensordot(x, y, 2)
                return output

        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_list(self):
        # [FUNCTION DOC] Exports tensordot with explicit dimension lists including negative
        # dims, on 5D integer tensors. Requires opset 12 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module contracting two tensors over explicit dimension lists.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the tensordot of x and y over dims 1,-2,-1 of x
                # and 1,0,3 of y.
                output = torch.tensordot(x, y, ([1, -2, -1], [1, 0, 3]))
                return output

        x = torch.randint(6, (7, 4, 3, 5, 2))
        y = torch.randint(6, (5, 4, 4, 2, 6))

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dynamic_dim(self):
        # [FUNCTION DOC] Exports tensordot with dynamic axes on all dims of both inputs and an
        # additional input pair of different sizes. Requires opset 12 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module contracting two tensors over two dimensions.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the tensordot of x and y with
                # two contracted dimensions.
                output = torch.tensordot(x, y, 2)
                return output

        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))

        new_x = torch.randint(6, (8, 6, 2, 5))
        new_y = torch.randint(6, (2, 5, 3, 4))

        self.run_test(
            M(),
            (x, y),
            additional_test_inputs=[(new_x, new_y)],
            input_names=["input_x", "input_y"],
            dynamic_axes={"input_x": [0, 1, 2, 3], "input_y": [0, 1, 2, 3]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_to_device(self):
        # [FUNCTION DOC] Exports moving a tensor to the device of another, optionally also
        # converting to long. Requires opset 9 or newer.
        class M_ToDevice(torch.nn.Module):
            # [CLASS DOC] HELPER: Module moving x to the device of y and returning it with y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x on the device of y, and y.
                return x.to(y.device), y

        class M_ToDeviceDtype(torch.nn.Module):
            # [CLASS DOC] HELPER: Module moving x to the device of y as long and returning
            # it with y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x on the device of y converted to long, and y.
                return x.to(y.device, dtype=torch.long), y

        x = torch.randn(6)
        y = torch.randn(6)

        self.run_test(M_ToDevice(), (x, y))
        self.run_test(M_ToDeviceDtype(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_fill(self):
        # [FUNCTION DOC] Exports in-place fill with an integer value input, a float value input,
        # and a scalar fill of a computed tensor. Requires opset 9 or newer.
        class FillModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling its input in place with an integer argument.
            def forward(self, x, filled_value: int):
                # [FUNCTION DOC] HELPER: Returns x after filling it with the value in place.
                return x.fill_(filled_value)

        x = torch.randn((4, 5, 6))
        filled_value = 7
        self.run_test(FillModule(), (x, filled_value))

        class FillFloatModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling its input in place with a float argument.
            def forward(self, x, filled_value: float):
                # [FUNCTION DOC] HELPER: Returns x after filling it with the value in place.
                return x.fill_(filled_value)

        x = torch.randn((4, 5, 6))
        filled_value = 7.5
        self.run_test(FillFloatModule(), (x, filled_value))

        class FillScalarModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module filling a derived tensor with 2.5 and returning it
            # with the original.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes x plus 2, fills that tensor with 2.5 in place
                # and returns it with x.
                res = x + 2
                res.fill_(2.5)
                return res, x

        x = torch.ones(2, 3, 4, dtype=torch.long)
        self.run_test(FillScalarModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_normal(self):
        # [FUNCTION DOC] Exports index_add_ along dims 0, 1 and 2 of ones tensors with different
        # index permutations and updates. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding fixed updates into the input at fixed indices
            # along a stored dimension.
            def __init__(self, dim, index, updates):
                # [FUNCTION DOC] HELPER: Stores the dimension, index tensor and updates.
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the updates into x along the stored dim at the
                # indices in place and returns x.
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = torch.ones(5, 1)
        updates = torch.tensor([[1], [4], [7], [3], [2]], dtype=torch.float)
        index = torch.tensor([0, 2, 3, 1, 4])
        self.run_test(M(0, index, updates), (x,))

        x = torch.ones(1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        index = torch.tensor([0, 2, 3, 1])
        self.run_test(M(1, index, updates), (x,))

        updates = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]], dtype=torch.float
        )
        index = torch.tensor([0, 2, 1])
        self.run_test(M(2, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dim_size_differ(self):
        # [FUNCTION DOC] Exports index_add_ where the index length differs from the dimension
        # size (see the cases below). Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding fixed updates into the input at fixed indices
            # along a stored dimension.
            def __init__(self, dim, index, updates):
                # [FUNCTION DOC] HELPER: Stores the dimension, index tensor and updates.
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the updates into x along the stored dim at the
                # indices in place and returns x.
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6]]], dtype=torch.float)
        index = torch.tensor([0, 2, 1])
        self.run_test(M(1, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_in_loop(self):
        # [FUNCTION DOC] Exports index_add_ inside a loop whose trip count is a random integer
        # below 20 chosen at test time, updating along dim 1. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module repeatedly adding fixed updates into the input at fixed
            # indices along a stored dimension.
            def __init__(self, dim, index, updates, loop_count):
                # [FUNCTION DOC] HELPER: Stores the dimension, index, updates and loop count.
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates
                self.loop_count = loop_count

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Performs the in-place index add the stored number of
                # times and returns x.
                for _ in range(self.loop_count):
                    x.index_add_(self.dim, self.index, self.updates)
                return x

        x = torch.ones(1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        index = torch.tensor([0, 2, 3, 1])
        loop_count = torch.randint(20, (1,))[0].item()
        self.run_test(M(1, index, updates, loop_count), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_if(self):
        # [FUNCTION DOC] Exports a scripted module that picks one of two index tensors for
        # index_add_ by a boolean tensor input. Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two alternative index tensors, selected
            # by a condition.
            def __init__(self, dim, updates, index_true, index_false):
                # [FUNCTION DOC] HELPER: Stores the dimension, updates and both index tensors.
                super().__init__()
                self.dim = dim
                self.updates = updates
                self.index_true = index_true
                self.index_false = index_false

            def forward(self, x, cond):
                # [FUNCTION DOC] HELPER: Adds the updates in place at the true-branch index or
                # false-branch index depending on cond and returns x.
                if cond:
                    x.index_add_(self.dim, self.index_true, self.updates)
                else:
                    x.index_add_(self.dim, self.index_false, self.updates)
                return x

        x = torch.ones(1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        index_true = torch.tensor([0, 2, 3, 1])
        index_false = torch.tensor([1, 0, 2, 3])
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(
            torch.jit.script(M(1, updates, index_true, index_false)), (x, cond)
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dynamic_axes(self):
        # [FUNCTION DOC] Exports index_add_ with dynamic axes on dims 0 and 1 of the input.
        # Requires opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding fixed updates into the input at fixed indices
            # along a stored dimension.
            def __init__(self, dim, index, updates):
                # [FUNCTION DOC] HELPER: Stores the dimension, index and updates.
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the updates into x along the stored dim at the
                # indices in place and returns x.
                x.index_add_(self.dim, self.index, self.updates)
                return x

        x = torch.ones(1, 4, 3)
        updates = torch.tensor(
            [[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float
        )
        index = torch.tensor([0, 2, 3, 1])

        self.run_test(
            M(1, index, updates),
            (x,),
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1]},
        )

    def test_roll(self):
        # [FUNCTION DOC] Exports roll with list shifts and dims, repeated dims, a scalar shift
        # and dim, and negative dims on a 2 x 3 x 4 tensor.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module rolling the input by stored shifts along stored dims.
            def __init__(self, shifts, dims):
                # [FUNCTION DOC] HELPER: Stores the shifts and dims.
                super().__init__()
                self.shifts = shifts
                self.dims = dims

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input rolled by the stored shifts along
                # the stored dims.
                return torch.roll(x, self.shifts, self.dims)

        x = torch.randn(2, 3, 4)
        self.run_test(M([1, 1], [1, 0]), (x,))
        self.run_test(M([0, 1, 2], [1, 0, 2]), (x,))
        self.run_test(M(2, 1), (x,))
        self.run_test(M([-1, 3], [-2, -1]), (x,))

    def test_sum(self):
        # [FUNCTION DOC] Exports the sum of all elements with a dynamic first axis.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing all elements of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the input.
                return torch.sum(x)

        x = torch.ones(12, 3)
        self.run_test(M(), (x,), input_names=["x"], dynamic_axes={"x": [0]})

    @skipShapeChecking
    def test_sum_empty_tensor(self):
        # [FUNCTION DOC] Exports sums of an empty slice and the whole tensor for a vector, a
        # tensor with a zero dim and an empty vector. Shape checking disabled.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of an empty slice and the sum
            # of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair: the sum of the first zero entries and
                # the sum of x.
                return x[0:0].sum(), x.sum()

        x = torch.ones(12)
        self.run_test(M(), (x,))

        x = torch.ones(2, 0, 3)
        self.run_test(M(), (x,))

        x = torch.ones(0)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_broad_cast_tensors(self):
        # [FUNCTION DOC] Exports broadcast_tensors on integer and float inputs with differing
        # ranks. Requires opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the broadcast versions of two tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the pair of tensors broadcast to
                # a common shape.
                m = torch.broadcast_tensors(x, y)
                return m

        x = torch.randint(5, (1,))
        y = torch.randint(5, (5,))

        self.run_test(M(), (x, y))

        x = torch.randint(5, (4, 2, 1, 4))
        y = torch.randint(5, (2, 3, 1))

        self.run_test(M(), (x, y))

        x = torch.randn(2, 1, 4)
        y = torch.randn(5, 2, 3, 1)

        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_scaled_dot_product_attention(self):
        # [FUNCTION DOC] Exports functional scaled dot product attention with scale 1.0 on
        # random query, key and value tensors of shape batch 2, 4 heads, length 5, head dim 8.
        # Requires opset 14 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing scaled dot product attention with scale 1.0.
            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Returns the attention output for q, k and v.
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=1.0
                )

        # Parameters
        batch_size = 2  # Number of samples in the batch
        num_heads = 4  # Number of attention heads
        seq_length = 5  # Sequence length
        head_dim = 8  # Dimensionality of each head

        # Create random query, key, and value tensors
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim)

        self.run_test(M(), (q, k, v))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal(self):
        # [FUNCTION DOC] Exports sampling from a Normal distribution built from tensor
        # parameters and returns only the sample's first dimension size plus the parameters, for
        # several broadcast shapes. Skipped under scripting; requires opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module sampling from a Normal distribution and reporting the
            # sample's leading size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first dimension size of a sample, with
                # x and y.
                return torch.distributions.Normal(x, y).sample().size(0), x, y

        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([[1.0], [2.0]])))
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([1.0])))

        self.run_test(
            M(),
            (
                torch.tensor([[[0.0], [10.0]], [[2.0], [8.0]], [[2.0], [8.0]]]),
                torch.tensor([[1.0], [3.0]]),
            ),
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal_correctness(self):
        # [FUNCTION DOC] Exports Normal sampling of 20000 draws with mean 5 and standard
        # deviation 10, runs it in ORT and checks the sample mean and standard deviation are
        # within 10 percent. Raises AssertionError if outside. Skipped under scripting; requires
        # opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module drawing 20000 samples from a Normal distribution
            # with tensor parameters.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns 20000 samples from a Normal distribution with
                # mean x and standard deviation y.
                return torch.distributions.Normal(x, y).sample([20000])

        expected_mean = 5.0
        expected_std = 10.0

        model_export = M()
        dummy_input = (torch.tensor([expected_mean]), torch.tensor([expected_std]))
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model_export,
            dummy_input,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)

        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        if not (abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1):
            raise AssertionError(
                "the gap of mean between ort outputs and expected one is unacceptable."
            )
        if not (abs(abs(actual_std) - expected_std) <= expected_std * 0.1):
            raise AssertionError(
                "the gap of variance between ort outputs and expected one is unacceptable."
            )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nn_init_normal_correctness(self):
        # [FUNCTION DOC] Exports a module that fills a new tensor of shape 1 x 400 x 50 with the
        # in-place normal initializer and checks the mean and standard deviation from ORT are
        # within 10 percent of 5 and 10. Raises AssertionError if outside. Skipped under
        # scripting; requires opset 11 or newer.
        expected_mean = 5.0
        expected_std = 10.0

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking no inputs that returns
            # a normal-initialized tensor.
            def forward(self):
                # [FUNCTION DOC] HELPER: Creates an empty tensor of shape 1 x 400 x 50, fills it
                # with normal values using the enclosing mean and std and returns it.
                x = torch.ones([]).new_empty(1, 400, 50)
                torch.nn.init.normal_(x, expected_mean, expected_std)
                return x

        model_export = M()
        model_onnx = io.BytesIO()
        test_inputs = ()
        torch.onnx.export(
            model_export,
            test_inputs,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=test_inputs)

        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)

        if not (abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1):
            raise AssertionError(
                "the gap of mean between ort outputs and expected one is unacceptable."
            )
        if not (abs(abs(actual_std) - expected_std) <= expected_std * 0.1):
            raise AssertionError(
                "the gap of variance between ort outputs and expected one is unacceptable."
            )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform(self):
        # [FUNCTION DOC] Exports sampling from Uniform distributions for several broadcast
        # shapes, returning only the sample's leading size and parameters. Skipped under
        # scripting; requires opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module sampling from a Uniform distribution and reporting the
            # sample's leading size.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first dimension size of a sample, with
                # x and y.
                return torch.distributions.Uniform(x, y).sample().size(0), x, y

        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([10.0])))
        self.run_test(M(), (torch.tensor([[0.0], [6.0]]), torch.tensor([[1.0], [7.0]])))
        self.run_test(
            M(), (torch.tensor([1.0]), torch.tensor([[10.0], [7.0], [9.0], [20.0]]))
        )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform_correctness(self):
        # [FUNCTION DOC] Exports Uniform sampling of 10000 draws between 5 and 10 and checks
        # in ORT that the minimum and maximum are within bounds and the mean is within 5
        # percent of 7.5. Raises AssertionError otherwise. Skipped under scripting; requires
        # opset 11 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module drawing 10000 samples from a Uniform distribution
            # with tensor bounds.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns 10000 samples from a Uniform distribution
                # between x and y.
                return torch.distributions.Uniform(x, y).sample([10000])

        expected_min = 5.0
        expected_max = 10.0
        expected_mean = (expected_min + expected_max) / 2

        model_export = M()
        dummy_input = (torch.tensor([expected_min]), torch.tensor([expected_max]))
        model_onnx = io.BytesIO()
        torch.onnx.export(
            model_export,
            dummy_input,
            model_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )
        ort_sess = verification._ort_session(model_onnx)

        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)
        actual_min = np.min(ort_out)
        actual_max = np.max(ort_out)
        actual_mean = np.mean(ort_out)

        if not (actual_min >= expected_min):
            raise AssertionError("the minimum value of ort outputs is out of scope.")
        if not (actual_max <= expected_max):
            raise AssertionError("the maximum value of ort outputs is out of scope.")
        if not (abs(actual_mean - expected_mean) <= expected_mean * 0.05):
            raise AssertionError("the mean value of ort outputs is out of scope.")

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_int(self):
        # [FUNCTION DOC] Exports a tensor built from a list comprehension of integers whose
        # length is the first dimension size, with int dtype. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input and an int tensor of twos with
            # length equal to its first dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a list of twos, one per row, converts it to an
                # int tensor and returns it with x.
                result = torch.tensor([2 for i in range(x.size()[0])], dtype=torch.int)
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_float(self):
        # [FUNCTION DOC] Exports a tensor built from a list of floats whose length is the first
        # dimension size. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input and a float tensor of 1.1 values
            # with length equal to its first dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a list of 1.1 values, one per row, converts it
                # to a float tensor and returns it with x.
                result = torch.tensor(
                    [1.1 for i in range(x.size()[0])], dtype=torch.float
                )
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_bool(self):
        # [FUNCTION DOC] Exports a tensor built from a list of booleans whose length is the
        # first dimension size. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input and a bool tensor of false values
            # with length equal to its first dimension.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a list of false values, one per row, converts it
                # to a bool tensor and returns it with x.
                result = torch.tensor(
                    [False for i in range(x.size()[0])], dtype=torch.bool
                )
                return x, result

        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    def test_tuple_output_from_if_with_raised_exception(self):
        # [FUNCTION DOC] Exports a scripted module with a branch that raises an exception
        # and a branch returning a pair of zero tensors, using a zero input so the second
        # branch is taken.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module raising an exception for a negative scalar input and
            # otherwise returning two zero vectors.
            def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
                # [FUNCTION DOC] HELPER: Converts the input to a float; if negative raises
                # an exception with message Negative input, else returns two zeros vectors
                # of length 5.
                if float(t) < 0:
                    raise Exception("Negative input")  # noqa: TRY002
                else:
                    return torch.zeros(5), torch.zeros(5)

        x = torch.zeros(1)
        self.run_test(torch.jit.script(M()), (x,))

    # NOTE: For quantization tests, choose scale and zero point carefully
    #       such that inputs and outputs do not always overflow/underflow.
    #       Otherwise test results could be inaccurate.
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_linear(self):
        # [FUNCTION DOC] Exports a quantized Linear layer with fixed quantized weight and bias
        # and a fixed quantized input, for a deterministic result. Requires opset 10 or newer.
        model = torch.ao.nn.quantized.Linear(4, 8)
        # Set fixed weight to avoid flaky test.
        weight = torch.quantize_per_tensor(
            torch.arange(32, dtype=torch.float).view(8, 4), 0.5, 0, torch.qint8
        )
        # Set non-zero bias.
        bias = torch.arange(8, dtype=torch.float)
        model.set_weight_bias(weight, bias)
        # Set fixed input to avoid flaky test.
        input = torch.randn(4, 4)
        input = torch.arange(16, dtype=torch.float).view(4, 4) - 8
        input_tensor = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, input_tensor)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d(self):
        # [FUNCTION DOC] Exports a quantized Conv1d layer with random quantized weight, a ramp
        # bias and a quantized random input. Requires opset 10 or newer.
        model = torch.ao.nn.quantized.Conv1d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d(self):
        # [FUNCTION DOC] Exports a quantized Conv2d layer with random quantized weight, a ramp
        # bias and a quantized random input. Requires opset 10 or newer.
        model = torch.ao.nn.quantized.Conv2d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d(self):
        # [FUNCTION DOC] Exports a quantized Conv3d layer with random quantized weight, a ramp
        # bias and a quantized random input. Skipped if the quantization backend is QNNPACK;
        # requires opset 10 or newer.
        model = torch.ao.nn.quantized.Conv3d(16, 33, [2, 3, 4], stride=[3, 1, 2])
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_adaptive_avg_pool2d(self):
        # [FUNCTION DOC] Exports adaptive average pooling to 5 x 7 on a quantized input.
        # Requires opset 10 or newer.
        model = torch.nn.AdaptiveAvgPool2d((5, 7))
        input = torch.randn(4, 3, 10, 14)
        q_input = torch.quantize_per_tensor(input, 0.2, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d_relu(self):
        # [FUNCTION DOC] Exports a quantized fused Conv1d plus ReLU layer with random quantized
        # weight and ramp bias. Requires opset 10 or newer.
        model = torch.ao.nn.intrinsic.quantized.ConvReLU1d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d_relu(self):
        # [FUNCTION DOC] Exports a quantized fused Conv2d plus ReLU layer with random quantized
        # weight and ramp bias. Requires opset 10 or newer.
        model = torch.ao.nn.intrinsic.quantized.ConvReLU2d(16, 33, 3, stride=2)
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d_relu(self):
        # [FUNCTION DOC] Exports a quantized fused Conv3d plus ReLU layer with random quantized
        # weight and ramp bias. Skipped if the quantization backend is QNNPACK; requires opset
        # 10 or newer.
        model = torch.ao.nn.intrinsic.quantized.ConvReLU3d(
            16, 33, [2, 3, 4], stride=[3, 1, 2]
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose1d(self):
        # [FUNCTION DOC] Exports a quantized ConvTranspose1d layer with output padding 1 and
        # stride 2, random quantized weight and ramp bias. Requires opset 10 or newer.
        model = torch.ao.nn.quantized.ConvTranspose1d(
            16, 33, 3, output_padding=1, stride=2
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose2d(self):
        # [FUNCTION DOC] Exports a quantized ConvTranspose2d layer with output padding and
        # stride 2, random quantized weight and ramp bias. Requires opset 10 or newer.
        model = torch.ao.nn.quantized.ConvTranspose2d(
            16, 33, 3, output_padding=(0, 1), stride=2
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 3, 3), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv_transpose3d(self):
        # [FUNCTION DOC] Exports a quantized ConvTranspose3d layer with output padding and
        # strides, random quantized weight and ramp bias. Skipped if the quantization backend is
        # QNNPACK; requires opset 10 or newer.
        model = torch.ao.nn.quantized.ConvTranspose3d(
            16, 33, [2, 3, 4], output_padding=(0, 1, 2), stride=[3, 1, 2]
        )
        # Manually initialize model weight and bias to random numbers.
        # By default all zeros.
        q_weight = torch.quantize_per_tensor(
            torch.randn(16, 33, 2, 3, 4), 0.5, 0, torch.qint8
        )
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @common_utils.parametrize(
        "function_or_module",
        [
            common_utils.subtest(
                torch.nn.ReLU(),
                name="relu",
            ),
            common_utils.subtest(
                torch.nn.LeakyReLU(),
                name="leaky_relu",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.LeakyReLU(2.0, 1),
                name="quantized_leaky_relu",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.Hardswish(2.0, 1),
                name="quantized_hardswish",
            ),
            common_utils.subtest(
                torch.nn.Sigmoid(),
                name="sigmoid",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.Sigmoid(2.0, 1),
                name="quantized_sigmoid",
            ),
            common_utils.subtest(
                torch.nn.Hardsigmoid(),
                name="hardsigmoid",
            ),
            common_utils.subtest(
                torch.nn.Tanh(),
                name="tanh",
            ),
            common_utils.subtest(
                torch.nn.Hardtanh(),
                name="hardtanh",
            ),
            common_utils.subtest(
                lambda x: torch.transpose(x, 0, 1),
                name="transpose",
            ),
            common_utils.subtest(
                lambda x: x.expand(2, 4, 2, 3),
                name="expand",
            ),
            common_utils.subtest(
                lambda x: x.view(1, 4, 6),
                name="view",
            ),
            common_utils.subtest(
                lambda x: x.select(1, 1),
                name="select",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.LayerNorm(
                    [4, 2, 3],
                    torch.nn.Parameter(torch.ones([4, 2, 3])),
                    torch.nn.Parameter(torch.zeros([4, 2, 3])),
                    2.0,
                    1,
                ),
                name="layer_norm",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.InstanceNorm1d(
                    2,
                    torch.nn.Parameter(torch.ones(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                name="instance_norm",
            ),
            common_utils.subtest(
                torch.ao.nn.quantized.GroupNorm(
                    2,
                    4,
                    torch.nn.Parameter(torch.zeros(4)),
                    torch.nn.Parameter(torch.zeros(4)),
                    2.0,
                    1,
                ),
                name="group_norm",
            ),
            common_utils.subtest(
                lambda x: torch.as_strided(x, (2, 2), (1, 2)),
                name="as_strided",
            ),
        ],
    )
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_unary_ops(self, function_or_module):
        # [FUNCTION DOC] Parameterized over many callables and modules (relu, leaky relu,
        # quantized leaky relu and hardswish, sigmoid, quantized sigmoid, hardsigmoid, tanh,
        # hardtanh, transpose, expand, view, select, quantized layer norm, instance norm, group
        # norm and as_strided). Wraps each in a module and exports it with a quantized uint8
        # input of shape 1 x 4 x 2 x 3. Skipped under scripting; requires opset 10 or newer.
        input = torch.randn(1, 4, 2, 3)
        q_input = torch.quantize_per_tensor(input, 0.26, 128, torch.quint8)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an arbitrary callable and applying it
            # to the input.
            def __init__(self, function_or_module):
                # [FUNCTION DOC] HELPER: Stores the callable.
                super().__init__()
                self.function_or_module = function_or_module

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the callable applied to the input.
                return self.function_or_module(x)

        self.run_test(Model(function_or_module), q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_flatten(self):
        # [FUNCTION DOC] Exports flatten of a quantized tensor. Requires opset 10 or newer.
        class FlattenModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening its input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the flattened input.
                return torch.flatten(input)

        x = torch.quantize_per_tensor(torch.randn(1, 2, 3, 4), 1, 0, torch.quint8)
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # torch.jit.frontend.FrontendError: Cannot instantiate class 'QFunctional' in a script function:
    def test_quantized_cat_when_concatinating_the_same_tensor(self):
        # [FUNCTION DOC] Exports quantized concatenation of a tensor with itself via the
        # quantized functional helper. Skipped under scripting since the helper class cannot be
        # instantiated in a script function; requires opset 10 or newer.
        class QuantizedSelfConcatenationModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating the input with itself along dim 1 using
            # the quantized functional helper.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates the helper object and returns the concatenation
                # of x with itself along dim 1.
                return torch.ao.nn.quantized.QFunctional().cat((x, x), dim=1)

        q_input = torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8)
        self.run_test(QuantizedSelfConcatenationModel(), q_input)

    @common_utils.parametrize(
        "x, y",
        [
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(
                        torch.zeros(1, 3), 0.26, 128, torch.quint8
                    ),
                ],
                name="different_shape",
            ),
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 42, 1, torch.quint8),
                ],
                name="different_scale",
            ),
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 63, torch.quint8),
                ],
                name="different_zero_point",
            ),
            common_utils.subtest(
                [
                    torch.quantize_per_tensor(
                        torch.ones(2, 3), 0.26, 128, torch.quint8
                    ),
                    torch.quantize_per_tensor(torch.ones(2, 3), 0.1, 63, torch.quint8),
                ],
                name="different_zero_point_and_scale",
            ),
        ],
    )
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # torch.jit.frontend.FrontendError: Cannot instantiate class 'QFunctional' in a script function:
    def test_quantized_cat(self, x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] Parameterized over four pairs of quantized tensors that differ in
        # shape, scale, zero point, or zero point and scale. Exports quantized concatenation
        # along dim 0. Skipped under scripting; requires opset 10 or newer.
        class QuantizedConcatenationModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating two quantized tensors along dim 0 using
            # the quantized functional helper.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Creates the helper object and returns the concatenation
                # of x and y along dim 0.
                return torch.ao.nn.quantized.QFunctional().cat((x, y), dim=0)

        self.run_test(QuantizedConcatenationModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    # torch.jit.frontend.FrontendError:
    # Cannot instantiate class 'QFunctional' in a script function
    @skipScriptTest()
    def test_quantized_arithmetic_qfunctional(self):
        # [FUNCTION DOC] Exports quantized add followed by multiply via the quantized
        # functional helper on two random quantized tensors. Skipped under scripting; requires
        # opset 10 or newer.
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)

        class ArithmeticModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding then multiplying quantized tensors with the
            # quantized functional helper.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the product of the quantized sum of x and y
                # with x, each computed through a fresh helper object.
                o = torch.ao.nn.quantized.QFunctional().add(x, y)
                o = torch.ao.nn.quantized.QFunctional().mul(o, x)
                return o

        self.run_test(ArithmeticModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_arithmetic(self):
        # [FUNCTION DOC] Exports quantized add and multiply through the quantized operator
        # library with output scale 0.4 and zero point 100. Requires opset 10 or newer.
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)

        class ArithmeticModel2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the quantized add and mul operators with fixed
            # output scale and zero point.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the quantized product of the quantized sum of x
                # and y with x, with scale 0.4 and zero point 100 at each step.
                o = torch.ops.quantized.add(x, y, 0.4, 100)
                o = torch.ops.quantized.mul(o, x, 0.4, 100)
                return o

        self.run_test(ArithmeticModel2(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantize_per_tensor(self):
        # [FUNCTION DOC] Exports per-tensor quantization of a float tensor into signed int8 and
        # unsigned uint8 outputs with scale 0.2. Requires opset 10 or newer.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module quantizing its input two ways.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair: the int8 quantization with zero point 0
                # and the uint8 quantization with zero point 128.
                return (
                    torch.quantize_per_tensor(x, 0.2, 0, torch.qint8),
                    torch.quantize_per_tensor(x, 0.2, 128, torch.quint8),
                )

        x = torch.randn(4, 6)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_dequantize(self):
        # [FUNCTION DOC] Exports dequantize of an int8 quantized tensor. Requires opset
        # 10 or newer.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dequantizing its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dequantized input.
                return torch.dequantize(x)

        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 0, torch.qint8)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_per_channel(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training linear layer with
        # per-channel weights, using fixed weight, bias and input values from the deterministic
        # tensor helper. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a 4 to 3 linear layer and
            # a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs and the linear layer.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 3)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies the linear layer and
                # dequantizes the result.
                x = self.quant(x)
                x = self.linear(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((3, 4))
        )
        model.linear.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((4, 4), offset=-8)
        self.run_test(model, input)

    @unittest.skip(
        "ORT fails with Validating no unexpected access using an invalid node_index on torch converted model"
    )
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_quantized_list_of_inputs_with_cat(self):
        # [FUNCTION DOC] Skipped test (ORT validation failure on converted models). Would
        # export a converted model concatenating a quantized tensor with itself. Requires opset
        # 13 or newer.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with quantization and dequantization stubs around a
            # concatenation of the input with itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two stubs.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, concatenates it with itself along
                # dim 1 and dequantizes the result.
                x = self.quant(x)
                x = torch.cat([x, x], 1)
                x = self.dequant(x)
                return x

        model = TestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        x = torch.randn(2, 4, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_relu(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training model with ReLU between
        # stubs. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a ReLU and
            # a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs and the ReLU.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies ReLU and dequantizes.
                x = self.quant(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        input = torch.randn(8, 4)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training Conv2d with
        # fixed weight, bias and input from the deterministic tensor helper. Requires
        # opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a 4 to 2 channel convolution
            # and a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs and the convolution.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies the
                # convolution and dequantizes.
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training Conv2d plus ReLU with
        # fixed weight, bias and input. Requires opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a convolution, a ReLU and
            # a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs, the convolution and the ReLU.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies the convolution and
                # ReLU, and dequantizes.
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model)
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu_fused(self):
        # [FUNCTION DOC] Exports a quantization-aware-training model with the convolution and
        # ReLU fused before preparation, with fixed weight, bias and input values. Requires
        # opset 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a convolution, a ReLU and a
            # dequantization stub, fused as conv and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs, the convolution and the ReLU.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies the convolution and
                # ReLU, and dequantizes.
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.fuse_modules(model.eval(), [["conv", "relu"]])
        model = torch.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.conv.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2)
        )
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test(
            (3, 4, 8, 8), offset=-384, max_val=12
        )
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_relu_fused(self):
        # [FUNCTION DOC] Exports a quantization-aware-training model with a Linear and ReLU
        # fused before preparation, with fixed weight, bias and input values. Requires opset
        # 13 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a 4 to 2 linear layer, a ReLU
            # and a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs, the linear layer and the ReLU.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies the linear layer and
                # ReLU, and dequantizes.
                x = self.quant(x)
                x = self.linear(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.fuse_modules(model.eval(), [["linear", "relu"]])
        model = torch.ao.quantization.prepare_qat(model.train())
        # Set fixed weight and bias to avoid flaky test.
        model.linear.weight = torch.nn.Parameter(
            _construct_tensor_for_quantization_test((2, 4), max_val=2)
        )
        model.linear.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((3, 4), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_maxpool2d(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training max pooling model on a
        # fixed deterministic input. Requires opset 10 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a quantization stub, a MaxPool2d (kernel 3, stride
            # 2, padding 1) and a dequantization stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs and the pooling layer.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, pools and dequantizes.
                x = self.quant(x)
                x = self.pool(x)
                x = self.dequant(x)
                return x

        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)

        # Set fixed input to avoid flaky test.
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # Scale and Zero-point must be a scalar in ORT:optimization
    def test_qat_avg_pool2d(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training sequential model with
        # average pooling (kernel 3, stride 2, padding 1) between stubs. Skipped under
        # scripting (scale and zero point must be scalars in ORT optimization); requires opset
        # 10 or newer.
        model = torch.nn.Sequential(
            torch.ao.quantization.QuantStub(),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            torch.ao.quantization.DeQuantStub(),
        )
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_qat_upsample_nearest2d(self):
        # [FUNCTION DOC] Exports a converted quantization-aware-training sequential model with
        # nearest upsampling by 1.5 between stubs. Requires opset 11 or newer.
        model = torch.nn.Sequential(
            torch.ao.quantization.QuantStub(),
            torch.nn.UpsamplingNearest2d(scale_factor=1.5),
            torch.ao.quantization.DeQuantStub(),
        )
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 3, 2, 2))
        self.run_test(model, input)

    def test_0d_tensor_broadcast(self):
        # [FUNCTION DOC] Exports a model combining an empty tensor and a one-element tensor in
        # add and multiply, checking broadcasting with a zero-size operand.
        class fn(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the sum of x and y to the square of y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y plus y times y.
                a = torch.add(x, y)
                b = torch.mul(y, y)
                return a + b

        x = torch.ones(0)
        y = torch.ones(1)
        self.run_test(fn(), (x, y), input_names=["x", "y"], output_names=["output"])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_convolution_allow_tf32(self):
        # [FUNCTION DOC] Exports the internal convolution operator with and without the extra
        # flag allowing TF32, on a 1 x 3 x 224 x 224 input with a random 32-channel weight and
        # relaxed tolerances. Requires opset 9 or newer.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the internal convolution operator with stride 2
            # and a random weight, passing an extra TF32 flag when enabled.
            def __init__(self, allow_tf32):
                # [FUNCTION DOC] HELPER: Stores the TF32 flag and creates the
                # random weight parameter.
                super().__init__()

                self.allow_tf32 = allow_tf32
                weight = torch.rand(32, 3, 3, 3)
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the internal convolution operator with the input
                # and weight, stride 2, no padding, no bias, and an argument list that has one
                # more trailing flag when TF32 is allowed.
                if self.allow_tf32:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                        True,
                    )
                else:
                    return torch._convolution(
                        x,
                        self.weight,
                        None,
                        [2, 2],
                        [0, 0],
                        [1, 1],
                        False,
                        [0, 0],
                        1,
                        False,
                        False,
                        True,
                    )

        x = torch.randn(1, 3, 224, 224)
        self.run_test(Module(False), x, rtol=1e-3, atol=1e-6)
        self.run_test(Module(True), x, rtol=1e-3, atol=1e-6)

    class AffineGridModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Class-level helper module computing an affine sampling grid with a
        # stored align_corners setting.
        def __init__(self, align_corners) -> None:
            # [FUNCTION DOC] HELPER: Stores the align_corners setting.
            super().__init__()
            self.align_corners = align_corners

        def forward(self, theta, size):
            # [FUNCTION DOC] HELPER: Returns the affine grid for the transform
            # matrices and size.
            return torch.nn.functional.affine_grid(theta, size, self.align_corners)

    @skipIfUnsupportedMinOpsetVersion(20)
    @skipScriptTest()
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                10,
                np.array([0.3, -0.5]),
                np.array([1.5, 0.5]),
            ),
            (
                60,
                np.array([-0.5, -0.5]),
                np.array([3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2], [2, 10, 2, 3]),
    )
    def test_affine_grid_2d(self, align_corners, theta_params, size):
        # [FUNCTION DOC] Parameterized over align_corners, two 2D transform parameter sets
        # (rotation angle, translation, scale) and two output sizes. Builds a batch of 2x3
        # affine matrices from the rotation, translation and scale, and exports the 2D affine
        # grid. Skipped under scripting; requires opset 20 or newer.
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian = (angle / 180.0) * np.pi
            theta = np.append(
                theta,
                [
                    np.cos(angle_radian) * scale[0],
                    -np.sin(angle_radian),
                    translation[0],
                    np.sin(angle_radian),
                    np.cos(angle_radian) * scale[1],
                    translation[1],
                ],
            )
        theta = theta.reshape(size[0], 2, 3)
        theta = torch.Tensor(theta)
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    @skipIfUnsupportedMinOpsetVersion(20)
    @skipScriptTest()
    @common_utils.parametrize(
        "align_corners",
        (True, False),
    )
    @common_utils.parametrize(
        "theta_params",
        (
            (
                [10, 20],
                np.array([0.3, -0.5, 1.8]),
                np.array([1.5, 2.0, 0.5]),
            ),
            (
                [60, -30],
                np.array([-0.5, -0.5, 0.3]),
                np.array([0.3, 3.0, 5.5]),
            ),
        ),
    )
    @common_utils.parametrize(
        "size",
        ([1, 1, 3, 2, 2], [2, 10, 2, 2, 3]),
    )
    def test_affine_grid_3d(self, align_corners, theta_params, size):
        # [FUNCTION DOC] Parameterized over align_corners, two 3D transform parameter sets (two
        # rotation angles, translation, scale) and two sizes. Builds a batch of 3x4 affine
        # matrices from rotations about two axes, scale and translation, and exports the 3D
        # affine grid. Skipped under scripting; requires opset 20 or newer.
        angle, translation, scale = theta_params
        theta = np.array([], dtype=np.float32)
        for _ in range(size[0]):
            angle_radian_x = (angle[0] / 180.0) * np.pi
            angle_radian_y = (angle[1] / 180.0) * np.pi
            rot_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_radian_x), -np.sin(angle_radian_x)],
                    [0, np.sin(angle_radian_x), np.cos(angle_radian_x)],
                ]
            )
            rot_matrix_y = np.array(
                [
                    [np.cos(angle_radian_y), 0, np.sin(angle_radian_y)],
                    [0, 1, 0],
                    [-np.sin(angle_radian_y), 0, np.cos(angle_radian_y)],
                ]
            )
            rot_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
            rot_matrix = rot_matrix * scale.reshape(3, 1)
            rot_matrix = np.append(rot_matrix, np.reshape(translation, (3, 1)), axis=1)
            theta = np.append(theta, rot_matrix.flatten())

        theta = theta.reshape(size[0], 3, 4)
        theta = torch.Tensor(theta)
        self.run_test(TestONNXRuntime.AffineGridModule(align_corners), (theta, size))

    @skipIfUnsupportedMinOpsetVersion(16)
    @common_utils.parametrize(
        "mode",
        ("bilinear", "nearest", "bicubic"),
    )
    @common_utils.parametrize(
        "padding_mode",
        ("zeros", "border", "reflection"),
    )
    @common_utils.parametrize(
        "align_corners",
        (True, False),
        name_fn=lambda align_corners: str(align_corners),
    )
    def test_grid_sample(self, mode, padding_mode, align_corners):
        # [FUNCTION DOC] Parameterized over interpolation mode, padding mode and align_corners.
        # Exports grid_sample on a 4D input with a random grid, using looser tolerances for
        # bicubic with border padding, then repeats for 5D volumetric input over the modes that
        # support it (see the loop below). Requires opset 16 or newer.
        n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4

        atol_rtol = {}
        if (mode, padding_mode) == ("bicubic", "border"):
            if align_corners:
                atol_rtol.update({"atol": 0.3, "rtol": 0.4})
            else:
                atol_rtol.update({"atol": 0.02, "rtol": 0.02})
        input, grid = torch.randn(n, c, h_in, w_in), torch.randn(n, h_out, w_out, 2)

        class GridSampleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying functional grid_sample with stored mode,
            # padding mode and align_corners.
            def __init__(self, mode, padding_mode, align_corners) -> None:
                # [FUNCTION DOC] HELPER: Stores the three settings.
                super().__init__()
                self.mode, self.padding_mode, self.align_corners = (
                    mode,
                    padding_mode,
                    align_corners,
                )

            def forward(self, input, grid):
                # [FUNCTION DOC] HELPER: Returns the grid sample of the input at the grid with
                # the stored settings.
                return torch.nn.functional.grid_sample(
                    input, grid, self.mode, self.padding_mode, self.align_corners
                )

        self.run_test(
            GridSampleModule(mode, padding_mode, align_corners),
            (input, grid),
            **atol_rtol,
        )

        # ONNX Opset 16 GridSample with 5D volumetric input is not supported.
        volumetric_input_tensor = torch.randn(n, c, d_in, h_in, w_in)
        volumetric_grid_tensor = torch.randn(n, d_out, h_out, w_out, 3)
        for mode, padding_mode, align_corners in itertools.product(
            (
                "bilinear",
                "nearest",
            ),  # PyTorch grid_sample "bicubic" mode does not support 5D volumetric input.
            (
                "zeros",
                "border",
                "reflection",
            ),
            (
                True,
                False,
            ),
        ):
            if self.opset_version < 20:
                with self.assertRaises(
                    torch.onnx.OnnxExporterError,
                ):
                    self.run_test(
                        GridSampleModule(mode, padding_mode, align_corners),
                        (volumetric_input_tensor, volumetric_grid_tensor),
                        **atol_rtol,
                    )
            else:
                self.run_test(
                    GridSampleModule(mode, padding_mode, align_corners),
                    (volumetric_input_tensor, volumetric_grid_tensor),
                    **atol_rtol,
                )

    class IfNoneInput(torch.nn.Module):
        # [CLASS DOC] HELPER: Module returning an optional tensor that is None unless the first
        # dimension exceeds one, in which case it is the input.
        def forward(self, x) -> Tensor | None:
            # [FUNCTION DOC] HELPER: Starts with None and sets the result to x if dim 0 is
            # greater than 1; returns the result.
            y: Tensor | None = None
            if x.size(0) > 1:
                y = x
            return y

    class IfNoneOutput(torch.nn.Module):
        # [CLASS DOC] HELPER: Module returning an optional tensor that starts as the input and
        # becomes None if the first dimension exceeds one.
        def forward(self, x) -> Tensor | None:
            # [FUNCTION DOC] HELPER: Starts with x and sets the result to None if dim 0 is
            # greater than 1; returns the result.
            y: Tensor | None = x
            if x.size(0) > 1:
                y = None
            return y

    class LoopNoneInput(torch.nn.Module):
        # [CLASS DOC] HELPER: Module returning an optional tensor that starts as None and is set
        # to the input in a loop over the first dimension.
        def forward(self, x) -> Tensor | None:
            # [FUNCTION DOC] HELPER: Starts with None and assigns x in each iteration of a loop
            # of dim 0 iterations; returns the result.
            y: Tensor | None = None
            for _ in range(x.size(0)):
                y = x
            return y

    class LoopNoneOutput(torch.nn.Module):
        # [CLASS DOC] HELPER: Module returning an optional tensor that starts as the input and
        # is set to None in a loop over the first dimension.
        def forward(self, x) -> Tensor | None:
            # [FUNCTION DOC] HELPER: Starts with x and assigns None in each iteration of a loop
            # of dim 0 iterations; returns the result.
            y: Tensor | None = x
            for _ in range(x.size(0)):
                y = None
            return y

    @common_utils.parametrize(
        "module_class",
        (IfNoneOutput, IfNoneInput, LoopNoneOutput, LoopNoneInput),
        name_fn=lambda module_class: module_class.__name__,
    )
    @common_utils.parametrize("x_size", (0, 1), name_fn=lambda x_size: str(x_size))
    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_optional_output(self, module_class: type[torch.nn.Module], x_size: int):
        # [FUNCTION DOC] Parameterized over four optional-returning modules and input sizes 0
        # and 1. Scripts the module, exports it with a dynamic first axis, loads the ONNX graph
        # and asserts that the graph output and the output of every conditional branch have the
        # optional tensor type with the dynamic dimension, then runs run_test. Skipped under
        # tracing; requires opset 16 or newer.
        # Need scripting to preserve control flow for this test to be
        # meaningful.
        model = torch.jit.script(module_class())
        f = io.BytesIO()
        x = torch.ones(x_size)
        dynamic_axis_name = "condition"
        torch.onnx.export(
            model,
            x,
            f,
            opset_version=self.opset_version,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
            dynamo=False,
        )
        exported = onnx.load_from_string(f.getvalue())
        expected_elem_type = JitScalarType.from_value(x).onnx_type()
        expected_output_type = onnx.helper.make_optional_type_proto(
            onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,))
        )
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        for node in exported.graph.node:
            # Both branches output types should match.
            if node.op_type == "If":
                for attr in node.attribute:
                    if attr.name in ("then_branch", "else_branch"):
                        self.assertEqual(expected_output_type, attr.g.output[0].type)

        self.run_test(
            module_class(),
            x,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_uninitialized_optional(self):
        # [FUNCTION DOC] Exports a model with an optional tensor input that is None-checked and
        # has nested conditionals, using dynamic axes on both dims of an int input. Skipped
        # under tracing; requires opset 16 or newer.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking an optional tensor and conditionally adding 4
            # in nested branches.
            def forward(self, y: Tensor | None) -> Tensor | None:
                # [FUNCTION DOC] HELPER: If y is present and its dim 1 is below 5: when dim 0 is
                # 1 adds 4 to y, otherwise returns y early; returns y.
                if y is not None:
                    if y.shape[1] < 5:
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            return y
                return y

        self.run_test(
            Module(),
            torch.ones((3, 4), dtype=torch.int),
            dynamic_axes={"y": {0: "y0", 1: "y1"}},
            input_names=["y"],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_device_eq(self):
        # [FUNCTION DOC] Exports a scripted model comparing a tensor device with the cpu device;
        # since ONNX has no devices the comparison is constant false, and the input is on cpu so
        # behavior matches. Uses a dynamic first axis so the model keeps an input. Requires
        # opset 9 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input if its device is not cpu and
            # otherwise zeros like the input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Compares the device of a with the cpu device; returns a
                # when different, else zeros of the same shape.
                # exercise both Tensor.device (prim::device)
                # and torch.device (prim::Constant).
                if a.device != torch.device("cpu"):
                    return a
                return torch.zeros_like(a)

        mod = torch.jit.script(M())  # preserve control flow

        self.run_test(
            mod,
            # In order for the ONNX model behavior to match the torch model, we
            # need to construct input that has the same device that is checked for
            # in forward(). In ONNX there is no such thing as a device, so the if
            # condition is always false.
            torch.randn(3, 3, device="cpu"),
            # Force dynamic axes so that the output shape depends on the input.
            # Otherwise the entire model will just return a constant and not have
            # any inputs.
            input_names=["a"],
            dynamic_axes={"a": {0: "a0"}},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lerp(self):
        # [FUNCTION DOC] Exports lerp with scalar, tensor and expression weights and with scalar
        # tensor end points. Requires opset 9 or newer.
        class LerpModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning seven lerp results with different end
            # and weight forms.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of seven interpolations of x toward
                # filled tensors or a scalar tensor with weights as float scalars, scalar
                # tensors or x divided by ten.
                return (
                    x.lerp(torch.full_like(x, 10), 0.4),
                    x.lerp(torch.full_like(x, 20), 0.7),
                    x.lerp(torch.full_like(x, 30), torch.tensor(0.4)),
                    x.lerp(torch.full_like(x, 40), x / 10.0),
                    x.lerp(torch.tensor(10.0), x / 10.0),
                    x.lerp(torch.tensor(10.0), 0.4),
                    x.lerp(torch.tensor(10.0), torch.tensor(0.4)),
                )

        self.run_test(LerpModel(), torch.rand(5, 4, 3))

    @common_utils.parametrize("input_dtype", [torch.cfloat, torch.float])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_print_tensor_within_torch_nn_module(self, input_dtype: torch.dtype):
        # [FUNCTION DOC] Parameterized over complex float and float inputs. Exports a model that
        # prints a slice and calls tolist, which resolve conjugation and negation as side
        # effects. With complex input expects RuntimeError, with real input expects success.
        # Requires opset 9 or newer.
        class PrintTensorOnMyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module printing a tensor and converting its input to a list as
            # side effects, returning the first column.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes the first column of x, prints it,
                # converts x to a list of floats (discarding it) and returns the column.
                # Prints to standard output.
                # 'print' has side effect calling 'resolve_conj' and 'resolve_neg'.
                x_firsts = x[:, 0]
                print(f"x_firsts: {x_firsts}")
                # 'tolist' has side effect calling 'resolve_conj' and 'resolve_neg'.
                # Annotation added to pass torch script.
                _: list[float] = x.tolist()
                return x_firsts

        m = PrintTensorOnMyModel()
        x = torch.randn(10, 5, dtype=input_dtype)
        if input_dtype == torch.cfloat:
            with self.assertRaises(RuntimeError):
                self.run_test(
                    m,
                    x,
                )
        else:
            self.run_test(
                m,
                x,
            )

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    @unittest.skipIf(
        not torch.hub._check_module_exists("torch_geometric"),
        "torch_geometric not installed.",
    )
    def test_sage_conv(self):
        # [FUNCTION DOC] Exports a graph neural network module with a SAGE convolution over
        # a k-nearest-neighbor graph built from random coordinates, with dynamic axes.
        # Skipped unless torch_geometric is installed; skipped under scripting; requires
        # opset 16 or newer.
        from torch_geometric import nn as torch_geometric_nn

        # Input
        coords0 = torch.randn(1, 6)
        coords1 = torch.randn(1, 6)
        coords = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
        adj = torch_geometric_nn.knn_graph(coords, k=2, batch=None, loop=True)
        edge_from = adj[0:1, :]
        edge_to = adj[1:, :]
        inputs = (coords0, coords1, edge_from, edge_to)

        class MySAGEConv(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a SAGE convolution block, a batch norm, ReLU, a
            # dense block and a sigmoid, whose forward uses only the convolution.
            # KNOWN ISSUE: the constructor references undefined names (Seq on the nn namespace
            # and Lin) that are only suppressed by a lint comment, so it fails if executed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers listed in the class doc.
                # KNOWN ISSUE: the dense layer line refers to undefined names and raises an
                # error when the module is built.
                super().__init__()
                self.SAGEConvBlock1 = torch_geometric_nn.SAGEConv(
                    2, 512, normalize=True
                )
                self.bano1 = torch_geometric_nn.BatchNorm(512)
                self.relu = torch.nn.ReLU()
                self.dense1 = torch.nn.Seq(Lin(512, 1))  # noqa: F821
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, coords0, coords1, edge_from, edge_to):
                # [FUNCTION DOC] HELPER: Builds an edge index from the two edge tensors, builds
                # a node feature matrix from the two coordinate tensors, applies the SAGE block
                # and returns the sum over all as a one-element tensor.
                adj = torch.cat((edge_from, edge_to), dim=0)
                gra = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
                x1 = self.SAGEConvBlock1(gra, edge_index=adj)
                x = torch.unsqueeze(torch.sum(x1), dim=0)
                return x

        input_names = ["coords0", "coords1", "edge_from", "edge_to"]
        output_names = ["outputs"]
        dynamic_axes = {
            "coords0": {0: "batch_size", 1: "features"},
            "coords1": {0: "batch_size", 1: "features"},
            "edge_from": {0: "batch_size", 1: "features"},
            "edge_to": {0: "batch_size", 1: "features"},
            "outputs": {0: "batch_size"},
        }
        self.run_test(
            MySAGEConv(),
            inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Cannot export with older opsets because of "ConstantFill" op
    # ConstantFill was a temp op removed at opset 8. This is no longer supported by onnxruntime
    # There are still some issues prevent us from enabling script test for these scenarios:
    # test_gru_*:
    #   Operator aten::as_tensor is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055382
    #   Operator aten::_pack_padded_sequence is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055384
    # test_elman_*:
    # Compiling in script mode fails with errors like:
    #   torch.jit.frontend.UnsupportedNodeError: annotated assignments
    #   without assigned value aren't supported
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    # test_lstm_*:
    #   Compiling in script mode fails with errors like:
    #   RuntimeError: Arguments for call are not valid.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1160723
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(9)
    @common_utils.parametrize(
        "name, nonlinearity",
        [
            ("elman", "relu"),
            ("elman", "tanh"),
            ("lstm", None),
            ("gru", None),
        ],
    )
    @common_utils.parametrize(**_parametrize_rnn_args("layers"))
    @common_utils.parametrize(**_parametrize_rnn_args("bidirectional"))
    @common_utils.parametrize(**_parametrize_rnn_args("initial_state"))
    @common_utils.parametrize(**_parametrize_rnn_args("packed_sequence"))
    @common_utils.parametrize(**_parametrize_rnn_args("dropout"))
    def test_rnn(self, *args, **kwargs):
        # [FUNCTION DOC] Parameterized over RNN kind (elman with relu or tanh, lstm, gru) and
        # over layer count, directionality, initial state, packed sequence mode and dropout
        # options. Dispatches to the matching RNN helper. Skipped under scripting; requires
        # opset 9 or newer.
        self._dispatch_rnn_test(*args, **kwargs)


if __name__ == "__main__":
    common_utils.TestCase._default_dtype_check_enabled = True
    common_utils.run_tests()
