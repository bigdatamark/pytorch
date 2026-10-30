# Owner(s): ["module: onnx"]
# [MODULE DOC] ONNX Runtime export tests for pretrained quantized torchvision classification
# models. Provides a helper that loads and preprocesses a sample image from the assets directory,
# a wrapper module reducing logits to the top-1 class id (since quantization noise makes full
# output comparison unreliable), and a test class parameterized over scripting on and off. Imports
# PIL, torchvision and parameterized at import time; runs the common test runner as a script.

import os
import unittest

import onnx_test_common
import parameterized
import PIL
import torchvision

import torch
from torch import nn
from torch.testing._internal import common_utils


def _get_test_image_tensor():
    # [FUNCTION DOC] HELPER: Opens the sample photo in the assets directory next to this file,
    # resizes to 256, center-crops to 224, converts to a tensor, normalizes with the ImageNet mean
    # and std, and returns it with a leading batch dimension of one. Raises if the file is
    # missing.
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    img_path = os.path.join(data_dir, "grace_hopper_517x606.jpg")
    input_image = PIL.Image.open(img_path)
    # Based on example from https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess(input_image).unsqueeze(0)


# Due to precision error from quantization, check only that the top prediction matches.
class _TopPredictor(nn.Module):
    # [CLASS DOC] HELPER: Wrapper module around a base model whose output is reduced to the index
    # of the top-1 class of the first batch element, so tests compare only the top prediction.
    # Holds the base model as an attribute.
    def __init__(self, base_model):
        # [FUNCTION DOC] HELPER: Stores the wrapped base model.
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the base model and returns the id of the highest-scoring
        # entry of the first batch element as a one-element tensor.
        x = self.base_model(x)
        _, topk_id = torch.topk(x[0], 1)
        return topk_id


# TODO: All torchvision quantized model test can be written as single parameterized test case,
# after per-parameter test decoration is supported via #79979, or after they are all enabled,
# whichever is first.
@parameterized.parameterized_class(
    ("is_script",),
    [(True,), (False,)],
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestQuantizedModelsONNXRuntime(onnx_test_common._TestONNXRuntime):
    # [CLASS DOC] ONNX Runtime tests for quantized torchvision models, generated twice by the
    # class decorator with is_script true and false, with generated class names from the common
    # name function. Extends the shared ONNX Runtime test base; every model is wrapped in the
    # top-1 predictor before checking.
    def run_test(self, model, inputs, *args, **kwargs):
        # [FUNCTION DOC] Overrides the base runner: wraps the given model in the top-1 predictor,
        # then delegates to the base class run_test with the same inputs and extra arguments,
        # returning its result.
        model = _TopPredictor(model)
        return super().run_test(model, inputs, *args, **kwargs)

    def test_mobilenet_v3(self):
        # [FUNCTION DOC] Loads the pretrained quantized MobileNet v3 large model from torchvision
        # and runs the shared run_test check on the preprocessed sample image.
        model = torchvision.models.quantization.mobilenet_v3_large(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_inception_v3(self):
        # [FUNCTION DOC] Quantized Inception v3 check; unconditionally skipped because the
        # quantized cat op is unsupported. Loads the pretrained quantized model and runs run_test
        # on the sample image.
        model = torchvision.models.quantization.inception_v3(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_googlenet(self):
        # [FUNCTION DOC] Quantized GoogLeNet check; unconditionally skipped because the quantized
        # cat op is unsupported. Loads the pretrained quantized model and runs run_test on the
        # sample image.
        model = torchvision.models.quantization.googlenet(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_shufflenet_v2_x0_5(self):
        # [FUNCTION DOC] Quantized ShuffleNet v2 0.5x check; unconditionally skipped because the
        # quantized cat op is unsupported. Loads the pretrained quantized model and runs run_test
        # on the sample image.
        model = torchvision.models.quantization.shufflenet_v2_x0_5(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    def test_resnet18(self):
        # [FUNCTION DOC] Loads the pretrained quantized ResNet18 from torchvision and runs
        # run_test on the preprocessed sample image.
        model = torchvision.models.quantization.resnet18(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnet50(self):
        # [FUNCTION DOC] Loads the pretrained quantized ResNet50 from torchvision and runs
        # run_test on the preprocessed sample image.
        model = torchvision.models.quantization.resnet50(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnext101_32x8d(self):
        # [FUNCTION DOC] Loads the pretrained quantized ResNeXt101 32x8d from torchvision and runs
        # run_test on the preprocessed sample image.
        model = torchvision.models.quantization.resnext101_32x8d(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())


if __name__ == "__main__":
    common_utils.run_tests()
