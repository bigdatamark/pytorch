#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]
# mypy: allow-untyped-defs
# [MODULE DOC] Test of bundling JPEG-compressed images as inflatable bundled inputs for a
# scripted module and decoding them after a save/load round trip. Requires cv2 and an internal
# operator library (loaded at import time); running the file as a script raises RuntimeError
# because the test is not enabled in the open-source test discovery.

import io

import cv2  # @manual

import torch
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase


torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")


def model_size(sm):
    # [FUNCTION DOC] HELPER: Returns the byte size of the module when serialized with the jit
    # save function into an in-memory buffer.
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    return len(buffer.getvalue())


def save_and_load(sm):
    # [FUNCTION DOC] HELPER: Serializes the scripted module into an in-memory buffer, rewinds it
    # and returns the module loaded back with the jit load function.
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)


"""Return an InflatableArg that contains a tensor of the compressed image and the way to decode it

    keyword arguments:
    img_tensor -- the raw image tensor in HWC or NCHW with pixel value of type unsigned int
                  if in NCHW format, N should be 1
    quality -- the quality needed to compress the image
"""


def bundle_jpeg_image(img_tensor, quality):
    # [FUNCTION DOC] HELPER: Wraps an image tensor as a bundled-inputs InflatableArg. A 4-D
    # tensor must have batch size one (else an AssertionError is raised) and is converted from
    # NCHW to HWC. The pixels are JPEG-encoded with cv2 at the given quality, flattened into a
    # byte tensor, and paired with a decode expression using the internal decode_bundled_image
    # operator.
    # turn NCHW to HWC
    if img_tensor.dim() == 4:
        if img_tensor.size(0) != 1:
            raise AssertionError(
                f"img_tensor.size(0) must be 1, got {img_tensor.size(0)}"
            )
        img_tensor = img_tensor[0].permute(1, 2, 0)
    pixels = img_tensor.numpy()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)
    enc_img_tensor = torch.from_numpy(enc_img)
    enc_img_tensor = torch.flatten(enc_img_tensor).byte()
    obj = torch.utils.bundled_inputs.InflatableArg(
        enc_img_tensor, "torch.ops.fb.decode_bundled_image({})"
    )
    return obj


def get_tensor_from_raw_BGR(im) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts a BGR image array into a float RGB tensor in NCHW layout
    # with a batch dimension of one and values scaled to the 0-1 range.
    raw_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    raw_data = torch.from_numpy(raw_data).float()
    raw_data = raw_data.permute(2, 0, 1)
    raw_data = torch.div(raw_data, 255).unsqueeze(0)
    return raw_data


class TestBundledImages(TestCase):
    # [CLASS DOC] Test case for bundling compressed images with
    # augment_model_with_bundled_inputs.
    def test_single_tensors(self):
        # [FUNCTION DOC] Reads a sample jpg, bundles it with quality 90 into an identity
        # scripted model, saves/loads the model and checks the inflated bundled input matches
        # the raw image tensor within loose tolerances. Then decodes the raw file bytes with the
        # internal NCHW decode operator (diagonal 1/255 weight, zero bias) and compares to the
        # same raw tensor. Uses hard-coded internal file paths.
        class SingleTensorModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward returns its single argument
            # unchanged.
            def forward(self, arg):
            # [FUNCTION DOC] HELPER: Returns the argument unchanged.
                return arg

        im = cv2.imread("caffe2/test/test_img/p1.jpg")
        tensor = torch.from_numpy(im)
        inflatable_arg = bundle_jpeg_image(tensor, 90)
        input = [(inflatable_arg,)]
        sm = torch.jit.script(SingleTensorModel())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, input)
        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        decoded_data = inflated[0][0]

        # raw image
        raw_data = get_tensor_from_raw_BGR(im)

        self.assertEqual(len(inflated), 1)
        self.assertEqual(len(inflated[0]), 1)
        self.assertEqual(raw_data.shape, decoded_data.shape)
        self.assertEqual(raw_data, decoded_data, atol=0.1, rtol=1e-01)

        # Check if fb::image_decode_to_NCHW works as expected
        with open("caffe2/test/test_img/p1.jpg", "rb") as fp:
            weight = torch.full((3,), 1.0 / 255.0).diag()
            bias = torch.zeros(3)
            byte_tensor = torch.tensor(list(fp.read())).byte()
            im2_tensor = torch.ops.fb.image_decode_to_NCHW(byte_tensor, weight, bias)
            self.assertEqual(raw_data.shape, im2_tensor.shape)
            self.assertEqual(raw_data, im2_tensor, atol=0.1, rtol=1e-01)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
