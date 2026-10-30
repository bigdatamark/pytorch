# Owner(s): ["module: onnx"]
# [MODULE DOC] ONNX Runtime export tests for larger torchvision models (detection models, RoI
# heads, a transformer encoder, MobileNet v3, ShuffleNet v2). Defines a module-level exportTest
# function that is injected into two dynamically created test classes (TestModels and
# TestModels_new_jit_API) built by copying the namespace of the test_models module test class, one
# with script testing disabled and one enabled. Also holds helpers that load sample images and
# build small detection components, and a parameterized test class run in script and non-script
# variants. Imports torchvision, PIL, parameterized and test_models at import time; runs the
# common test runner as a script.

import os
import unittest
from collections import OrderedDict
from collections.abc import Mapping

import onnx_test_common
import parameterized
import PIL
import pytorch_test_common
import test_models
import torchvision
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion, skipScriptTest
from torchvision import ops
from torchvision.models.detection import (
    faster_rcnn,
    image_list,
    keypoint_rcnn,
    mask_rcnn,
    roi_heads,
    rpn,
    transform,
)

import torch
from torch import nn
from torch.testing._internal import common_utils


def exportTest(
    self,
    model,
    inputs,
    rtol=1e-2,
    atol=1e-7,
    opset_versions=None,
    acceptable_error_percentage=None,
):
    # [FUNCTION DOC] Shared export check, used as the exportTest method of the generated classes.
    # For each requested opset version (default 7 through 14) it sets the opset and
    # shape-inference attributes on the test instance and calls run_model_test with the model,
    # inputs, tolerances and acceptable error percentage. When the instance has script testing
    # enabled and the opset is above 11, it also scripts the model and repeats the check on the
    # scripted model. Mutates attributes of the test case.
    opset_versions = opset_versions if opset_versions else [7, 8, 9, 10, 11, 12, 13, 14]

    for opset_version in opset_versions:
        self.opset_version = opset_version
        self.onnx_shape_inference = True
        onnx_test_common.run_model_test(
            self,
            model,
            input_args=inputs,
            rtol=rtol,
            atol=atol,
            acceptable_error_percentage=acceptable_error_percentage,
        )

        if self.is_script_test_enabled and opset_version > 11:
            script_model = torch.jit.script(model)
            onnx_test_common.run_model_test(
                self,
                script_model,
                input_args=inputs,
                rtol=rtol,
                atol=atol,
                acceptable_error_percentage=acceptable_error_percentage,
            )


TestModels = type(
    "TestModels",
    (pytorch_test_common.ExportTestCase,),
    dict(
        test_models.TestModels.__dict__,
        is_script_test_enabled=False,
        is_script=False,
        exportTest=exportTest,
    ),
)


# model tests for scripting with new JIT APIs and shape inference
TestModels_new_jit_API = type(
    "TestModels_new_jit_API",
    (pytorch_test_common.ExportTestCase,),
    dict(
        TestModels.__dict__,
        exportTest=exportTest,
        is_script_test_enabled=True,
        is_script=True,
        onnx_shape_inference=True,
    ),
)


def _get_image(rel_path: str, size: tuple[int, int]) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Opens an image from the assets directory next to this file (the
    # relative path is split on slashes), converts to RGB, resizes with bilinear filtering to the
    # given width and height pair, and returns it as a tensor in the range zero to one. Raises if
    # the file is missing.
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    path = os.path.join(data_dir, *rel_path.split("/"))
    image = PIL.Image.open(path).convert("RGB").resize(size, PIL.Image.BILINEAR)

    return torchvision.transforms.ToTensor()(image)


def _get_test_images() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Returns two lists each holding one image tensor: one from the Grace
    # Hopper photo resized to 100 by 320 and one from the PyTorch logo resized to 250 by 380
    # (width and height order of the resize call).
    return (
        [_get_image("grace_hopper_517x606.jpg", (100, 320))],
        [_get_image("rgb_pytorch.png", (250, 380))],
    )


def _get_features(images):
    # [FUNCTION DOC] HELPER: Builds an ordered mapping from the names zero to four to random
    # feature maps with 256 channels and batch size 2, whose spatial sizes are the last two
    # dimensions of the images divided by 4, 8, 16, 32 and 64 (integer division). Only the shape
    # of the images is used.
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


def _init_test_generalized_rcnn_transform():
    # [FUNCTION DOC] HELPER: Returns a torchvision GeneralizedRCNNTransform configured with min
    # size 100, max size 200 and the ImageNet mean and std.
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    return transform.GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)


def _init_test_rpn():
    # [FUNCTION DOC] HELPER: Builds a torchvision region proposal network with five anchor sizes
    # (32 to 512, one per level), three aspect ratios, a head with 256 channels, IoU thresholds
    # 0.7 and 0.3, batch size 256 per image, positive fraction 0.5, 2000 pre and post NMS
    # candidates in training and 1000 in testing, NMS threshold 0.7 and score threshold 0. Returns
    # it.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    out_channels = 256
    rpn_head = rpn.RPNHead(
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

    return rpn.RegionProposalNetwork(
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


def _init_test_roi_heads_faster_rcnn():
    # [FUNCTION DOC] HELPER: Builds a Faster R-CNN style torchvision RoIHeads with a multi-scale
    # RoI align pooling over feature maps zero to three (output size 7, sampling ratio 2), a
    # two-layer MLP box head with representation size 1024, a box predictor for 91 classes, IoU
    # thresholds 0.5, batch size 512, positive fraction 0.25, score threshold 0.05, NMS threshold
    # 0.5 and at most 100 detections. Returns it.
    out_channels = 256
    num_classes = 91

    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100

    box_roi_pool = ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = faster_rcnn.TwoMLPHead(out_channels * resolution**2, representation_size)

    representation_size = 1024
    box_predictor = faster_rcnn.FastRCNNPredictor(representation_size, num_classes)

    return roi_heads.RoIHeads(
        box_roi_pool,
        box_head,
        box_predictor,
        box_fg_iou_thresh,
        box_bg_iou_thresh,
        box_batch_size_per_image,
        box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
    )


@parameterized.parameterized_class(
    ("is_script",),
    [(True,), (False,)],
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestModelsONNXRuntime(onnx_test_common._TestONNXRuntime):
    # [CLASS DOC] ONNX Runtime tests for large torchvision models; the decorator generates one
    # class with is_script true and one with false, named by the common name function. Extends the
    # shared ONNX Runtime test base and uses its run_test method.
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Faster RCNN model is not scriptable
    def test_faster_rcnn(self):
        # [FUNCTION DOC] Exports Faster R-CNN ResNet50 FPN (min size 200, max size 300, eval mode)
        # and runs run_test several times: plain with two random images, with explicit input and
        # output names and dynamic axes, and twice with additional test inputs including a dummy
        # image and the sample images. Requires opset 11 or newer; skipped in scripting mode
        # because the model is not scriptable.
        model = faster_rcnn.fasterrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=True, min_size=200, max_size=300
        )
        model.eval()
        x1 = torch.randn(3, 200, 300, requires_grad=True)
        x2 = torch.randn(3, 200, 300, requires_grad=True)
        self.run_test(model, ([x1, x2],), rtol=1e-3, atol=1e-5)
        self.run_test(
            model,
            ([x1, x2],),
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        images, test_images = _get_test_images()
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        self.run_test(
            model,
            (dummy_image,),
            additional_test_inputs=[(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["outputs"],
            dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )

    @unittest.skip("Failing after ONNX 1.13.0")
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_mask_rcnn(self):
        # [FUNCTION DOC] Mask R-CNN export check mirroring the Faster R-CNN test, with named
        # outputs for boxes, labels, scores and masks and per-output dynamic axes, on the sample
        # images and a dummy image. Unconditionally skipped (failing after ONNX 1.13.0); also
        # needs opset 11 and no scripting.
        model = mask_rcnn.maskrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=True, min_size=200, max_size=300
        )
        images, test_images = _get_test_images()
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        self.run_test(
            model,
            (images,),
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_image,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )
        self.run_test(
            model,
            (dummy_image,),
            additional_test_inputs=[(dummy_image,), (images,)],
            input_names=["images_tensors"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images_tensors": [0, 1, 2],
                "boxes": [0, 1],
                "labels": [0],
                "scores": [0],
                "masks": [0, 1, 2],
            },
            rtol=1e-3,
            atol=1e-5,
        )

    @unittest.skip("Failing, see https://github.com/pytorch/pytorch/issues/66528")
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_keypoint_rcnn(self):
        # [FUNCTION DOC] Keypoint R-CNN export check with four generic output names and dynamic
        # axes on the input images, using looser rtol 5e-3 for the extra-input runs.
        # Unconditionally skipped (see the linked issue); also needs opset 11 and no scripting.
        model = keypoint_rcnn.keypointrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False, min_size=200, max_size=300
        )
        images, test_images = _get_test_images()
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        self.run_test(
            model,
            (images,),
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=1e-3,
            atol=1e-5,
        )
        dummy_images = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(
            model,
            (images,),
            additional_test_inputs=[(images,), (test_images,), (dummy_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=5e-3,
            atol=1e-5,
        )
        self.run_test(
            model,
            (dummy_images,),
            additional_test_inputs=[(dummy_images,), (test_images,)],
            input_names=["images_tensors"],
            output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
            dynamic_axes={"images_tensors": [0, 1, 2]},
            rtol=5e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_roi_heads(self):
        # [FUNCTION DOC] Builds a model that wraps the transform, region proposal network and RoI
        # heads, runs it once eagerly on random 100 by 100 images with generated features, then
        # checks export with six named inputs (the images and the five feature maps) all having
        # fully dynamic axes, with an additional input set built from 150 by 150 images. Requires
        # opset 11 or newer; skipped in scripting mode.
        class RoIHeadsModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module chaining the generalized RCNN transform, the RPN and the
            # RoI heads to turn images and a mapping of feature maps into post-processed
            # detections.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the transform, region proposal network and RoI
                # heads using the module-level builder helpers.
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()
                self.rpn = _init_test_rpn()
                self.roi_heads = _init_test_roi_heads_faster_rcnn()

            def forward(self, images, features: Mapping[str, torch.Tensor]):
                # [FUNCTION DOC] HELPER: Records the original size pair of each image, wraps the
                # images into an image list, gets proposals from the RPN, runs the RoI heads on
                # features and proposals, post-processes the detections back to the original
                # sizes, and returns the detections.
                original_image_sizes = [
                    (img.shape[-1], img.shape[-2]) for img in images
                ]

                images_m = image_list.ImageList(
                    images, [(i.shape[-1], i.shape[-2]) for i in images]
                )
                proposals, _ = self.rpn(images_m, features)
                detections, _ = self.roi_heads(
                    features, proposals, images_m.image_sizes
                )
                detections = self.transform.postprocess(
                    detections, images_m.image_sizes, original_image_sizes
                )
                return detections

        images = torch.rand(2, 3, 100, 100)
        features = _get_features(images)
        images2 = torch.rand(2, 3, 150, 150)
        test_features = _get_features(images2)

        model = RoIHeadsModule()
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
        )

    @skipScriptTest()  # TODO: #75625
    @skipIfUnsupportedMinOpsetVersion(20)
    def test_transformer_encoder(self):
        # [FUNCTION DOC] Exports a small transformer encoder (embedding 512, 8 heads, hidden 2048,
        # no dropout, 3 layers) on a random sequence input of shape 10 by 32 by 512 with absolute
        # tolerance 1e-5. Requires opset 20 or newer; skipped in scripting mode (see the
        # referenced issue).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a stack of transformer encoder layers built from
            # the given sizes.
            def __init__(self, ninp, nhead, nhid, dropout, nlayers):
                # [FUNCTION DOC] HELPER: Builds one transformer encoder layer from the embedding
                # size, heads, hidden size and dropout, and stacks nlayers of them into an
                # encoder.
                super().__init__()
                encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, nlayers
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the output of the transformer encoder for the
                # input.
                return self.transformer_encoder(input)

        x = torch.rand(10, 32, 512)
        self.run_test(MyModule(512, 8, 2048, 0.0, 3), (x,), atol=1e-5)

    @skipScriptTest()
    def test_mobilenet_v3(self):
        # [FUNCTION DOC] Exports the non-pretrained quantization-variant MobileNet v3 large model
        # on a random single 224 by 224 image through run_test; skipped in scripting mode.
        model = torchvision.models.quantization.mobilenet_v3_large(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        self.run_test(model, (dummy_input,))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_shufflenet_v2_dynamic_axes(self):
        # [FUNCTION DOC] Exports a non-pretrained ShuffleNet v2 0.5x with an extra test input of
        # batch size 3, named input and output, and a batch-size dynamic-axes mapping. Requires
        # opset 11 or newer; skipped in scripting mode.
        # KNOWN ISSUE: the dynamic axes mapping uses the key output while the declared output name
        #   is outputs, so the output axis mapping does not match any output name.
        model = torchvision.models.shufflenet_v2_x0_5(weights=None)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
        self.run_test(
            model,
            (dummy_input,),
            additional_test_inputs=[(dummy_input,), (test_inputs,)],
            input_names=["input_images"],
            output_names=["outputs"],
            dynamic_axes={
                "input_images": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            rtol=1e-3,
            atol=1e-5,
        )


if __name__ == "__main__":
    common_utils.run_tests()
