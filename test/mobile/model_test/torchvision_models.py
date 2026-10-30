# [MODULE DOC] Model factories for mobile tests built on torchvision. Each class has a method
# that loads pretrained ImageNet weights (needs torchvision and weight download), traces the
# model with a zero example input, optimizes it for mobile, attaches bundled inputs and runs
# it once. Importing requires torchvision; no other import-time side effects.
from torchvision import models

import torch
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import optimize_for_mobile


class MobileNetV2Module:
    # [CLASS DOC] Factory for a mobile-optimized, traced MobileNetV2 (CPU backend).
    def getModule(self):
        # [FUNCTION DOC] Loads pretrained MobileNetV2 in eval mode, traces it on a zero tensor
        # of shape 1x3x224x224, applies the default mobile optimizer, adds that example as
        # bundled input, runs the module once as a smoke check and returns it.
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module)
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module


class MobileNetV2VulkanModule:
    # [CLASS DOC] Factory for a traced MobileNetV2 optimized for the Vulkan backend.
    def getModule(self):
        # [FUNCTION DOC] Same as the CPU MobileNetV2 factory but calls the mobile optimizer
        # with the vulkan backend. Returns the optimized module after one smoke-test run on
        # the example input (which is also attached as bundled input).
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module, backend="vulkan")
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module


class Resnet18Module:
    # [CLASS DOC] Factory for a mobile-optimized, traced ResNet18 (CPU backend).
    def getModule(self):
        # [FUNCTION DOC] Loads pretrained ResNet18 in eval mode, traces it on a zero tensor
        # of shape 1x3x224x224, optimizes for mobile, attaches the example as bundled input,
        # runs it once and returns the optimized module.
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module)
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module
