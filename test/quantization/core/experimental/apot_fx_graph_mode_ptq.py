# [MODULE DOC] Manual script (not a unit test) that evaluates ResNet-18 on ImageNet with
# APoT and uniform quantization configs applied only to Linear layers via FX graph mode.
# Importing it runs everything: loads ImageNet from a fixed home-directory path, builds a
# pretrained float model, then prints top-1/top-5 accuracy for the float model and for
# six qconfig variants (uniform, APoT weights, APoT activations and weights; 8 and 4 bit),
# plus the eager-mode quantized torchvision model.
# KNOWN ISSUE: convert_fx is never imported and quantized_model1 is never defined, so the
# script raises NameError at the first convert/evaluate step; the APoT cases also evaluate
# the prepared (not converted) model.
import torch
import torch.nn as nn
import torch.ao.quantization
from torchvision.models.quantization.resnet import resnet18
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,
    prepare_data_loaders
)

# validation dataset: full ImageNet dataset
data_path = '~/my_imagenet/'

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = resnet18(pretrained=True)
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

"""
Prepare models
"""

# Note that this is temporary, we'll expose these functions to torch.ao.quantization after official releasee
from torch.ao.quantization.quantize_fx import prepare_qat_fx

def calibrate(model, data_loader):
    # [FUNCTION DOC] Runs the model in eval mode with gradients disabled over every batch of the
    # given data loader, discarding outputs, so observers in the prepared model collect
    # statistics. Mutates observer state; returns nothing.
    model.eval()
    with torch.no_grad():
        for image, _ in data_loader:
            model(image)

from torch.ao.quantization.experimental.qconfig import (
    uniform_qconfig_8bit,
    apot_weights_qconfig_8bit,
    apot_qconfig_8bit,
    uniform_qconfig_4bit,
    apot_weights_qconfig_4bit,
    apot_qconfig_4bit
)

"""
Prepare full precision model
"""
full_precision_model = float_model

top1, top5 = evaluate(full_precision_model, criterion, data_loader_test)
print(f"Model #0 Evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model PTQ for specified qconfig for torch.nn.Linear
"""
def prepare_ptq_linear(qconfig):
    # [FUNCTION DOC] Builds a qconfig mapping that applies the given qconfig to all Linear
    # modules, deep-copies the global float_model, prepares it with prepare_qat_fx (despite the
    # PTQ name), then calibrates it on the global test loader. Returns the prepared model.
    # KNOWN ISSUE: uses the QAT preparation entry point for a post-training flow.
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}
    prepared_model = prepare_qat_fx(copy.deepcopy(float_model), qconfig_dict)  # fuse modules and insert observers
    calibrate(prepared_model, data_loader_test)  # run calibration on sample data
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_8bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model  # noqa: F821

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_4bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model  # noqa: F821

top1, top5 = evaluate(quantized_model1, criterion, data_loader_test)  # noqa: F821
print(f"Model #1 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")


"""
Prepare model with APoT activation and weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with APoT activation and weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare eager mode quantized model
"""
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print(f"Eager mode quantized model evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")
