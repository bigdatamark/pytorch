# [MODULE DOC] Benchmark script comparing per-sample gradient computation of a ResNet-18
# (10 classes, CUDA, batch 128, 32x32 images) between functorch (grad + vmap over a functional
# model) and Opacus (grad_sample hooks). Requires opacus, torchvision and a CUDA device.
# Import-time side effects: builds both models with the same seed, raises an error if their
# parameters differ, creates a PrivacyEngine, runs six timed runs of each method printing
# seconds, and prints the number of functorch gradient tensors. No results are asserted equal
# (see the TODO at the end).
import time

from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from torchvision import models

import torch
import torch.nn as nn
from functorch import grad, make_functional, vmap


device = "cuda"
batch_size = 128
torch.manual_seed(0)

model_functorch = convert_batchnorm_modules(models.resnet18(num_classes=10))
model_functorch = model_functorch.to(device)
criterion = nn.CrossEntropyLoss()

images = torch.randn(batch_size, 3, 32, 32, device=device)
targets = torch.randint(0, 10, (batch_size,), device=device)
func_model, weights = make_functional(model_functorch)


def compute_loss(weights, image, target):
    # [FUNCTION DOC] HELPER: Loss of the functional model for one sample. Adds a batch dimension
    # of size one to image and target, calls the global func_model with weights, and returns the
    # cross-entropy loss tensor.
    images = image.unsqueeze(0)
    targets = target.unsqueeze(0)
    output = func_model(weights, images)
    loss = criterion(output, targets)
    return loss


def functorch_per_sample_grad():
    # [FUNCTION DOC] HELPER: Compute per-sample gradients with functorch: vmap of grad of
    # compute_loss, mapping over the global images and targets (weights shared). Synchronizes
    # CUDA. Returns a tuple of (the gradients result, elapsed wall-clock seconds).
    compute_grad = grad(compute_loss)
    compute_per_sample_grad = vmap(compute_grad, (None, 0, 0))

    start = time.time()
    result = compute_per_sample_grad(weights, images, targets)
    torch.cuda.synchronize()
    end = time.time()

    return result, end - start  # end - start in seconds


torch.manual_seed(0)
model_opacus = convert_batchnorm_modules(models.resnet18(num_classes=10))
model_opacus = model_opacus.to(device)
criterion = nn.CrossEntropyLoss()
for p_f, p_o in zip(model_functorch.parameters(), model_opacus.parameters()):
    if not torch.allclose(p_f, p_o):  # Sanity check
        raise AssertionError("Parameters do not match")

privacy_engine = PrivacyEngine(
    model_opacus,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1,
    max_grad_norm=10000.0,
)


def opacus_per_sample_grad():
    # [FUNCTION DOC] HELPER: Run forward and backward of the Opacus model on the global images
    # and targets, synchronize CUDA, and time it. Collects each parameter's grad_sample into a
    # list, then deletes the grad_sample attribute and clears grad so runs can repeat.
    # Returns (list of per-sample gradients, elapsed seconds).
    start = time.time()
    output = model_opacus(images)
    loss = criterion(output, targets)
    loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    expected = [p.grad_sample for p in model_opacus.parameters()]
    for p in model_opacus.parameters():
        delattr(p, "grad_sample")
        p.grad = None
    return expected, end - start


for _ in range(5):
    _, seconds = functorch_per_sample_grad()
    print(seconds)

result, seconds = functorch_per_sample_grad()
print(seconds)

for _ in range(5):
    _, seconds = opacus_per_sample_grad()
    print(seconds)

expected, seconds = opacus_per_sample_grad()
print(seconds)

result = [r.detach() for r in result]
print(len(result))

# TODO: The following shows that the per-sample-grads computed are different.
# This concerns me a little; we should compare to a source of truth.
# for i, (r, e) in enumerate(list(zip(result, expected))[::-1]):
#     if torch.allclose(r, e, rtol=1e-5):
#         continue
#     print(-(i+1), ((r - e)/(e + 0.000001)).abs().max())
