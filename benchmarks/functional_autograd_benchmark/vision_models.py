# [MODULE DOC] Getter functions for the functional autograd benchmark vision models. Each getter
# builds a model (resnet18, an FCN resnet50, or DETR from the local torchvision_models module),
# optionally swaps batch norm for functorch-friendly versions, extracts the weights, and returns
# a pure loss function of the parameters together with the parameter tuple. Import-time side
# effect: probes for functorch and stores the result in has_functorch.
from typing import cast

import torchvision_models as models
from utils import check_for_functorch, extract_weights, GetterReturnType, load_weights

import torch
from torch import Tensor


has_functorch = check_for_functorch()


def get_resnet18(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build the resnet18 benchmark case on the given device. Creates an untrained
    # resnet18, replaces batch norm modules if functorch is available, moves the model to the
    # device and extracts its weights. Inputs are a random batch of 32 images (3x224x224) and
    # random integer class labels in 0..9 (random floats scaled by 10 then cast to long).
    # Returns a pair of (function, params); the function takes the parameters, reloads them into
    # the model, runs the forward pass and returns the cross entropy loss as a scalar tensor.
    N = 32
    model = models.resnet18(pretrained=False)

    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_

        replace_all_batch_norm_modules_(model)

    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand([N, 3, 224, 224], device=device)
    labels = torch.rand(N, device=device).mul(10).long()

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Closure over the model, inputs, labels and criterion. Reloads
        # the supplied parameter tensors into the model, runs the batch through it and returns
        # the cross entropy loss tensor.
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    return forward, params


def get_fcn_resnet(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build the FCN-resnet50 segmentation benchmark case on the given device.
    # Uses an untrained model without pretrained backbone, batch size 8. If functorch is
    # available, batch norm modules are replaced and the model is put in eval mode (the eval
    # call is inside the functorch branch, so without functorch dropout stays active). Inputs
    # are random 8x3x480x480 images and random dense targets of shape 8x21x480x480. Returns the
    # loss function (mean squared error on the "out" entry of the model output) and
    # the parameters.
    N = 8
    criterion = torch.nn.MSELoss()
    model = models.fcn_resnet50(pretrained=False, pretrained_backbone=False)

    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_

        replace_all_batch_norm_modules_(model)
        # disable dropout for consistency checking
        model.eval()

    model.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand([N, 3, 480, 480], device=device)
    # Given model has 21 classes
    labels = torch.rand([N, 21, 480, 480], device=device)

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Closure that reloads the supplied parameters into the model,
        # runs the model on the fixed inputs, and returns the mean squared error between the
        # "out" entry of the output and the fixed random targets.
        load_weights(model, names, new_params)
        out = model(inputs)["out"]

        loss = criterion(out, labels)
        return loss

    return forward, params


def get_detr(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build the DETR object detection benchmark case on the given device. The
    # hyperparameters (91 classes, hidden size 256, 8 heads, 6 encoder and 6 decoder layers) are
    # the CLI defaults of the original DETR project. Builds the model, a Hungarian matcher and a
    # set criterion with label, box and cardinality losses, batch size 2, and random 800x1200
    # inputs. Per-image targets are random labels and boxes with 5 to 9 objects each; box
    # coordinates are swapped where needed so minimum corners come first. Returns the weighted
    # loss function and the extracted parameters. Uses torch random numbers (not seeded here).
    # All values below are from CLI defaults in https://github.com/facebookresearch/detr
    N = 2
    num_classes = 91
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    model = models.DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_

        replace_all_batch_norm_modules_(model)

    losses = ["labels", "boxes", "cardinality"]
    eos_coef = 0.1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    weight_dict = {
        "loss_ce": 1,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }
    matcher = models.HungarianMatcher(1, 5, 2)
    criterion = models.SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    )

    model = model.to(device)
    criterion = criterion.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand(N, 3, 800, 1200, device=device)
    labels = []
    for idx in range(N):
        targets = {}
        n_targets: int = int(torch.randint(5, 10, size=()).item())
        label = torch.randint(5, 10, size=(n_targets,), device=device)
        targets["labels"] = label
        boxes = torch.randint(100, 800, size=(n_targets, 4), device=device)
        for t in range(n_targets):
            if boxes[t, 0] > boxes[t, 2]:
                boxes[t, 0], boxes[t, 2] = boxes[t, 2], boxes[t, 0]
            if boxes[t, 1] > boxes[t, 3]:
                boxes[t, 1], boxes[t, 3] = boxes[t, 3], boxes[t, 1]
        targets["boxes"] = boxes.float()
        labels.append(targets)

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Closure that reloads the supplied parameters, runs the model on
        # the fixed inputs, evaluates the set criterion against the fixed targets and returns
        # the sum of each loss term multiplied by its weight (only terms present in the weight
        # mapping count).
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        weight_dict = criterion.weight_dict
        final_loss = cast(
            Tensor,
            sum(loss[k] * weight_dict[k] for k in loss if k in weight_dict),
        )
        return final_loss

    return forward, params
