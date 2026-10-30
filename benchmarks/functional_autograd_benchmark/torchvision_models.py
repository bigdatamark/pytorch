# Taken from https://github.com/pytorch/vision
# So that we don't need torchvision to be installed
# [MODULE DOC] Local copies of the torchvision pieces needed by the functional autograd
# benchmark so torchvision need not be installed: ResNet-18 and ResNet-50, an intermediate
# layer extractor, a fully convolutional segmentation model (FCN with a dilated ResNet-50
# backbone), and a demo DETR detector together with its set criterion, Hungarian matcher and
# box utilities. Pretrained weight loading is stubbed out, so the pretrained flags have no
# effect on weights. Import-time side effect: tries to import the scipy assignment solver
# and records its availability in a module flag; the DETR matcher fails at run time
# without scipy.
from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import Dict
from torch.nn import functional as F


try:
    from scipy.optimize import linear_sum_assignment

    scipy_available = True
except Exception:
    scipy_available = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # [FUNCTION DOC] HELPER: Build a bias-free 3x3 convolution whose padding equals the
    # dilation, with the given stride, groups and dilation. Returns the module.
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    # [FUNCTION DOC] HELPER: Build a bias-free 1x1 convolution with the given stride. Returns
    # the module.
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # [CLASS DOC] Two-convolution residual block used by shallow ResNets (expansion factor one).
    # Applies a 3x3 convolution with batch norm and relu, a second 3x3 convolution with batch
    # norm, adds the identity (or a downsampled identity) and applies relu.
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        # [FUNCTION DOC] HELPER: Create the two convolutions and norm layers (batch norm by
        # default) and store the optional downsample module and stride. Raises a ValueError if
        # groups is not one or the base width is not 64, and a NotImplementedError if dilation
        # is above one.
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # [FUNCTION DOC] Compute the residual output: two conv/norm stages with relu between,
        # plus the identity path (passed through the downsample module if present), followed by
        # relu. The addition is done in place on the branch output.
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # [CLASS DOC] Three-convolution bottleneck residual block (expansion factor four) in the
    # ResNet V1.5 style, where the stride sits in the 3x3 convolution. A 1x1 reduction, a 3x3
    # grouped convolution and a 1x1 expansion each followed by normalization, with a
    # residual connection.
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        # [FUNCTION DOC] HELPER: Compute the inner width from the planes, base width and groups,
        # and create the three convolutions with their norm layers (batch norm by default),
        # relu, optional downsample module and stride.
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # [FUNCTION DOC] Compute the bottleneck residual output: reduce, convolve, expand with
        # normalization and relu in between, add the identity path (downsampled if a module was
        # given) in place, and apply relu.
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # [CLASS DOC] ResNet image classifier assembled from a block type and per-stage block
    # counts: a 7x7 stem with batch norm, relu and max pooling, four stages of residual blocks,
    # global average pooling and a linear classifier. Supports grouped convolutions, optional
    # replacement of stage strides by dilation and zero initialization of the last norm in each
    # residual branch. Convolutions use Kaiming normal initialization and norm layers start at
    # identity scale.
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        # [FUNCTION DOC] HELPER: Build the stem, the four stages through the layer builder, the
        # pooling and the classifier, then initialize weights. Raises a ValueError if the
        # dilation replacement list does not have exactly three entries. If zero-init is
        # requested, the last norm weight of every block is set to zero.
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # [FUNCTION DOC] HELPER: Build one stage as a sequential of blocks. When dilating, the
        # running dilation is multiplied by the stride and the stride becomes one. A projection
        # shortcut (1x1 convolution plus norm) is created when the stride is not one or the
        # channel count changes. The first block receives the shortcut and the previous
        # dilation; the remaining blocks use the new channel count. Updates the running input
        # plane count as a side effect and returns the sequential.
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # [FUNCTION DOC] HELPER: Forward pass: stem, four stages, global average pooling,
        # flatten from dimension one and the final linear layer. Returns class logits.
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        # [FUNCTION DOC] Return class logits by delegating to the shared forward implementation.
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # [FUNCTION DOC] HELPER: Construct a ResNet from a block class and layer counts, forwarding
    # keyword arguments. The architecture name, pretrained flag and progress flag are ignored
    # because loading of pretrained weights is commented out. Returns the new model.
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    # [FUNCTION DOC] Build a ResNet-18 (basic blocks, 2-2-2-2 layout). The pretrained and
    # progress arguments have no effect; extra keyword arguments go to the model constructor.
    # Returns the model.
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    # [FUNCTION DOC] Build a ResNet-50 (bottleneck blocks, 3-4-6-3 layout). The pretrained and
    # progress arguments have no effect; extra keyword arguments go to the model constructor.
    # Returns the model.
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


class IntermediateLayerGetter(nn.ModuleDict):
    # [CLASS DOC] Wrapper that keeps only the children of a model up to the last requested layer
    # and returns the activations of the requested layers as an ordered mapping. Children are
    # executed in registration order, so it assumes each module is used once and in that order,
    # and only direct children can be selected. The mapping from child name to output name is
    # kept as an attribute.
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        # [FUNCTION DOC] HELPER: Check that every requested layer is a direct child (ValueError
        # if not), then copy the children in order into a module dictionary, stopping after the
        # last requested layer. Stores the original return layers mapping.
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        # [FUNCTION DOC] Run the kept children in order on the input and collect, under the
        # configured output names, the activation after each requested layer. Returns an ordered
        # mapping of output name to tensor.
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class _SimpleSegmentationModel(nn.Module):
    # [CLASS DOC] Base class for segmentation models: a backbone returning a mapping of feature
    # maps, a classifier for the main feature map and an optional auxiliary classifier. The
    # output maps are resized back to the input resolution.
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone, classifier, aux_classifier=None):
        # [FUNCTION DOC] HELPER: Store the backbone, classifier and optional
        # auxiliary classifier.
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        # [FUNCTION DOC] Run the backbone, classify the feature map named out, upsample the
        # prediction bilinearly to the input height and width, and put it under the out key of
        # an ordered result. If an auxiliary classifier exists the feature map named aux is
        # processed the same way under the aux key. Returns the ordered mapping.
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result


class FCN(_SimpleSegmentationModel):
    # [CLASS DOC] Fully convolutional network for semantic segmentation. It adds no behavior
    # beyond its base class; the backbone must return a mapping with an out entry and, if an
    # auxiliary classifier is used, an aux entry.
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """


class FCNHead(nn.Sequential):
    # [CLASS DOC] Segmentation head: a sequential of a 3x3 convolution reducing channels to a
    # quarter, batch norm, relu, dropout with probability 0.1 and a 1x1 convolution to the
    # requested channel count.
    def __init__(self, in_channels, channels):
        # [FUNCTION DOC] HELPER: Build the layer list for the head and initialize the sequential
        # base with it.
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    # [FUNCTION DOC] HELPER: Assemble a segmentation model with a ResNet-50 backbone using
    # dilation in the last two stages. Only the name resnet50 is accepted for the backbone
    # (AssertionError otherwise) and only the fcn model name is in the lookup (KeyError
    # otherwise). The backbone is wrapped to return the last stage as the out entry and, if aux
    # is true, the third stage as aux with an auxiliary head of 1024 input channels; the main
    # head takes 2048 channels. Returns the model.
    # backbone = resnet.__dict__[backbone_name](
    #     pretrained=pretrained_backbone,
    #     replace_stride_with_dilation=[False, True, True])
    # Hardcoded resnet 50
    if backbone_name != "resnet50":
        raise AssertionError(
            f"Expected backbone_name='resnet50', but got '{backbone_name}'"
        )
    backbone = resnet50(
        pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True]
    )

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        # 'deeplabv3': (DeepLabHead, DeepLabV3), # Not used
        "fcn": (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(
    arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs
):
    # [FUNCTION DOC] HELPER: Create a segmentation model. When the pretrained flag is set the
    # auxiliary classifier is forced on, but no weights are downloaded or loaded (that code is
    # commented out); the progress flag is unused. Extra keyword arguments are passed to the
    # model builder. Returns the model.
    # KNOWN ISSUE: the pretrained flag only changes whether an auxiliary classifier is created
    # and does not load any weights.
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    # if pretrained:
    #     arch = arch_type + '_' + backbone + '_coco'
    #     model_url = model_urls[arch]
    #     if model_url is None:
    #         raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    #     else:
    #         state_dict = load_state_dict_from_url(model_url, progress=progress)
    #         model.load_state_dict(state_dict)
    return model


def fcn_resnet50(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    # [FUNCTION DOC] Build an FCN with a ResNet-50 backbone for 21 classes by default. The
    # pretrained and progress flags are accepted for API compatibility without loading weights;
    # the auxiliary loss option is None by default (no auxiliary head); keyword arguments such
    # as the backbone pretraining flag are forwarded. Returns the model.
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model(
        "fcn", "resnet50", pretrained, progress, num_classes, aux_loss, **kwargs
    )


# Taken from @fmassa example slides and https://github.com/facebookresearch/detr
class DETR(nn.Module):
    # [CLASS DOC] Minimal DETR object detector: a ResNet-50 backbone without its classifier, a
    # 1x1 convolution to the hidden size, a stock transformer, learned query, row and column
    # position parameters, a linear class head with one extra no-object class and a linear box
    # head. Positional embedding tables hold 50 rows and 50 columns, so feature maps larger than
    # that are not supported.
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        # [FUNCTION DOC] HELPER: Build the backbone (deleting its classifier), the projection,
        # transformer with the given sizes, class and box heads, and the random parameters for
        # 100 object queries and the row and column embeddings.
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # [FUNCTION DOC] Run the detector on an image batch. Features come from the backbone
        # stages, are projected to the hidden size, and combined with a learned two dimensional
        # positional encoding built from column and row embeddings (the features are scaled by
        # 0.1 before adding it). The flattened feature positions feed the transformer encoder
        # and the object queries (expanded across the batch) feed the decoder. Returns a
        # dictionary with class logits of shape batch, 100, classes plus one, and sigmoid box
        # predictions of shape batch, 100, 4.
        # KNOWN ISSUE: the class docstring says only batch size 1 is supported, but the query
        # expansion handles larger batches.
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # propagate through the transformer
        # TODO (alband) Why this is not automatically broadcasted? (had to add the repeat)
        f = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        s = self.query_pos.unsqueeze(1)
        s = s.expand(s.size(0), inputs.size(0), s.size(2))
        h = self.transformer(f, s).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }


def generalized_box_iou(boxes1, boxes2):
    # [FUNCTION DOC] Compute the pairwise generalized IoU matrix between two sets of boxes in
    # corner format (shapes N and M). Raises an AssertionError if either set contains a box
    # whose second corner is below the first. Returns an N by M tensor.
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise AssertionError("boxes1 has invalid box coordinates (x1 < x0 or y1 < y0)")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise AssertionError("boxes2 has invalid box coordinates (x1 < x0 or y1 < y0)")
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    # [FUNCTION DOC] Convert boxes from center x, center y, width, height format (last dimension
    # of size four) to corner format and return the stacked tensor.
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    # [FUNCTION DOC] Return the area of each box in an N by 4 tensor of corner coordinates.
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    # [FUNCTION DOC] Compute pairwise intersection over union between two sets of corner-format
    # boxes (modified from torchvision to also return the union). Returns a pair: the N by M IoU
    # matrix and the N by M union areas.
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def is_dist_avail_and_initialized():
    # [FUNCTION DOC] HELPER: Stub for distributed detection; always returns False.
    return False


def get_world_size():
    # [FUNCTION DOC] HELPER: Return the world size, which is one when distributed training is
    # not available (always, given the stub above).
    if not is_dist_avail_and_initialized():
        return 1


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    # [FUNCTION DOC] Compute top-k precision percentages of logits against integer targets,
    # without gradients. For an empty target it returns a list holding a single zero tensor;
    # otherwise it returns one value per requested k, as a percentage of the batch size.
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    # [CLASS DOC] Loss module for DETR. Matches predictions to targets with the provided matcher
    # and then computes the requested losses (classification, box, cardinality or mask) for the
    # last layer and any auxiliary decoder outputs. Holds the class count, matcher, loss weight
    # mapping, no-object weight and the list of loss names; a buffer gives the no-object class a
    # reduced weight in the classification loss.
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        # [FUNCTION DOC] HELPER: Store the arguments and register the class weight buffer (ones,
        # with the last entry set to the no-object coefficient).
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        # [FUNCTION DOC] Weighted cross entropy over query classes. Matched predictions get
        # their target labels and all others the no-object class. Requires the logits entry in
        # outputs (AssertionError). Returns a dictionary with the cross entropy and, when
        # logging is on, a class error percentage.
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if "pred_logits" not in outputs:
            raise AssertionError("outputs must contain 'pred_logits' key")
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        # [FUNCTION DOC] Compute (without gradients) the absolute difference between the number
        # of predictions not classified as no-object and the number of target boxes per image,
        # averaged over the batch. For logging only. Returns a dictionary with the
        # cardinality error.
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # [FUNCTION DOC] Compute box losses for matched pairs: the summed L1 distance and one
        # minus the generalized IoU, each divided by the number of boxes. Requires the boxes
        # entry in outputs (AssertionError). Returns a dictionary with the two losses.
        # KNOWN ISSUE: the docstring names the target format as center x, center y, height,
        # width, but the conversion function used treats the last two values as width
        # then height.
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise AssertionError("outputs must contain 'pred_boxes' key")
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        # [FUNCTION DOC] Compute mask losses (focal and dice) for matched pairs. Requires the
        # masks entry in outputs (AssertionError).
        # KNOWN ISSUE: the helper functions it calls (nested tensor builder, interpolation,
        # focal loss, dice loss) are not defined or imported in this file, so calling it raises
        # a NameError; masks are never used in the benchmark.
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if "pred_masks" not in outputs:
            raise AssertionError("outputs must contain 'pred_masks' key")

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(  # noqa: F821
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(  # noqa: F821
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(  # noqa: F821
                src_masks, target_masks, num_boxes
            ),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),  # noqa: F821
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # [FUNCTION DOC] HELPER: Return a pair of tensors (batch index per matched prediction,
        # prediction index) built by concatenating the matcher's source indices over the batch.
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # [FUNCTION DOC] HELPER: Return a pair of tensors (batch index per matched target,
        # target index) built by concatenating the matcher's target indices over the batch.
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # [FUNCTION DOC] Dispatch to one loss function by name from a fixed table of labels,
        # cardinality, boxes and masks. Raises an AssertionError for an unknown name. Extra
        # keyword arguments are forwarded. Returns the dictionary of losses.
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise AssertionError(
                f"Unknown loss type '{loss}'. Available: {list(loss_map.keys())}"
            )
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        # [FUNCTION DOC] Compute all configured losses. Auxiliary outputs are excluded from
        # matching, the matcher gives the assignment, and the number of target boxes is clamped
        # to at least one (and averaged over workers, which is a single worker here). Each
        # requested loss is evaluated and the dictionaries merged. For every auxiliary decoder
        # output the matching is repeated and the losses are added with an index suffix on each
        # key, skipping mask losses and disabling the classification logging. Returns the
        # dictionary of losses.
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcher(nn.Module):
    # [CLASS DOC] Computes a one-to-one assignment between predicted queries and ground-truth
    # objects by minimizing a weighted sum of classification, L1 box and generalized IoU costs.
    # Unmatched predictions are treated as no-object. The assignment itself uses the scipy
    # linear sum assignment solver.
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        # [FUNCTION DOC] HELPER: Store the three cost weights. Raises an AssertionError if all
        # three are zero.
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise AssertionError(
                "At least one of cost_class, cost_bbox, or cost_giou must be non-zero"
            )

    @torch.no_grad()
    def forward(self, outputs, targets):
        # [FUNCTION DOC] Compute the assignment for a batch, without gradients. The class scores
        # of all queries are flattened across the batch, costs against the concatenated targets
        # are computed for all pairs (negative class probability, L1 distance of boxes, negative
        # generalized IoU), combined with the weights, reshaped per image and split by target
        # counts. For each image the scipy solver finds the optimal matching. Returns a list
        # with one pair of int64 index tensors (predictions, targets) per image, each of length
        # equal to the smaller of the query and target counts. Raises a RuntimeError if scipy is
        # not available (checked after the cost computation).
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        if not scipy_available:
            raise RuntimeError(
                "The 'detr' model requires scipy to run. Please make sure you have it installed"
                " if you enable the 'detr' model."
            )
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
