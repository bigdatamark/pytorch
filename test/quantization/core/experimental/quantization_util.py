# [MODULE DOC] Helper utilities for the APoT PTQ/QAT example scripts: an average meter,
# top-k accuracy, model evaluation, loading/size-reporting of ResNet-18, ImageNet data
# loader construction and a simple Adam training loop. Import-time side effects: sets
# warning filters (ignore DeprecationWarning globally, show warnings from the quantization
# package), seeds torch's RNG, and defines the global train and eval batch sizes.
import torch
import torchvision
import torchvision.transforms.transforms as transforms
import os
import torch.ao.quantization
from torchvision.models.quantization.resnet import resnet18
from torch.autograd import Variable

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

"""
Define helper functions for APoT PTQ and QAT
"""

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

train_batch_size = 30
eval_batch_size = 50

class AverageMeter:
    # [CLASS DOC] Running-statistics holder: tracks the latest value, the weighted sum, the
    # count and the running average for one named metric, plus a format string used when
    # printing. Call reset to zero it and update to add observations.
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        # [FUNCTION DOC] Stores the metric name and format spec, then resets the counters.
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        # [FUNCTION DOC] Zeroes the latest value, average, sum and count.
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # [FUNCTION DOC] Records val as the latest value, adds val times n to the sum and n to the
        # count, and recomputes the average as sum over count. Raises ZeroDivisionError if the
        # total count is still zero (first call with n equal to zero).
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # [FUNCTION DOC] Formats the name, latest value and average using the stored format spec
        # by filling a template with the instance attribute dictionary.
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    # [FUNCTION DOC] For logits output and integer class targets, returns a list with one
    # one-element tensor per k in topk: the percentage of samples whose true class is among
    # the k highest-scoring predictions. Runs without gradients.
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    # [FUNCTION DOC] Puts the model in eval mode and iterates the loader without gradients,
    # accumulating top-1 and top-5 accuracy in two AverageMeter objects weighted by batch
    # size (the loss is computed but unused). Prints an empty line and returns the two meters.
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)

            loss = criterion(output, target)  # noqa: F841
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print()

    return top1, top5

def load_model(model_file):
    # [FUNCTION DOC] Builds an untrained ResNet-18, loads the state dict stored at the given
    # file path, moves the model to CPU and returns it.
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    # [FUNCTION DOC] Reports serialized model size in megabytes: saves the module (scripting it
    # first unless already a recursive script module) to a temporary file in the current
    # directory, prints its size, then deletes the file.
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):
# [FUNCTION DOC] Builds ImageNet train (random resized crop, flip, normalization) and
# validation (resize, center crop, normalization) datasets under data_path, with a random
# sampler for training and sequential sampler for validation. Returns a tuple of the train
# and validation data loaders using the global batch sizes.

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(data_path,
                                            split="train",
                                            transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          normalize]))
    dataset_test = torchvision.datasets.ImageNet(data_path,
                                                 split="val",
                                                 transform=transforms.Compose([transforms.Resize(256),
                                                                               transforms.CenterCrop(224),
                                                                               transforms.ToTensor(),
                                                                               normalize]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

def training_loop(model, criterion, data_loader):
    # [FUNCTION DOC] Trains the model for ten passes over the loader using Adam with learning
    # rate 0.001, in train mode. Each step wraps the loss in a fresh variable requiring grad
    # before backward. Returns the accumulated loss sum, number of correct predictions and
    # number of samples seen.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    model.train()
    for _ in range(10):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total
