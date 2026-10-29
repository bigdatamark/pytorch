# mypy: ignore-errors
# [MODULE DOC] Manual script (not a unit test) that trains an MNIST convnet on the lazy tensor
# TorchScript backend. Requires torchvision. At import time it initializes the lazy ts backend.
# When run as main it builds a DataLoader over MNIST (downloaded to ./data), moves a Net to the
# lazy device, trains with Adadelta and a StepLR schedule for 14 epochs, and calls mark_step per
# batch. Setting the LTC_TS_CUDA environment variable adds worker, pin-memory and shuffle options.

import os

from torchvision import datasets, transforms

import torch
import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


torch._lazy.ts_backend.init()


class Net(nn.Module):
    # [CLASS DOC] Small MNIST convolutional classifier: two conv layers, max pooling, two dropouts and
    # two linear layers (9216 -> 128 -> 10), producing log-probabilities over 10 classes.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the conv, dropout and linear submodules used by forward.
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # [FUNCTION DOC] Runs the forward pass on a batch of single-channel 28x28 images x and returns
        # log-softmax class scores of shape (batch, 10).
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # [FUNCTION DOC] Trains the model for one epoch over train_loader: for each batch moves data to device,
    # zeroes grads, computes nll loss, backpropagates, steps the optimizer and calls the lazy mark_step.
    # Prints loss every log_interval batches. Mutates model and optimizer state; returns nothing.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        torch._lazy.mark_step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


if __name__ == "__main__":
    bsz = 64
    device = "lazy"
    epochs = 14
    log_interval = 10
    lr = 1
    gamma = 0.7
    train_kwargs = {"batch_size": bsz}
    # if we want to use CUDA
    if "LTC_TS_CUDA" in os.environ:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
            "batch_size": bsz,
        }
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        scheduler.step()
