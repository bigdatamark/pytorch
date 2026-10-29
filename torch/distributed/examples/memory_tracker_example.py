# mypy: allow-untyped-defs
# [MODULE DOC] Example script showing how to use MemoryTracker on one training iteration of a model (resnet34 on random CUDA input when run as main). Importing has no side effects beyond imports; running as main requires torchvision and a CUDA device.
import torch
from torch.distributed._tools import MemoryTracker


def run_one_model(net: torch.nn.Module, input: torch.Tensor, device: str = "cuda"):
    # [FUNCTION DOC] Move the network and input to the device, start a MemoryTracker monitor on the network, run one forward/backward pass (the output is the model output, or its out entry when a dict; summed to a scalar for backward), clear gradients, then stop the tracker, print its summary and show operator-level memory traces. Returns None; side effects are printing and plotting.
    net.to(device)
    input = input.to(device)

    # Create the memory Tracker
    mem_tracker = MemoryTracker()
    # start_monitor before the training iteration starts
    mem_tracker.start_monitor(net)

    # run one training iteration
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss["out"]
    loss.sum().backward()
    net.zero_grad(set_to_none=True)

    # stop monitoring after the training iteration ends
    mem_tracker.stop()
    # print the memory stats summary
    mem_tracker.summary()
    # plot the memory traces at operator level
    mem_tracker.show_traces()


if __name__ == "__main__":
    import torchvision

    dev = "cuda"
    run_one_model(
        torchvision.models.resnet34(),
        torch.rand(32, 3, 224, 224, device=dev),
        device=dev,
    )
