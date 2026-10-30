# [MODULE DOC] Script that profiles memory use of one forward pass of a freshly built ResNet-18
# on a random batch of five images, first on cpu and then, if cuda is available, on gpu (without
# cuda, the second pass repeats the cpu run). Prints a table of averaged profiler events grouped
# by input shape, sorted by self memory usage, with no row limit. Everything runs at import
# time; there are no functions or classes.
from torchvision import models

import torch
import torch.autograd.profiler as profiler


for with_cuda in [False, True]:
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    sort_key = "self_cpu_memory_usage"
    if with_cuda and torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        sort_key = "self_cuda_memory_usage"
        print("Profiling CUDA Resnet model")
    else:
        print("Profiling CPU Resnet model")

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with profiler.record_function("root"):
            model(inputs)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key, row_limit=-1
        )
    )
