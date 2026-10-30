# [MODULE DOC] Reveal-type sample checking that the adaptive average and max pooling 2D modules
# accept an output size containing None in either position (None in the second or first slot),
# and that the resulting variable types are revealed as AdaptiveAvgPool2d and AdaptiveMaxPool2d
# respectively, per the expected-output markers.
import torch


avg_pool1 = torch.nn.AdaptiveAvgPool2d((1, None))
reveal_type(avg_pool1)  # E: {AdaptiveAvgPool2d}
avg_pool2 = torch.nn.AdaptiveAvgPool2d((None, 1))
reveal_type(avg_pool2)  # E: {AdaptiveAvgPool2d}
max_pool1 = torch.nn.AdaptiveMaxPool2d((1, None))
reveal_type(max_pool1)  # E: {AdaptiveMaxPool2d}
max_pool2 = torch.nn.AdaptiveMaxPool2d((None, 1))
reveal_type(max_pool2)  # E: {AdaptiveMaxPool2d}
