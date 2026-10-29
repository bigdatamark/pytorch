# [PACKAGE DOC]
# torch.signal: signal-processing utilities. Currently exports only the windows subpackage
# (window functions such as hann, hamming, kaiser). Importing it has no other side effects.
from . import windows


__all__ = ["windows"]
