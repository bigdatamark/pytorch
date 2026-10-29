# [PACKAGE DOC] Custom operator library internals: imports autograd, fake_impl, simple_registry and utils
# submodules and re-exports register_fake_class, capture_triton, triton_op and wrap_triton.
import torch._library.autograd
import torch._library.fake_impl
import torch._library.simple_registry
import torch._library.utils
from torch._library.fake_class_registry import register_fake_class
from torch._library.triton import capture_triton, triton_op, wrap_triton
