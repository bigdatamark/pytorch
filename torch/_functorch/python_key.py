# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC] Backward-compatibility shim re-exporting make_fx, dispatch_trace and
# PythonKeyTracer from torch.fx.experimental.proxy_tensor, and aliasing decompose as
# pythonkey_decompose. No logic of its own.
__all__ = ["make_fx", "dispatch_trace", "PythonKeyTracer", "pythonkey_decompose"]
from torch.fx.experimental.proxy_tensor import (
    decompose,
    dispatch_trace,
    make_fx,
    PythonKeyTracer,
)


pythonkey_decompose = decompose
