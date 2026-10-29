# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments
# [MODULE DOC] Reruns the CUDA graph trees test suite with the CUDA caching allocator
# configured for expandable segments. It imports the existing cudagraph trees test case
# class (so its tests are collected here) and, when run as a script on a CUDA machine that
# is not Jetson or Windows and has Triton, it downloads the disabled-tests list, enables
# the expandable_segments allocator setting and then runs the tests. The repo root is
# temporarily put on the import path only to import the disabled-tests helper.

import os
import pathlib
import sys

import torch
from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if HAS_CUDA_AND_TRITON:
    try:
        from .test_cudagraph_trees import CudaGraphTreeTests
    except ImportError:
        from test_cudagraph_trees import (  # noqa: F401  # @manual=fbcode//caffe2/test/inductor:cudagraph_trees-library
            CudaGraphTreeTests,
        )

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.stats.import_test_stats import get_disabled_tests  # @manual


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if (
        torch.cuda.is_available()
        and not IS_JETSON
        and not IS_WINDOWS
        and HAS_CUDA_AND_TRITON
    ):
        get_disabled_tests(".")

        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        run_tests()
