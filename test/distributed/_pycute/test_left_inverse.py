# ruff: noqa: PGH004, G004, F403
# flake8: noqa
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Unit tests for the pycute left_inverse function. One TestCase with a round-trip
# helper and one test over several layouts. Star-imports the pycute package; creates a logger.
#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Unit tests for _pycute.left_inverse
"""

import logging

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


_LOGGER = logging.getLogger(__name__)


class TestLeftInverse(TestCase):
    # [CLASS DOC] Test case for left_inverse; stateless, one helper and one test.
    def helper_test_left_inverse(self, layout):
        # [FUNCTION DOC] HELPER: Computes the left inverse of the layout and asserts that
        # applying the inverse to the layout's output for every index in the layout's size
        # recovers that index. Logs the pair at debug level.
        inv_layout = left_inverse(layout)

        _LOGGER.debug(f"{layout}  =>  {inv_layout}")

        for i in range(size(layout)):
            self.assertEqual(inv_layout(layout(i)), i)

    def test_left_inverse(self):
        # [FUNCTION DOC] Runs the round-trip check on size-1, flat strided, and multi-mode
        # layouts with column-major, row-major and gapped strides.
        test = Layout(1, 0)
        self.helper_test_left_inverse(test)

        test = Layout((1, 1), (0, 0))
        self.helper_test_left_inverse(test)

        test = Layout(1, 1)
        self.helper_test_left_inverse(test)

        test = Layout(4, 1)
        self.helper_test_left_inverse(test)

        test = Layout(4, 2)
        self.helper_test_left_inverse(test)

        test = Layout((8, 4), (1, 8))
        self.helper_test_left_inverse(test)

        test = Layout((8, 4), (4, 1))
        self.helper_test_left_inverse(test)

        test = Layout((2, 4, 6), (1, 2, 8))
        self.helper_test_left_inverse(test)

        test = Layout((2, 4, 6), (4, 1, 8))
        self.helper_test_left_inverse(test)

        test = Layout((4, 2), (1, 16))
        self.helper_test_left_inverse(test)


if __name__ == "__main__":
    run_tests()
