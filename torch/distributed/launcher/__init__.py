#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# [PACKAGE DOC] Package for the torchrun/elastic launcher API. Re-exports
# elastic_launch, launch_agent and LaunchConfig from
# torch.distributed.launcher.api.
from torch.distributed.launcher.api import elastic_launch, launch_agent, LaunchConfig
